#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "Pillow>=10.0.0",
#   "numpy>=1.26.0",
#   "rembg>=2.0.0",
#   "onnxruntime>=1.17.0",
# ]
#
# [tool.uv]
# override-dependencies = ["numba>=0.60.0"]
# ///
"""
Muninn Logo Post-Processor — deterministic image pipeline.

Takes a GenAI-generated raven icon and applies deterministic edits:
  - Background removal (HuggingFace segmentation model)
  - Text overlay with custom fonts, colour, opacity (PIL)
  - Futhark rune watermark overlay
  - Composite assembly (icon + wordmark → final logo)

This saves API costs by doing text/overlay work locally instead of
re-generating entire images for typographic changes.

Usage:
    # Remove background from a generated image
    uv run scripts/process_logo.py remove-bg docs/logo/muninn_20260211_235216_0.png

    # Add "MUNINN" wordmark to the right of the icon
    uv run scripts/process_logo.py wordmark docs/logo/muninn_20260211_235216_0.png

    # Add Futhark rune overlay at low opacity
    uv run scripts/process_logo.py runes docs/logo/muninn_20260211_235216_0.png

    # Full pipeline: remove bg → add wordmark → save
    uv run scripts/process_logo.py composite docs/logo/muninn_20260211_235216_0.png

Auth:
    No API keys needed — runs entirely offline.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rembg import remove as rembg_remove

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

SCRIPT = Path(__file__)
SCRIPT_NAME = SCRIPT.stem
SCRIPT_DIR = SCRIPT.parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_ROOT / "docs" / "logo"
DEFAULT_CONFIG_FILE = SCRIPT_DIR / "logo_config.json"

# Background removal via rembg (U2-Net, no auth required)

# Brand colours from docs/plans/brand_identity.md
CHARCOAL = (43, 45, 66)  # #2B2D42
AMBER_GOLD = (244, 162, 97)  # #F4A261
SLATE_GREY = (85, 91, 110)  # #555B6E

# Elder Futhark runes for "Muninn"
FUTHARK_MUNINN = "ᛗᚢᚾᛁᚾᚾ"

# Default font (macOS system font, override with --font)
DEFAULT_FONT = "/System/Library/Fonts/Helvetica.ttc"
FALLBACK_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
    "C:/Windows/Fonts/arial.ttf",  # Windows
]

# Anchor name → (horizontal_factor, vertical_factor) for positioning
ANCHORS = {
    "top_left": (0.0, 0.0),
    "top_center": (0.5, 0.0),
    "top_right": (1.0, 0.0),
    "center_left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "center_right": (1.0, 0.5),
    "bottom_left": (0.0, 1.0),
    "bottom_center": (0.5, 1.0),
    "bottom_right": (1.0, 1.0),
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: Path | None = None) -> dict:
    """Load JSON config, falling back to defaults if not found."""
    path = config_path or DEFAULT_CONFIG_FILE
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
        log.info("Loaded config from %s", path)
        return cfg
    log.debug("No config file at %s, using built-in defaults", path)
    return {}


def resolve_position(
    pos: tuple[float, float],
    anchor: str,
    canvas_w: int,
    canvas_h: int,
    content_w: int,
    content_h: int,
) -> tuple[int, int]:
    """Convert normalized (0.0-1.0) position + anchor → absolute top-left pixel coords.

    pos=(0.0, 0.0) → top-left, (0.5, 0.5) → centred, (1.0, 1.0) → bottom-right.
    Anchor determines which point of the content sits at that position.
    """
    ax, ay = ANCHORS.get(anchor, (0.0, 0.0))
    abs_x = int(pos[0] * canvas_w) - int(ax * content_w)
    abs_y = int(pos[1] * canvas_h) - int(ay * content_h)
    return (abs_x, abs_y)


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------


def load_font(font_path: str | None, size: int, weight: int | None = None) -> ImageFont.FreeTypeFont:
    """Load a TrueType font, with fallback chain."""
    candidates = [font_path] if font_path else []
    candidates.extend([DEFAULT_FONT] + FALLBACK_FONTS)

    for path in candidates:
        if path and Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                if weight is not None and hasattr(font, "set_variation_by_axes"):
                    font.set_variation_by_axes([weight])
                log.info("Loaded font: %s (size=%d)", path, size)
                return font
            except Exception as e:
                log.debug("Font %s failed: %s", path, e)
                continue

    log.warning("No TrueType fonts found, using default bitmap font")
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------


def _auto_crop(arr: np.ndarray, pad: int = 4) -> np.ndarray:
    """Crop RGBA array to the bounding box of non-transparent pixels."""
    h, w = arr.shape[:2]
    alpha = arr[:, :, 3]
    fg_rows = np.any(alpha > 10, axis=1)
    fg_cols = np.any(alpha > 10, axis=0)
    if np.any(fg_rows) and np.any(fg_cols):
        rmin, rmax = np.where(fg_rows)[0][[0, -1]]
        cmin, cmax = np.where(fg_cols)[0][[0, -1]]
        rmin = max(0, rmin - pad)
        rmax = min(h - 1, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(w - 1, cmax + pad)
        arr = arr[rmin : rmax + 1, cmin : cmax + 1]
        log.info("Auto-cropped to [%d:%d, %d:%d] → %dx%d", rmin, rmax + 1, cmin, cmax + 1, arr.shape[1], arr.shape[0])
    return arr


def remove_background(
    input_path: Path,
    output_path: Path | None = None,
    tolerance: int = 30,
    edge_softness: int = 2,
    dark_border_threshold: int = 40,
    mode: str = "model",
) -> Path:
    """Remove background from an image, producing a transparent PNG.

    Three modes:
      - "model": U2-Net segmentation via rembg. Semantic foreground/background
        separation — preserves white details in the subject (eyes, highlights).
      - "global": Threshold all near-white pixels regardless of position.
        Fast, no model download, but removes ALL white incl. subject details.
      - "flood": Flood-fill from image edges only. Better for photos where
        interior white areas should be preserved.

    Args:
        tolerance: (global/flood) How far from white (per-channel, 0-255) counts as bg.
        edge_softness: Pixels of alpha gradient at foreground/background boundary.
        dark_border_threshold: (flood mode only) Max RGB for dark border pixels.
        mode: "model", "global", or "flood".
    """
    img = Image.open(input_path).convert("RGBA")
    log.info(
        "Removing background from %s (%dx%d) mode=%s tolerance=%d", input_path, img.width, img.height, mode, tolerance
    )

    arr = np.array(img)
    h, w = arr.shape[:2]

    if mode == "model":
        log.info("Running U2-Net segmentation (first run downloads ~170MB model)...")
        result_img = rembg_remove(img)
        arr = np.array(result_img.convert("RGBA"))

        if edge_softness > 0:
            alpha = arr[:, :, 3]
            alpha_img = Image.fromarray(alpha, mode="L")
            alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=edge_softness))
            arr[:, :, 3] = np.array(alpha_img)

        fg_count = int(np.sum(arr[:, :, 3] > 128))
        log.info("Model segmentation: %d/%d foreground pixels (%.1f%%)", fg_count, h * w, 100.0 * fg_count / (h * w))

        arr = _auto_crop(arr)

        result = Image.fromarray(arr)
        if output_path is None:
            output_path = input_path.with_stem(input_path.stem + "_nobg")
        result.save(output_path)
        log.info("Saved: %s (%dx%d)", output_path, result.width, result.height)
        return output_path

    if mode == "global":
        rgb = arr[:, :, :3].astype(np.float32)
        # Per-pixel max channel distance from white (Chebyshev distance).
        # Pure white → 0, pure black → 255.
        max_dist = np.max(255.0 - rgb, axis=2)

        # Smooth alpha ramp: fully transparent below tolerance, fully opaque
        # above tolerance + transition, linear blend in between.
        tol = float(tolerance)
        transition = max(float(edge_softness * 5), 1.0)
        alpha_float = np.clip((max_dist - tol) / transition, 0.0, 1.0)
        alpha = (alpha_float * 255).astype(np.uint8)

        if edge_softness > 0:
            alpha_img = Image.fromarray(alpha, mode="L")
            alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=edge_softness))
            alpha = np.array(alpha_img)

        bg_count = int(np.sum(alpha < 128))
        log.info("Global threshold: %d/%d pixels transparent (%.1f%%)", bg_count, h * w, 100.0 * bg_count / (h * w))

        arr[:, :, 3] = alpha
        arr = _auto_crop(arr)

        result = Image.fromarray(arr)
        if output_path is None:
            output_path = input_path.with_stem(input_path.stem + "_nobg")
        result.save(output_path)
        log.info("Saved: %s (%dx%d)", output_path, result.width, result.height)
        return output_path

    # --- Flood mode (original algorithm) ---
    # Classify pixels
    rgb = arr[:, :, :3].astype(np.int16)
    is_white = np.all(rgb >= (255 - tolerance), axis=2)
    is_dark = np.all(rgb <= dark_border_threshold, axis=2)

    visited = np.zeros((h, w), dtype=bool)
    bg_mask = np.zeros((h, w), dtype=bool)

    # --- Pass 1: Flood dark border from corners ---
    # Images with rounded-corner borders have dark (near-black) corners.
    # This pass peels away that border layer.
    corner_seeds = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    dark_stack = []
    for r, c in corner_seeds:
        if is_dark[r, c] and not visited[r, c]:
            visited[r, c] = True
            dark_stack.append((r, c))

    # Also seed from edges (catches non-square borders)
    for x in range(0, w, 4):
        for row in [0, h - 1]:
            if is_dark[row, x] and not visited[row, x]:
                visited[row, x] = True
                dark_stack.append((row, x))
    for y in range(0, h, 4):
        for col in [0, w - 1]:
            if is_dark[y, col] and not visited[y, col]:
                visited[y, col] = True
                dark_stack.append((y, col))

    # Collect white pixels at the border boundary — these seed pass 2
    border_boundary = []

    while dark_stack:
        r, c = dark_stack.pop()
        bg_mask[r, c] = True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                if is_dark[nr, nc]:
                    visited[nr, nc] = True
                    dark_stack.append((nr, nc))
                elif is_white[nr, nc]:
                    # Don't mark visited yet — pass 2 will process these
                    border_boundary.append((nr, nc))

    dark_pixels = int(bg_mask.sum())
    log.info("Pass 1 (dark border): removed %d pixels", dark_pixels)

    # --- Pass 2: Flood white background ---
    # Seeds come from three sources:
    #   1. Border boundary — white pixels where dark border meets white bg
    #   2. Edge pixels that are already white (no dark border on that edge)
    #   3. Inset corners — safety net, 50px in from each corner
    white_stack = []

    # Source 1: border boundary (most important — bridges the two passes)
    for r, c in border_boundary:
        if not visited[r, c]:
            visited[r, c] = True
            white_stack.append((r, c))

    # Source 2: edge pixels that are white
    for x in range(0, w, 2):
        for row in [0, h - 1]:
            if is_white[row, x] and not visited[row, x]:
                visited[row, x] = True
                white_stack.append((row, x))
    for y in range(0, h, 2):
        for col in [0, w - 1]:
            if is_white[y, col] and not visited[y, col]:
                visited[y, col] = True
                white_stack.append((y, col))

    # Source 3: inset corners (past any border radius)
    inset = 50
    for r, c in [(inset, inset), (inset, w - 1 - inset), (h - 1 - inset, inset), (h - 1 - inset, w - 1 - inset)]:
        if 0 <= r < h and 0 <= c < w and is_white[r, c] and not visited[r, c]:
            visited[r, c] = True
            white_stack.append((r, c))

    log.debug("Pass 2 seeds: %d border_boundary, total %d white seeds", len(border_boundary), len(white_stack))

    while white_stack:
        r, c = white_stack.pop()
        bg_mask[r, c] = True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and is_white[nr, nc]:
                visited[nr, nc] = True
                white_stack.append((nr, nc))

    white_pixels = int(bg_mask.sum()) - dark_pixels
    total_pixels = h * w
    log.info("Pass 2 (white bg): removed %d pixels", white_pixels)

    # --- Pass 3: Edge-margin cleanup ---
    # Anti-aliased borders create a gradient fringe (e.g. RGB 24,24,36) that
    # falls between the dark and white thresholds. Iteratively dilate bg_mask
    # within the border margin to eat this fringe without touching the subject.
    border_margin = 60
    edge_zone = np.zeros((h, w), dtype=bool)
    edge_zone[:border_margin, :] = True
    edge_zone[-border_margin:, :] = True
    edge_zone[:, :border_margin] = True
    edge_zone[:, -border_margin:] = True

    pre_cleanup = int(bg_mask.sum())
    for _ in range(5):
        mask_img = Image.fromarray(bg_mask.astype(np.uint8) * 255, mode="L")
        dilated = mask_img.filter(ImageFilter.MaxFilter(size=3))
        dilated_arr = np.array(dilated) > 128
        bg_mask = bg_mask | (dilated_arr & edge_zone)
    cleanup_pixels = int(bg_mask.sum()) - pre_cleanup
    log.info("Pass 3 (edge cleanup): removed %d fringe pixels", cleanup_pixels)

    log.info(
        "Total background: %d/%d pixels (%.1f%%)", int(bg_mask.sum()), total_pixels, 100 * bg_mask.sum() / total_pixels
    )

    # Create alpha channel: 0 for background, 255 for foreground
    alpha = np.where(bg_mask, 0, 255).astype(np.uint8)

    # Soften edges with a blur on the alpha boundary
    if edge_softness > 0:
        alpha_img = Image.fromarray(alpha, mode="L")
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=edge_softness))
        alpha = np.array(alpha_img)
        # Re-clamp: fully bg stays 0, fully fg stays 255, only edges get softened
        alpha = np.where(bg_mask & (alpha < 128), 0, alpha)

    arr[:, :, 3] = alpha
    arr = _auto_crop(arr)

    result = Image.fromarray(arr)

    if output_path is None:
        output_path = input_path.with_stem(input_path.stem + "_nobg")
    result.save(output_path)
    log.info("Saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Layer segmentation
# ---------------------------------------------------------------------------


def segment_layers(
    input_path: Path,
    output_dir: Path,
    white_tolerance: int = 15,
    edge_softness: int = 2,
    grey_reference: int = 30,
    sharpen: int = 12,
) -> Path:
    """Segment image into layers, save intermediates, produce final transparent PNG.

    Combines two signals to decide what stays:
      1. U2-Net semantic mask — identifies the main subject (raven, graph).
         Preserves white subject details like eyes that color thresholds miss.
      2. Color distance from white — identifies ANY non-white content (grey
         lines, runes, design elements) that U2-Net considers "background."

    Final alpha = max(u2net_signal, color_signal).  Only pixels that are BOTH
    semantically background AND near-pure-white become transparent.

    Intermediate outputs (numbered for inspection order):
      01_u2net_mask.png      — U2-Net foreground probability (grayscale)
      02_color_distance.png  — Per-pixel distance from white (grayscale)
      03_edge_scharr.png     — Scharr edge magnitude (4× amplified)
      03b_edge_dilated.png   — Edges dilated (spatial envelope)
      03c_edge_refined.png   — Edge mask × color alpha (grey→opaque, white→transparent)
      04_fg_semantic.png     — Original masked by U2-Net only
      05_fg_color.png        — Original masked by color distance only
      06_fg_edge.png         — Original masked by edge detection only
      07_combined_mask.png   — Union of all three signals
      08_final.png           — Final transparent image, auto-cropped
    """
    img = Image.open(input_path).convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Segmenting %s (%dx%d) into layers → %s", input_path, w, h, output_dir)

    # --- Layer 1: U2-Net semantic foreground ---
    log.info("Running U2-Net segmentation...")
    fg_result = rembg_remove(img)
    fg_arr = np.array(fg_result.convert("RGBA"))
    u2net_alpha = fg_arr[:, :, 3].astype(np.float32) / 255.0

    u2net_mask_path = output_dir / "01_u2net_mask.png"
    Image.fromarray((u2net_alpha * 255).astype(np.uint8), mode="L").save(u2net_mask_path)
    log.info("Saved: %s (U2-Net foreground probability)", u2net_mask_path)

    # --- Layer 2: Color distance from white ---
    rgb = arr[:, :, :3].astype(np.float32)
    max_dist = np.max(255.0 - rgb, axis=2)

    # Visualise: scale so max_dist=100 maps to 255 (makes faint elements visible)
    color_dist_viz = np.clip(max_dist * (255.0 / 100.0), 0, 255).astype(np.uint8)
    color_dist_path = output_dir / "02_color_distance.png"
    Image.fromarray(color_dist_viz, mode="L").save(color_dist_path)
    log.info("Saved: %s (distance from white, scaled ×2.55)", color_dist_path)

    # Color signal: smooth ramp from 0 (pure white) to 1 (clearly non-white)
    tol = float(white_tolerance)
    transition = max(float(edge_softness * 5), 1.0)
    color_signal = np.clip((max_dist - tol) / transition, 0.0, 1.0)

    # --- Layer 3: Scharr edge detection ---
    # Detects gradient transitions even in very faint lines where color
    # distance alone can't distinguish them from the white background.
    gray = np.mean(arr[:, :, :3], axis=2).astype(np.float32)

    # Scharr kernels (better rotational symmetry than Sobel)
    scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)
    scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32)

    padded = np.pad(gray, 1, mode="edge")
    dx = sum(scharr_x[i, j] * padded[i : i + h, j : j + w] for i in range(3) for j in range(3))
    dy = sum(scharr_y[i, j] * padded[i : i + h, j : j + w] for i in range(3) for j in range(3))
    edge_mag = np.sqrt(dx**2 + dy**2)

    # Normalize to 0-1
    edge_max = edge_mag.max() if edge_mag.max() > 0 else 1.0
    edge_norm = edge_mag / edge_max

    # Save raw edge map (amplified for visibility)
    edge_viz = np.clip(edge_norm * 4.0, 0, 1)  # 4× boost to see faint edges
    edge_viz_path = output_dir / "03_edge_scharr.png"
    Image.fromarray((edge_viz * 255).astype(np.uint8), mode="L").save(edge_viz_path)
    log.info("Saved: %s (Scharr edge magnitude, 4× amplified)", edge_viz_path)

    # Threshold + dilate: edges define spatial regions where lines exist
    edge_threshold = 0.02  # sensitive — catches faint grey lines
    edge_binary = (edge_norm > edge_threshold).astype(np.uint8) * 255
    edge_dilated_img = Image.fromarray(edge_binary, mode="L")
    edge_dilated_img = edge_dilated_img.filter(ImageFilter.MaxFilter(size=5))
    edge_mask = np.array(edge_dilated_img).astype(np.float32) / 255.0

    edge_mask_path = output_dir / "03b_edge_dilated.png"
    Image.fromarray((edge_mask * 255).astype(np.uint8), mode="L").save(edge_mask_path)
    log.info("Saved: %s (edges dilated 5px — spatial envelope)", edge_mask_path)

    # Within edge regions, compute per-pixel alpha from three color channels:
    #   1. Grey alpha:  distance from white → catches neutral grey lines
    #   2. Warmth alpha: R-B difference → catches gold/amber anti-aliased edges
    #   3. Dark alpha:  inverted brightness → catches black outline anti-aliased edges
    # Each channel handles AA boundary pixels that the others miss.
    grey_ref = float(grey_reference)
    grey_alpha = np.clip(max_dist / grey_ref, 0.0, 1.0)

    # Gold/amber warmth: R channel minus B channel, normalized.
    # Pure white → 0, gold (244,162,97) → 1.0, neutral grey → 0.
    warmth = np.clip((rgb[:, :, 0] - rgb[:, :, 2]) / 40.0, 0.0, 1.0)

    # Darkness: inverted brightness, normalized.
    # Pure white → 0, dark outlines → 1.0. Threshold at 220 avoids
    # catching light grey (already handled by grey_alpha).
    brightness = np.mean(rgb, axis=2)
    darkness = np.clip((220.0 - brightness) / 150.0, 0.0, 1.0)

    # Any channel can claim the pixel — take the max
    element_alpha = np.maximum(np.maximum(grey_alpha, warmth), darkness)
    edge_signal = edge_mask * element_alpha

    # Save intermediate visualisations
    Image.fromarray((grey_alpha * 255).astype(np.uint8), mode="L").save(output_dir / "03c_alpha_grey.png")
    Image.fromarray((warmth * 255).astype(np.uint8), mode="L").save(output_dir / "03d_alpha_warmth.png")
    Image.fromarray((darkness * 255).astype(np.uint8), mode="L").save(output_dir / "03e_alpha_darkness.png")
    Image.fromarray((edge_signal * 255).astype(np.uint8), mode="L").save(output_dir / "03f_edge_refined.png")
    log.info(
        "Saved: 03c-f (grey/warmth/darkness/refined alpha, grey_ref=%d)",
        grey_reference,
    )

    # --- Layer 4: Semantic foreground only ---
    fg_semantic = arr.copy()
    fg_semantic[:, :, 3] = (u2net_alpha * 255).astype(np.uint8)
    fg_semantic_path = output_dir / "04_fg_semantic.png"
    Image.fromarray(fg_semantic).save(fg_semantic_path)
    log.info("Saved: %s (U2-Net foreground only)", fg_semantic_path)

    # --- Layer 5: Color elements only ---
    fg_color = arr.copy()
    fg_color[:, :, 3] = (color_signal * 255).astype(np.uint8)
    fg_color_path = output_dir / "05_fg_color.png"
    Image.fromarray(fg_color).save(fg_color_path)
    log.info("Saved: %s (non-white content only)", fg_color_path)

    # --- Layer 6: Edge-detected elements only ---
    fg_edge = arr.copy()
    fg_edge[:, :, 3] = (edge_signal * 255).astype(np.uint8)
    fg_edge_path = output_dir / "06_fg_edge.png"
    Image.fromarray(fg_edge).save(fg_edge_path)
    log.info("Saved: %s (edge-detected structure only)", fg_edge_path)

    # --- Layer 7: Combined mask = max(semantic, color, edge) ---
    combined = np.maximum(np.maximum(u2net_alpha, color_signal), edge_signal)

    combined_raw = (combined * 255).astype(np.uint8)
    combined_raw_path = output_dir / "07_combined_mask.png"
    Image.fromarray(combined_raw, mode="L").save(combined_raw_path)
    log.info("Saved: %s (union of all three signals, pre-sharpen)", combined_raw_path)

    # --- Layer 7b: Sigmoid sharpening for crisp rotoscoped edges ---
    # Sigmoid contrast curve pushes soft alpha gradients toward 0 or 1,
    # producing a clean hard-edged mask without jagged staircase artifacts.
    #   σ(x) = 1 / (1 + exp(-k * (x - midpoint)))
    # k = sharpen steepness: 0 = no change, 12 = crisp, 30+ = near-binary.
    if sharpen > 0:
        midpoint = 0.5
        combined = 1.0 / (1.0 + np.exp(-sharpen * (combined - midpoint)))
        # Re-normalize: sigmoid doesn't exactly hit 0/1 at the extremes
        sig_min, sig_max = combined.min(), combined.max()
        if sig_max > sig_min:
            combined = (combined - sig_min) / (sig_max - sig_min)
        # Minimal anti-alias pass: 0.5px Gaussian prevents staircase on diagonal edges
        sharp_u8 = (combined * 255).astype(np.uint8)
        sharp_img = Image.fromarray(sharp_u8, mode="L")
        sharp_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=0.5))
        combined = np.array(sharp_img).astype(np.float32) / 255.0
        log.info("Sigmoid sharpen applied (k=%d, midpoint=%.2f)", sharpen, midpoint)

    combined_mask = (combined * 255).astype(np.uint8)
    combined_sharp_path = output_dir / "07b_combined_sharpened.png"
    Image.fromarray(combined_mask, mode="L").save(combined_sharp_path)
    log.info("Saved: %s (sharpened mask, k=%d)", combined_sharp_path, sharpen)

    # --- Layer 8: Final composite ---
    final = arr.copy()
    final[:, :, 3] = combined_mask
    final = _auto_crop(final)

    final_path = output_dir / "08_final.png"
    Image.fromarray(final).save(final_path)
    log.info("Saved: %s (%dx%d)", final_path, final.shape[1], final.shape[0])

    # Summary stats
    fg_u2net = int(np.sum(u2net_alpha > 0.5))
    fg_color_count = int(np.sum(color_signal > 0.5))
    fg_edge = int(np.sum(edge_signal > 0.5))
    fg_combined = int(np.sum(combined > 0.5))
    rescued = fg_combined - fg_u2net
    log.info(
        "Pixel counts — U2-Net: %d, color: %d, edge: %d, combined: %d, rescued: %d (%.1f%%)",
        fg_u2net,
        fg_color_count,
        fg_edge,
        fg_combined,
        rescued,
        100.0 * rescued / (h * w),
    )

    return final_path


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def render_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int, int],
    stroke_color: tuple[int, int, int, int] | None = None,
    stroke_width: int = 0,
) -> Image.Image:
    """Render text to a tight RGBA image."""
    # Measure text bounds
    temp = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(temp)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    left, top, right, bottom = bbox

    padding = stroke_width + 4
    width = (right - left) + padding * 2
    height = (bottom - top) + padding * 2

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text(
        (padding - left, padding - top),
        text,
        font=font,
        fill=color,
        stroke_width=stroke_width,
        stroke_fill=stroke_color,
    )
    return img


def add_wordmark(
    icon_path: Path,
    output_path: Path | None = None,
    config: dict | None = None,
    text: str | None = None,
    font_path: str | None = None,
    font_size: int | None = None,
    font_weight: int | None = None,
    split_at: int | None = None,
    color_left: tuple[int, int, int] | None = None,
    color_right: tuple[int, int, int] | None = None,
    icon_pos: tuple[float, float] | None = None,
    icon_anchor: str | None = None,
    text_pos: tuple[float, float] | None = None,
    text_anchor: str | None = None,
    canvas_size: tuple[int, int] | None = None,
) -> Path:
    """Compose icon + two-tone wordmark on a canvas using normalized positions.

    Positions are normalized 0.0-1.0: (0.0, 0.0)=top-left, (0.5, 0.5)=centre,
    (1.0, 1.0)=bottom-right. Anchors control which point of the element sits
    at the position coordinate.
    """
    cfg = config or {}
    icon_cfg = cfg.get("icon", {})
    wm_cfg = cfg.get("wordmark", {})
    canvas_cfg = cfg.get("canvas", {})

    # Resolve values: CLI arg > config > built-in default
    text = text or wm_cfg.get("text", "MUNINN")
    split_at = split_at if split_at is not None else wm_cfg.get("split_at", 3)
    color_left = tuple(color_left or wm_cfg.get("color_left", list(CHARCOAL)))
    color_right = tuple(color_right or wm_cfg.get("color_right", list(AMBER_GOLD)))
    icon_pos = tuple(icon_pos or icon_cfg.get("position", [0.05, 0.5]))
    icon_anchor = icon_anchor or icon_cfg.get("anchor", "center_left")
    text_pos = tuple(text_pos or wm_cfg.get("position", [0.55, 0.5]))
    text_anchor = text_anchor or wm_cfg.get("anchor", "center_left")
    icon_scale = icon_cfg.get("scale", 0.9)

    icon = Image.open(icon_path).convert("RGBA")

    # Canvas dimensions
    if canvas_size:
        cw, ch = canvas_size
    else:
        cw = canvas_cfg.get("width", 2048)
        ch = canvas_cfg.get("height", 1024)

    bg_color = tuple(cfg.get("background", [255, 255, 255, 0]))

    # Scale icon to fit canvas height
    scaled_h = int(ch * icon_scale)
    scale_factor = scaled_h / icon.height
    scaled_w = int(icon.width * scale_factor)
    icon_scaled = icon.resize((scaled_w, scaled_h), Image.LANCZOS)

    # Auto-size font relative to canvas height
    font_size_ratio = wm_cfg.get("font_size_ratio", 0.25)
    if font_size is None:
        font_size = int(ch * font_size_ratio)
    font_path = font_path or wm_cfg.get("font")
    font_weight = font_weight if font_weight is not None else wm_cfg.get("font_weight")

    font = load_font(font_path, font_size, font_weight)

    # Render two-tone text
    text_left = text[:split_at]
    text_right = text[split_at:]

    stroke_w = wm_cfg.get("stroke_width", 0)
    stroke_c = tuple(wm_cfg["stroke_color"]) if wm_cfg.get("stroke_color") else None

    img_left = render_text(text_left, font, (*color_left, 255), stroke_color=stroke_c, stroke_width=stroke_w)
    img_right = render_text(text_right, font, (*color_right, 255), stroke_color=stroke_c, stroke_width=stroke_w)

    # Combine text halves into one image
    text_w = img_left.width + img_right.width
    text_h = max(img_left.height, img_right.height)
    text_combined = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
    text_combined.paste(img_left, (0, 0), img_left)
    text_combined.paste(img_right, (img_left.width, 0), img_right)

    # Assemble on canvas
    canvas = Image.new("RGBA", (cw, ch), bg_color)

    ix, iy = resolve_position(icon_pos, icon_anchor, cw, ch, scaled_w, scaled_h)
    canvas.paste(icon_scaled, (ix, iy), icon_scaled)

    tx, ty = resolve_position(text_pos, text_anchor, cw, ch, text_w, text_h)
    canvas.paste(text_combined, (tx, ty), text_combined)

    if output_path is None:
        output_path = OUTPUT_DIR / f"{icon_path.stem}_wordmark.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    log.info("Saved: %s (%dx%d)", output_path, canvas.width, canvas.height)
    return output_path


# ---------------------------------------------------------------------------
# Futhark rune overlay
# ---------------------------------------------------------------------------


def add_rune_overlay(
    icon_path: Path,
    output_path: Path | None = None,
    config: dict | None = None,
    runes: str | None = None,
    opacity: float | None = None,
    font_path: str | None = None,
    font_size: int | None = None,
    pos: tuple[float, float] | None = None,
    anchor: str | None = None,
) -> Path:
    """Add Futhark runes as a low-opacity overlay on the icon."""
    cfg = config or {}
    ru_cfg = cfg.get("runes", {})

    runes = runes or ru_cfg.get("text", FUTHARK_MUNINN)
    opacity = opacity if opacity is not None else ru_cfg.get("opacity", 0.15)
    pos = tuple(pos or ru_cfg.get("position", [0.5, 0.85]))
    anchor = anchor or ru_cfg.get("anchor", "center")

    icon = Image.open(icon_path).convert("RGBA")

    font_size_ratio = ru_cfg.get("font_size_ratio", 0.12)
    if font_size is None:
        font_size = int(icon.height * font_size_ratio)

    # Futhark needs a Unicode-capable font; try Noto Sans Runic or DejaVu
    font_path = font_path or ru_cfg.get("font")
    rune_font_candidates = [
        font_path,
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansRunic-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font = None
    for candidate in rune_font_candidates:
        if candidate and Path(candidate).exists():
            try:
                font = ImageFont.truetype(candidate, font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()
        log.warning("No Futhark-capable font found; runes may render as boxes")

    # Render runes
    rune_img = render_text(runes, font, (128, 128, 128, int(255 * opacity)))

    # Position runes using normalized coords
    rx, ry = resolve_position(pos, anchor, icon.width, icon.height, rune_img.width, rune_img.height)

    # Paste runes UNDER the icon (runes first, then icon on top)
    base = Image.new("RGBA", icon.size, (0, 0, 0, 0))
    base.paste(rune_img, (rx, ry), rune_img)
    composite = Image.alpha_composite(base, icon)

    if output_path is None:
        output_path = OUTPUT_DIR / f"{icon_path.stem}_runes.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    composite.save(output_path)
    log.info("Saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Full composite pipeline
# ---------------------------------------------------------------------------


def composite_pipeline(
    icon_path: Path,
    output_path: Path | None = None,
    config: dict | None = None,
    remove_bg: bool = True,
    add_text: bool = True,
    text: str | None = None,
    font_path: str | None = None,
    font_size: int | None = None,
    font_weight: int | None = None,
) -> Path:
    """Run the full pipeline: remove bg → add wordmark → save."""
    current = icon_path

    if remove_bg:
        bg_removed = icon_path.with_stem(icon_path.stem + "_nobg")
        current = remove_background(current, bg_removed)

    if add_text:
        if output_path is None:
            output_path = OUTPUT_DIR / f"{icon_path.stem}_final.png"
        current = add_wordmark(
            current,
            output_path,
            config=config,
            text=text,
            font_path=font_path,
            font_size=font_size,
            font_weight=font_weight,
        )

    return current


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_pos(s: str) -> tuple[float, float]:
    """Parse 'x,y' normalized position string."""
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Position must be 'x,y' (got: {s!r})")
    return (float(parts[0]), float(parts[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Muninn logo post-processor — deterministic image pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Positioning uses normalised coordinates (0.0-1.0):
  (0.0, 0.0) = top-left       (0.5, 0.5) = centre
  (1.0, 1.0) = bottom-right   (0.8, 0.5) = 80%% across, vertically centred

Defaults are loaded from scripts/logo_config.json. CLI flags override config.

Examples:
  uv run %(prog)s remove-bg docs/logo/input.png
  uv run %(prog)s wordmark docs/logo/input.png --text-pos 0.55,0.5
  uv run %(prog)s wordmark docs/logo/input.png --icon-pos 0.05,0.5 --icon-anchor center_left
  uv run %(prog)s runes docs/logo/input.png --pos 0.5,0.85 --opacity 0.2
  uv run %(prog)s composite docs/logo/input.png --config scripts/logo_config.json
        """,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # Shared options
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("input", type=Path, help="Input image path")
    shared.add_argument("-o", "--output", type=Path, default=None, help="Output path")
    shared.add_argument(
        "--config", type=Path, default=None, help="JSON config file (default: scripts/logo_config.json)"
    )
    shared.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    shared.add_argument("-q", "--quiet", action="store_true", help="Errors only")

    anchor_choices = list(ANCHORS.keys())

    # remove-bg
    bg = sub.add_parser("remove-bg", parents=[shared], help="Remove white background, producing transparent PNG")
    bg.add_argument(
        "--mode",
        choices=["model", "global", "flood"],
        default="model",
        help="'model' uses U2-Net segmentation (best quality, needs ~170MB model download); "
        "'global' thresholds all white pixels; 'flood' fills from edges only (default: model)",
    )
    bg.add_argument(
        "--tolerance", type=int, default=30, help="How far from white (0-255) still counts as bg (default: 30)"
    )
    bg.add_argument("--edge-softness", type=int, default=2, help="Pixels of alpha gradient at edges (default: 2)")
    bg.add_argument(
        "--dark-border-threshold", type=int, default=40, help="Max RGB value for dark border pixels (default: 40)"
    )

    # wordmark
    wm = sub.add_parser("wordmark", parents=[shared], help="Add two-tone wordmark text next to icon")
    wm.add_argument("--text", default=None, help="Wordmark text (default from config: MUNINN)")
    wm.add_argument("--font", default=None, help="Path to .ttf/.otf font file")
    wm.add_argument("--font-size", type=int, default=None, help="Font size in px")
    wm.add_argument("--font-weight", type=int, default=None, help="Variable font weight (100-900)")
    wm.add_argument(
        "--split-at", type=int, default=None, help="Character index to split colours (default: 3 → MUN|INN)"
    )
    wm.add_argument(
        "--icon-pos", type=_parse_pos, default=None, help="Icon position as 'x,y' normalised (e.g. 0.05,0.5)"
    )
    wm.add_argument("--icon-anchor", default=None, choices=anchor_choices, help="Icon anchor point")
    wm.add_argument(
        "--text-pos", type=_parse_pos, default=None, help="Text position as 'x,y' normalised (e.g. 0.55,0.5)"
    )
    wm.add_argument("--text-anchor", default=None, choices=anchor_choices, help="Text anchor point")
    wm.add_argument(
        "--canvas", type=_parse_pos, default=None, help="Canvas size as 'width,height' in px (e.g. 2048,1024)"
    )

    # runes
    ru = sub.add_parser("runes", parents=[shared], help="Add Futhark rune overlay")
    ru.add_argument("--opacity", type=float, default=None, help="Rune opacity (0.0-1.0)")
    ru.add_argument("--rune-font", default=None, help="Font with Futhark support")
    ru.add_argument("--rune-size", type=int, default=None, help="Rune font size")
    ru.add_argument("--pos", type=_parse_pos, default=None, help="Rune position as 'x,y' normalised")
    ru.add_argument("--anchor", default=None, choices=anchor_choices, help="Rune anchor point")

    # segment
    seg = sub.add_parser(
        "segment",
        parents=[shared],
        help="Segment into layers with intermediate outputs for inspection",
    )
    seg.add_argument(
        "--white-tolerance",
        type=int,
        default=15,
        help="Max distance from white to count as blank background (default: 15)",
    )
    seg.add_argument(
        "--grey-ref",
        type=int,
        default=30,
        help="Color distance from white that maps to full opacity in edge regions (default: 30)",
    )
    seg.add_argument(
        "--sharpen",
        type=int,
        default=12,
        help="Sigmoid steepness for edge sharpening (0=off, 12=crisp, 30+=near-binary; default: 12)",
    )
    seg.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for intermediate layer images (default: docs/logo/tweaks/)",
    )

    # composite
    comp = sub.add_parser("composite", parents=[shared], help="Full pipeline: remove-bg → wordmark → save")
    comp.add_argument("--text", default=None, help="Wordmark text")
    comp.add_argument("--font", default=None, help="Path to .ttf/.otf font file")
    comp.add_argument("--font-size", type=int, default=None, help="Font size in px")
    comp.add_argument("--font-weight", type=int, default=None, help="Variable font weight")
    comp.add_argument("--no-remove-bg", action="store_true", help="Skip background removal")

    return p.parse_args()


def main():
    args = parse_args()

    # Configure logging
    if args.quiet:
        level = logging.ERROR
    elif args.verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    if not args.input.exists():
        log.error("Input file not found: %s", args.input)
        sys.exit(1)

    cfg = load_config(args.config)

    if args.command == "remove-bg":
        remove_background(
            args.input,
            args.output,
            tolerance=args.tolerance,
            edge_softness=args.edge_softness,
            dark_border_threshold=args.dark_border_threshold,
            mode=args.mode,
        )

    elif args.command == "wordmark":
        canvas_size = (int(args.canvas[0]), int(args.canvas[1])) if args.canvas else None
        add_wordmark(
            args.input,
            args.output,
            config=cfg,
            text=args.text,
            font_path=args.font,
            font_size=args.font_size,
            font_weight=args.font_weight,
            split_at=args.split_at,
            icon_pos=args.icon_pos,
            icon_anchor=args.icon_anchor,
            text_pos=args.text_pos,
            text_anchor=args.text_anchor,
            canvas_size=canvas_size,
        )

    elif args.command == "runes":
        add_rune_overlay(
            args.input,
            args.output,
            config=cfg,
            opacity=args.opacity,
            font_path=args.rune_font,
            font_size=args.rune_size,
            pos=args.pos,
            anchor=args.anchor,
        )

    elif args.command == "segment":
        out_dir = args.output_dir or (PROJECT_ROOT / "docs" / "logo" / "tweaks")
        segment_layers(
            args.input,
            out_dir,
            white_tolerance=args.white_tolerance,
            edge_softness=2,
            grey_reference=args.grey_ref,
            sharpen=args.sharpen,
        )

    elif args.command == "composite":
        composite_pipeline(
            args.input,
            args.output,
            config=cfg,
            remove_bg=not args.no_remove_bg,
            text=args.text,
            font_path=args.font,
            font_size=args.font_size,
            font_weight=args.font_weight,
        )


if __name__ == "__main__":
    main()
