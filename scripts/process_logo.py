#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "Pillow>=10.0.0",
#   "numpy>=1.26.0",
# ]
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
CHARCOAL = (43, 45, 66)       # #2B2D42
AMBER_GOLD = (244, 162, 97)   # #F4A261
SLATE_GREY = (85, 91, 110)    # #555B6E

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
    "top_left":      (0.0, 0.0),
    "top_center":    (0.5, 0.0),
    "top_right":     (1.0, 0.0),
    "center_left":   (0.0, 0.5),
    "center":        (0.5, 0.5),
    "center_right":  (1.0, 0.5),
    "bottom_left":   (0.0, 1.0),
    "bottom_center": (0.5, 1.0),
    "bottom_right":  (1.0, 1.0),
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

def remove_background(
    input_path: Path,
    output_path: Path | None = None,
    tolerance: int = 30,
    edge_softness: int = 2,
    dark_border_threshold: int = 40,
) -> Path:
    """Remove background using two-pass flood-fill: dark border + white background.

    Handles images with dark rounded-corner borders wrapping a white background.
    Pass 1 floods dark border pixels from corners, then pass 2 floods white
    background pixels from the boundary where border meets white.

    Args:
        tolerance: How far from white a pixel can be and still count as bg (0-255).
        edge_softness: Pixels of alpha gradient at foreground/background boundary.
        dark_border_threshold: Max RGB value to consider a pixel part of a dark border.
    """
    img = Image.open(input_path).convert("RGBA")
    log.info("Removing background from %s (%dx%d) tolerance=%d border_thresh=%d",
             input_path, img.width, img.height, tolerance, dark_border_threshold)

    arr = np.array(img)
    h, w = arr.shape[:2]

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
    for r, c in [(inset, inset), (inset, w - 1 - inset),
                 (h - 1 - inset, inset), (h - 1 - inset, w - 1 - inset)]:
        if 0 <= r < h and 0 <= c < w and is_white[r, c] and not visited[r, c]:
            visited[r, c] = True
            white_stack.append((r, c))

    log.debug("Pass 2 seeds: %d border_boundary, total %d white seeds",
              len(border_boundary), len(white_stack))

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

    log.info("Total background: %d/%d pixels (%.1f%%)",
             int(bg_mask.sum()), total_pixels, 100 * bg_mask.sum() / total_pixels)

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

    # Auto-crop to foreground bounding box (removes border artifacts at edges)
    fg_rows = np.any(alpha > 10, axis=1)
    fg_cols = np.any(alpha > 10, axis=0)
    if np.any(fg_rows) and np.any(fg_cols):
        rmin, rmax = np.where(fg_rows)[0][[0, -1]]
        cmin, cmax = np.where(fg_cols)[0][[0, -1]]
        pad = 4
        rmin = max(0, rmin - pad)
        rmax = min(h - 1, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(w - 1, cmax + pad)
        arr = arr[rmin:rmax + 1, cmin:cmax + 1]
        log.info("Auto-cropped to [%d:%d, %d:%d] → %dx%d",
                 rmin, rmax + 1, cmin, cmax + 1, arr.shape[1], arr.shape[0])

    result = Image.fromarray(arr)

    if output_path is None:
        output_path = input_path.with_stem(input_path.stem + "_nobg")
    result.save(output_path)
    log.info("Saved: %s", output_path)
    return output_path


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
        current = add_wordmark(current, output_path, config=config,
                               text=text, font_path=font_path,
                               font_size=font_size, font_weight=font_weight)

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
    shared.add_argument("--config", type=Path, default=None,
                        help="JSON config file (default: scripts/logo_config.json)")
    shared.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    shared.add_argument("-q", "--quiet", action="store_true", help="Errors only")

    anchor_choices = list(ANCHORS.keys())

    # remove-bg
    bg = sub.add_parser("remove-bg", parents=[shared],
                        help="Remove white background via flood-fill")
    bg.add_argument("--tolerance", type=int, default=30,
                    help="How far from white (0-255) still counts as bg (default: 30)")
    bg.add_argument("--edge-softness", type=int, default=2,
                    help="Pixels of alpha gradient at edges (default: 2)")
    bg.add_argument("--dark-border-threshold", type=int, default=40,
                    help="Max RGB value for dark border pixels (default: 40)")

    # wordmark
    wm = sub.add_parser("wordmark", parents=[shared],
                        help="Add two-tone wordmark text next to icon")
    wm.add_argument("--text", default=None, help="Wordmark text (default from config: MUNINN)")
    wm.add_argument("--font", default=None, help="Path to .ttf/.otf font file")
    wm.add_argument("--font-size", type=int, default=None, help="Font size in px")
    wm.add_argument("--font-weight", type=int, default=None, help="Variable font weight (100-900)")
    wm.add_argument("--split-at", type=int, default=None,
                    help="Character index to split colours (default: 3 → MUN|INN)")
    wm.add_argument("--icon-pos", type=_parse_pos, default=None,
                    help="Icon position as 'x,y' normalised (e.g. 0.05,0.5)")
    wm.add_argument("--icon-anchor", default=None, choices=anchor_choices,
                    help="Icon anchor point")
    wm.add_argument("--text-pos", type=_parse_pos, default=None,
                    help="Text position as 'x,y' normalised (e.g. 0.55,0.5)")
    wm.add_argument("--text-anchor", default=None, choices=anchor_choices,
                    help="Text anchor point")
    wm.add_argument("--canvas", type=_parse_pos, default=None,
                    help="Canvas size as 'width,height' in px (e.g. 2048,1024)")

    # runes
    ru = sub.add_parser("runes", parents=[shared],
                        help="Add Futhark rune overlay")
    ru.add_argument("--opacity", type=float, default=None, help="Rune opacity (0.0-1.0)")
    ru.add_argument("--rune-font", default=None, help="Font with Futhark support")
    ru.add_argument("--rune-size", type=int, default=None, help="Rune font size")
    ru.add_argument("--pos", type=_parse_pos, default=None,
                    help="Rune position as 'x,y' normalised")
    ru.add_argument("--anchor", default=None, choices=anchor_choices,
                    help="Rune anchor point")

    # composite
    comp = sub.add_parser("composite", parents=[shared],
                          help="Full pipeline: remove-bg → wordmark → save")
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
        remove_background(args.input, args.output,
                          tolerance=args.tolerance, edge_softness=args.edge_softness,
                          dark_border_threshold=args.dark_border_threshold)

    elif args.command == "wordmark":
        canvas_size = (int(args.canvas[0]), int(args.canvas[1])) if args.canvas else None
        add_wordmark(
            args.input, args.output, config=cfg,
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
            args.input, args.output, config=cfg,
            opacity=args.opacity,
            font_path=args.rune_font,
            font_size=args.rune_size,
            pos=args.pos,
            anchor=args.anchor,
        )

    elif args.command == "composite":
        composite_pipeline(
            args.input, args.output, config=cfg,
            remove_bg=not args.no_remove_bg,
            text=args.text,
            font_path=args.font,
            font_size=args.font_size,
            font_weight=args.font_weight,
        )


if __name__ == "__main__":
    main()
