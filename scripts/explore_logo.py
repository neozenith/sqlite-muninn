#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "google-genai>=1.0.0",
#   "Pillow>=10.0.0",
# ]
# ///
"""
Muninn Logo Explorer — iterative image generation using Google's GenAI APIs.

Supports two backends:
  - gemini   : Nano Banana models (gemini-2.5-flash-image, gemini-3-pro-image-preview)
                Best for iterative refinement via multi-turn chat.
  - imagen   : Imagen 4 models (imagen-4.0-generate-001, ultra, fast)
                Best for high-fidelity standalone generation.

Usage:
    # Interactive chat mode (default — great for iteration)
    uv run scripts/explore_logo.py

    # One-shot from a prompt file (edit the file, re-run to iterate)
    uv run scripts/explore_logo.py --prompt-file scripts/prompt.md

    # One-shot generation with an inline prompt
    uv run scripts/explore_logo.py --prompt "A minimalist raven silhouette logo"

    # Use Imagen 4 for high-quality output
    uv run scripts/explore_logo.py --backend imagen --prompt-file scripts/prompt.md

    # Use the pro model for complex prompts
    uv run scripts/explore_logo.py --model pro --prompt-file scripts/prompt.md

    # Generate multiple Imagen variants
    uv run scripts/explore_logo.py --backend imagen --starter minimal --count 4

    # Specify resolution and aspect ratio
    uv run scripts/explore_logo.py --size 2K --aspect 1:1

Auth:
    Set GOOGLE_API_KEY env var, or authenticate via:
        gcloud auth application-default login
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

SCRIPT = Path(__file__)
SCRIPT_NAME = SCRIPT.stem
SCRIPT_DIR = SCRIPT.parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_ROOT / "docs" / "logo" / "gen"
DEFAULT_PROMPT_FILE = SCRIPT_DIR / "prompt.md"

# Vertex AI defaults — the google-genai SDK reads these automatically.
# Override by setting env vars before running the script.
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "jpeak-nam-poc")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

GEMINI_MODELS = {
    "flash": "gemini-2.5-flash-image",
    "pro": "gemini-3-pro-image-preview",
}

IMAGEN_MODELS = {
    "standard": "imagen-4.0-generate-001",
    "ultra": "imagen-4.0-ultra-generate-001",
    "fast": "imagen-4.0-fast-generate-001",
}

ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9"]
IMAGE_SIZES = ["1K", "2K", "4K"]  # 4K only for Gemini

# Pre-built prompts based on docs/plans/brand_identity.md
STARTER_PROMPTS = {
    "minimal": (
        "Design a minimalist logo for 'Muninn', a developer tool. "
        "The logo is a raven head in profile, rendered as a clean geometric silhouette "
        "in deep charcoal. The raven's eye is a single glowing amber/gold node with "
        "faint graph edges radiating from it. "
        "Style: flat vector, no gradients, suitable for favicon and README header. "
        "Background: transparent or pure white."
    ),
    "graph-wing": (
        "Design a logo for 'Muninn', a graph database extension. "
        "A raven in flight, viewed from below, wings spread wide. "
        "The wing feathers are subtly formed from interconnected graph nodes and edges, "
        "creating a network pattern. Deep charcoal raven with amber/gold node highlights. "
        "Style: modern tech logo, clean lines, works at small sizes. "
        "Background: transparent."
    ),
    "perched": (
        "Design a logo for 'Muninn', a knowledge graph library. "
        "A stylized raven perched on a small graph structure (3-4 connected nodes). "
        "The raven is sleek and geometric, rendered in deep charcoal/black. "
        "The graph nodes glow amber/gold. "
        "Norse-inspired but modern and technical. "
        "Style: clean vector art, suitable for a GitHub repo social preview. "
        "Background: dark navy (#1a1a2e)."
    ),
    "carrying": (
        "Design a logo for 'Muninn', a vector search engine. "
        "A raven diving downward, carrying a glowing golden orb (representing a vector "
        "embedding) in its talons. Trailing behind the orb are faint connection lines "
        "to other smaller nodes. Deep charcoal raven, amber/gold accents. "
        "Style: modern startup logo, bold silhouette, works in monochrome too. "
        "Background: transparent."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_client() -> genai.Client:
    """Create a GenAI client. Config comes from env vars set above (or overridden by caller)."""
    log.info(
        "Client: vertexai=%s project=%s location=%s",
        os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"),
        os.environ.get("GOOGLE_CLOUD_PROJECT"),
        os.environ.get("GOOGLE_CLOUD_LOCATION"),
    )
    return genai.Client()


def save_image(image: Image.Image, prompt: str, model: str, index: int = 0) -> Path:
    """Save a generated image with metadata sidecar."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"muninn_{ts}_{index}"
    img_path = OUTPUT_DIR / f"{stem}.png"
    meta_path = OUTPUT_DIR / f"{stem}.json"

    image.save(img_path)
    meta_path.write_text(json.dumps({
        "prompt": prompt,
        "model": model,
        "timestamp": ts,
        "size": f"{image.width}x{image.height}",
    }, indent=2), encoding="utf-8")

    log.info("Saved %s (%dx%d)", img_path, image.width, image.height)
    return img_path


def list_starters():
    """Print available starter prompts to stdout (user-facing output)."""
    sys.stdout.write("\nStarter prompts (use --starter NAME):\n\n")
    for name, prompt in STARTER_PROMPTS.items():
        preview = prompt[:80].replace("\n", " ") + "..."
        sys.stdout.write(f"  {name:15s}  {preview}\n")
    sys.stdout.write("\n")


def load_prompt_file(path: Path) -> str:
    """Read a prompt from a markdown file, stripping comment lines (starting with #)."""
    raw = path.read_text(encoding="utf-8")
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        # Skip markdown headings and HTML comments — they're structural, not prompt content
        if stripped.startswith("#") or stripped.startswith("<!--"):
            continue
        lines.append(line)
    prompt = "\n".join(lines).strip()
    if not prompt:
        log.error("Prompt file is empty (after stripping comments): %s", path)
        sys.exit(1)
    log.info("Loaded prompt from %s (%d chars)", path, len(prompt))
    log.debug("Prompt text:\n%s", prompt)
    return prompt


def _build_image_config(aspect: str | None, size: str | None) -> types.ImageConfig | None:
    """Build an ImageConfig from aspect/size, or None if neither is set."""
    kwargs = {}
    if aspect:
        kwargs["aspect_ratio"] = aspect
    if size:
        kwargs["image_size"] = size
    return types.ImageConfig(**kwargs) if kwargs else None


# ---------------------------------------------------------------------------
# Gemini backend — supports multi-turn chat for iterative refinement
# ---------------------------------------------------------------------------

def gemini_generate(client: genai.Client, prompt: str, model: str,
                    aspect: str | None, size: str | None,
                    ref_images: list[Path] | None = None) -> list[Path]:
    """Generate image(s) with a Gemini model (single turn)."""
    contents: list = [prompt]
    if ref_images:
        for p in ref_images:
            contents.append(Image.open(p))

    img_config = _build_image_config(aspect, size)
    config = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
    if img_config:
        config.image_config = img_config

    log.info("Generating with %s ...", model)
    response = client.models.generate_content(model=model, contents=contents, config=config)

    saved = []
    idx = 0
    for part in response.parts:
        if part.text is not None:
            sys.stdout.write(f"\n  Model says: {part.text}\n")
        elif part.inline_data is not None:
            img = Image.open(BytesIO(part.inline_data.data))
            path = save_image(img, prompt, model, idx)
            sys.stdout.write(f"  Saved: {path}\n")
            saved.append(path)
            idx += 1

    return saved


def gemini_chat(client: genai.Client, model: str, aspect: str | None, size: str | None):
    """Interactive multi-turn chat for iterative logo refinement."""
    sys.stdout.write(f"\n--- Muninn Logo Studio (chat mode) ---\n")
    sys.stdout.write(f"Model: {model}\n")
    sys.stdout.write("Type your prompt, or use these commands:\n")
    sys.stdout.write("  /starters     — show pre-built prompts\n")
    sys.stdout.write("  /use NAME     — use a starter prompt\n")
    sys.stdout.write("  /ref PATH     — attach a reference image to next message\n")
    sys.stdout.write("  /aspect RATIO — change aspect ratio (e.g. 1:1, 16:9)\n")
    sys.stdout.write("  /size SIZE    — change size (1K, 2K, 4K)\n")
    sys.stdout.write("  /model NAME   — switch model (flash, pro)\n")
    sys.stdout.write("  /quit         — exit\n\n")

    img_config = _build_image_config(aspect, size)
    config = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
    if img_config:
        config.image_config = img_config

    chat = client.chats.create(model=model, config=config)
    ref_images: list[Path] = []
    turn = 0

    while True:
        try:
            user_input = input(f"\n[{turn}] You> ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\nBye!\n")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input == "/quit":
            sys.stdout.write("Bye!\n")
            break
        elif user_input == "/starters":
            list_starters()
            continue
        elif user_input.startswith("/use "):
            name = user_input[5:].strip()
            if name in STARTER_PROMPTS:
                user_input = STARTER_PROMPTS[name]
                sys.stdout.write(f"  Using prompt: {user_input[:80]}...\n")
            else:
                sys.stdout.write(f"  Unknown starter: {name}\n")
                list_starters()
                continue
        elif user_input.startswith("/ref "):
            path = Path(user_input[5:].strip())
            if path.exists():
                ref_images.append(path)
                sys.stdout.write(f"  Reference image queued: {path}\n")
            else:
                sys.stdout.write(f"  File not found: {path}\n")
            continue
        elif user_input.startswith("/aspect "):
            new_aspect = user_input[8:].strip()
            if new_aspect in ASPECT_RATIOS:
                aspect = new_aspect
                sys.stdout.write(f"  Aspect ratio set to: {aspect}\n")
            else:
                sys.stdout.write(f"  Invalid. Choose from: {', '.join(ASPECT_RATIOS)}\n")
            continue
        elif user_input.startswith("/size "):
            new_size = user_input[6:].strip().upper()
            if new_size in IMAGE_SIZES:
                size = new_size
                sys.stdout.write(f"  Size set to: {size}\n")
            else:
                sys.stdout.write(f"  Invalid. Choose from: {', '.join(IMAGE_SIZES)}\n")
            continue
        elif user_input.startswith("/model "):
            alias = user_input[7:].strip()
            if alias in GEMINI_MODELS:
                model = GEMINI_MODELS[alias]
                img_config = _build_image_config(aspect, size)
                config = types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
                if img_config:
                    config.image_config = img_config
                chat = client.chats.create(model=model, config=config)
                sys.stdout.write(f"  Switched to: {model} (new chat session)\n")
                turn = 0
            else:
                sys.stdout.write(f"  Unknown model alias. Choose from: {', '.join(GEMINI_MODELS)}\n")
            continue

        # Build message contents
        contents: list = [user_input]
        for ref in ref_images:
            contents.append(Image.open(ref))
        ref_images.clear()

        # Send message
        sys.stdout.write("  Generating...\n")
        try:
            response = chat.send_message(contents)
        except Exception as e:
            log.error("Generation failed: %s", e)
            continue

        idx = 0
        for part in response.parts:
            if part.text is not None:
                sys.stdout.write(f"  Model: {part.text}\n")
            elif part.inline_data is not None:
                img = Image.open(BytesIO(part.inline_data.data))
                path = save_image(img, user_input, model, idx)
                sys.stdout.write(f"  Saved: {path}\n")
                idx += 1

        turn += 1


# ---------------------------------------------------------------------------
# Imagen backend — high-fidelity standalone generation
# ---------------------------------------------------------------------------

def imagen_generate(client: genai.Client, prompt: str, model: str,
                    count: int, aspect: str | None, size: str | None) -> list[Path]:
    """Generate image(s) with an Imagen 4 model."""
    config_kwargs: dict = {"number_of_images": count}
    if aspect:
        config_kwargs["aspect_ratio"] = aspect
    # Imagen supports 1K and 2K only (not 4K)
    if size and size in ("1K", "2K"):
        config_kwargs["image_size"] = size

    log.info("Generating %d image(s) with %s ...", count, model)
    response = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=types.GenerateImagesConfig(**config_kwargs),
    )

    saved = []
    for idx, generated in enumerate(response.generated_images):
        img = generated.image
        # genai.types.Image → PIL.Image conversion
        pil_img = Image.open(BytesIO(img.image_bytes))
        path = save_image(pil_img, prompt, model, idx)
        sys.stdout.write(f"  Saved: {path}\n")
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Muninn logo explorer — iterate on raven logos with Google GenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run %(prog)s                              # Auto-loads scripts/prompt.md if it exists
  uv run %(prog)s --prompt-file scripts/prompt.md   # Explicit prompt file
  uv run %(prog)s --starter minimal            # Use a pre-built starter prompt
  uv run %(prog)s --backend imagen --count 4   # 4 Imagen variants
  uv run %(prog)s --prompt "A raven logo"      # Inline one-shot generation
  uv run %(prog)s --list-starters              # Show all starter prompts

Prompt file workflow:
  1. Edit scripts/prompt.md with your prompt text
  2. Run: uv run %(prog)s
  3. Review output in docs/logo/
  4. Refine prompt, re-run. Lines starting with # are ignored.
        """,
    )
    p.add_argument("--backend", choices=["gemini", "imagen"], default="gemini",
                   help="Generation backend (default: gemini)")
    p.add_argument("--model", default=None,
                   help="Model alias (flash/pro for gemini, standard/ultra/fast for imagen) or raw model ID")
    p.add_argument("--prompt", default=None,
                   help="Text prompt (if omitted, enters interactive chat mode for gemini)")
    p.add_argument("--prompt-file", default=None, type=Path,
                   help="Load prompt from a markdown file (default: scripts/prompt.md if it exists)")
    p.add_argument("--starter", default=None, choices=list(STARTER_PROMPTS),
                   help="Use a pre-built starter prompt")
    p.add_argument("--list-starters", action="store_true",
                   help="List available starter prompts and exit")
    p.add_argument("--ref", action="append", default=[], type=Path,
                   help="Reference image path (can be repeated, gemini only)")
    p.add_argument("--count", type=int, default=1,
                   help="Number of images to generate (imagen only, 1-4)")
    p.add_argument("--aspect", default="1:1", choices=ASPECT_RATIOS,
                   help="Aspect ratio (default: 1:1)")
    p.add_argument("--size", default=None, choices=IMAGE_SIZES,
                   help="Image resolution (1K, 2K, 4K)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Debug logging")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="Errors only")
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

    if args.list_starters:
        list_starters()
        return

    # Resolve prompt: --prompt > --prompt-file > --starter > default file > interactive
    prompt = args.prompt
    if prompt:
        log.debug("Using inline --prompt")
    elif args.prompt_file:
        prompt = load_prompt_file(args.prompt_file)
    elif args.starter:
        prompt = STARTER_PROMPTS[args.starter]
    elif DEFAULT_PROMPT_FILE.exists():
        prompt = load_prompt_file(DEFAULT_PROMPT_FILE)
        log.info("Auto-loaded default prompt file: %s", DEFAULT_PROMPT_FILE)

    # Resolve model
    if args.backend == "gemini":
        if args.model and args.model in GEMINI_MODELS:
            model = GEMINI_MODELS[args.model]
        elif args.model:
            model = args.model  # Allow raw model ID
        else:
            model = GEMINI_MODELS["pro"]
    else:  # imagen
        if args.model and args.model in IMAGEN_MODELS:
            model = IMAGEN_MODELS[args.model]
        elif args.model:
            model = args.model
        else:
            model = IMAGEN_MODELS["standard"]

    log.info("Backend: %s", args.backend)
    log.info("Model:   %s", model)
    log.info("Output:  %s/", OUTPUT_DIR)

    client = make_client()

    if args.backend == "gemini":
        if prompt:
            gemini_generate(client, prompt, model, args.aspect, args.size,
                            ref_images=args.ref or None)
        else:
            gemini_chat(client, model, args.aspect, args.size)
    else:
        if not prompt:
            log.error("--prompt or --starter required for imagen backend")
            sys.exit(1)
        imagen_generate(client, prompt, model, args.count, args.aspect, args.size)


if __name__ == "__main__":
    main()
