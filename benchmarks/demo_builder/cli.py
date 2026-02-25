"""CLI for the demo_builder package.

Subcommands:
    manifest     Show permutation status and generate commands
    build        Build demo database(s)
    list-books   List discovered books with metadata
    list-models  List available embedding models

Usage:
    uv run -m benchmarks.demo_builder manifest
    uv run -m benchmarks.demo_builder manifest --missing --commands
    uv run -m benchmarks.demo_builder build --output-folder wasm/assets/
    uv run -m benchmarks.demo_builder build --output-folder wasm/assets/ --book-id 3300 --embedding-model MiniLM
    uv run -m benchmarks.demo_builder list-books
    uv run -m benchmarks.demo_builder list-models
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

from benchmarks.demo_builder.constants import EMBEDDING_MODELS, MUNINN_PATH, PROJECT_ROOT
from benchmarks.demo_builder.discovery import discover_book_ids, print_books, print_models
from benchmarks.demo_builder.manifest import permutation_manifest, print_manifest

log = logging.getLogger(__name__)


def _resolve_output_folder(args: argparse.Namespace) -> Path:
    """Resolve --output-folder to an absolute path."""
    output_folder = Path(args.output_folder)
    if not output_folder.is_absolute():
        output_folder = PROJECT_ROOT / output_folder
    return output_folder


def _cmd_manifest(args: argparse.Namespace) -> None:
    """Handle the 'manifest' subcommand."""
    output_folder = _resolve_output_folder(args)
    entries = permutation_manifest(output_folder)
    print_manifest(
        entries,
        output_folder,
        missing=args.missing,
        done=args.done,
        sort=args.sort,
        limit=args.limit,
        commands=args.commands,
        force=args.force,
    )


def _cmd_build(args: argparse.Namespace) -> None:
    """Handle the 'build' subcommand.

    Deferred imports: models.py, build.py, and phases.py are only
    imported here so that info subcommands work without ML dependencies.
    """
    from benchmarks.demo_builder.build import DemoBuild  # noqa: E402 — deferred for ML deps
    from benchmarks.demo_builder.models import ModelPool  # noqa: E402 — deferred for ML deps

    # ── Discover available books ──────────────────────────────────
    available_books = discover_book_ids()
    if not available_books:
        log.error(
            "No books found with both text and chunks. Ensure both"
            " benchmarks/texts/gutenberg_{id}.txt and"
            " benchmarks/kg/{id}_chunks.db exist."
            " Run: uv run -m benchmarks.harness prep texts"
            " && uv run -m benchmarks.harness prep kg-chunks"
        )
        sys.exit(1)

    if args.book_id is not None:
        assert args.book_id in available_books, f"Book {args.book_id} not available. Discovered: {available_books}"
        book_ids = [args.book_id]
    else:
        book_ids = available_books

    model_names = [args.embedding_model] if args.embedding_model else list(EMBEDDING_MODELS.keys())

    # ── Build permutation list ────────────────────────────────────
    output_folder = _resolve_output_folder(args)

    permutations: list[tuple[int, str]] = []
    skipped: list[str] = []
    for bid in book_ids:
        for mname in model_names:
            perm_id = f"{bid}_{mname}"
            out = output_folder / f"{perm_id}.db"
            if out.exists() and not args.force:
                skipped.append(out.name)
                continue
            if out.exists() and args.force:
                log.info("Will overwrite: %s", out.name)
                out.unlink()
            # Clean stale staging dir on --force
            stale_staging = output_folder / "_build" / perm_id
            if stale_staging.exists() and args.force:
                log.info("Removing stale staging dir: %s", stale_staging)
                shutil.rmtree(stale_staging)
            permutations.append((bid, mname))

    if skipped:
        log.info(
            "Skipping %d existing database(s): %s (use --force to overwrite)",
            len(skipped),
            ", ".join(skipped),
        )

    if not permutations:
        log.info("Nothing to build -- all databases already exist.")
        return

    log.info("Building %d database(s):", len(permutations))
    for bid, mname in permutations:
        log.info("  %s_%s", bid, mname)

    # ── Pre-flight checks ─────────────────────────────────────────
    assert Path(MUNINN_PATH).with_suffix(".dylib").exists() or Path(MUNINN_PATH).with_suffix(".so").exists(), (
        f"Muninn extension not found at {MUNINN_PATH}. Run: make all"
    )

    # ── Pre-load ML models (shared across all permutations) ──────
    needed_st_models = sorted({m for _, m in permutations})
    pool = ModelPool(needed_st_models)
    pool.load_all()

    # ── Build each permutation ────────────────────────────────────
    t_total = time.monotonic()
    for i, (bid, mname) in enumerate(permutations, 1):
        log.info("\n[%d/%d] Building %s_%s", i, len(permutations), bid, mname)
        build = DemoBuild(bid, mname, output_folder, pool)
        build.run()

    elapsed = time.monotonic() - t_total
    log.info("All done! Built %d database(s) in %.1fs", len(permutations), elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build KG demo databases for WASM and viz demos.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # ── manifest ──────────────────────────────────────────────────
    manifest_p = subparsers.add_parser("manifest", help="Show permutation status and generate commands")
    manifest_p.add_argument(
        "--output-folder",
        type=str,
        default="wasm/assets",
        help="Output folder to check (default: wasm/assets)",
    )
    manifest_p.add_argument("--missing", action="store_true", help="Only show missing permutations")
    manifest_p.add_argument("--done", action="store_true", help="Only show completed permutations")
    manifest_p.add_argument(
        "--sort",
        choices=["size", "name"],
        default="size",
        help="Sort order: 'size' (smallest/cheapest first, default) or 'name' (alphabetical)",
    )
    manifest_p.add_argument("--limit", type=int, default=None, help="Limit to first N entries")
    manifest_p.add_argument("--commands", action="store_true", help="Print runnable commands instead of a table")
    manifest_p.add_argument(
        "--force",
        action="store_true",
        help="With --commands: append --force to each generated command",
    )

    # ── build ─────────────────────────────────────────────────────
    build_p = subparsers.add_parser("build", help="Build demo database(s)")
    build_p.add_argument(
        "--output-folder",
        type=str,
        default="wasm/assets",
        help="Output folder for generated databases (default: wasm/assets)",
    )
    build_p.add_argument(
        "--book-id",
        type=int,
        default=None,
        help="Filter to a specific Gutenberg book ID (default: all discovered)",
    )
    build_p.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        choices=list(EMBEDDING_MODELS.keys()),
        help="Filter to a specific embedding model (default: all models)",
    )
    build_p.add_argument("--force", action="store_true", help="Overwrite existing output files")

    # ── list-books ────────────────────────────────────────────────
    subparsers.add_parser("list-books", help="List discovered books with metadata")

    # ── list-models ───────────────────────────────────────────────
    subparsers.add_parser("list-models", help="List available embedding models")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "manifest":
        _cmd_manifest(args)
    elif args.command == "build":
        _cmd_build(args)
    elif args.command == "list-books":
        print_books()
    elif args.command == "list-models":
        print_models()
