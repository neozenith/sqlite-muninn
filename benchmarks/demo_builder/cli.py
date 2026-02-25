"""CLI for the demo_builder package.

Subcommands:
    manifest     Show permutation status and generate commands
    build        Build a single demo database
    list-books   List discovered books with metadata
    list-models  List available embedding models

Usage:
    uv run -m benchmarks.demo_builder manifest
    uv run -m benchmarks.demo_builder manifest --missing --commands
    uv run -m benchmarks.demo_builder build --output-folder wasm/assets/ --book-id 3300 --embedding-model MiniLM
    uv run -m benchmarks.demo_builder list-books
    uv run -m benchmarks.demo_builder list-models
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
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

    Builds a single permutation (book_id + embedding_model).
    Deferred import: build.py (and thus phases/) is only imported here
    so that info subcommands work without ML dependencies.
    """
    from benchmarks.demo_builder.build import DemoBuild  # noqa: E402 — deferred for ML deps

    # ── Validate book exists ──────────────────────────────────────
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

    assert args.book_id in available_books, f"Book {args.book_id} not available. Discovered: {sorted(available_books)}"

    # ── Handle existing output / --force ──────────────────────────
    output_folder = _resolve_output_folder(args)
    perm_id = f"{args.book_id}_{args.embedding_model}"
    final_path = output_folder / f"{perm_id}.db"

    if final_path.exists() and not args.force:
        log.info("Already exists: %s (use --force to overwrite)", final_path.name)
        return

    if final_path.exists() and args.force:
        log.info("Will overwrite: %s", final_path.name)
        final_path.unlink()

    # Clean stale staging dir on --force
    stale_staging = output_folder / "_build" / perm_id
    if stale_staging.exists() and args.force:
        log.info("Removing stale staging dir: %s", stale_staging)
        shutil.rmtree(stale_staging)

    # ── Pre-flight checks ─────────────────────────────────────────
    assert Path(MUNINN_PATH).with_suffix(".dylib").exists() or Path(MUNINN_PATH).with_suffix(".so").exists(), (
        f"Muninn extension not found at {MUNINN_PATH}. Run: make all"
    )

    # ── Build ─────────────────────────────────────────────────────
    log.info("Building %s_%s", args.book_id, args.embedding_model)
    build = DemoBuild(args.book_id, args.embedding_model, output_folder)
    build.setup()
    try:
        build.run()
    finally:
        build.teardown()


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
    build_p = subparsers.add_parser("build", help="Build a single demo database")
    build_p.add_argument(
        "--output-folder",
        type=str,
        default="wasm/assets",
        help="Output folder for generated databases (default: wasm/assets)",
    )
    build_p.add_argument(
        "--book-id",
        type=int,
        required=True,
        help="Gutenberg book ID to build",
    )
    build_p.add_argument(
        "--embedding-model",
        type=str,
        required=True,
        choices=list(EMBEDDING_MODELS.keys()),
        help="Embedding model to use",
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
