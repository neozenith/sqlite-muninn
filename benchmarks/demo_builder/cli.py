"""CLI for the demo_builder package.

Subcommands:
    manifest        Show permutation status and generate commands
    build           Build a single demo database (sequential, all phases)
    run-phase       Run a single named phase against an existing staging DB
    write-manifest  Generate manifest.json from existing built DBs
    list-books      List discovered books with metadata
    list-models     List available embedding models

Usage:
    uv run -m benchmarks.demo_builder manifest
    uv run -m benchmarks.demo_builder manifest --missing --commands
    uv run -m benchmarks.demo_builder manifest --makefile --limit 3 > Makefile
    uv run -m benchmarks.demo_builder build --book-id 3300 --embedding-model MiniLM
    uv run -m benchmarks.demo_builder build --book-id 3300 --embedding-model MiniLM --status
    uv run -m benchmarks.demo_builder run-phase --book-id 3300 --embedding-model MiniLM chunks
    uv run -m benchmarks.demo_builder write-manifest
    uv run -m benchmarks.demo_builder list-books
    uv run -m benchmarks.demo_builder list-models
"""

from __future__ import annotations

import argparse
import datetime
import logging
import shutil
import sqlite3
import sys
from pathlib import Path

from benchmarks.demo_builder.constants import (
    DEFAULT_OUTPUT_FOLDER,
    EMBEDDING_MODELS,
    MUNINN_PATH,
    PHASE_NAMES,
    PROJECT_ROOT,
)
from benchmarks.demo_builder.discovery import discover_book_ids, print_books, print_models
from benchmarks.demo_builder.manifest import (
    generate_makefile,
    permutation_manifest,
    print_manifest,
    write_manifest_json,
)

log = logging.getLogger(__name__)


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


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

    # Apply the same filtering + sorting + limiting as the table/commands modes.
    if args.book_id is not None:
        entries = [e for e in entries if e["book_id"] == args.book_id]
    if args.embedding_model is not None:
        entries = [e for e in entries if e["model_name"] == args.embedding_model]
    if args.missing:
        entries = [e for e in entries if not e["done"]]
    if args.done:
        entries = [e for e in entries if e["done"]]
    if args.sort == "name":
        entries = sorted(entries, key=lambda e: e["permutation_id"])
    else:
        entries = sorted(entries, key=lambda e: e["sort_key"])
    if args.limit is not None:
        entries = entries[: args.limit]

    if args.makefile:
        print(generate_makefile(entries, output_folder))
        return

    print_manifest(
        entries,
        output_folder,
        missing=False,  # filtering already applied above
        done=False,
        sort="name",  # already sorted
        limit=None,  # already limited
        commands=args.commands,
        force=args.force,
    )


def _cmd_run_phase(args: argparse.Namespace) -> None:
    """Handle the 'run-phase' subcommand.

    Runs a single named phase against the staging DB for the given permutation.
    The Makefile calls this for each phase and manages the sentinel files.

    Lifecycle:
    - For the first phase (chunks): creates staging dir + DB if missing.
    - For all other phases: staging DB must already exist (fails loudly if not).
    - For all phases before the target: restore_ctx() is called to hydrate ctx.
    - For the last phase (metadata): finalizes (VACUUM + atomic move + joblib).
    """
    from benchmarks.demo_builder.build import DemoBuild, PhaseContext  # deferred for ML deps
    from benchmarks.demo_builder.common import load_muninn  # noqa: E402 — deferred (common.py imports numpy/spacy)
    from benchmarks.demo_builder.phases import (  # noqa: E402 — deferred for ML deps
        PhaseChunks,
        PhaseChunksEmbeddings,
        PhaseChunksUMAP,
        PhaseCommunities,
        PhaseCommunityNaming,
        PhaseEntitiesUMAP,
        PhaseEntityEmbeddings,
        PhaseEntityResolution,
        PhaseMetadata,
        PhaseNER,
        PhaseNode2Vec,
        PhaseRE,
    )

    phase_slug = args.phase
    if phase_slug not in PHASE_NAMES:
        log.error("Unknown phase %r. Valid phases: %s", phase_slug, PHASE_NAMES)
        sys.exit(1)

    output_folder = _resolve_output_folder(args)
    book_id = args.book_id
    model_name = args.embedding_model

    assert Path(MUNINN_PATH).with_suffix(".dylib").exists() or Path(MUNINN_PATH).with_suffix(".so").exists(), (
        f"Muninn extension not found at {MUNINN_PATH}. Run: make all"
    )

    # Build a DemoBuild instance just for its path properties.
    build = DemoBuild(book_id, model_name, output_folder)
    perm_id = build.perm_id
    staging_db_path = build.staging_db_path
    phase_idx = PHASE_NAMES.index(phase_slug)
    is_first = phase_idx == 0
    is_last = phase_idx == len(PHASE_NAMES) - 1

    # ── Open / create staging DB ──────────────────────────────────────
    if is_first:
        build.staging_dir.mkdir(parents=True, exist_ok=True)
    else:
        assert staging_db_path.exists(), (
            f"Staging DB not found: {staging_db_path}\nRun the 'chunks' phase first to create it."
        )

    conn = sqlite3.connect(str(staging_db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    # Allow up to 2 minutes of retries when another run-phase process holds the
    # write lock. Required for parallel builds where ner || chunks_embeddings
    # both write to the same staging DB concurrently.
    conn.execute("PRAGMA busy_timeout = 120000")
    load_muninn(conn)

    if is_first:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _build_progress (
                phase INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                completed_at TEXT NOT NULL
            )
        """)
        conn.commit()

    # ── Build ctx ────────────────────────────────────────────────────
    ctx = PhaseContext()
    ctx.db_path = staging_db_path

    # Instantiate all phases so we can call restore_ctx on preceding ones.
    muninn_model = getattr(args, "muninn_model", None)
    if muninn_model:
        ner_backend = "muninn"
        re_backend = "muninn"
    elif args.legacy_models:
        ner_backend = "gliner"
        re_backend = "glirel"
    else:
        ner_backend = "gliner2"
        re_backend = "gliner2"
    all_phases = [
        PhaseChunks(book_id, model_name),
        PhaseChunksEmbeddings(book_id, model_name),
        PhaseChunksUMAP(),
        PhaseNER(backend=ner_backend, gguf_model=muninn_model),
        PhaseRE(backend=re_backend),
        PhaseEntityEmbeddings(model_name),
        PhaseEntitiesUMAP(),
        PhaseEntityResolution(),
        PhaseNode2Vec(),
        PhaseCommunities(),
        PhaseCommunityNaming(),
        PhaseMetadata(book_id, model_name),
    ]

    # Restore ctx from all phases that precede the target.
    for preceding_phase in all_phases[:phase_idx]:
        preceding_phase.restore_ctx(conn, ctx)

    # ── Run target phase ──────────────────────────────────────────────
    target_phase = all_phases[phase_idx]
    log.info("run-phase [%d/%d] %s — %s", phase_idx + 1, len(PHASE_NAMES), phase_slug, perm_id)
    target_phase(conn, ctx)
    conn.execute(
        "INSERT OR REPLACE INTO _build_progress (phase, name, completed_at) VALUES (?, ?, ?)",
        (phase_idx + 1, phase_slug, datetime.datetime.now(datetime.UTC).isoformat()),
    )
    conn.commit()

    # ── Finalize on last phase ────────────────────────────────────────
    if is_last:
        log.info("  Finalizing: VACUUM + atomic move")
        conn.execute("DROP TABLE IF EXISTS _build_progress")
        conn.commit()
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("VACUUM")
        conn.close()

        output_folder.mkdir(parents=True, exist_ok=True)
        final_path = build.final_path
        shutil.move(str(staging_db_path), str(final_path))
        for joblib_file in build.staging_dir.glob("*.joblib"):
            shutil.move(str(joblib_file), str(output_folder / joblib_file.name))

        db_size = final_path.stat().st_size
        log.info("Done! %s (%.1f MB)", final_path.name, db_size / 1e6)
        write_manifest_json(output_folder)
        return

    if conn is not None:
        conn.close()


def _cmd_build_status(args: argparse.Namespace) -> None:
    """Print build status for a single permutation without running anything."""
    from benchmarks.demo_builder.build import DemoBuild, get_build_status

    output_folder = _resolve_output_folder(args)
    build = DemoBuild(args.book_id, args.embedding_model, output_folder)

    # Check final path first, then staging path.
    if build.final_path.exists():
        db_path = build.final_path
        location = "final"
    elif build.staging_db_path.exists():
        db_path = build.staging_db_path
        location = "staging"
    else:
        print(f"Database not found for {build.perm_id}")
        print(f"  Final   : {build.final_path}")
        print(f"  Staging : {build.staging_db_path}")
        print("Run 'build' to create it.")
        return

    s = get_build_status(db_path)
    completed = s["completed_phases"]
    n_phases = len(PHASE_NAMES)

    print()
    print("=== demo_builder Build Status ===")
    print(f"Database : {s['db_path']}")
    print(f"Location : {location}")
    print(f"Size     : {_fmt_bytes(s['db_size_bytes'])}")
    print()

    # ── Phase progress ────────────────────────────────────────────
    print("Phase Progress:")
    for i, name in enumerate(PHASE_NAMES, 1):
        if s["build_finalized"]:
            # Build complete — all phases are done (table was dropped).
            status_icon = "✓"
            state_str = "complete"
        elif name in completed:
            status_icon = "✓"
            _phase_num, ts = completed[name]
            ts_str = ts[11:19] if len(ts) >= 19 else ts
            state_str = f"last:{ts_str}"
        else:
            status_icon = " "
            state_str = "[pending]"
        counts = s["phase_counts"].get(name)
        if counts is not None:
            done_val, pend_val = counts
            counts_str = f"  done:{done_val:>8,}  pend:{pend_val:>8,}"
        else:
            counts_str = ""
        print(f"  [{status_icon}] {i:>2}/{n_phases}  {name:<22}  {state_str:<24}{counts_str}")
    print()

    # ── Current data ──────────────────────────────────────────────
    print("Current Data:")
    print(f"  Chunks              : {s['chunks']:>10,}  ({s['chunks_embedded']:,} embedded)")
    print(f"  Entities            : {s['entities']:>10,}  ({s['entities_embedded']:,} embedded)")
    print(f"  Relations           : {s['relations']:>10,}")
    print(f"  Nodes / Edges       : {s['nodes']:,} / {s['edges']:,}")
    print(f"  N2V embeddings      : {s['n2v_embeddings']:>10,}")
    print(f"  UMAP (chunks)       : {s['chunks_umap']:>10,}")
    print(f"  UMAP (entities)     : {s['entities_umap']:>10,}")
    print(f"  Entity clusters     : {s['entity_clusters']:>10,}")
    print()


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
            "No books found. Ensure benchmarks/texts/gutenberg_{id}.txt exists."
            " Run: uv run -m benchmarks.harness prep texts"
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
    muninn_model = getattr(args, "muninn_model", None)
    build = DemoBuild(
        args.book_id,
        args.embedding_model,
        output_folder,
        legacy_models=args.legacy_models,
        muninn_model=muninn_model,
    )
    build.setup()
    try:
        build.run()
    finally:
        build.teardown()

    # ── Update manifest.json after successful build ───────────────
    write_manifest_json(output_folder)


def _cmd_write_manifest(args: argparse.Namespace) -> None:
    """Handle the 'write-manifest' subcommand."""
    output_folder = _resolve_output_folder(args)
    if not output_folder.exists():
        log.error("Output folder does not exist: %s", output_folder)
        sys.exit(1)
    write_manifest_json(output_folder)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build KG demo databases for viz demos.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # ── manifest ──────────────────────────────────────────────────
    manifest_p = subparsers.add_parser("manifest", help="Show permutation status and generate commands")
    manifest_p.add_argument(
        "--output-folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Output folder to check (default: {DEFAULT_OUTPUT_FOLDER})",
    )
    manifest_p.add_argument("--missing", action="store_true", help="Only show missing permutations")
    manifest_p.add_argument("--done", action="store_true", help="Only show completed permutations")
    manifest_p.add_argument(
        "--book-id",
        type=int,
        default=None,
        dest="book_id",
        help="Filter to a specific Gutenberg book ID",
    )
    manifest_p.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        dest="embedding_model",
        choices=list(EMBEDDING_MODELS.keys()),
        help="Filter to a specific embedding model",
    )
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
    manifest_p.add_argument(
        "--makefile",
        action="store_true",
        help=(
            "Print a Make-managed parallel build Makefile to stdout. "
            "Pipe directly to make with: "
            "manifest [--book-id N] [--embedding-model M] --makefile | make -f - all -j8"
        ),
    )

    # ── build ─────────────────────────────────────────────────────
    build_p = subparsers.add_parser("build", help="Build a single demo database")
    build_p.add_argument(
        "--status",
        action="store_true",
        help="Show build status and table counts without running anything",
    )
    build_p.add_argument(
        "--output-folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Output folder for generated databases (default: {DEFAULT_OUTPUT_FOLDER})",
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
    build_p.add_argument(
        "--legacy-models",
        action="store_true",
        dest="legacy_models",
        default=False,
        help="Use legacy GLiNER + GLiREL + spaCy stack instead of GLiNER2 for NER and RE phases",
    )
    build_p.add_argument(
        "--muninn-model",
        type=str,
        default=None,
        dest="muninn_model",
        help="Use muninn LLM NER+RE via this GGUF model filename (e.g. Qwen3-4B-Q4_K_M.gguf)",
    )

    # ── write-manifest ────────────────────────────────────────────
    wm_p = subparsers.add_parser("write-manifest", help="Generate manifest.json from existing built DBs")
    wm_p.add_argument(
        "--output-folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Output folder containing built databases (default: {DEFAULT_OUTPUT_FOLDER})",
    )

    # ── run-phase ─────────────────────────────────────────────────
    rp = subparsers.add_parser(
        "run-phase",
        help="Run a single named phase against the staging DB (called by generated Makefiles)",
    )
    rp.add_argument(
        "--output-folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Output folder for generated databases (default: {DEFAULT_OUTPUT_FOLDER})",
    )
    rp.add_argument("--book-id", type=int, required=True, help="Gutenberg book ID")
    rp.add_argument(
        "--embedding-model",
        type=str,
        required=True,
        choices=list(EMBEDDING_MODELS.keys()),
        help="Embedding model to use",
    )
    rp.add_argument(
        "phase",
        choices=PHASE_NAMES,
        help=f"Phase to run. One of: {', '.join(PHASE_NAMES)}",
    )
    rp.add_argument(
        "--legacy-models",
        action="store_true",
        dest="legacy_models",
        default=False,
        help="Use legacy GLiNER + GLiREL + spaCy stack instead of GLiNER2 for NER and RE phases",
    )
    rp.add_argument(
        "--muninn-model",
        type=str,
        default=None,
        dest="muninn_model",
        help="Use muninn LLM NER+RE via this GGUF model filename (e.g. Qwen3-4B-Q4_K_M.gguf)",
    )

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
        if args.status:
            _cmd_build_status(args)
        else:
            _cmd_build(args)
    elif args.command == "run-phase":
        _cmd_run_phase(args)
    elif args.command == "write-manifest":
        _cmd_write_manifest(args)
    elif args.command == "list-books":
        print_books()
    elif args.command == "list-models":
        print_models()
