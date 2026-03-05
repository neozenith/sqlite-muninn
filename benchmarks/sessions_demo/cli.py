"""CLI interface for sessions_demo — build, run-phase, and cache subcommands.

Usage:
    uv run -m benchmarks.sessions_demo build
    uv run -m benchmarks.sessions_demo build --status
    uv run -m benchmarks.sessions_demo build --output-folder viz/frontend/public/demos
    uv run -m benchmarks.sessions_demo run-phase ner
    uv run -m benchmarks.sessions_demo run-phase entity_resolution
    uv run -m benchmarks.sessions_demo run-phase metadata
    uv run -m benchmarks.sessions_demo cache init
    uv run -m benchmarks.sessions_demo cache update
    uv run -m benchmarks.sessions_demo cache rebuild
    uv run -m benchmarks.sessions_demo cache clear
    uv run -m benchmarks.sessions_demo cache status
"""

from __future__ import annotations

import argparse
import logging
import sys

from benchmarks.sessions_demo.constants import (
    ALL_MESSAGE_TYPES,
    DEFAULT_DB_NAME,
    DEFAULT_MESSAGE_TYPES,
    DEFAULT_OUTPUT_FOLDER,
    PHASE_NAMES,
)

log = logging.getLogger(__name__)


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def _cmd_build_status(args: argparse.Namespace) -> None:
    """Print build status and pending-work delta without running anything."""
    from benchmarks.sessions_demo.build import SessionsBuild

    db_path = args.output_folder / args.db_name
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run 'build' to create it.")
        return

    build = SessionsBuild(db_path)
    build.setup()
    try:
        s = build.get_build_status()
    finally:
        build.teardown()

    completed = s["completed_phases"]  # {name: (phase_num, completed_at)}

    print()
    print("=== sessions_demo Build Status ===")
    print(f"Database : {s['db_path']}")
    print(f"Size     : {_fmt_bytes(s['db_size_bytes'])}")
    print()

    phase_stale = s.get("phase_stale", {})
    phase_counts = s.get("phase_counts", {})
    n_phases = len(PHASE_NAMES)

    # ── Phase progress ────────────────────────────────────────────
    print("Phase Progress:")
    for i, name in enumerate(PHASE_NAMES, 1):
        stale = phase_stale.get(name, True)
        status_icon = "✓" if not stale else " "
        ts_str = ""
        if name in completed:
            _phase_num, ts = completed[name]
            ts_str = ts[11:19] if len(ts) >= 19 else ts
        if ts_str:
            state_str = f"last:{ts_str}" + (" [stale]" if stale else "")
        else:
            state_str = "[pending]" if stale else ""
        counts = phase_counts.get(name)
        if counts is not None:
            done_val, pend_val = counts
            counts_str = f"  done:{done_val:>8,}  pend:{pend_val:>8,}"
        else:
            counts_str = ""
        print(f"  [{status_icon}] {i:>2}/{n_phases}  {name:<22}  {state_str:<24}{counts_str}")
    print()

    # ── Current data ──────────────────────────────────────────────
    print("Current Data:")
    print(f"  Events            : {s['events']:>10,}")
    print(f"  Chunks            : {s['chunks']:>10,}  ({s['chunks_embedded']:,} embedded)")
    print(f"  Entities          : {s['entities']:>10,}")
    print(f"  Relations         : {s['relations']:>10,}")
    print(f"  Nodes / Edges     : {s['nodes']:,} / {s['edges']:,}")
    print(f"  N2V embeddings    : {s['n2v_embeddings']:>10,}")
    print()

    # ── Pending work ──────────────────────────────────────────────
    print("Pending Work:")
    if s["jsonl_files_changed"] >= 0:
        print(f"  JSONL files changed   : {s['jsonl_files_changed']:>6}  (of {s['jsonl_files_total']} total)")
    else:
        print("  JSONL files changed   :      ? (could not check)")
    print(f"  Chunks to embed       : {s['chunks_to_embed']:>6}")
    if s["kg_rebuild_needed"]:
        incomplete = ", ".join(s["kg_incomplete_phases"]) if s["kg_incomplete_phases"] else "new chunks detected"
        print(f"  KG rebuild needed     :    YES  ({incomplete})")
    else:
        print("  KG rebuild needed     :     NO")
    print()


def _cmd_cache(args: argparse.Namespace) -> None:
    """Handle cache subcommands."""
    from benchmarks.sessions_demo.cache import CacheManager

    db_path = args.output_folder / args.db_name
    cache = CacheManager(db_path)

    try:
        if args.cache_command == "init":
            cache.init_schema()
            print(f"Cache initialized: {db_path}")

        elif args.cache_command == "update":
            if cache.needs_rebuild():
                log.info("Schema version mismatch — destroying and reinitializing")
                cache.destroy()
                cache.init_schema()
            result = cache.update()
            print(f"Updated: {result['files_updated']} files, {result['events_added']} events")

        elif args.cache_command == "rebuild":
            cache.destroy()
            cache.init_schema()
            result = cache.update()
            print(f"Rebuilt: {result['files_updated']} files, {result['events_added']} events")

        elif args.cache_command == "clear":
            cache.clear()
            print("Cache cleared")

        elif args.cache_command == "status":
            if cache.needs_rebuild():
                print("Cache needs rebuild (schema version mismatch)")
                return
            status = cache.get_status()
            print(f"Database: {status['db_path']}")
            print(f"Size: {status['db_size_mb']} MB")
            print(f"Source files: {status['source_files']}")
            print(f"Projects: {status['projects']}")
            print(f"Sessions: {status['sessions']}")
            print(f"Events: {status['events']}")
            print(f"Event edges: {status['event_edges']}")
            print(f"Created: {status['created_at']}")
            print(f"Last update: {status['last_update_at']}")

    finally:
        cache.close()


def _parse_message_types(raw: str) -> list[str]:
    """Parse and validate a comma-separated message types string."""
    types = [t.strip() for t in raw.split(",") if t.strip()]
    unknown = [t for t in types if t not in ALL_MESSAGE_TYPES]
    if unknown:
        print(f"Error: unknown message type(s): {', '.join(unknown)}")
        print(f"Valid types: {', '.join(ALL_MESSAGE_TYPES)}")
        sys.exit(1)
    return types


def _cmd_run_phase(args: argparse.Namespace) -> None:
    """Handle run-phase subcommand — run a single phase standalone."""
    from benchmarks.demo_builder.manifest import write_manifest_json
    from benchmarks.sessions_demo.build import SessionsBuild

    message_types = _parse_message_types(args.message_types)
    db_path = args.output_folder / args.db_name
    if not db_path.exists():
        print(f"Error: database not found: {db_path}")
        print("Run 'build' first to create it.")
        sys.exit(1)

    build = SessionsBuild(db_path)
    build.setup()
    try:
        build.run_single_phase(args.phase, message_types=message_types, legacy_models=args.legacy_models)
    finally:
        build.teardown()

    # Update manifest if metadata was (re-)written so viz frontend sees fresh counts.
    if args.phase == "metadata":
        write_manifest_json(args.output_folder)
        log.info("Updated manifest.json in %s", args.output_folder)


def _cmd_build(args: argparse.Namespace) -> None:
    """Handle build subcommand."""
    from benchmarks.demo_builder.manifest import write_manifest_json
    from benchmarks.sessions_demo.build import SessionsBuild

    if args.run_from and args.run_from not in PHASE_NAMES:
        print(f"Error: unknown phase {args.run_from!r}")
        print(f"Valid phase names: {', '.join(PHASE_NAMES)}")
        sys.exit(1)

    message_types = _parse_message_types(args.message_types)

    db_path = args.output_folder / args.db_name
    build = SessionsBuild(db_path)
    build.setup()
    try:
        build.run(
            run_from=args.run_from,
            message_types=message_types,
            legacy_models=args.legacy_models,
        )
    finally:
        build.teardown()

    # Update shared manifest.json so the viz frontend discovers sessions_demo.db
    # alongside any demo_builder DBs in the same output folder.
    write_manifest_json(args.output_folder)
    log.info("Updated manifest.json in %s", args.output_folder)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the sessions_demo CLI."""
    parser = argparse.ArgumentParser(
        prog="sessions_demo",
        description="Build a sessions demo DB from Claude Code JSONL logs",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--output-folder",
        type=lambda p: __import__("pathlib").Path(p),
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Output directory for generated database (default: {DEFAULT_OUTPUT_FOLDER})",
    )
    parser.add_argument(
        "--db-name",
        default=DEFAULT_DB_NAME,
        help=f"Database filename (default: {DEFAULT_DB_NAME})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── build subcommand ─────────────────────────────────────────
    build_parser = subparsers.add_parser(
        "build",
        help="Run full build pipeline (ingest + chunks + embeddings + KG pipeline)",
    )
    build_parser.add_argument(
        "--status",
        action="store_true",
        help="Show build status and pending work delta without running anything",
    )
    build_parser.add_argument(
        "--run-from",
        metavar="PHASE",
        default=None,
        dest="run_from",
        help=(
            "Force-skip all phases before PHASE (restoring their ctx from the DB) "
            "and start execution from PHASE. Useful for testing incremental behaviour "
            f"of downstream phases. Valid names: {', '.join(PHASE_NAMES)}"
        ),
    )
    build_parser.add_argument(
        "--message-types",
        default=",".join(DEFAULT_MESSAGE_TYPES),
        metavar="TYPES",
        help=(
            "Comma-separated filter for which events to chunk and feed into the KG. "
            f"Default: {','.join(DEFAULT_MESSAGE_TYPES)}. "
            "Use 'human' for genuine human-typed prompts only (user events with string "
            "content and isMeta=False — excludes tool results, skill injections, wrappers). "
            f"Other values: {', '.join(t for t in ALL_MESSAGE_TYPES if t != 'human')}. "
            "To switch filters on an existing DB, re-run with --run-from chunks."
        ),
    )
    build_parser.add_argument(
        "--legacy-models",
        action="store_true",
        dest="legacy_models",
        default=False,
        help="Use legacy GLiNER + GLiREL + spaCy stack instead of GLiNER2 for NER and RE phases",
    )
    # ── run-phase subcommand ──────────────────────────────────────
    run_phase_parser = subparsers.add_parser(
        "run-phase",
        help="Run a single build phase by name (restores ctx from DB for preceding phases)",
    )
    run_phase_parser.add_argument(
        "phase",
        metavar="PHASE",
        choices=PHASE_NAMES,
        help=f"Phase to run. Valid names: {', '.join(PHASE_NAMES)}",
    )
    run_phase_parser.add_argument(
        "--message-types",
        default=",".join(DEFAULT_MESSAGE_TYPES),
        metavar="TYPES",
        help=(
            "Comma-separated message types. Only relevant when running the 'chunks' phase. "
            f"Default: {','.join(DEFAULT_MESSAGE_TYPES)}."
        ),
    )
    run_phase_parser.add_argument(
        "--legacy-models",
        action="store_true",
        dest="legacy_models",
        default=False,
        help="Use legacy GLiNER + GLiREL + spaCy stack instead of GLiNER2 (only affects ner/relations phases)",
    )

    # ── cache subcommand ─────────────────────────────────────────
    cache_parser = subparsers.add_parser("cache", help="Cache management commands")
    cache_sub = cache_parser.add_subparsers(dest="cache_command", required=True)
    cache_sub.add_parser("init", help="Initialize SQLite cache schema")
    cache_sub.add_parser("update", help="Incremental update from JSONL files")
    cache_sub.add_parser("rebuild", help="Full cache rebuild (clear + update)")
    cache_sub.add_parser("clear", help="Clear all cached data")
    cache_sub.add_parser("status", help="Show cache status")

    args = parser.parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "build":
        if args.status:
            _cmd_build_status(args)
        else:
            _cmd_build(args)
    elif args.command == "run-phase":
        _cmd_run_phase(args)
    elif args.command == "cache":
        _cmd_cache(args)
    else:
        parser.print_help()
        sys.exit(1)
