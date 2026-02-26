"""CLI interface for sessions_demo — build and cache subcommands.

Usage:
    uv run -m benchmarks.sessions_demo build
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

from benchmarks.sessions_demo.constants import DEFAULT_DB_NAME, DEFAULT_OUTPUT_DIR

log = logging.getLogger(__name__)


def _cmd_cache(args: argparse.Namespace) -> None:
    """Handle cache subcommands."""
    from benchmarks.sessions_demo.cache import CacheManager

    db_path = args.output_dir / args.db_name
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


def _cmd_build(args: argparse.Namespace) -> None:
    """Handle build subcommand."""
    from benchmarks.sessions_demo.build import SessionsBuild

    db_path = args.output_dir / args.db_name
    build = SessionsBuild(db_path)
    build.setup()
    try:
        build.run()
    finally:
        build.teardown()


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
        "--output-dir",
        type=lambda p: __import__("pathlib").Path(p),
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--db-name",
        default=DEFAULT_DB_NAME,
        help=f"Database filename (default: {DEFAULT_DB_NAME})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── build subcommand ─────────────────────────────────────────
    subparsers.add_parser("build", help="Run full build pipeline (ingest + embeddings + chunks)")

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
        _cmd_build(args)
    elif args.command == "cache":
        _cmd_cache(args)
    else:
        parser.print_help()
        sys.exit(1)
