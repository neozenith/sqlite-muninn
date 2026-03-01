"""Phase 1: Ingest — Cache JSONL files into events + event_edges + FTS."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.sessions_demo.cache import CacheManager

if TYPE_CHECKING:
    from benchmarks.sessions_demo.build import PhaseContext

log = logging.getLogger(__name__)


class PhaseIngest:
    """Ingest JSONL session logs into the events table.

    Uses CacheManager for incremental discovery and parsing of
    ~/.claude/projects/**/*.jsonl files. Creates events, event_edges,
    events_fts, projects, and sessions tables.
    """

    @property
    def name(self) -> str:
        return "ingest"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        """Return True if any JSONL files are new or modified since last ingest."""
        try:
            # No source files recorded yet → definitely stale.
            count = conn.execute("SELECT count(*) FROM source_files").fetchone()[0]
            if count == 0:
                return True
            # CacheManager.get_files_needing_update() uses cached["mtime"] (dict-style
            # column access), which requires row_factory=sqlite3.Row.
            conn.row_factory = sqlite3.Row
            # Compare disk mtimes against source_files records via CacheManager.
            # PRAGMA database_list row: (seq, name, file) — seq=0 is the main DB.
            db_path = Path(conn.execute("PRAGMA database_list").fetchone()[2])
            cache = CacheManager(db_path)
            cache._conn = conn  # share the open connection, skip auto-connect
            all_files = cache.discover_files()
            return len(cache.get_files_needing_update(all_files)) > 0
        except Exception:
            return True

    def restore_ctx(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        """Nothing new was ingested; counts stay at zero (no new events this run)."""
        ctx.events_ingested = 0
        ctx.files_updated = 0

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        pass

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        # CacheManager needs row_factory=Row for dict-style column access
        conn.row_factory = sqlite3.Row
        cache = CacheManager(ctx.db_path)
        cache._conn = conn  # Share the existing connection

        # init_schema is idempotent — CREATE TABLE IF NOT EXISTS.
        # For schema version changes, use `cache rebuild` first.
        cache.init_schema()

        result = cache.update()
        ctx.events_ingested = result["events_added"]
        ctx.files_updated = result["files_updated"]

        status = cache.get_status()
        log.info(
            "Ingest complete: %d events, %d edges, %d projects, %d sessions",
            status["events"],
            status["event_edges"],
            status["projects"],
            status["sessions"],
        )

    def teardown(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        pass

    def __call__(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        self.setup(conn, ctx)
        self.run(conn, ctx)
        self.teardown(conn, ctx)
