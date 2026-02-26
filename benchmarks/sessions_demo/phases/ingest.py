"""Phase 1: Ingest — Cache JSONL files into events + event_edges + FTS."""

from __future__ import annotations

import logging
import sqlite3
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
