"""SessionsBuild: lifecycle manager for building the sessions demo database.

Handles schema initialization, phase orchestration, and cleanup.

Public API follows a setup/run/teardown lifecycle:
    build = SessionsBuild(db_path)
    build.setup()
    try:
        build.run()
    finally:
        build.teardown()
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.sessions_demo.phases import Phase, default_phases

log = logging.getLogger(__name__)


@dataclass
class PhaseContext:
    """Mutable bag carrying inter-phase data.

    Each phase writes its outputs here; later phases read what they need.
    """

    db_path: Path = Path("")
    events_ingested: int = 0
    files_updated: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0


class SessionsBuild:
    """Manages the lifecycle of building the sessions demo database.

    Unlike DemoBuild (which uses staging dirs and atomic moves), this builder
    operates in-place on the target DB since the cache is designed for
    incremental updates.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._ctx = PhaseContext(db_path=db_path)
        self._conn: sqlite3.Connection | None = None
        self._log = logging.getLogger("benchmarks.sessions_demo.build")

    @property
    def _db(self) -> sqlite3.Connection:
        """Return the open database connection; raises if not yet opened."""
        assert self._conn is not None, "Database not opened — call setup() first"
        return self._conn

    # ── Public API ────────────────────────────────────────────────

    def setup(self) -> None:
        """Create output directory and open DB."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_build_progress()
        self._log.info("Database opened: %s", self._db_path)

    def run(self, phases: list[Phase] | None = None) -> None:
        """Execute phases sequentially. Default: default_phases()."""
        if phases is None:
            phases = default_phases()

        t_total = time.monotonic()

        self._log.info("=" * 60)
        self._log.info("Building sessions demo: %s", self._db_path.name)
        self._log.info("=" * 60)

        for i, phase in enumerate(phases, 1):
            self._log.info("Phase %d/%d: %s", i, len(phases), phase.name)
            t0 = time.monotonic()
            phase(self._db, self._ctx)
            self._db.commit()
            self._record_phase(i, phase.name)
            self._db.commit()
            self._log.info("Phase %d complete (%.1fs)", i, time.monotonic() - t0)

        elapsed = time.monotonic() - t_total
        self._log.info("All phases complete (%.1fs total)", elapsed)
        self._drop_build_progress()

    def teardown(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._log.info("Database closed")

    # ── Build progress tracking ──────────────────────────────────

    def _init_build_progress(self) -> None:
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS _build_progress (
                phase INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                completed_at TEXT NOT NULL
            )
        """)
        self._db.commit()

    def _record_phase(self, phase_num: int, phase_name: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO _build_progress (phase, name, completed_at) VALUES (?, ?, ?)",
            (phase_num, phase_name, datetime.datetime.now(datetime.UTC).isoformat()),
        )

    def _drop_build_progress(self) -> None:
        """Drop internal bookkeeping table (not a deliverable)."""
        self._db.execute("DROP TABLE IF EXISTS _build_progress")
        self._db.commit()
