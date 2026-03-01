"""Phase ABC — the contract for all build phases.

Each phase follows a setup/run/teardown lifecycle:
- setup(): Create tables, load models, prepare preconditions (default: no-op)
- run(): Execute main phase logic (abstract — must implement)
- teardown(): Clean up temporary tables/resources (default: no-op)

DAG-aware incremental execution:
- is_stale(conn): Return True if this phase has pending work (default: True).
  Must be read-only DB queries — no model loading or side effects.
- restore_ctx(conn, ctx): Repopulate ctx fields from existing DB data when the
  phase is skipped (default: no-op). Downstream phases read ctx for their inputs.

Phases are callable: phase(conn, ctx) invokes the full lifecycle.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.sessions_demo.build import PhaseContext


class Phase(ABC):
    """Abstract base class for build phases."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable phase name."""

    def is_stale(self, conn: sqlite3.Connection) -> bool:  # noqa: B027
        """Return True if this phase has pending work. Default: always stale."""
        return True

    def restore_ctx(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:  # noqa: B027
        """Repopulate ctx fields from DB when phase is skipped. Default: no-op."""

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:  # noqa: B027
        """Create tables, load models, prepare preconditions. Default: no-op."""

    @abstractmethod
    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        """Execute main phase logic."""

    def teardown(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:  # noqa: B027
        """Clean up temporary tables/resources. Default: no-op."""

    def __call__(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        """Entry point: setup -> run -> teardown."""
        self.setup(conn, ctx)
        self.run(conn, ctx)
        self.teardown(conn, ctx)
