"""Phase ABC — the contract for all build phases.

Each phase follows a setup/run/teardown lifecycle:
- setup(): Create tables, load models, prepare preconditions (default: no-op)
- run(): Execute main phase logic (abstract — must implement)
- teardown(): Clean up temporary tables/resources (default: no-op)

Phases are callable: phase(conn, ctx) invokes the full lifecycle.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext


class Phase(ABC):
    """Abstract base class for build phases."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable phase name (matches PHASE_NAMES constant)."""

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
