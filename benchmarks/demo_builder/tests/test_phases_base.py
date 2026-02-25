"""Tests for the Phase ABC contract."""

from __future__ import annotations

import sqlite3

import pytest

from benchmarks.demo_builder.build import PhaseContext
from benchmarks.demo_builder.phases.base import Phase


class ConcretePhase(Phase):
    """Minimal concrete Phase for testing the ABC contract."""

    def __init__(self) -> None:
        self.setup_called = False
        self.run_called = False
        self.teardown_called = False
        self.call_order: list[str] = []

    @property
    def name(self) -> str:
        return "test_phase"

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        self.setup_called = True
        self.call_order.append("setup")

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        self.run_called = True
        self.call_order.append("run")

    def teardown(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        self.teardown_called = True
        self.call_order.append("teardown")


class DefaultLifecyclePhase(Phase):
    """Phase with only run() implemented — tests that default setup/teardown are no-ops."""

    @property
    def name(self) -> str:
        return "default_lifecycle"

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        pass


class TestPhaseABC:
    def test_call_invokes_lifecycle_in_order(self) -> None:
        phase = ConcretePhase()
        conn = sqlite3.connect(":memory:")
        ctx = PhaseContext()

        phase(conn, ctx)

        assert phase.setup_called
        assert phase.run_called
        assert phase.teardown_called
        assert phase.call_order == ["setup", "run", "teardown"]
        conn.close()

    def test_default_setup_teardown_are_noops(self) -> None:
        phase = DefaultLifecyclePhase()
        conn = sqlite3.connect(":memory:")
        ctx = PhaseContext()

        # Should not raise
        phase(conn, ctx)
        conn.close()

    def test_name_property(self) -> None:
        phase = ConcretePhase()
        assert phase.name == "test_phase"

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            Phase()  # type: ignore[abstract]

    def test_phase_is_callable(self) -> None:
        phase = ConcretePhase()
        assert callable(phase)
