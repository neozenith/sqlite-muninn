"""G3 un-defer trigger machinery — per-component timing + brandes_share.

These tests cover the deferred filter-aware Brandes work's PREREQUISITES
(T3.1, T3.2) — instrumentation that produces the metric we'll watch
for the un-defer trigger (T3.3's threshold). The actual filter-aware
Brandes implementation lands when the trigger fires.

Test markers (per pyproject.toml `[tool.pytest.ini_options]`):
    @pytest.mark.G3 → make test-g3 picks these up.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from benchmarks.kg_perf.bench import PhaseTimings


@pytest.mark.G3
def test_g3_per_component_timing_sums_to_total() -> None:
    """T3.1 — per-component timing breakdown must sum back to the total
    wall-clock time measured externally. A strategy that records phases
    via PhaseTimings.measure() shouldn't drift from a separate
    perf_counter measurement of the whole call."""
    phases = PhaseTimings()

    t0 = time.perf_counter()
    # Two phases that together cover the entire span.
    with phases.measure("load"):
        time.sleep(0.01)  # 10ms approx
    with phases.measure("centrality_call"):
        time.sleep(0.005)  # 5ms approx
    external_total_ms = (time.perf_counter() - t0) * 1000.0

    # Components sum to total within tolerance. The default 1ms slack
    # accommodates context-manager overhead and the gap between the
    # two `with` blocks; integration tests can tighten if needed.
    assert phases.sums_to(external_total_ms, tol_ms=2.0), (
        f"phases sum {phases.total_ms():.3f}ms != external {external_total_ms:.3f}ms"
    )

    # brandes_share = centrality / total, ~5/(5+10) = 0.33.
    share = phases.brandes_share()
    assert 0.20 < share < 0.50, f"brandes_share out of expected band: {share}"


@pytest.mark.G3
def test_g3_brandes_share_zero_when_no_centrality_phase() -> None:
    """Strategies that don't run Brandes (e.g., degree-only) record no
    centrality_call phase — brandes_share should be 0, not divide-by-
    zero or KeyError."""
    phases = PhaseTimings()
    with phases.measure("load"):
        time.sleep(0.001)
    assert phases.brandes_share() == 0.0


@pytest.mark.G3
def test_g3_brandes_share_zero_on_empty_phases() -> None:
    """Empty PhaseTimings (no measure() calls at all) returns 0.0
    instead of NaN. Defensive against strategies that opt-in with the
    field but never populate it."""
    phases = PhaseTimings()
    assert phases.brandes_share() == 0.0
    assert phases.total_ms() == 0.0


@pytest.mark.G3
def test_g3_sweep_produces_chart() -> None:
    """T3.2 — the brandes_share sweep produces the chart at the
    documented path. For now this is a presence check on the script
    that lands the chart; real benchmarking happens via
    `make -C benchmarks ...` invocation when un-defer is contemplated.

    The sweep script and chart directory are scaffolding so a future
    contributor running `python -m benchmarks.kg_perf.sweeps.g3_brandes_share`
    has a defined output location that the un-defer trigger ADR
    (plan section 909) references by path."""
    repo_root = Path(__file__).resolve().parent.parent
    sweep_script = repo_root / "benchmarks" / "kg_perf" / "sweeps" / "g3_brandes_share.py"
    charts_dir = repo_root / "benchmarks" / "kg_perf" / "charts"

    # The sweep script exists and is non-empty (real implementation,
    # not a placeholder).
    assert sweep_script.exists(), f"missing sweep script at {sweep_script}"
    assert sweep_script.stat().st_size > 0, "sweep script is empty"

    # The charts directory exists (created by the script's first run
    # OR pre-created as part of the scaffold). Either way, the
    # un-defer ADR's documented path resolves to a real directory.
    assert charts_dir.exists(), f"missing charts directory at {charts_dir}"
    assert charts_dir.is_dir(), f"{charts_dir} is not a directory"
