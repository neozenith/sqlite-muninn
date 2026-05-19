"""Timing + fidelity scoring + JSONL persistence for one (strategy, workload) run."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import statistics
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from shlex import split

import sqlite_muninn
from benchmarks.kg_perf.constants import REPETITIONS, RESULTS_DIR, WARMUP
from benchmarks.kg_perf.strategies._base import Result, Strategy
from benchmarks.kg_perf.workload import Workload

log = logging.getLogger(__name__)


# G3 T3.1 — Per-component timing. Strategies that opt in stuff a
# PhaseTimings into Result.extras["phases"]; bench.py reads it,
# computes brandes_share, and includes both in the JSONL record.
# The un-defer trigger fires on three consecutive runs where
# brandes_share > MUNINN_BRANDES_SHARE_THRESHOLD (T3.3).
@dataclass
class PhaseTimings:
    """Per-component wall-clock breakdown for one strategy.run() call.

    Phases are arbitrary string keys; the only one bench.py treats
    specially is ``centrality_call``, which becomes the numerator of
    brandes_share. Strategies use the ``measure(name)`` context manager
    to accumulate elapsed time per phase across multiple invocations.
    """

    phases_ms: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.phases_ms[name] = self.phases_ms.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0

    def total_ms(self) -> float:
        return sum(self.phases_ms.values())

    def brandes_share(self) -> float:
        """Ratio of centrality_call time to total. Returns 0.0 if no
        centrality phase was recorded — strategies that don't run
        Brandes (e.g., degree-only strategies) should land here."""
        total = self.total_ms()
        if total <= 0.0:
            return 0.0
        return self.phases_ms.get("centrality_call", 0.0) / total

    def sums_to(self, expected_total_ms: float, tol_ms: float = 1.0) -> bool:
        """Consistency check used by T3.1's property test: the sum of
        per-component times must match the externally-measured total
        within tolerance (small slack for context-manager overhead)."""
        return abs(self.total_ms() - expected_total_ms) <= tol_ms


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_muninn.load(conn)
    conn.enable_load_extension(False)
    return conn


def _result_signature(r: Result) -> dict[str, object]:
    nodes_blob = ",".join(sorted(r.node_set)).encode("utf-8")
    edges_blob = ",".join(sorted(f"{a}\x00{b}" for a, b in r.edge_set)).encode("utf-8")
    seeds_blob = ",".join(sorted(r.seed_set)).encode("utf-8")
    return {
        "node_count": len(r.node_set),
        "edge_count": len(r.edge_set),
        "seed_count": len(r.seed_set),
        "node_hash": hashlib.sha256(nodes_blob).hexdigest()[:16],
        "edge_hash": hashlib.sha256(edges_blob).hexdigest()[:16],
        "seed_hash": hashlib.sha256(seeds_blob).hexdigest()[:16],
        # Persist the actual lists (tiny — top_k seeds, ~100s of nodes)
        # so we can compute Jaccard / set-similarity later without re-running.
        "seeds": sorted(r.seed_set),
        "nodes": sorted(r.node_set),
    }


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def _git_sha() -> str:
    try:
        return subprocess.check_output(split("git rev-parse --short HEAD"), text=True).strip()
    except Exception:
        return "unknown"


def time_one(strategy: Strategy, workload: Workload) -> dict[str, object]:
    """Execute one strategy+workload pair, return a JSONL-friendly record."""
    conn = open_db(workload.db_path)
    try:
        strategy.prepare(conn)
        for _ in range(WARMUP):
            strategy.run(conn, workload)

        times_ms: list[float] = []
        last_result: Result | None = None
        for _ in range(REPETITIONS):
            t0 = time.perf_counter()
            last_result = strategy.run(conn, workload)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        assert last_result is not None
    finally:
        conn.close()

    record: dict[str, object] = {
        "permutation_id": f"{strategy.name}__{workload.slug}",
        "strategy": strategy.name,
        "db_stem": workload.db_path.stem,
        "filter": {"project_id": workload.filter.project_id, "days": workload.filter.days},
        "query": {
            "metric": workload.query.metric,
            "top_k": workload.query.top_k,
            "depth": workload.query.depth,
            "min_degree": workload.query.min_degree,
        },
        "wall_ms": {
            "p50": statistics.median(times_ms),
            "p95": _p95(times_ms),
            "min": min(times_ms),
            "mean": statistics.mean(times_ms),
        },
        "samples_ms": [round(t, 3) for t in times_ms],
        "signature": _result_signature(last_result),
        "git_sha": _git_sha(),
        "timestamp": datetime.now(UTC).isoformat(),
        "repetitions": REPETITIONS,
    }

    # G3 T3.1 — record phase breakdown when the strategy provided it.
    # Strategies opt in by stashing a PhaseTimings into
    # Result.extras["phases"]; missing extras leaves the fields absent
    # so legacy strategies aren't disturbed.
    phases = last_result.extras.get("phases") if last_result else None
    if isinstance(phases, PhaseTimings) and phases.phases_ms:
        record["phases_ms"] = {k: round(v, 3) for k, v in phases.phases_ms.items()}
        record["brandes_share"] = round(phases.brandes_share(), 6)
    return record


def _p95(xs: list[float]) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    idx = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return s[idx]


def append_record(record: dict[str, object]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{record['strategy']}.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return out


def fidelity_against(record: dict[str, object], baseline: dict[str, object]) -> dict[str, float]:
    """Soft fidelity: Jaccard on seeds + Jaccard on expanded subgraph nodes.

    A semantically-different strategy (e.g., Brandes on induced subgraph vs full
    graph) WILL diverge on exact equality but should still produce a high-Jaccard
    answer set. Anything <0.5 means the strategy is answering a meaningfully
    different question.
    """
    sig = cast(dict, record["signature"])
    base = cast(dict, baseline["signature"])
    return {
        "seed_jaccard": _jaccard(frozenset(sig.get("seeds", [])), frozenset(base.get("seeds", []))),
        "node_jaccard": _jaccard(frozenset(sig.get("nodes", [])), frozenset(base.get("nodes", []))),
    }


from typing import cast  # placed after definition to keep diff minimal  # noqa: E402

__all__ = ["PhaseTimings", "append_record", "fidelity_against", "open_db", "time_one"]
