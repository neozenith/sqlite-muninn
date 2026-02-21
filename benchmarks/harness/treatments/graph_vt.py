"""Graph VT (virtual table) caching treatment: 5 approaches compared.

Ports from benchmarks/scripts/benchmark_adjacency.py.
Approaches: tvf, csr, csr_full_rebuild, csr_incremental, csr_blocked.
"""

import logging
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

from benchmarks.harness.common import (
    GRAPH_VT_BLOCK_SIZE,
    generate_barabasi_albert,
    generate_erdos_renyi,
    load_muninn,
)
from benchmarks.harness.treatments.base import Treatment

log = logging.getLogger(__name__)

ALGORITHMS = ["degree", "betweenness", "closeness", "leiden"]

EDGE_TABLE = "bench_edges"


class GraphVtTreatment(Treatment):
    """Single graph VT benchmark permutation."""

    def __init__(self, approach: str, workload: str, n_nodes: int, target_edges: int, graph_model: str) -> None:
        self._approach = approach
        self._workload = workload
        self._n_nodes = n_nodes
        self._target_edges = target_edges
        self._graph_model = graph_model
        self._edges: list[tuple[int, int, float]] | None = None

    @property
    def category(self) -> str:
        return "graph_vt"

    @property
    def permutation_id(self) -> str:
        return f"graph_vt_{self._approach}_{self._workload}_{self._graph_model}"

    @property
    def label(self) -> str:
        return f"Graph VT: {self._approach} / {self._workload} / {self._graph_model} / N={self._n_nodes}"

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (self._n_nodes, self._target_edges, self._approach)

    def params_dict(self) -> dict[str, Any]:
        return {
            "approach": self._approach,
            "workload": self._workload,
            "n_nodes": self._n_nodes,
            "target_edges": self._target_edges,
            "graph_model": self._graph_model,
        }

    def setup(self, conn: sqlite3.Connection, db_path: Path) -> dict[str, Any]:
        load_muninn(conn)

        avg_degree = self._target_edges / self._n_nodes
        if self._graph_model == "erdos_renyi":
            self._edges, _ = generate_erdos_renyi(self._n_nodes, avg_degree, weighted=True, seed=42)
        else:
            m = max(1, int(avg_degree))
            self._edges, _ = generate_barabasi_albert(self._n_nodes, m, weighted=True, seed=42)

        conn.execute(f"CREATE TABLE IF NOT EXISTS {EDGE_TABLE}(src INTEGER, dst INTEGER, weight REAL)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_src ON {EDGE_TABLE}(src)")
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_dst ON {EDGE_TABLE}(dst)")
        conn.executemany(f"INSERT INTO {EDGE_TABLE}(src, dst, weight) VALUES (?, ?, ?)", self._edges)
        conn.commit()

        actual_edges = conn.execute(f"SELECT COUNT(*) FROM {EDGE_TABLE}").fetchone()[0]
        return {"actual_edges": actual_edges, "n_nodes": self._n_nodes}

    def run(self, conn: sqlite3.Connection) -> dict[str, Any]:
        approach = self._approach
        algos = ALGORITHMS if self._n_nodes <= 5000 else ["degree", "leiden"]

        if approach == "tvf":
            return self._run_tvf(conn, algos)
        elif approach == "csr":
            return self._run_csr(conn, algos)
        elif approach == "csr_full_rebuild":
            return self._run_csr_rebuild(conn, algos, mode="full")
        elif approach == "csr_incremental":
            return self._run_csr_rebuild(conn, algos, mode="incremental")
        elif approach == "csr_blocked":
            return self._run_csr_rebuild(conn, algos, mode="blocked")
        else:
            raise ValueError(f"Unknown approach: {approach}")

    def teardown(self, conn: sqlite3.Connection) -> None:
        self._edges = None

    # ── Metric helpers ──────────────────────────────────────────────

    def _measure_disk_bytes(self, conn: sqlite3.Connection, vtab_name: str = "g") -> int:
        """Measure shadow table disk size in bytes via SQLite dbstat.

        Returns total pgsize across all shadow tables (``{vtab_name}_%``),
        or ``-1`` if ``dbstat`` is not compiled into this SQLite build.
        """
        try:
            row = conn.execute(
                "SELECT SUM(pgsize) FROM dbstat WHERE name LIKE ?",
                (f"{vtab_name}_%",),
            ).fetchone()
            return row[0] or 0
        except Exception:
            return -1

    def _measure_trigger_overhead(self, conn: sqlite3.Connection, vtab_name: str = "g") -> dict[str, Any]:
        """Measure overhead of auto-installed graph_adjacency triggers.

        Times an INSERT + DELETE batch with triggers active, then drops
        the triggers and repeats the same batch.  The difference (floored
        at zero) is the trigger overhead.
        """
        n_batch = max(10, min(1000, self._target_edges // 10))
        rng = random.Random(400)
        offset = 1_000_000  # high IDs to avoid collision with real graph nodes
        trigger_edges = [
            (offset + rng.randint(0, self._n_nodes - 1), offset + rng.randint(0, self._n_nodes - 1), 1.0)
            for _ in range(n_batch)
        ]
        cleanup_sql = f"DELETE FROM {EDGE_TABLE} WHERE src >= {offset}"

        # With triggers (auto-installed by graph_adjacency)
        t0 = time.perf_counter()
        conn.executemany(f"INSERT INTO {EDGE_TABLE} VALUES (?, ?, ?)", trigger_edges)
        conn.execute(cleanup_sql)
        conn.commit()
        with_ms = (time.perf_counter() - t0) * 1000

        # Drop triggers
        for suffix in ("ai", "ad", "au"):
            conn.execute(f"DROP TRIGGER IF EXISTS {vtab_name}_{suffix}")

        # Without triggers
        t0 = time.perf_counter()
        conn.executemany(f"INSERT INTO {EDGE_TABLE} VALUES (?, ?, ?)", trigger_edges)
        conn.execute(cleanup_sql)
        conn.commit()
        without_ms = (time.perf_counter() - t0) * 1000

        overhead_ms = max(0.0, with_ms - without_ms)
        return {
            "trigger_overhead_ms": round(overhead_ms, 3),
            "trigger_with_ms": round(with_ms, 3),
            "trigger_without_ms": round(without_ms, 3),
            "trigger_batch_size": n_batch,
        }

    # ── Per-approach runners ────────────────────────────────────────

    def _run_tvf(self, conn: sqlite3.Connection, algos: list[str]) -> dict[str, Any]:
        """Run algorithms directly via TVFs (no caching)."""
        results: dict[str, Any] = {"disk_bytes": 0}  # no shadow tables
        for algo in algos:
            sql = self._algo_sql(algo, EDGE_TABLE)
            t0 = time.perf_counter()
            conn.execute(sql).fetchall()
            elapsed = (time.perf_counter() - t0) * 1000
            results[f"{algo}_ms"] = round(elapsed, 3)
        return results

    def _run_csr(self, conn: sqlite3.Connection, algos: list[str]) -> dict[str, Any]:
        """Build CSR cache, then run algorithms, then measure disk + trigger overhead."""
        t0 = time.perf_counter()
        conn.execute(
            f"CREATE VIRTUAL TABLE g USING graph_adjacency("
            f"edge_table='{EDGE_TABLE}', src_col='src', dst_col='dst', weight_col='weight')"
        )
        build_ms = (time.perf_counter() - t0) * 1000

        results: dict[str, Any] = {"build_ms": round(build_ms, 3)}
        for algo in algos:
            sql = self._algo_sql(algo, "g")
            t0 = time.perf_counter()
            conn.execute(sql).fetchall()
            elapsed = (time.perf_counter() - t0) * 1000
            results[f"{algo}_ms"] = round(elapsed, 3)

        # Shadow table disk usage
        disk_bytes = self._measure_disk_bytes(conn)
        if disk_bytes >= 0:
            results["disk_bytes"] = disk_bytes

        # Trigger overhead (measured last — drops triggers)
        results.update(self._measure_trigger_overhead(conn))

        return results

    def _run_csr_rebuild(self, conn: sqlite3.Connection, algos: list[str], mode: str) -> dict[str, Any]:
        """Build CSR, mutate, rebuild, then run algorithms."""
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS g USING graph_adjacency("
            f"edge_table='{EDGE_TABLE}', src_col='src', dst_col='dst', weight_col='weight')"
        )

        # Add delta edges
        n_delta = max(10, self._target_edges // 100)
        rng = random.Random(99)

        if mode == "blocked":
            limit = min(GRAPH_VT_BLOCK_SIZE, self._n_nodes)
            deltas = [
                (rng.randint(0, limit - 1), rng.randint(0, limit - 1), round(rng.uniform(0.1, 5.0), 2))
                for _ in range(n_delta)
            ]
        else:
            deltas = [
                (
                    rng.randint(0, self._n_nodes - 1),
                    rng.randint(0, self._n_nodes - 1),
                    round(rng.uniform(0.1, 5.0), 2),
                )
                for _ in range(n_delta)
            ]

        conn.executemany(f"INSERT INTO {EDGE_TABLE} VALUES (?, ?, ?)", deltas)
        conn.commit()

        rebuild_cmd = "rebuild" if mode == "full" else "incremental_rebuild"
        t0 = time.perf_counter()
        conn.execute(f"INSERT INTO g(g) VALUES ('{rebuild_cmd}')")
        rebuild_ms = (time.perf_counter() - t0) * 1000

        results: dict[str, Any] = {"rebuild_ms": round(rebuild_ms, 3), "delta_count": n_delta}
        for algo in algos:
            sql = self._algo_sql(algo, "g")
            t0 = time.perf_counter()
            conn.execute(sql).fetchall()
            elapsed = (time.perf_counter() - t0) * 1000
            results[f"{algo}_ms"] = round(elapsed, 3)

        # Shadow table disk usage (post-rebuild)
        disk_bytes = self._measure_disk_bytes(conn)
        if disk_bytes >= 0:
            results["disk_bytes"] = disk_bytes

        # Trigger overhead (measured last — drops triggers)
        results.update(self._measure_trigger_overhead(conn))

        return results

    @staticmethod
    def _algo_sql(algo: str, table: str) -> str:
        """Return SQL for running a graph algorithm against a table."""
        tvf_map = {
            "degree": "graph_degree",
            "betweenness": "graph_betweenness",
            "closeness": "graph_closeness",
            "leiden": "graph_leiden",
        }
        tvf = tvf_map.get(algo)
        if tvf is None:
            raise ValueError(f"Unknown algorithm: {algo}")
        return f"SELECT COUNT(*) FROM {tvf}('{table}', 'src', 'dst', 'weight')"
