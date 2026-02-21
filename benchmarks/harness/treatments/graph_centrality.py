"""Graph centrality treatment: degree, betweenness, closeness.

Ports centrality benchmarks from benchmarks/scripts/benchmark_graph.py.
"""

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

from benchmarks.harness.common import generate_barabasi_albert, generate_erdos_renyi, load_muninn
from benchmarks.harness.treatments.base import Treatment

log = logging.getLogger(__name__)


class GraphCentralityTreatment(Treatment):
    """Single graph centrality benchmark permutation."""

    def __init__(self, operation: str, graph_model: str, n_nodes: int, avg_degree: int) -> None:
        self._operation = operation
        self._graph_model = graph_model
        self._n_nodes = n_nodes
        self._avg_degree = avg_degree
        self._edges: list[tuple[int, int, float]] | None = None

    @property
    def category(self) -> str:
        return "centrality"

    @property
    def permutation_id(self) -> str:
        return f"centrality_muninn_{self._operation}_{self._graph_model}_n{self._n_nodes}_deg{self._avg_degree}"

    @property
    def label(self) -> str:
        return f"Centrality: {self._operation} / {self._graph_model} / N={self._n_nodes} / deg={self._avg_degree}"

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (self._n_nodes, self._avg_degree, self._operation)

    def params_dict(self) -> dict[str, Any]:
        return {
            "engine": "muninn",
            "operation": self._operation,
            "graph_model": self._graph_model,
            "n_nodes": self._n_nodes,
            "avg_degree": self._avg_degree,
        }

    def setup(self, conn: sqlite3.Connection, db_path: Path) -> dict[str, Any]:
        if self._graph_model == "erdos_renyi":
            self._edges, _ = generate_erdos_renyi(self._n_nodes, self._avg_degree, seed=42)
        else:
            self._edges, _ = generate_barabasi_albert(self._n_nodes, int(self._avg_degree), seed=42)

        n_edges = len(self._edges)
        load_muninn(conn)

        conn.execute("CREATE TABLE IF NOT EXISTS bench_edges(src INTEGER, dst INTEGER, weight REAL)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_src ON bench_edges(src)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dst ON bench_edges(dst)")
        conn.executemany("INSERT INTO bench_edges(src, dst, weight) VALUES (?, ?, ?)", self._edges)
        conn.commit()

        return {"n_edges": n_edges}

    def run(self, conn: sqlite3.Connection) -> dict[str, Any]:
        op = self._operation
        tvf_map = {
            "degree": "graph_degree",
            "betweenness": "graph_betweenness",
            "closeness": "graph_closeness",
        }
        tvf = tvf_map[op]

        direction_clause = " AND direction = 'both'" if op in ("betweenness", "closeness") else ""

        t0 = time.perf_counter()
        rows = conn.execute(
            f"SELECT node, centrality FROM {tvf}"
            f" WHERE edge_table = 'bench_edges' AND src_col = 'src' AND dst_col = 'dst'"
            f"{direction_clause}",
        ).fetchall()
        elapsed = (time.perf_counter() - t0) * 1000

        return {
            "query_time_ms": round(elapsed, 3),
            "nodes_computed": len(rows),
        }

    def teardown(self, conn: sqlite3.Connection) -> None:
        self._edges = None
