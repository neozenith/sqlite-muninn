"""Graph traversal treatment: BFS, DFS, shortest_path, components, pagerank.

Ports from benchmarks/scripts/benchmark_graph.py -- the muninn TVF runners.
"""

import logging
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

from benchmarks.harness.common import generate_barabasi_albert, generate_erdos_renyi, load_muninn
from benchmarks.harness.treatments.base import Treatment

log = logging.getLogger(__name__)

N_PER_QUERY_OPS = 50


class GraphTraversalTreatment(Treatment):
    """Single graph traversal benchmark permutation."""

    def __init__(self, engine: str, operation: str, graph_model: str, n_nodes: int, avg_degree: int) -> None:
        self._engine = engine
        self._operation = operation
        self._graph_model = graph_model
        self._n_nodes = n_nodes
        self._avg_degree = avg_degree
        self._edges: list[tuple[int, int, float]] | None = None
        self._adj: dict[int, list[tuple[int, float]]] | None = None
        self._start_nodes: list[int] | None = None
        self._end_nodes: list[int] | None = None

    @property
    def category(self) -> str:
        return "graph"

    @property
    def permutation_id(self) -> str:
        return f"graph_{self._engine}_{self._operation}_{self._graph_model}_n{self._n_nodes}_deg{self._avg_degree}"

    @property
    def label(self) -> str:
        return (
            f"Graph: {self._engine} / {self._operation} / "
            f"{self._graph_model} / N={self._n_nodes} / deg={self._avg_degree}"
        )

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (self._n_nodes, self._avg_degree, self._operation, self._engine)

    def params_dict(self) -> dict[str, Any]:
        return {
            "engine": self._engine,
            "operation": self._operation,
            "graph_model": self._graph_model,
            "n_nodes": self._n_nodes,
            "avg_degree": self._avg_degree,
        }

    def setup(self, conn: sqlite3.Connection, db_path: Path) -> dict[str, Any]:
        if self._graph_model == "erdos_renyi":
            self._edges, self._adj = generate_erdos_renyi(self._n_nodes, self._avg_degree, seed=42)
        else:
            self._edges, self._adj = generate_barabasi_albert(self._n_nodes, int(self._avg_degree), seed=42)

        n_edges = len(self._edges)

        # Pick start/end nodes
        rng = random.Random(42)
        node_list = sorted(self._adj.keys())
        n_queries = min(N_PER_QUERY_OPS, len(node_list))
        self._start_nodes = rng.sample(node_list, n_queries)
        self._end_nodes = rng.sample(node_list, n_queries)

        # Load edges into table
        if self._engine == "muninn":
            load_muninn(conn)

        conn.execute("CREATE TABLE IF NOT EXISTS bench_edges(src INTEGER, dst INTEGER, weight REAL)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_src ON bench_edges(src)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dst ON bench_edges(dst)")
        conn.executemany("INSERT INTO bench_edges(src, dst, weight) VALUES (?, ?, ?)", self._edges)
        conn.commit()

        return {"n_edges": n_edges, "n_nodes": self._n_nodes}

    def run(self, conn: sqlite3.Connection) -> dict[str, Any]:
        op = self._operation
        if self._engine == "muninn":
            return self._run_muninn(conn, op)
        elif self._engine == "graphqlite":
            return self._run_graphqlite(op)
        else:
            raise ValueError(f"Unknown graph engine: {self._engine}")

    def teardown(self, conn: sqlite3.Connection) -> None:
        self._edges = None
        self._adj = None

    def _run_muninn(self, conn: sqlite3.Connection, operation: str) -> dict[str, Any]:
        """Run graph operation using muninn TVFs."""
        assert self._start_nodes is not None
        assert self._end_nodes is not None
        if operation in ("bfs", "dfs"):
            times: list[float] = []
            total_visited = 0
            tvf = "graph_bfs" if operation == "bfs" else "graph_dfs"
            for start in self._start_nodes:
                t0 = time.perf_counter()
                rows = conn.execute(
                    f"SELECT node, depth FROM {tvf}"
                    " WHERE edge_table = 'bench_edges' AND src_col = 'src' AND dst_col = 'dst'"
                    " AND start_node = ? AND max_depth = 1000 AND direction = 'forward'",
                    (start,),
                ).fetchall()
                times.append(time.perf_counter() - t0)
                total_visited += len(rows)

            mean_ms = (sum(times) / len(times)) * 1000 if times else 0
            return {
                "query_time_ms": round(mean_ms, 3),
                "n_queries": len(times),
                "nodes_visited_mean": round(total_visited / len(times), 1) if times else 0,
            }

        elif operation == "shortest_path":
            times = []
            for start, end in zip(self._start_nodes, self._end_nodes, strict=False):
                t0 = time.perf_counter()
                conn.execute(
                    "SELECT node, distance, path_order FROM graph_shortest_path"
                    " WHERE edge_table = 'bench_edges' AND src_col = 'src' AND dst_col = 'dst'"
                    " AND start_node = ? AND end_node = ?",
                    (start, end),
                ).fetchall()
                times.append(time.perf_counter() - t0)

            mean_ms = (sum(times) / len(times)) * 1000 if times else 0
            return {"query_time_ms": round(mean_ms, 3), "n_queries": len(times)}

        elif operation == "components":
            t0 = time.perf_counter()
            rows = conn.execute(
                "SELECT node, component_id FROM graph_components"
                " WHERE edge_table = 'bench_edges' AND src_col = 'src' AND dst_col = 'dst'",
            ).fetchall()
            elapsed = (time.perf_counter() - t0) * 1000
            n_components = len({r[1] for r in rows})
            return {"query_time_ms": round(elapsed, 3), "n_queries": 1, "n_components": n_components}

        elif operation == "pagerank":
            t0 = time.perf_counter()
            rows = conn.execute(
                "SELECT node, rank FROM graph_pagerank"
                " WHERE edge_table = 'bench_edges' AND src_col = 'src' AND dst_col = 'dst'"
                " AND damping = 0.85 AND iterations = 100",
            ).fetchall()
            elapsed = (time.perf_counter() - t0) * 1000
            return {"query_time_ms": round(elapsed, 3), "n_queries": 1, "nodes_ranked": len(rows)}

        raise ValueError(f"Unknown operation: {operation}")

    def _run_graphqlite(self, operation: str) -> dict[str, Any]:
        """Run graph operation using GraphQLite."""
        from graphqlite import Graph

        assert self._edges is not None
        assert self._start_nodes is not None
        assert self._end_nodes is not None
        g = Graph(":memory:")
        nodes_batch = [(str(i), {}, "Node") for i in range(self._n_nodes)]
        edges_batch = [(str(src), str(dst), {"weight": w}, "EDGE") for src, dst, w in self._edges if src < dst]
        g.upsert_nodes_batch(nodes_batch)
        g.upsert_edges_batch(edges_batch)

        if operation == "bfs":
            times: list[float] = []
            for start in self._start_nodes:
                t0 = time.perf_counter()
                g.bfs(start_id=str(start), max_depth=-1)
                times.append(time.perf_counter() - t0)
            mean_ms = (sum(times) / len(times)) * 1000 if times else 0
            return {"query_time_ms": round(mean_ms, 3), "n_queries": len(times)}

        elif operation == "dfs":
            times = []
            for start in self._start_nodes:
                t0 = time.perf_counter()
                g.dfs(start_id=str(start), max_depth=-1)
                times.append(time.perf_counter() - t0)
            mean_ms = (sum(times) / len(times)) * 1000 if times else 0
            return {"query_time_ms": round(mean_ms, 3), "n_queries": len(times)}

        elif operation == "shortest_path":
            times = []
            for start, end in zip(self._start_nodes, self._end_nodes, strict=False):
                t0 = time.perf_counter()
                g.shortest_path(source_id=str(start), target_id=str(end))
                times.append(time.perf_counter() - t0)
            mean_ms = (sum(times) / len(times)) * 1000 if times else 0
            return {"query_time_ms": round(mean_ms, 3), "n_queries": len(times)}

        elif operation == "components":
            t0 = time.perf_counter()
            _ = g.weakly_connected_components()
            elapsed = (time.perf_counter() - t0) * 1000
            return {"query_time_ms": round(elapsed, 3), "n_queries": 1}

        elif operation == "pagerank":
            t0 = time.perf_counter()
            _ = g.pagerank(damping=0.85, iterations=100)
            elapsed = (time.perf_counter() - t0) * 1000
            return {"query_time_ms": round(elapsed, 3), "n_queries": 1}

        raise ValueError(f"Unknown operation: {operation}")
