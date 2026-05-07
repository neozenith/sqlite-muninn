"""TopKCache strategy: persist (filter+query) -> result records in a SQLite table,
keyed on a stable signature. On hit: O(1) lookup. On miss: defer to chunk_canonical
and store the result for next time.

In production this signature would also include the GII generation counter so cached
entries auto-invalidate on edge mutation. For the harness, edges are static so the
signature is just (filter_signature, query_slug).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

from benchmarks.kg_perf.strategies._base import Result, Strategy
from benchmarks.kg_perf.strategies.chunk_canonical import (
    EVENT_CANONICAL_DDL,
    EVENT_CANONICAL_FILL,
    _adjacency_from_subset,
    _bfs_expand,
    _build_filtered_edges_via_index,
    _compute_seeds_on_subset,
    _induced_edges_from_subset,
    _prune_min_degree,
)
from benchmarks.kg_perf.workload import Filter, QuerySpec, Workload

CACHE_DDL = """
    CREATE TABLE IF NOT EXISTS kgperf_topk_cache (
        signature TEXT PRIMARY KEY,
        seeds_json TEXT NOT NULL,
        nodes_json TEXT NOT NULL,
        edges_json TEXT NOT NULL,
        edge_generation INTEGER NOT NULL,
        cached_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
"""


class TopKCacheStrategy(Strategy):
    name = "topk_cache"

    def prepare(self, conn: sqlite3.Connection) -> None:
        # Same provenance index as chunk_canonical.
        conn.execute(EVENT_CANONICAL_DDL)
        cur = conn.execute("SELECT COUNT(*) FROM event_canonical_idx")
        if cur.fetchone()[0] == 0:
            conn.execute(EVENT_CANONICAL_FILL)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS event_canonical_idx_proj_ts "
                "ON event_canonical_idx(project_id, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS event_canonical_idx_canonical "
                "ON event_canonical_idx(canonical)"
            )
        conn.execute(CACHE_DDL)
        conn.commit()

    def run(self, conn: sqlite3.Connection, workload: Workload) -> Result:
        sig = _signature(workload.filter, workload.query, _edge_generation(conn))

        cached = conn.execute(
            "SELECT seeds_json, nodes_json, edges_json FROM kgperf_topk_cache WHERE signature = ?",
            (sig,),
        ).fetchone()
        if cached is not None:
            seeds = json.loads(cached[0])
            nodes = json.loads(cached[1])
            edges = [tuple(e) for e in json.loads(cached[2])]
            return Result(nodes=nodes, edges=edges, seeds=seeds, extras={"cache": "hit"})

        # Miss path: run the chunk_canonical pipeline and cache the result.
        _build_filtered_edges_via_index(conn, workload.filter)
        try:
            seeds = _compute_seeds_on_subset(conn, workload.query)
            if not seeds:
                result = Result(nodes=[], edges=[], seeds=[], extras={"cache": "miss"})
            else:
                adj = _adjacency_from_subset(conn)
                visited = _bfs_expand(seeds, adj, workload.query.depth)
                visited = _prune_min_degree(visited, adj, workload.query.min_degree)
                nodes = sorted(visited)
                edges = _induced_edges_from_subset(conn, visited)
                result = Result(nodes=nodes, edges=edges, seeds=seeds, extras={"cache": "miss"})

            conn.execute(
                "INSERT OR REPLACE INTO kgperf_topk_cache "
                "(signature, seeds_json, nodes_json, edges_json, edge_generation) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    sig,
                    json.dumps(result.seeds),
                    json.dumps(result.nodes),
                    json.dumps(result.edges),
                    _edge_generation(conn),
                ),
            )
            conn.commit()
            return result
        finally:
            conn.execute("DROP TABLE IF EXISTS _kgperf_filtered_edges")
            conn.execute("DROP TABLE IF EXISTS _kgperf_allowed_nodes")
            conn.commit()


def _signature(flt: Filter, query: QuerySpec, edge_gen: int) -> str:
    """Stable signature: filter + query + edge generation. Anything that would change
    the output must be in here.
    """
    parts = (
        flt.project_id or "",
        str(flt.days or ""),
        query.metric,
        str(query.top_k),
        str(query.depth),
        str(query.min_degree),
        str(edge_gen),
    )
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:32]


def _edge_generation(conn: sqlite3.Connection) -> int:
    """Stand-in for the GII generation counter: hash of edge-table row count.

    For a static-edges benchmark this is constant. In production with GII this
    would read the namespace-scoped generation field from `_config`.
    """
    row = conn.execute("SELECT COUNT(*) FROM edges").fetchone()
    return int(row[0]) if row else 0


__all__ = ["TopKCacheStrategy"]
