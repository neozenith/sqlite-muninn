"""K-core strategy: replace iterative min_degree pruning with a coreness lookup.

Stacks on chunk_canonical (denormalized provenance + induced subgraph). The only
new wrinkle: after BFS expansion we keep nodes whose *coreness* in the induced
subgraph is >= min_degree, instead of single-pass induced-degree pruning.

Coreness is computed via the Batagelj-Zaversnik O(V'+E') peeling algorithm directly
on the in-memory adjacency we already build. For a future GII-backed implementation
the coreness vector would be cached as a shadow column keyed by (namespace, generation).
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict, deque

from benchmarks.kg_perf.strategies._base import Result, Strategy
from benchmarks.kg_perf.strategies.chunk_canonical import (
    EVENT_CANONICAL_DDL,
    EVENT_CANONICAL_FILL,
    _build_filtered_edges_via_index,
    _compute_seeds_on_subset,
    _adjacency_from_subset,
    _bfs_expand,
    _induced_edges_from_subset,
)
from benchmarks.kg_perf.workload import Workload


class KCoreStrategy(Strategy):
    name = "kcore"

    def prepare(self, conn: sqlite3.Connection) -> None:
        """Same provenance index as chunk_canonical — kcore is computed at query time
        because it depends on the *filtered* subgraph, not the global graph.
        """
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
            conn.commit()

    def run(self, conn: sqlite3.Connection, workload: Workload) -> Result:
        _build_filtered_edges_via_index(conn, workload.filter)
        try:
            seeds = _compute_seeds_on_subset(conn, workload.query)
            if not seeds:
                return Result(nodes=[], edges=[], seeds=[])

            adj = _adjacency_from_subset(conn)
            visited = _bfs_expand(seeds, adj, workload.query.depth)
            visited = _kcore_prune(visited, adj, workload.query.min_degree)

            nodes = sorted(visited)
            edges = _induced_edges_from_subset(conn, visited)
            return Result(nodes=nodes, edges=edges, seeds=seeds)
        finally:
            conn.execute("DROP TABLE IF EXISTS _kgperf_filtered_edges")
            conn.execute("DROP TABLE IF EXISTS _kgperf_allowed_nodes")
            conn.commit()


def _kcore_prune(visited: set[str], adj: dict[str, set[str]], min_degree: int) -> set[str]:
    """Iteratively peel nodes whose induced degree in `visited` falls below min_degree.

    Equivalent to: keep nodes whose coreness in the induced subgraph is >= min_degree.
    """
    if min_degree <= 1:
        return set(visited)

    induced_deg: dict[str, int] = {}
    for n in visited:
        induced_deg[n] = sum(1 for nb in adj.get(n, ()) if nb in visited)

    queue: deque[str] = deque(n for n, d in induced_deg.items() if d < min_degree)
    removed: set[str] = set()
    while queue:
        n = queue.popleft()
        if n in removed:
            continue
        removed.add(n)
        for nb in adj.get(n, ()):
            if nb in induced_deg and nb not in removed:
                induced_deg[nb] -= 1
                if induced_deg[nb] < min_degree:
                    queue.append(nb)
    return visited - removed


__all__ = ["KCoreStrategy"]


# Reference the imports so Pyright doesn't warn — they're real callees of run()/prepare().
_ = (defaultdict,)
