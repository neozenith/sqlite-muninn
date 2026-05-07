"""SQL-subset strategy: materialize a TEMP filtered_edges view, then run centrality
TVFs over THAT instead of the full `edges` table.

Hypothesis: Brandes on the induced subgraph is dramatically cheaper than Brandes on
the full graph followed by a Python set-membership filter. This is the no-muninn-change
upper bound for what a filter-aware TVF could deliver.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict, deque

from benchmarks.kg_perf.strategies._base import Result, Strategy
from benchmarks.kg_perf.workload import Filter, QuerySpec, Workload


class SqlSubsetStrategy(Strategy):
    name = "sql_subset"

    def run(self, conn: sqlite3.Connection, workload: Workload) -> Result:
        _build_filtered_edges(conn, workload.filter)

        try:
            seeds = _compute_seeds_on_subset(conn, workload.query)
            if not seeds:
                return Result(nodes=[], edges=[], seeds=[])

            adj = _adjacency_from_subset(conn)
            visited = _bfs_expand(seeds, adj, workload.query.depth)
            visited = _prune_min_degree(visited, adj, workload.query.min_degree)

            nodes = sorted(visited)
            edges = _induced_edges_from_subset(conn, visited)
            return Result(nodes=nodes, edges=edges, seeds=seeds)
        finally:
            conn.execute("DROP TABLE IF EXISTS _kgperf_filtered_edges")
            conn.execute("DROP TABLE IF EXISTS _kgperf_allowed_nodes")
            conn.commit()


def _build_filtered_edges(conn: sqlite3.Connection, flt: Filter) -> None:
    """Build _kgperf_allowed_nodes and _kgperf_filtered_edges for this filter."""
    where = ["1=1"]
    params: list[object] = []
    if flt.project_id is not None:
        where.append("e.project_id = ?")
        params.append(flt.project_id)
    if flt.days is not None:
        where.append(
            "e.timestamp >= datetime("
            "(SELECT MAX(timestamp) FROM events), "
            f"'-{int(flt.days)} days')"
        )

    conn.execute("DROP TABLE IF EXISTS _kgperf_allowed_nodes")
    conn.execute(
        f"""
        CREATE TABLE _kgperf_allowed_nodes AS
        SELECT DISTINCT COALESCE(ec.canonical, ent.name) AS canonical
        FROM events e
        JOIN event_message_chunks emc ON emc.event_id = e.id
        JOIN entities ent ON ent.chunk_id = emc.chunk_id
        LEFT JOIN entity_clusters ec ON ec.name = ent.name
        WHERE {' AND '.join(where)}
        """,
        params,
    )
    conn.execute("CREATE INDEX _kgperf_allowed_nodes_idx ON _kgperf_allowed_nodes(canonical)")

    conn.execute("DROP TABLE IF EXISTS _kgperf_filtered_edges")
    conn.execute(
        """
        CREATE TABLE _kgperf_filtered_edges AS
        SELECT e.src, e.dst, e.rel_type, e.weight
        FROM edges e
        JOIN _kgperf_allowed_nodes an_s ON an_s.canonical = e.src
        JOIN _kgperf_allowed_nodes an_d ON an_d.canonical = e.dst
        """
    )
    conn.execute("CREATE INDEX _kgperf_filtered_edges_src ON _kgperf_filtered_edges(src)")
    conn.execute("CREATE INDEX _kgperf_filtered_edges_dst ON _kgperf_filtered_edges(dst)")


def _compute_seeds_on_subset(conn: sqlite3.Connection, query: QuerySpec) -> list[str]:
    """Run the centrality TVF directly over the filtered edge view."""
    if query.metric == "node_betweenness":
        rows = conn.execute(
            "SELECT node, centrality FROM graph_node_betweenness "
            "WHERE edge_table='_kgperf_filtered_edges' AND src_col='src' AND dst_col='dst' AND direction='both'"
        ).fetchall()
        ranked = [(c, n) for n, c in rows]
    elif query.metric == "edge_betweenness":
        rows = conn.execute(
            "SELECT src, dst, centrality FROM graph_edge_betweenness "
            "WHERE edge_table='_kgperf_filtered_edges' AND src_col='src' AND dst_col='dst' AND direction='both'"
        ).fetchall()
        per_node: dict[str, float] = defaultdict(float)
        for s, d, c in rows:
            per_node[s] += c
            per_node[d] += c
        ranked = [(c, n) for n, c in per_node.items()]
    elif query.metric == "degree":
        rows = conn.execute(
            "SELECT node, degree FROM graph_degree "
            "WHERE edge_table='_kgperf_filtered_edges' AND src_col='src' AND dst_col='dst' AND direction='both'"
        ).fetchall()
        ranked = [(float(d), n) for n, d in rows]
    else:
        raise ValueError(f"unknown metric: {query.metric}")

    ranked.sort(reverse=True)
    return [n for _, n in ranked[: query.top_k]]


def _adjacency_from_subset(conn: sqlite3.Connection) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for s, d in conn.execute("SELECT src, dst FROM _kgperf_filtered_edges"):
        if s != d:
            adj[s].add(d)
            adj[d].add(s)
    return adj


def _bfs_expand(seeds: list[str], adj: dict[str, set[str]], depth: int) -> set[str]:
    if depth <= 0:
        return set(seeds)
    visited = set(seeds)
    frontier = deque((s, 0) for s in seeds)
    while frontier:
        node, hop = frontier.popleft()
        if hop >= depth:
            continue
        for nb in adj.get(node, ()):
            if nb not in visited:
                visited.add(nb)
                frontier.append((nb, hop + 1))
    return visited


def _prune_min_degree(visited: set[str], adj: dict[str, set[str]], min_degree: int) -> set[str]:
    if min_degree <= 1:
        return visited
    return {n for n in visited if len(adj.get(n, set()) & visited) >= min_degree}


def _induced_edges_from_subset(conn: sqlite3.Connection, nodes: set[str]) -> list[tuple[str, str]]:
    if not nodes:
        return []
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for s, d in conn.execute("SELECT src, dst FROM _kgperf_filtered_edges"):
        if s in nodes and d in nodes and s != d:
            key = (s, d) if s < d else (d, s)
            if key not in seen:
                seen.add(key)
                out.append((s, d))
    return out
