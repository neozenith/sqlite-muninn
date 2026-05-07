"""Baseline strategy — exactly mirrors claude-code-sessions/.../kg/payload.py.

1. Walk events -> chunks -> entities -> entity_clusters in Python to build
   `allowed_canonicals` (a Python set).
2. Call `graph_node_betweenness` / `graph_edge_betweenness` over the FULL `edges`
   table; post-filter results in Python by allowed_canonicals.
3. Take top-K seeds, BFS expand to `depth` hops, prune by min_degree.

This is the "naive but correct" reference. Every other strategy must match its
result set (Jaccard >= floor) while being faster.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict, deque

from benchmarks.kg_perf.strategies._base import Result, Strategy
from benchmarks.kg_perf.workload import Filter, QuerySpec, Workload


class BaselineStrategy(Strategy):
    name = "baseline"

    def run(self, conn: sqlite3.Connection, workload: Workload) -> Result:
        allowed = _allowed_canonicals(conn, workload.filter)
        if not allowed:
            return Result(nodes=[], edges=[], seeds=[])

        seeds = _compute_seeds(conn, workload.query, allowed)
        if not seeds:
            return Result(nodes=[], edges=[], seeds=[])

        adj = _adjacency_for(conn, allowed)
        visited = _bfs_expand(seeds, adj, workload.query.depth)
        visited = _prune_min_degree(visited, adj, workload.query.min_degree)

        nodes = sorted(visited)
        edges = _induced_edges(conn, visited)
        return Result(nodes=nodes, edges=edges, seeds=seeds)


def _allowed_canonicals(conn: sqlite3.Connection, flt: Filter) -> set[str]:
    """events --(filter)--> chunks --> entities --> entity_clusters.canonical.

    The `days` window is computed relative to MAX(timestamp) in this DB so the
    workload is reproducible regardless of wall-clock time.
    """
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

    sql = f"""
        SELECT DISTINCT COALESCE(ec.canonical, ent.name) AS canonical
        FROM events e
        JOIN event_message_chunks emc ON emc.event_id = e.id
        JOIN entities ent ON ent.chunk_id = emc.chunk_id
        LEFT JOIN entity_clusters ec ON ec.name = ent.name
        WHERE {' AND '.join(where)}
    """
    return {row[0] for row in conn.execute(sql, params) if row[0] is not None}


def _compute_seeds(conn: sqlite3.Connection, query: QuerySpec, allowed: set[str]) -> list[str]:
    """Run the centrality TVF over the FULL graph, then post-filter — exactly like payload.py."""
    if query.metric == "node_betweenness":
        rows = conn.execute(
            "SELECT node, centrality FROM graph_node_betweenness "
            "WHERE edge_table='edges' AND src_col='src' AND dst_col='dst' AND direction='both'"
        ).fetchall()
        ranked = [(c, n) for n, c in rows if n in allowed]
    elif query.metric == "edge_betweenness":
        rows = conn.execute(
            "SELECT src, dst, centrality FROM graph_edge_betweenness "
            "WHERE edge_table='edges' AND src_col='src' AND dst_col='dst' AND direction='both'"
        ).fetchall()
        per_node: dict[str, float] = defaultdict(float)
        for s, d, c in rows:
            if s in allowed:
                per_node[s] += c
            if d in allowed:
                per_node[d] += c
        ranked = [(c, n) for n, c in per_node.items()]
    elif query.metric == "degree":
        rows = conn.execute(
            "SELECT node, degree FROM graph_degree "
            "WHERE edge_table='edges' AND src_col='src' AND dst_col='dst' AND direction='both'"
        ).fetchall()
        ranked = [(float(d), n) for n, d in rows if n in allowed]
    else:
        raise ValueError(f"unknown metric: {query.metric}")

    ranked.sort(reverse=True)
    return [n for _, n in ranked[: query.top_k]]


def _adjacency_for(conn: sqlite3.Connection, allowed: set[str]) -> dict[str, set[str]]:
    """Undirected adjacency restricted to nodes in `allowed`."""
    adj: dict[str, set[str]] = defaultdict(set)
    for s, d in conn.execute("SELECT src, dst FROM edges"):
        if s in allowed and d in allowed and s != d:
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


def _induced_edges(conn: sqlite3.Connection, nodes: set[str]) -> list[tuple[str, str]]:
    if not nodes:
        return []
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for s, d in conn.execute("SELECT src, dst FROM edges"):
        if s in nodes and d in nodes and s != d:
            key = (s, d) if s < d else (d, s)
            if key not in seen:
                seen.add(key)
                out.append((s, d))
    return out
