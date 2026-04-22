"""Knowledge-graph payload for the Cytoscape viz.

Two table variants:
  * base — raw nodes + edges from NER / RE output (keyed by entity name)
  * er   — entity-resolved: collapse nodes by entity_clusters.canonical

The payload includes a seed-and-expand mechanism: the backend picks N seed
nodes by a centrality metric (degree / node-betweenness / edge-betweenness)
from the FULL graph, then BFS-expands up to `max_depth` hops through the
undirected view of the edge set. `max_depth=0` is unlimited (yields every
connected component containing a seed).

Each returned node and edge carries its full-graph betweenness centrality
score so the client can size nodes/edges by those metrics without
recomputing.
"""

import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Literal

import networkx as nx
from pydantic import BaseModel

from server.db import table_exists

KG_TABLES = ("base", "er")
DEFAULT_RESOLUTION = 0.25
DEFAULT_TOP_N = 50
DEFAULT_SEED_METRIC = "edge_betweenness"
DEFAULT_MAX_DEPTH = 0

SeedMetric = Literal["degree", "node_betweenness", "edge_betweenness"]
VALID_SEED_METRICS: tuple[SeedMetric, ...] = ("degree", "node_betweenness", "edge_betweenness")

# Approximate BC is used for large graphs to bound latency. k=150 gives a
# stable top-K ranking on our DBs; smaller values were noisy at V>5000.
_BC_SAMPLE_THRESHOLD = 2000
_BC_SAMPLE_K = 150


class KGNode(BaseModel):
    id: str
    label: str
    entity_type: str | None = None
    community_id: int | None = None
    mention_count: int | None = None
    node_betweenness: float | None = None


class KGEdge(BaseModel):
    source: str
    target: str
    rel_type: str | None = None
    weight: float | None = None
    edge_betweenness: float | None = None


class KGCommunity(BaseModel):
    id: int
    label: str | None = None
    member_count: int
    node_ids: list[str]


class KGPayload(BaseModel):
    table_id: str
    resolution: float
    seed_metric: str
    max_depth: int
    node_count: int
    edge_count: int
    community_count: int
    total_node_count: int
    total_edge_count: int
    nodes: list[KGNode]
    edges: list[KGEdge]
    communities: list[KGCommunity]


class UnknownKGTable(ValueError):
    pass


class KGDataMissing(RuntimeError):
    pass


def _available_resolutions(conn: sqlite3.Connection) -> list[float]:
    rows = conn.execute("SELECT DISTINCT resolution FROM leiden_communities ORDER BY resolution").fetchall()
    return [float(r[0]) for r in rows]


def _pick_resolution(conn: sqlite3.Connection, requested: float | None) -> float:
    available = _available_resolutions(conn)
    if not available:
        raise KGDataMissing("leiden_communities has no rows")
    if requested is None:
        return DEFAULT_RESOLUTION if DEFAULT_RESOLUTION in available else available[0]
    if requested not in available:
        raise ValueError(f"resolution {requested} not in {available}")
    return requested


def _load_base(
    conn: sqlite3.Connection, resolution: float
) -> tuple[list[KGNode], list[KGEdge], list[KGCommunity]]:
    if not table_exists(conn, "nodes") or not table_exists(conn, "edges"):
        raise KGDataMissing("base KG requires `nodes` and `edges` tables")

    community_map: dict[str, int] = {
        str(row["node"]): int(row["community_id"])
        for row in conn.execute(
            "SELECT node, community_id FROM leiden_communities WHERE resolution = ?",
            (resolution,),
        )
    }

    node_meta: dict[str, tuple[str | None, int | None]] = {
        str(row["name"]): (row["entity_type"], row["mention_count"])
        for row in conn.execute("SELECT name, entity_type, mention_count FROM nodes")
    }

    edges_rows = conn.execute("SELECT src, dst, rel_type, weight FROM edges").fetchall()
    all_names: set[str] = set(node_meta) | set(community_map)
    for row in edges_rows:
        all_names.add(str(row["src"]))
        all_names.add(str(row["dst"]))

    nodes = [
        KGNode(
            id=name,
            label=name,
            entity_type=node_meta.get(name, (None, None))[0],
            community_id=community_map.get(name),
            mention_count=node_meta.get(name, (None, None))[1],
        )
        for name in sorted(all_names)
    ]

    edges = [
        KGEdge(
            source=str(row["src"]),
            target=str(row["dst"]),
            rel_type=row["rel_type"],
            weight=row["weight"],
        )
        for row in edges_rows
    ]

    communities = _build_communities(conn, resolution, community_map)
    return nodes, edges, communities


def _load_er(
    conn: sqlite3.Connection, resolution: float
) -> tuple[list[KGNode], list[KGEdge], list[KGCommunity]]:
    if (
        not table_exists(conn, "entity_clusters")
        or not table_exists(conn, "edges")
        or not table_exists(conn, "nodes")
    ):
        raise KGDataMissing("ER KG requires `entity_clusters`, `edges`, and `nodes`")

    canonical_map: dict[str, str] = {
        str(row["name"]): str(row["canonical"])
        for row in conn.execute("SELECT name, canonical FROM entity_clusters")
    }

    cluster_labels: dict[str, str] = {}
    if table_exists(conn, "entity_cluster_labels"):
        cluster_labels = {
            str(row["canonical"]): str(row["label"])
            for row in conn.execute(
                "SELECT canonical, label FROM entity_cluster_labels WHERE label IS NOT NULL"
            )
        }

    type_counter: dict[str, Counter[str]] = defaultdict(Counter)
    mention_sum: dict[str, int] = defaultdict(int)
    for row in conn.execute("SELECT name, entity_type, mention_count FROM nodes"):
        name = str(row["name"])
        canonical = canonical_map.get(name, name)
        if row["entity_type"]:
            type_counter[canonical][row["entity_type"]] += 1
        mention_sum[canonical] += int(row["mention_count"] or 0)

    community_rows = conn.execute(
        "SELECT node, community_id FROM leiden_communities WHERE resolution = ?",
        (resolution,),
    ).fetchall()
    canonical_communities: dict[str, Counter[int]] = defaultdict(Counter)
    for row in community_rows:
        node_name = str(row["node"])
        canonical = canonical_map.get(node_name, node_name)
        canonical_communities[canonical][int(row["community_id"])] += 1

    raw_edges = conn.execute("SELECT src, dst, rel_type, weight FROM edges").fetchall()
    seen: dict[tuple[str, str, str], float] = {}
    for row in raw_edges:
        src = str(row["src"])
        dst = str(row["dst"])
        src_c = canonical_map.get(src, src)
        dst_c = canonical_map.get(dst, dst)
        if src_c == dst_c:
            continue
        rel = str(row["rel_type"] or "")
        key = (src_c, dst_c, rel)
        seen[key] = seen.get(key, 0.0) + float(row["weight"] or 0.0)
    edges = [KGEdge(source=s, target=d, rel_type=r or None, weight=w) for (s, d, r), w in seen.items()]

    all_canonicals: set[str] = set(canonical_map.values())
    all_canonicals.update(canonical_communities.keys())
    for e in edges:
        all_canonicals.add(e.source)
        all_canonicals.add(e.target)

    nodes = [
        KGNode(
            id=c,
            label=cluster_labels.get(c, c),
            entity_type=(
                type_counter[c].most_common(1)[0][0] if type_counter.get(c) else None
            ),
            community_id=(
                canonical_communities[c].most_common(1)[0][0]
                if canonical_communities.get(c)
                else None
            ),
            mention_count=mention_sum[c] if mention_sum[c] else None,
        )
        for c in sorted(all_canonicals)
    ]

    flat_map = {
        c: (counter.most_common(1)[0][0] if counter else None)
        for c, counter in canonical_communities.items()
    }
    communities = _build_communities(conn, resolution, flat_map)
    return nodes, edges, communities


def _build_communities(
    conn: sqlite3.Connection,
    resolution: float,
    node_to_community: Mapping[str, int | None],
) -> list[KGCommunity]:
    members: dict[int, list[str]] = defaultdict(list)
    for node_id, community_id in node_to_community.items():
        if community_id is None:
            continue
        members[community_id].append(node_id)

    labels: dict[int, str] = {}
    if table_exists(conn, "community_labels"):
        for row in conn.execute(
            "SELECT community_id, label FROM community_labels WHERE resolution = ? AND label IS NOT NULL",
            (resolution,),
        ):
            labels[int(row["community_id"])] = row["label"]

    return [
        KGCommunity(
            id=cid,
            label=labels.get(cid),
            member_count=len(ids),
            node_ids=sorted(ids),
        )
        for cid, ids in sorted(members.items())
    ]


def _build_graph(nodes: list[KGNode], edges: list[KGEdge]) -> nx.DiGraph:
    """Build a DiGraph from the payload.

    Duplicate (source, target) edges collapse by summing their weights; self-
    loops are dropped. networkx's betweenness_centrality doesn't run on
    MultiDiGraph, hence the collapse.
    """
    g: nx.DiGraph = nx.DiGraph()
    for n in nodes:
        g.add_node(n.id)
    for e in edges:
        if e.source == e.target:
            continue
        if not g.has_node(e.source) or not g.has_node(e.target):
            continue
        w_in = float(e.weight or 1.0)
        if g.has_edge(e.source, e.target):
            g[e.source][e.target]["weight"] = g[e.source][e.target].get("weight", 1.0) + w_in
        else:
            g.add_edge(e.source, e.target, weight=w_in)
    return g


def _compute_betweenness(
    g: nx.DiGraph,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Node and edge betweenness centrality over the full graph.

    Uses k-sampled approximation when the graph is large (V > 2000) to keep
    latency bounded. `seed=42` fixes the sample for reproducibility.
    """
    v = g.number_of_nodes()
    if v == 0:
        return {}, {}
    k: int | None = min(v, _BC_SAMPLE_K) if v > _BC_SAMPLE_THRESHOLD else None
    node_bc = nx.betweenness_centrality(g, k=k, normalized=True, seed=42)
    edge_bc_raw = nx.edge_betweenness_centrality(g, k=k, normalized=True, seed=42)
    # edge_bc keys are (u, v) tuples — normalize to our (source, target) form.
    edge_bc: dict[tuple[str, str], float] = {(u, v): float(b) for (u, v), b in edge_bc_raw.items()}
    return {n: float(b) for n, b in node_bc.items()}, edge_bc


def _seed_scores(
    nodes: list[KGNode],
    g: nx.DiGraph,
    node_bc: dict[str, float],
    edge_bc: dict[tuple[str, str], float],
    metric: SeedMetric,
) -> dict[str, float]:
    """Per-node ranking score for the chosen seed metric.

    For edge_betweenness the node score is the sum of BC across all incident
    edges — that way highly-central nodes tend to lie on many shortest paths.
    """
    if metric == "degree":
        return {n.id: float(g.degree(n.id)) if g.has_node(n.id) else 0.0 for n in nodes}
    if metric == "node_betweenness":
        return {n.id: node_bc.get(n.id, 0.0) for n in nodes}
    # edge_betweenness
    incident: dict[str, float] = defaultdict(float)
    for (u, v), bc in edge_bc.items():
        incident[u] += bc
        incident[v] += bc
    return {n.id: incident.get(n.id, 0.0) for n in nodes}


def _bfs_expand(g: nx.DiGraph, seeds: set[str], max_depth: int) -> set[str]:
    """Expand seeds via undirected BFS. max_depth=0 → unlimited expansion."""
    undirected = g.to_undirected(as_view=True)
    if max_depth == 0:
        result: set[str] = set()
        for seed in seeds:
            if seed in result or seed not in undirected:
                continue
            result |= nx.node_connected_component(undirected, seed)
        # Seeds that aren't in any component edge still count.
        result |= {s for s in seeds if s not in undirected}
        return result

    result = set(seeds)
    frontier = {s for s in seeds if s in undirected}
    for _ in range(max_depth):
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbor in undirected.neighbors(node):
                if neighbor not in result:
                    next_frontier.add(neighbor)
        if not next_frontier:
            break
        result |= next_frontier
        frontier = next_frontier
    return result


# Cache keyed by the hashable `(db_file_path, table_id, resolution)` tuple.
# Stores the built graph + full-graph BC scores so successive requests with
# different top_n/max_depth/seed_metric don't repay the BC cost.
_GraphCache = tuple[nx.DiGraph, dict[str, float], dict[tuple[str, str], float]]
_GRAPH_CACHE: dict[tuple[str, str, float], _GraphCache] = {}


def _graph_with_metrics(
    cache_key: tuple[str, str, float] | None,
    nodes: list[KGNode],
    edges: list[KGEdge],
) -> _GraphCache:
    if cache_key is not None and cache_key in _GRAPH_CACHE:
        return _GRAPH_CACHE[cache_key]
    g = _build_graph(nodes, edges)
    node_bc, edge_bc = _compute_betweenness(g)
    result: _GraphCache = (g, node_bc, edge_bc)
    if cache_key is not None:
        _GRAPH_CACHE[cache_key] = result
    return result


def _select_and_expand(
    nodes: list[KGNode],
    edges: list[KGEdge],
    top_n: int,
    seed_metric: SeedMetric,
    max_depth: int,
    cache_key: tuple[str, str, float] | None = None,
) -> tuple[set[str], dict[str, float], dict[tuple[str, str], float]]:
    """Pick seeds by the metric, BFS-expand, and return kept ids + metrics."""
    g, node_bc, edge_bc = _graph_with_metrics(cache_key, nodes, edges)

    if top_n <= 0 or top_n >= len(nodes):
        return {n.id for n in nodes}, node_bc, edge_bc

    score = _seed_scores(nodes, g, node_bc, edge_bc, seed_metric)
    ranked = sorted(
        nodes,
        key=lambda n: (score.get(n.id, 0.0), n.mention_count or 0),
        reverse=True,
    )
    seeds = {n.id for n in ranked[:top_n]}
    kept = _bfs_expand(g, seeds, max_depth)
    return kept, node_bc, edge_bc


def _cache_key_for_conn(
    conn: sqlite3.Connection, table_id: str, resolution: float
) -> tuple[str, str, float] | None:
    """Stable cache key per (sqlite file path, table, resolution).

    Returns None for `:memory:` or other non-file backings — no point caching
    tests' disposable in-memory DBs, which may share the same Python id across
    test cases and collide.
    """
    for row in conn.execute("PRAGMA database_list"):
        if row["name"] == "main":
            path = str(row["file"])
            if not path:
                return None
            return (path, table_id, resolution)
    return None


def load_kg_graph(
    conn: sqlite3.Connection,
    table_id: str,
    resolution: float | None = None,
    top_n: int = DEFAULT_TOP_N,
    seed_metric: SeedMetric = DEFAULT_SEED_METRIC,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> KGPayload:
    """Assemble a KG payload: pick seeds, BFS-expand, attach BC metrics.

    `top_n` selects the N highest-scoring seed nodes by `seed_metric`. BFS
    then expands from those seeds through the undirected view up to
    `max_depth` hops (0 = unlimited — yields every connected component
    containing a seed). `total_node_count` / `total_edge_count` reflect the
    full graph for the UI's "showing N of M" banner.
    """
    if table_id not in KG_TABLES:
        raise UnknownKGTable(f"unknown KG table: {table_id!r}. Expected one of {KG_TABLES}")
    if seed_metric not in VALID_SEED_METRICS:
        raise ValueError(
            f"invalid seed_metric {seed_metric!r}; expected one of {list(VALID_SEED_METRICS)}"
        )
    if max_depth < 0:
        raise ValueError(f"max_depth must be >= 0, got {max_depth}")

    if not table_exists(conn, "leiden_communities"):
        raise KGDataMissing("leiden_communities table missing")

    resolved = _pick_resolution(conn, resolution)

    if table_id == "base":
        nodes, edges, communities = _load_base(conn, resolved)
    else:
        nodes, edges, communities = _load_er(conn, resolved)

    total_nodes = len(nodes)
    total_edges = len(edges)

    cache_key = _cache_key_for_conn(conn, table_id, resolved)
    kept, node_bc, edge_bc = _select_and_expand(
        nodes, edges, top_n, seed_metric, max_depth, cache_key=cache_key
    )

    kept_nodes = [
        KGNode(
            id=n.id,
            label=n.label,
            entity_type=n.entity_type,
            community_id=n.community_id,
            mention_count=n.mention_count,
            node_betweenness=node_bc.get(n.id),
        )
        for n in nodes
        if n.id in kept
    ]
    kept_edges = [
        KGEdge(
            source=e.source,
            target=e.target,
            rel_type=e.rel_type,
            weight=e.weight,
            edge_betweenness=edge_bc.get((e.source, e.target)),
        )
        for e in edges
        if e.source in kept and e.target in kept
    ]
    kept_communities = [
        KGCommunity(
            id=c.id,
            label=c.label,
            member_count=len([nid for nid in c.node_ids if nid in kept]),
            node_ids=[nid for nid in c.node_ids if nid in kept],
        )
        for c in communities
        if any(nid in kept for nid in c.node_ids)
    ]

    return KGPayload(
        table_id=table_id,
        resolution=resolved,
        seed_metric=seed_metric,
        max_depth=max_depth,
        node_count=len(kept_nodes),
        edge_count=len(kept_edges),
        community_count=len(kept_communities),
        total_node_count=total_nodes,
        total_edge_count=total_edges,
        nodes=kept_nodes,
        edges=kept_edges,
        communities=kept_communities,
    )
