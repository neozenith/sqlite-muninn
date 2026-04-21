"""Knowledge-graph payload for the Cytoscape viz.

Two table variants:
  * base — raw nodes + edges from NER / RE output (keyed by entity name)
  * er   — entity-resolved: collapse nodes by entity_clusters.canonical

Both payloads include communities (leiden_communities at a chosen
resolution) as a separate list; the client decides whether to render
them as cytoscape compound parents.
"""

import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping

from pydantic import BaseModel

from server.db import table_exists

KG_TABLES = ("base", "er")
DEFAULT_RESOLUTION = 0.25
DEFAULT_TOP_N = 500


class KGNode(BaseModel):
    id: str
    label: str
    entity_type: str | None = None
    community_id: int | None = None
    mention_count: int | None = None


class KGEdge(BaseModel):
    source: str
    target: str
    rel_type: str | None = None
    weight: float | None = None


class KGCommunity(BaseModel):
    id: int
    label: str | None = None
    member_count: int
    node_ids: list[str]


class KGPayload(BaseModel):
    table_id: str
    resolution: float
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
        # prefer default if present, else the smallest available
        return DEFAULT_RESOLUTION if DEFAULT_RESOLUTION in available else available[0]
    # exact match required — callers get an explicit error if they pass junk
    if requested not in available:
        raise ValueError(f"resolution {requested} not in {available}")
    return requested


def _load_base(
    conn: sqlite3.Connection, resolution: float
) -> tuple[list[KGNode], list[KGEdge], list[KGCommunity]]:
    if not table_exists(conn, "nodes") or not table_exists(conn, "edges"):
        raise KGDataMissing("base KG requires `nodes` and `edges` tables")

    # node → community_id at the chosen resolution
    community_map: dict[str, int] = {
        str(row["node"]): int(row["community_id"])
        for row in conn.execute(
            "SELECT node, community_id FROM leiden_communities WHERE resolution = ?",
            (resolution,),
        )
    }

    # Enriched metadata keyed by name (from the nodes table)
    node_meta: dict[str, tuple[str | None, int | None]] = {
        str(row["name"]): (row["entity_type"], row["mention_count"])
        for row in conn.execute("SELECT name, entity_type, mention_count FROM nodes")
    }

    # The real node set = union of every appearance across nodes / edges / leiden.
    # Upstream demo_builder output has case drift between these tables, so if we
    # took `nodes.name` as the source of truth the KG would have dangling
    # community members and unresolvable edge endpoints.
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

    # name → canonical. entity_clusters is not-null-both-columns by construction.
    canonical_map: dict[str, str] = {
        str(row["name"]): str(row["canonical"])
        for row in conn.execute("SELECT name, canonical FROM entity_clusters")
    }

    # canonical → display label (from entity_cluster_labels if available)
    cluster_labels: dict[str, str] = {}
    if table_exists(conn, "entity_cluster_labels"):
        cluster_labels = {
            str(row["canonical"]): str(row["label"])
            for row in conn.execute(
                "SELECT canonical, label FROM entity_cluster_labels WHERE label IS NOT NULL"
            )
        }

    # Aggregate entity_type + mention counts across members of each canonical
    type_counter: dict[str, Counter[str]] = defaultdict(Counter)
    mention_sum: dict[str, int] = defaultdict(int)
    for row in conn.execute("SELECT name, entity_type, mention_count FROM nodes"):
        name = str(row["name"])
        canonical = canonical_map.get(name, name)
        if row["entity_type"]:
            type_counter[canonical][row["entity_type"]] += 1
        mention_sum[canonical] += int(row["mention_count"] or 0)

    # Community assignment per canonical: most-common community_id across members
    community_rows = conn.execute(
        "SELECT node, community_id FROM leiden_communities WHERE resolution = ?",
        (resolution,),
    ).fetchall()
    canonical_communities: dict[str, Counter[int]] = defaultdict(Counter)
    for row in community_rows:
        node_name = str(row["node"])
        canonical = canonical_map.get(node_name, node_name)
        canonical_communities[canonical][int(row["community_id"])] += 1

    # Collapse edges by canonical mapping; drop self-loops; sum weights for duplicates
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

    # Node set = union of canonical forms observed anywhere. Covers the case
    # where an edge endpoint or community member isn't in entity_clusters.
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

    # Build communities from canonical_communities
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
    """Group nodes by community_id + join in community_labels for display."""
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


def _rank_by_degree(nodes: list[KGNode], edges: list[KGEdge]) -> dict[str, int]:
    """Count edge-endpoints per node. Tie-break with mention_count (more is better)."""
    degree: dict[str, int] = {n.id: 0 for n in nodes}
    for e in edges:
        if e.source in degree:
            degree[e.source] += 1
        if e.target in degree:
            degree[e.target] += 1
    return degree


def _filter_top_n(
    nodes: list[KGNode],
    edges: list[KGEdge],
    communities: list[KGCommunity],
    top_n: int,
) -> tuple[list[KGNode], list[KGEdge], list[KGCommunity]]:
    """Keep the N highest-degree nodes and every edge + community they appear in."""
    if top_n <= 0 or len(nodes) <= top_n:
        return nodes, edges, communities

    degree = _rank_by_degree(nodes, edges)
    ranked = sorted(
        nodes,
        key=lambda n: (degree.get(n.id, 0), n.mention_count or 0),
        reverse=True,
    )
    kept = {n.id for n in ranked[:top_n]}
    kept_nodes = [n for n in nodes if n.id in kept]
    kept_edges = [e for e in edges if e.source in kept and e.target in kept]
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
    return kept_nodes, kept_edges, kept_communities


def load_kg_graph(
    conn: sqlite3.Connection,
    table_id: str,
    resolution: float | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> KGPayload:
    """Assemble a KG payload for Cytoscape rendering.

    `top_n` caps the node count to the highest-degree-N nodes; the full
    counts are exposed via `total_node_count` / `total_edge_count` so the
    UI can render a "showing N of M" banner. Pass top_n=0 to disable.
    """
    if table_id not in KG_TABLES:
        raise UnknownKGTable(f"unknown KG table: {table_id!r}. Expected one of {KG_TABLES}")

    if not table_exists(conn, "leiden_communities"):
        raise KGDataMissing("leiden_communities table missing")

    resolved = _pick_resolution(conn, resolution)

    if table_id == "base":
        nodes, edges, communities = _load_base(conn, resolved)
    else:
        nodes, edges, communities = _load_er(conn, resolved)

    total_nodes = len(nodes)
    total_edges = len(edges)
    nodes, edges, communities = _filter_top_n(nodes, edges, communities, top_n)

    return KGPayload(
        table_id=table_id,
        resolution=resolved,
        node_count=len(nodes),
        edge_count=len(edges),
        community_count=len(communities),
        total_node_count=total_nodes,
        total_edge_count=total_edges,
        nodes=nodes,
        edges=edges,
        communities=communities,
    )
