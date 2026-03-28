"""HNSW embedding and KNN blocking — common pre-step for all pipelines.

Embeds entity names via muninn_embed() into an HNSW index, then
finds candidate pairs via KNN search with cosine distance filtering.
"""

import logging
import sqlite3

from .datasets import Entity
from .models import EMBED_MODEL

log = logging.getLogger(__name__)


def embed_and_block(
    conn: sqlite3.Connection,
    entities: list[Entity],
    k: int = 10,
    dist_threshold: float = 0.4,
) -> tuple[dict[int, str], dict[int, str], dict[tuple[int, int], float]]:
    """Create entities table, embed into HNSW, run KNN blocking.

    Returns:
        id_map: rowid -> entity_id
        name_map: rowid -> entity name
        candidate_pairs: (min_rowid, max_rowid) -> cosine_distance
    """
    conn.execute("CREATE TABLE entities(entity_id TEXT, name TEXT, source TEXT)")
    for e in entities:
        conn.execute("INSERT INTO entities VALUES(?, ?, ?)", (e.id, e.name, e.source))

    dim = conn.execute("SELECT muninn_model_dim(?)", (EMBED_MODEL.name,)).fetchone()[0]
    conn.execute(f"CREATE VIRTUAL TABLE entity_vecs USING hnsw_index(dimensions={dim}, metric=cosine)")
    conn.execute(
        "INSERT INTO entity_vecs(rowid, vector) SELECT rowid, muninn_embed(?, name) FROM entities",
        (EMBED_MODEL.name,),
    )
    log.info("Embedded %d entities (dim=%d)", len(entities), dim)

    id_map: dict[int, str] = {}
    name_map: dict[int, str] = {}
    for row in conn.execute("SELECT rowid, entity_id, name FROM entities"):
        id_map[row[0]] = row[1]
        name_map[row[0]] = row[2]

    candidate_pairs: dict[tuple[int, int], float] = {}
    for rowid in id_map:
        vec = conn.execute("SELECT vector FROM entity_vecs WHERE rowid = ?", (rowid,)).fetchone()[0]
        neighbors = conn.execute(
            "SELECT rowid, distance FROM entity_vecs WHERE vector MATCH ? AND k = ?",
            (vec, k + 1),
        ).fetchall()
        for nid, dist in neighbors:
            if nid != rowid and dist <= dist_threshold:
                pair = (min(rowid, nid), max(rowid, nid))
                if pair not in candidate_pairs or dist < candidate_pairs[pair]:
                    candidate_pairs[pair] = dist

    log.info("HNSW blocking: %d candidate pairs (k=%d, dist<=%.2f)", len(candidate_pairs), k, dist_threshold)
    return id_map, name_map, candidate_pairs


def leiden_cluster(
    conn: sqlite3.Connection,
    entities: list[Entity],
    match_edges: list[tuple[str, str, float]],
) -> dict[str, int]:
    """Run Leiden clustering on match edges. Returns entity_id -> cluster_id."""
    conn.execute("CREATE TEMP TABLE _match_edges(src TEXT, dst TEXT, weight REAL)")
    for src, dst, w in match_edges:
        conn.execute("INSERT INTO _match_edges VALUES(?, ?, ?)", (src, dst, w))
        conn.execute("INSERT INTO _match_edges VALUES(?, ?, ?)", (dst, src, w))

    clusters: dict[str, int] = {}
    next_id = 0

    if match_edges:
        leiden_results = conn.execute(
            "SELECT node, community_id FROM graph_leiden"
            " WHERE edge_table = '_match_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND weight_col = 'weight'"
        ).fetchall()
        comm_to_id: dict[int, int] = {}
        for node, comm_id in leiden_results:
            if comm_id not in comm_to_id:
                comm_to_id[comm_id] = next_id
                next_id += 1
            clusters[node] = comm_to_id[comm_id]

    for e in entities:
        if e.id not in clusters:
            clusters[e.id] = next_id
            next_id += 1

    n_multi = sum(1 for sz in _cluster_sizes(clusters).values() if sz > 1)
    log.info("Leiden: %d clusters (%d multi-member)", len(set(clusters.values())), n_multi)
    return clusters


def _cluster_sizes(clusters: dict[str, int]) -> dict[int, int]:
    from collections import defaultdict

    sizes: dict[int, int] = defaultdict(int)
    for cid in clusters.values():
        sizes[cid] += 1
    return dict(sizes)
