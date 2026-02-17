"""Knowledge Graph pipeline stage queries and GraphRAG execution."""

import logging
from typing import Any

import numpy as np

try:
    import pysqlite3 as sqlite3  # type: ignore[import-not-found]
except ImportError:
    import sqlite3

try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

log = logging.getLogger(__name__)

# ── Embedding model cache ─────────────────────────────────────────────

_embedding_model: Any = None


def _get_embedding_model() -> Any:
    """Lazy-load and cache the sentence-transformers model."""
    global _embedding_model
    if _embedding_model is None:
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError("sentence-transformers not installed. pip install sentence-transformers")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Loaded embedding model: all-MiniLM-L6-v2")
    return _embedding_model


def _pack_vector(v: np.ndarray) -> bytes:
    """Pack a numpy array into a float32 blob for SQLite."""
    return v.astype(np.float32).tobytes()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Check if a table exists."""
    row = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", (name,)
    ).fetchone()
    return bool(row[0] > 0)


def _safe_count(conn: sqlite3.Connection, table: str) -> int:
    """Count rows in a table, returning 0 if table doesn't exist."""
    if not _table_exists(conn, table):
        return 0
    return int(conn.execute(f"SELECT count(*) FROM [{table}]").fetchone()[0])


def get_pipeline_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    """Get a summary of all pipeline stages."""
    stages = []

    # Stage 1: Chunking
    chunk_count = _safe_count(conn, "chunks")
    stages.append(
        {
            "stage": 1,
            "name": "Chunking",
            "description": "Text split into paragraphs",
            "count": chunk_count,
            "available": chunk_count > 0,
        }
    )

    # Stage 2: Embedding
    chunks_vec_count = _safe_count(conn, "chunks_vec_nodes")
    stages.append(
        {
            "stage": 2,
            "name": "Embedding",
            "description": "Chunk embeddings via HNSW",
            "count": chunks_vec_count,
            "available": chunks_vec_count > 0,
        }
    )

    # Stage 3: Entity Extraction
    entity_count = _safe_count(conn, "entities")
    stages.append(
        {
            "stage": 3,
            "name": "Entity Extraction",
            "description": "Named entities extracted from chunks",
            "count": entity_count,
            "available": entity_count > 0,
        }
    )

    # Stage 4: Relation Extraction
    relation_count = _safe_count(conn, "relations")
    stages.append(
        {
            "stage": 4,
            "name": "Relation Extraction",
            "description": "Entity-to-entity relationships",
            "count": relation_count,
            "available": relation_count > 0,
        }
    )

    # Stage 5: Entity Resolution
    cluster_count = _safe_count(conn, "entity_clusters")
    stages.append(
        {
            "stage": 5,
            "name": "Entity Resolution",
            "description": "Synonym detection via HNSW + Leiden",
            "count": cluster_count,
            "available": cluster_count > 0,
        }
    )

    # Stage 6: Graph Construction
    node_count = _safe_count(conn, "nodes")
    edge_count = _safe_count(conn, "edges")
    stages.append(
        {
            "stage": 6,
            "name": "Graph Construction",
            "description": "Coalesced knowledge graph",
            "count": node_count,
            "available": node_count > 0,
            "extra": {"node_count": node_count, "edge_count": edge_count},
        }
    )

    # Stage 7: Node2Vec
    n2v_count = _safe_count(conn, "node2vec_emb_nodes")
    stages.append(
        {
            "stage": 7,
            "name": "Node2Vec",
            "description": "Structural graph embeddings",
            "count": n2v_count,
            "available": n2v_count > 0,
        }
    )

    return {"stages": stages}


def get_stage_detail(conn: sqlite3.Connection, stage_num: int) -> dict[str, Any]:
    """Get detailed data for a pipeline stage."""
    if stage_num == 1:
        return _stage_chunks(conn)
    if stage_num == 2:
        return _stage_embeddings(conn)
    if stage_num == 3:
        return _stage_entities(conn)
    if stage_num == 4:
        return _stage_relations(conn)
    if stage_num == 5:
        return _stage_resolution(conn)
    if stage_num == 6:
        return _stage_graph(conn)
    if stage_num == 7:
        return _stage_node2vec(conn)
    return {"error": f"Unknown stage: {stage_num}"}


def _stage_chunks(conn: sqlite3.Connection) -> dict[str, Any]:
    """Stage 1: Chunk details."""
    if not _table_exists(conn, "chunks"):
        return {"stage": 1, "available": False}

    count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    # Text length distribution
    lengths = conn.execute("SELECT length(text) FROM chunks").fetchall()
    lens = [r[0] for r in lengths]

    sample = conn.execute("SELECT chunk_id, text FROM chunks LIMIT 5").fetchall()
    samples = [{"chunk_id": r["chunk_id"], "text_preview": r["text"][:200]} for r in sample]

    return {
        "stage": 1,
        "name": "Chunking",
        "count": count,
        "text_lengths": {
            "min": min(lens) if lens else 0,
            "max": max(lens) if lens else 0,
            "mean": round(sum(lens) / len(lens), 1) if lens else 0,
        },
        "samples": samples,
    }


def _stage_embeddings(conn: sqlite3.Connection) -> dict[str, Any]:
    """Stage 2: Embedding index stats."""
    if not _table_exists(conn, "chunks_vec_config"):
        return {"stage": 2, "available": False}

    config_rows = conn.execute("SELECT key, value FROM chunks_vec_config").fetchall()
    config = {r["key"]: r["value"] for r in config_rows}
    count = _safe_count(conn, "chunks_vec_nodes")

    return {
        "stage": 2,
        "name": "Embedding",
        "count": count,
        "config": config,
    }


def _stage_entities(conn: sqlite3.Connection) -> dict[str, Any]:
    """Stage 3: Entity extraction details."""
    if not _table_exists(conn, "entities"):
        return {"stage": 3, "available": False}

    count = conn.execute("SELECT count(*) FROM entities").fetchone()[0]

    # Breakdown by entity type (if column exists)
    type_breakdown = []
    try:
        rows = conn.execute(
            "SELECT entity_type, count(*) as cnt FROM entities GROUP BY entity_type ORDER BY cnt DESC"
        ).fetchall()
        type_breakdown = [{"type": r["entity_type"], "count": r["cnt"]} for r in rows]
    except Exception:
        pass

    # Strategy breakdown (if column exists)
    strategy_breakdown = []
    try:
        rows = conn.execute(
            "SELECT strategy, count(*) as cnt FROM entities GROUP BY strategy ORDER BY cnt DESC"
        ).fetchall()
        strategy_breakdown = [{"strategy": r["strategy"], "count": r["cnt"]} for r in rows]
    except Exception:
        pass

    return {
        "stage": 3,
        "name": "Entity Extraction",
        "count": count,
        "by_type": type_breakdown,
        "by_strategy": strategy_breakdown,
    }


def _stage_relations(conn: sqlite3.Connection) -> dict[str, Any]:
    """Stage 4: Relation extraction details."""
    if not _table_exists(conn, "relations"):
        return {"stage": 4, "available": False}

    count = conn.execute("SELECT count(*) FROM relations").fetchone()[0]

    type_breakdown = []
    try:
        rows = conn.execute(
            "SELECT rel_type, count(*) as cnt FROM relations GROUP BY rel_type ORDER BY cnt DESC LIMIT 20"
        ).fetchall()
        type_breakdown = [{"type": r["rel_type"], "count": r["cnt"]} for r in rows]
    except Exception:
        pass

    sample = conn.execute("SELECT src, dst, rel_type, weight FROM relations LIMIT 10").fetchall()
    samples = [{"src": r["src"], "dst": r["dst"], "rel_type": r["rel_type"], "weight": r["weight"]} for r in sample]

    return {
        "stage": 4,
        "name": "Relation Extraction",
        "count": count,
        "by_type": type_breakdown,
        "samples": samples,
    }


def _stage_resolution(conn: sqlite3.Connection) -> dict[str, Any]:
    """Stage 5: Entity resolution (cluster) details."""
    if not _table_exists(conn, "entity_clusters"):
        return {"stage": 5, "available": False}

    total = conn.execute("SELECT count(*) FROM entity_clusters").fetchone()[0]
    canonical_count = conn.execute("SELECT count(DISTINCT canonical) FROM entity_clusters").fetchone()[0]
    merged = conn.execute("SELECT count(*) FROM entity_clusters WHERE name != canonical").fetchone()[0]

    # Largest clusters
    clusters = conn.execute("""
        SELECT canonical, count(*) as size
        FROM entity_clusters
        GROUP BY canonical
        HAVING count(*) > 1
        ORDER BY size DESC
        LIMIT 10
    """).fetchall()

    cluster_list = []
    for row in clusters:
        members = conn.execute("SELECT name FROM entity_clusters WHERE canonical = ?", (row["canonical"],)).fetchall()
        cluster_list.append(
            {
                "canonical": row["canonical"],
                "size": row["size"],
                "members": [m["name"] for m in members],
            }
        )

    return {
        "stage": 5,
        "name": "Entity Resolution",
        "total_entities": total,
        "canonical_count": canonical_count,
        "merged_count": merged,
        "merge_ratio": round(merged / total, 3) if total > 0 else 0,
        "largest_clusters": cluster_list,
    }


def _stage_graph(conn: sqlite3.Connection) -> dict[str, Any]:
    """Stage 6: Graph construction details."""
    node_count = _safe_count(conn, "nodes")
    edge_count = _safe_count(conn, "edges")

    if node_count == 0:
        return {"stage": 6, "available": False}

    # Degree distribution
    degree_dist = []
    try:
        rows = conn.execute("""
            SELECT degree, count(*) as cnt FROM (
                SELECT src as node, count(*) as degree FROM edges GROUP BY src
                UNION ALL
                SELECT dst as node, count(*) as degree FROM edges GROUP BY dst
            )
            GROUP BY degree ORDER BY degree
        """).fetchall()
        degree_dist = [{"degree": r["degree"], "count": r["cnt"]} for r in rows]
    except Exception:
        pass

    # Weight distribution
    weight_stats: dict[str, Any] = {}
    try:
        row = conn.execute("SELECT min(weight), max(weight), avg(weight) FROM edges").fetchone()
        weight_stats = {"min": row[0], "max": row[1], "mean": round(row[2], 2) if row[2] else 0}
    except Exception:
        pass

    return {
        "stage": 6,
        "name": "Graph Construction",
        "node_count": node_count,
        "edge_count": edge_count,
        "degree_distribution": degree_dist,
        "weight_stats": weight_stats,
    }


def _stage_node2vec(conn: sqlite3.Connection) -> dict[str, Any]:
    """Stage 7: Node2Vec embedding stats."""
    if not _table_exists(conn, "node2vec_emb_config"):
        return {"stage": 7, "available": False}

    config_rows = conn.execute("SELECT key, value FROM node2vec_emb_config").fetchall()
    config = {r["key"]: r["value"] for r in config_rows}
    count = _safe_count(conn, "node2vec_emb_nodes")

    return {
        "stage": 7,
        "name": "Node2Vec",
        "count": count,
        "config": config,
    }


def _build_n2v_rowid_map(conn: sqlite3.Connection) -> dict[str, int]:
    """Replicate node2vec_train's first-seen ordering to map names to rowids."""
    try:
        rows = conn.execute("SELECT src, dst FROM edges").fetchall()
    except Exception:
        return {}

    name_to_idx: dict[str, int] = {}
    for row in rows:
        src, dst = row["src"], row["dst"]
        if src not in name_to_idx:
            name_to_idx[src] = len(name_to_idx)
        if dst not in name_to_idx:
            name_to_idx[dst] = len(name_to_idx)

    return {name: idx + 1 for name, idx in name_to_idx.items()}


def run_graphrag_query(
    conn: sqlite3.Connection,
    query_text: str,
    *,
    k: int = 10,
    max_depth: int = 2,
) -> dict[str, Any]:
    """Run a GraphRAG query using pre-embedded chunks (no embedding model needed).

    Uses chunks_fts full-text search as the entry point instead of HNSW vector search,
    so this endpoint works without sentence-transformers installed.
    """
    result: dict[str, Any] = {"query": query_text, "stages": {}}

    # Stage 1: Full-text search on chunks (no embedding model needed)
    seed_chunks: list[dict[str, Any]] = []
    if _table_exists(conn, "chunks_fts"):
        try:
            rows = conn.execute(
                """
                SELECT chunk_id, text FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?
            """,
                (query_text, k),
            ).fetchall()
            seed_chunks = [{"chunk_id": r["chunk_id"], "text_preview": r["text"][:200]} for r in rows]
        except Exception as e:
            log.warning("FTS search failed: %s", e)

    result["stages"]["1_fts_chunks"] = {"count": len(seed_chunks), "chunks": seed_chunks[:5]}

    # Stage 2: Seed entities from matching chunks
    seed_entities: set[str] = set()
    seed_chunk_ids = [c["chunk_id"] for c in seed_chunks]
    if seed_chunk_ids and _table_exists(conn, "entity_clusters"):
        placeholders = ",".join("?" * len(seed_chunk_ids))
        try:
            entity_rows = conn.execute(
                f"""
                SELECT DISTINCT ec.canonical
                FROM entities e
                JOIN entity_clusters ec ON e.name = ec.name
                WHERE e.chunk_id IN ({placeholders})
            """,
                seed_chunk_ids,
            ).fetchall()
            seed_entities = {r["canonical"] for r in entity_rows}
        except Exception as e:
            log.warning("Seed entity lookup failed: %s", e)

    result["stages"]["2_seed_entities"] = {
        "count": len(seed_entities),
        "entities": sorted(seed_entities)[:20],
    }

    # Stage 3: BFS 2-hop expansion
    expanded_entities = set(seed_entities)
    for seed in list(seed_entities)[:10]:
        try:
            bfs_rows = conn.execute(
                """
                SELECT node, depth FROM graph_bfs
                WHERE edge_table = 'edges'
                  AND src_col = 'src'
                  AND dst_col = 'dst'
                  AND start_node = ?
                  AND max_depth = ?
            """,
                (seed, max_depth),
            ).fetchall()
            for row in bfs_rows:
                expanded_entities.add(row["node"])
        except Exception:
            pass

    newly_discovered = expanded_entities - seed_entities
    result["stages"]["3_bfs_expansion"] = {
        "seed_count": len(seed_entities),
        "expanded_count": len(expanded_entities),
        "newly_discovered": sorted(newly_discovered)[:20],
    }

    # Stage 4: Centrality ranking
    centrality_scores: dict[str, dict[str, float]] = {}
    for measure in ("betweenness", "degree", "closeness"):
        tvf = f"graph_{measure}"
        direction_clause = "AND direction = 'both'" if measure != "degree" else ""
        try:
            rows = conn.execute(f"""
                SELECT node, centrality FROM [{tvf}]
                WHERE edge_table = 'edges'
                  AND src_col = 'src'
                  AND dst_col = 'dst'
                  {direction_clause}
            """).fetchall()
            for row in rows:
                node = row["node"]
                if node in expanded_entities:
                    centrality_scores.setdefault(node, {})[measure] = round(row["centrality"], 6)
        except Exception:
            pass

    ranked: list[dict[str, Any]] = []
    for node, scores in centrality_scores.items():
        combined = sum(scores.values()) / max(len(scores), 1)
        ranked.append({"node": node, "combined_score": round(combined, 6), **scores})
    ranked.sort(key=lambda x: -x["combined_score"])

    result["stages"]["4_centrality"] = {"count": len(ranked), "top_bridges": ranked[:10]}

    # Stage 5: Leiden communities
    community_entities: set[str] = set()
    try:
        communities = conn.execute("""
            SELECT node, community_id FROM graph_leiden
            WHERE edge_table = 'edges'
              AND src_col = 'src'
              AND dst_col = 'dst'
              AND weight_col = 'weight'
        """).fetchall()
        node_community = {row["node"]: row["community_id"] for row in communities}
        seed_communities = {node_community[s] for s in seed_entities if s in node_community}
        for row in communities:
            if row["community_id"] in seed_communities:
                community_entities.add(row["node"])
    except Exception:
        pass

    result["stages"]["5_leiden_communities"] = {
        "community_entities": len(community_entities),
        "newly_added": sorted(community_entities - expanded_entities)[:20],
    }

    # Stage 6: Node2Vec structural similarity
    n2v_similar: set[str] = set()
    n2v_rowid_map = _build_n2v_rowid_map(conn)
    if n2v_rowid_map and _table_exists(conn, "node2vec_emb_nodes"):
        rowid_to_name = {v: k for k, v in n2v_rowid_map.items()}
        for seed in list(seed_entities)[:5]:
            seed_rowid = n2v_rowid_map.get(seed)
            if seed_rowid is None:
                continue
            try:
                vec_row = conn.execute("SELECT vector FROM node2vec_emb_nodes WHERE id = ?", (seed_rowid,)).fetchone()
                if vec_row is None:
                    continue
                knn = conn.execute(
                    "SELECT rowid, distance FROM node2vec_emb WHERE vector MATCH ? AND k = 10 AND ef_search = 64",
                    (vec_row["vector"],),
                ).fetchall()
                for r in knn:
                    name = rowid_to_name.get(r["rowid"])
                    if name and name != seed:
                        n2v_similar.add(name)
            except Exception:
                pass

    result["stages"]["6_node2vec"] = {
        "structurally_similar": len(n2v_similar),
        "newly_added": sorted(n2v_similar - expanded_entities - community_entities)[:20],
    }

    # Stage 7: Assembly
    all_entities = expanded_entities | community_entities | n2v_similar

    # Collect relevant passages
    top_passages: list[dict[str, Any]] = []
    for entity in sorted(all_entities)[:50]:
        try:
            rows = conn.execute(
                """
                SELECT DISTINCT c.chunk_id, c.text
                FROM entities e
                JOIN entity_clusters ec ON e.name = ec.name
                JOIN chunks c ON e.chunk_id = c.chunk_id
                WHERE ec.canonical = ?
            """,
                (entity,),
            ).fetchall()
            for r in rows:
                top_passages.append(
                    {
                        "chunk_id": r["chunk_id"],
                        "text_preview": r["text"][:300],
                        "entity": entity,
                    }
                )
        except Exception:
            pass

    # Deduplicate by chunk_id
    seen_chunks: set[int] = set()
    unique_passages: list[dict[str, Any]] = []
    for p in top_passages:
        if p["chunk_id"] not in seen_chunks:
            seen_chunks.add(p["chunk_id"])
            unique_passages.append(p)

    result["stages"]["7_assembly"] = {
        "total_entities": len(all_entities),
        "total_passages": len(unique_passages),
        "top_passages": unique_passages[:10],
    }

    return result


# ── KG Search (with Python embedding + CTE graph query) ──────────────


_CTE_NODES_SQL = """
WITH
vss_matches AS (
    SELECT rowid AS vec_rowid, distance AS cosine_distance
    FROM entities_vec
    WHERE vector MATCH ? AND k = ?
),
vss_entities AS (
    SELECT m.name, v.cosine_distance, (1.0 - v.cosine_distance) AS similarity
    FROM vss_matches v
    JOIN entity_vec_map m ON m.rowid = v.vec_rowid
),
anchor AS (
    SELECT name FROM vss_entities ORDER BY cosine_distance ASC LIMIT 1
),
bfs_neighbors AS (
    SELECT node, depth
    FROM graph_bfs
    WHERE edge_table = 'relations' AND src_col = 'src' AND dst_col = 'dst'
      AND start_node = (SELECT name FROM anchor)
      AND max_depth = ? AND direction = 'both'
),
scored AS (
    SELECT b.node, b.depth,
           COALESCE(v.cosine_distance, 1.0) AS cosine_distance,
           COALESCE(v.similarity, 0.0) AS similarity
    FROM bfs_neighbors b
    LEFT JOIN vss_entities v ON v.name = b.node
)
SELECT node AS name, depth, similarity FROM scored
"""


def run_kg_search(
    conn: sqlite3.Connection,
    query_text: str,
    *,
    k: int = 10,
) -> dict[str, Any]:
    """Run a KG search with server-side embedding, FTS, VSS + UMAP, and CTE graph query.

    Degrades gracefully: FTS always works; VSS and graph need sentence-transformers.
    """
    result: dict[str, Any] = {"query": query_text}

    # Embed query if model available; VSS + graph need this
    query_blob: bytes | None = None
    try:
        model = _get_embedding_model()
        embedding = model.encode([query_text], normalize_embeddings=True)[0]
        query_blob = _pack_vector(embedding)
    except RuntimeError:
        log.warning("sentence-transformers not available; VSS and graph search disabled")

    # 1. FTS results (no embedding needed)
    fts_results: list[dict[str, Any]] = []
    if _table_exists(conn, "chunks_fts"):
        try:
            rows = conn.execute(
                "SELECT chunk_id, text FROM chunks WHERE chunk_id IN "
                "(SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 20)",
                (query_text,),
            ).fetchall()
            fts_results = [{"chunk_id": r["chunk_id"], "text": r["text"]} for r in rows]
        except Exception as e:
            log.warning("FTS search failed: %s", e)
    result["fts_results"] = fts_results

    # 2. VSS results on chunks_vec + UMAP coords (needs embedding)
    vss_results: list[dict[str, Any]] = []
    if query_blob is not None and _table_exists(conn, "chunks_vec_nodes"):
        try:
            rows = conn.execute(
                "SELECT cv.rowid, cv.distance FROM chunks_vec cv WHERE cv.vector MATCH ? AND cv.k = ?",
                (query_blob, k),
            ).fetchall()
            rowids = [r["rowid"] for r in rows]
            distances = {r["rowid"]: r["distance"] for r in rows}

            if rowids:
                placeholders = ",".join("?" * len(rowids))
                # Join with chunks for text and umap for 3D coords
                chunk_rows = conn.execute(
                    f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({placeholders})",
                    rowids,
                ).fetchall()
                chunk_map = {r["chunk_id"]: r["text"] for r in chunk_rows}

                umap_map: dict[int, dict[str, float]] = {}
                if _table_exists(conn, "chunks_vec_umap"):
                    umap_rows = conn.execute(
                        f"SELECT id, x3d, y3d, z3d FROM chunks_vec_umap WHERE id IN ({placeholders})",
                        rowids,
                    ).fetchall()
                    umap_map = {r["id"]: {"x3d": r["x3d"], "y3d": r["y3d"], "z3d": r["z3d"]} for r in umap_rows}

                for rid in rowids:
                    dist = distances[rid]
                    coords = umap_map.get(rid, {})
                    vss_results.append(
                        {
                            "chunk_id": rid,
                            "similarity": round(1.0 - dist, 4),
                            "distance": round(dist, 4),
                            "text": chunk_map.get(rid, ""),
                            "x3d": coords.get("x3d"),
                            "y3d": coords.get("y3d"),
                            "z3d": coords.get("z3d"),
                        }
                    )
        except Exception as e:
            log.warning("VSS search failed: %s", e)
    result["vss_results"] = vss_results

    # 3. CTE graph search: VSS on entities_vec → anchor → BFS → scored nodes
    #    Split into two queries to avoid UNION ALL + virtual table CTE bug in SQLite
    #    (CTEs referencing virtual tables aren't reliably materialized across UNION ALL)
    graph_nodes: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []
    if query_blob is not None and _table_exists(conn, "entities_vec_nodes") and _table_exists(conn, "entity_vec_map"):
        try:
            # Query 1: Get nodes via CTE (references virtual tables once)
            rows = conn.execute(
                _CTE_NODES_SQL,
                (query_blob, 50, 1),
            ).fetchall()
            node_names: list[str] = []
            for row in rows:
                graph_nodes.append(
                    {
                        "name": row["name"],
                        "depth": row["depth"],
                        "similarity": round(float(row["similarity"]), 4),
                        "is_anchor": row["depth"] == 0,
                    }
                )
                node_names.append(row["name"])

            # Query 2: Get edges using collected node names (plain SQL, no virtual tables)
            if len(node_names) > 1:
                placeholders = ",".join("?" * len(node_names))
                edge_rows = conn.execute(
                    f"SELECT src, rel_type, dst FROM relations "
                    f"WHERE src IN ({placeholders}) AND dst IN ({placeholders})",
                    node_names + node_names,
                ).fetchall()
                for row in edge_rows:
                    graph_edges.append(
                        {
                            "src": row["src"],
                            "rel": row["rel_type"],
                            "dst": row["dst"],
                        }
                    )
        except Exception as e:
            log.warning("CTE graph search failed: %s", e)
    result["graph_nodes"] = graph_nodes
    result["graph_edges"] = graph_edges

    return result
