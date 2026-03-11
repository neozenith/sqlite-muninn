"""Knowledge Graph search: FTS + VSS + CTE graph query."""

import logging
import sqlite3
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from server.services.db import sanitize_fts_query

log = logging.getLogger(__name__)

# ── Embedding model config ────────────────────────────────────────────

# NomicEmbed is an asymmetric model: documents use "search_document: " prefix
# (applied at index-build time in demo_builder), queries use "search_query: ".
# MiniLM is symmetric — no prefixes needed.
_EMBEDDING_CONFIGS: dict[str, dict[str, Any]] = {
    "MiniLM": {
        "st_name": "all-MiniLM-L6-v2",
        "query_prefix": "",
    },
    "NomicEmbed": {
        "st_name": "nomic-ai/nomic-embed-text-v1.5",
        "query_prefix": "search_query: ",
        "trust_remote_code": True,
    },
}

# Map raw model slugs (from manifest model field or meta table) to config keys.
# The GGUF slug from sessions_demo maps to NomicEmbed because it uses the same
# base model (nomic-embed-text-v1.5) — just quantized via llama.cpp instead of
# full-precision via sentence-transformers.
_SLUG_TO_CONFIG: dict[str, str] = {
    "MiniLM": "MiniLM",
    "all-MiniLM-L6-v2": "MiniLM",
    "NomicEmbed": "NomicEmbed",
    "nomic-ai/nomic-embed-text-v1.5": "NomicEmbed",
    "nomic-embed-text-v1.5.Q8_0.gguf": "NomicEmbed",
}

# Per-slug model cache (lazy-loaded on first use of each slug)
_embedding_model_cache: dict[str, Any] = {}
_active_config_key: str | None = None  # None = auto-detect from DB meta table


def set_active_embedding_model(model_slug: str) -> None:
    """Update the active embedding model when the database is switched."""
    global _active_config_key
    config_key = _SLUG_TO_CONFIG.get(model_slug, "MiniLM")
    _active_config_key = config_key
    log.info("Active embedding model: %s (from slug: %s)", config_key, model_slug)


def _detect_model_from_db(conn: sqlite3.Connection) -> str:
    """Read embedding_model from the meta table and map to a config key."""
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
        if row:
            slug = row["value"] if isinstance(row, sqlite3.Row) else row[0]
            config_key = _SLUG_TO_CONFIG.get(slug, "MiniLM")
            log.info("Auto-detected embedding model from meta table: %s → %s", slug, config_key)
            return config_key
    except Exception:
        pass
    return "MiniLM"


def _get_embedding_model(conn: sqlite3.Connection | None = None) -> tuple[Any, str]:
    """Lazy-load the correct sentence-transformers model for the active DB.

    Returns (model, query_prefix). The query_prefix must be prepended to
    query text before encoding for asymmetric models like NomicEmbed.
    """
    global _embedding_model_cache, _active_config_key

    key = _active_config_key
    if key is None and conn is not None:
        key = _detect_model_from_db(conn)
        _active_config_key = key
    elif key is None:
        key = "MiniLM"

    if key not in _embedding_model_cache:
        cfg = _EMBEDDING_CONFIGS.get(key, _EMBEDDING_CONFIGS["MiniLM"])
        kwargs: dict[str, Any] = {}
        if cfg.get("trust_remote_code"):
            kwargs["trust_remote_code"] = True
        model = SentenceTransformer(cfg["st_name"], **kwargs)
        _embedding_model_cache[key] = model
        log.info("Loaded embedding model: %s", cfg["st_name"])

    cfg = _EMBEDDING_CONFIGS.get(key, _EMBEDDING_CONFIGS["MiniLM"])
    return _embedding_model_cache[key], cfg.get("query_prefix", "")


def _pack_vector(v: np.ndarray) -> bytes:
    """Pack a numpy array into a float32 blob for SQLite."""
    return v.astype(np.float32).tobytes()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Check if a table exists."""
    row = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?", (name,)
    ).fetchone()
    return bool(row[0] > 0)


# ── CTE graph query SQL ──────────────────────────────────────────────

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
    """Run a KG search with server-side embedding, FTS, VSS + UMAP, and CTE graph query."""
    result: dict[str, Any] = {"query": query_text}

    # Embed query if model available; VSS + graph need this.
    # For asymmetric models (NomicEmbed), prepend the query prefix so it
    # lands in the same embedding subspace as the indexed documents.
    query_blob: bytes | None = None
    try:
        model, query_prefix = _get_embedding_model(conn)
        text_to_embed = query_prefix + query_text if query_prefix else query_text
        embedding = model.encode([text_to_embed], normalize_embeddings=True)[0]
        query_blob = _pack_vector(embedding)
    except RuntimeError:
        log.warning("sentence-transformers not available; VSS and graph search disabled")

    # 1. FTS results (no embedding needed)
    fts_results: list[dict[str, Any]] = []
    fts_query = sanitize_fts_query(query_text)
    if fts_query and _table_exists(conn, "chunks_fts"):
        try:
            rows = conn.execute(
                "SELECT chunk_id, text FROM chunks WHERE chunk_id IN "
                "(SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 20)",
                (fts_query,),
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
                (query_blob, 50, 3),
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

    # 4. Leiden community detection scoped to BFS nodes
    node_community: dict[str, int] = {}
    if graph_nodes and _table_exists(conn, "relations"):
        try:
            node_set = {n["name"] for n in graph_nodes}
            leiden_rows = conn.execute(
                "SELECT node, community_id FROM graph_leiden(  'relations', 'src', 'dst', 'both', 1.0)"
            ).fetchall()
            for row in leiden_rows:
                if row["node"] in node_set:
                    node_community[row["node"]] = row["community_id"]
        except Exception as e:
            log.warning("Leiden community detection failed: %s", e)
    result["node_community"] = node_community
    result["community_count"] = len(set(node_community.values())) if node_community else 0

    return result
