"""VSS (Vector Similarity Search) endpoints."""

import logging
import sqlite3
import struct
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from server.services.db import db_session, discover_hnsw_indexes, sanitize_fts_query
from server.services.embeddings import get_projector
from server.services.validation import validate_identifier

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["vss"])


@router.get("/indexes")
def list_indexes(conn: sqlite3.Connection = Depends(db_session)) -> list[dict[str, Any]]:
    """List all discovered HNSW indexes."""
    return discover_hnsw_indexes(conn)


def _get_index_info(conn: sqlite3.Connection, name: str) -> dict[str, Any]:
    """Get a specific index info or raise 400."""
    try:
        validate_identifier(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    indexes = discover_hnsw_indexes(conn)
    match = next((i for i in indexes if i["name"] == name), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"Index not found: {name}")
    return match


def _load_vectors(conn: sqlite3.Connection, index_name: str, dimensions: int) -> list[tuple[int, list[float]]]:
    """Load all vectors from an HNSW index's _nodes shadow table."""
    rows = conn.execute(f"SELECT id, vector FROM [{index_name}_nodes]").fetchall()

    result = []
    for row in rows:
        node_id = row["id"]
        blob = row["vector"]
        vec = list(struct.unpack(f"{dimensions}f", blob))
        result.append((node_id, vec))
    return result


def _get_metadata(conn: sqlite3.Connection, index_name: str, node_id: int) -> dict[str, Any]:
    """Get metadata for a node based on the index type."""
    metadata: dict[str, Any] = {"id": node_id}

    if index_name == "chunks_vec":
        row = conn.execute("SELECT text FROM chunks WHERE chunk_id = ?", (node_id,)).fetchone()
        if row:
            metadata["text"] = row["text"][:200]
            metadata["type"] = "chunk"
    elif index_name == "entities_vec":
        row = conn.execute("SELECT name FROM entity_vec_map WHERE rowid = ?", (node_id,)).fetchone()
        if row:
            metadata["name"] = row["name"]
            metadata["type"] = "entity"
    elif index_name == "node2vec_emb":
        # Node2Vec uses first-seen ordering from edge table
        metadata["type"] = "node2vec"
        # Name is set externally via _build_n2v_name_map

    return metadata


def _build_n2v_name_map(conn: sqlite3.Connection) -> dict[int, str]:
    """Build rowid → node name map for node2vec_emb using first-seen edge ordering."""
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
    return {idx + 1: name for name, idx in name_to_idx.items()}


@router.get("/vss/{index_name}/embeddings")
def get_embeddings(
    index_name: str,
    dimensions: int = Query(default=2, ge=2, le=3),
    conn: sqlite3.Connection = Depends(db_session),
) -> dict[str, Any]:
    """Get UMAP-projected embeddings for an HNSW index."""
    info = _get_index_info(conn, index_name)

    vectors_raw = _load_vectors(conn, index_name, info["dimensions"])
    if not vectors_raw:
        return {"index": index_name, "count": 0, "points": []}

    ids = [v[0] for v in vectors_raw]
    vectors = [v[1] for v in vectors_raw]

    # Build node name map for node2vec embeddings
    n2v_name_map: dict[int, str] = {}
    if index_name == "node2vec_emb":
        n2v_name_map = _build_n2v_name_map(conn)

    # UMAP project
    projector = get_projector()
    projected = projector.fit_transform(vectors, n_components=dimensions)

    points = []
    for i, node_id in enumerate(ids):
        point: dict[str, Any] = {
            "id": node_id,
            "x": round(float(projected[i][0]), 4),
            "y": round(float(projected[i][1]), 4),
        }
        if dimensions == 3:
            point["z"] = round(float(projected[i][2]), 4)

        metadata = _get_metadata(conn, index_name, node_id)

        # For node2vec, inject the node name from the edge-ordering map
        if index_name == "node2vec_emb" and node_id in n2v_name_map:
            metadata["name"] = n2v_name_map[node_id]

        point["label"] = metadata.get("name") or metadata.get("text", "")[:50]
        point["metadata"] = metadata

        points.append(point)

    return {
        "index": index_name,
        "count": len(points),
        "original_dimensions": info["dimensions"],
        "projected_dimensions": dimensions,
        "points": points,
    }


@router.get("/vss/{index_name}/search")
def search_vss(
    index_name: str,
    query_id: int = Query(...),
    k: int = Query(default=20, ge=1, le=100),
    ef_search: int = Query(default=64, ge=1, le=500),
    conn: sqlite3.Connection = Depends(db_session),
) -> dict[str, Any]:
    """KNN search from a given node ID in an HNSW index."""
    _get_index_info(conn, index_name)  # validates index exists

    # Get the query vector
    row = conn.execute(f"SELECT vector FROM [{index_name}_nodes] WHERE id = ?", (query_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Node {query_id} not found in {index_name}")

    query_blob = row["vector"]

    # KNN search via the virtual table
    results = conn.execute(
        f"SELECT rowid, distance FROM [{index_name}] WHERE vector MATCH ? AND k = ? AND ef_search = ?",
        (query_blob, k, ef_search),
    ).fetchall()

    neighbors = []
    for r in results:
        neighbor = {
            "id": r["rowid"],
            "distance": round(r["distance"], 6),
        }
        metadata = _get_metadata(conn, index_name, r["rowid"])
        neighbor["label"] = metadata.get("name") or metadata.get("text", "")[:50]
        neighbor["metadata"] = metadata
        neighbors.append(neighbor)

    return {
        "index": index_name,
        "query_id": query_id,
        "k": k,
        "count": len(neighbors),
        "neighbors": neighbors,
    }


@router.get("/vss/{index_name}/search_text")
def search_text(
    index_name: str,
    q: str = Query(..., min_length=1),
    k: int = Query(default=20, ge=1, le=100),
    ef_search: int = Query(default=64, ge=1, le=500),
    conn: sqlite3.Connection = Depends(db_session),
) -> dict[str, Any]:
    """Text search: FTS5 match → centroid vector → HNSW KNN search.

    Only works for indexes with a companion FTS5 table (e.g., chunks_vec + chunks_fts).
    """
    _get_index_info(conn, index_name)  # validates index exists

    # Determine FTS5 table name (e.g., chunks_vec → chunks_fts)
    fts_table = index_name.replace("_vec", "_fts")

    # Check that the FTS5 table exists
    table_exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (fts_table,)).fetchone()
    if not table_exists:
        raise HTTPException(
            status_code=400,
            detail=f"No FTS5 companion table '{fts_table}' found for index '{index_name}'",
        )

    # FTS5 match → get chunk IDs (rowid maps to content_rowid)
    fts_q = sanitize_fts_query(q)
    if not fts_q:
        return {"index": index_name, "query": q, "k": k, "count": 0, "results": []}
    try:
        fts_rows = conn.execute(
            f"SELECT rowid FROM [{fts_table}] WHERE [{fts_table}] MATCH ?",
            (fts_q,),
        ).fetchall()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"FTS5 query failed: {e}") from e

    if not fts_rows:
        return {
            "index": index_name,
            "query": q,
            "k": k,
            "count": 0,
            "neighbors": [],
        }

    # Load vectors for matched chunk IDs
    chunk_ids = [row[0] for row in fts_rows]
    placeholders = ",".join("?" * len(chunk_ids))
    vec_rows = conn.execute(
        f"SELECT id, vector FROM [{index_name}_nodes] WHERE id IN ({placeholders})",
        chunk_ids,
    ).fetchall()

    if not vec_rows:
        return {
            "index": index_name,
            "query": q,
            "k": k,
            "count": 0,
            "neighbors": [],
        }

    # Compute centroid vector
    vectors = []
    for row in vec_rows:
        vec = np.frombuffer(row["vector"], dtype=np.float32)
        vectors.append(vec)
    centroid = np.mean(vectors, axis=0).astype(np.float32)
    centroid_blob = centroid.tobytes()

    # HNSW KNN search using centroid
    results = conn.execute(
        f"SELECT rowid, distance FROM [{index_name}] WHERE vector MATCH ? AND k = ? AND ef_search = ?",
        (centroid_blob, k, ef_search),
    ).fetchall()

    neighbors = []
    for r in results:
        neighbor = {
            "id": r["rowid"],
            "distance": round(r["distance"], 6),
        }
        metadata = _get_metadata(conn, index_name, r["rowid"])
        neighbor["label"] = metadata.get("name") or metadata.get("text", "")[:50]
        neighbor["metadata"] = metadata
        neighbors.append(neighbor)

    return {
        "index": index_name,
        "query": q,
        "k": k,
        "count": len(neighbors),
        "neighbors": neighbors,
    }
