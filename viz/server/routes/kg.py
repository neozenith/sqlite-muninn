"""Knowledge Graph pipeline endpoints."""

import logging
import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from server.services.db import db_session
from server.services.kg import get_pipeline_summary, get_stage_detail, run_kg_search

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kg", tags=["kg"])


@router.get("/pipeline")
def pipeline_summary(conn: sqlite3.Connection = Depends(db_session)) -> dict[str, Any]:
    """Get summary of all KG pipeline stages."""
    return get_pipeline_summary(conn)


@router.get("/stage/{stage_num}")
def stage_detail(stage_num: int, conn: sqlite3.Connection = Depends(db_session)) -> dict[str, Any]:
    """Get detailed data for a specific pipeline stage (1-7)."""
    if stage_num < 1 or stage_num > 7:
        raise HTTPException(status_code=400, detail="Stage number must be between 1 and 7")

    return get_stage_detail(conn, stage_num)


# Stage → (table_name, columns, filter_column)
_STAGE_TABLE_MAP: dict[int, tuple[str, list[str], str]] = {
    1: ("chunks", ["chunk_id", "text"], "text"),
    3: ("entities", ["name", "entity_type", "chunk_id"], "name"),
    4: ("relations", ["src", "dst", "rel_type", "weight"], "src"),
    5: ("entity_clusters", ["canonical", "name"], "name"),
}


@router.get("/stage/{stage_num}/items")
def stage_items(
    stage_num: int,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    q: str = Query(default=""),
    conn: sqlite3.Connection = Depends(db_session),
) -> dict[str, Any]:
    """Get paginated rows from a pipeline stage's table."""
    if stage_num not in _STAGE_TABLE_MAP:
        raise HTTPException(status_code=400, detail=f"Stage {stage_num} does not support item listing")

    table, columns, filter_col = _STAGE_TABLE_MAP[stage_num]

    # Check table exists
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    if not exists:
        return {"stage": stage_num, "items": [], "total": 0, "page": page, "page_size": page_size}

    # Discover which columns actually exist (graceful if schema differs)
    pragma_rows = conn.execute(f"PRAGMA table_info([{table}])").fetchall()
    actual_cols = {row["name"] for row in pragma_rows}
    select_cols = [c for c in columns if c in actual_cols]
    if not select_cols:
        select_cols = list(actual_cols)[:4]  # Fallback to first 4 columns

    col_expr = ", ".join(f"[{c}]" for c in select_cols)

    # Filter
    where = ""
    params: list[Any] = []
    if q and filter_col in actual_cols:
        where = f"WHERE [{filter_col}] LIKE ?"
        params.append(f"%{q}%")

    # Total count
    total = conn.execute(f"SELECT count(*) FROM [{table}] {where}", params).fetchone()[0]

    # Paginated query
    offset = (page - 1) * page_size
    rows = conn.execute(
        f"SELECT {col_expr} FROM [{table}] {where} LIMIT ? OFFSET ?",
        params + [page_size, offset],
    ).fetchall()

    items = [dict(row) for row in rows]

    return {
        "stage": stage_num,
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/stage/3/entities-grouped")
def entities_grouped(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    q: str = Query(default=""),
    conn: sqlite3.Connection = Depends(db_session),
) -> dict[str, Any]:
    """Get entities grouped by (name, entity_type) with aggregated chunk_ids."""
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='entities'").fetchone()
    if not exists:
        return {"items": [], "total": 0, "page": page, "page_size": page_size}

    where = ""
    params: list[Any] = []
    if q:
        where = "WHERE [name] LIKE ?"
        params.append(f"%{q}%")

    total = conn.execute(
        f"SELECT count(*) FROM (SELECT 1 FROM [entities] {where} GROUP BY [name], [entity_type])",
        params,
    ).fetchone()[0]

    offset = (page - 1) * page_size
    rows = conn.execute(
        f"""SELECT [name], [entity_type],
                   GROUP_CONCAT([chunk_id]) AS chunk_ids,
                   count(*) AS mention_count
            FROM [entities] {where}
            GROUP BY [name], [entity_type]
            ORDER BY count(*) DESC
            LIMIT ? OFFSET ?""",
        params + [page_size, offset],
    ).fetchall()

    items = [
        {
            "name": r["name"],
            "entity_type": r["entity_type"],
            "chunk_ids": r["chunk_ids"].split(",") if r["chunk_ids"] else [],
            "mention_count": r["mention_count"],
        }
        for r in rows
    ]

    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/stage/3/entities-by-chunk")
def entities_by_chunk(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    q: str = Query(default=""),
    conn: sqlite3.Connection = Depends(db_session),
) -> dict[str, Any]:
    """Get entities grouped by chunk_id."""
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='entities'").fetchone()
    if not exists:
        return {"items": [], "total": 0, "page": page, "page_size": page_size}

    # Get distinct chunk_ids from entities
    where = ""
    params: list[Any] = []
    if q:
        where = "WHERE [name] LIKE ?"
        params.append(f"%{q}%")

    total = conn.execute(
        f"SELECT count(DISTINCT [chunk_id]) FROM [entities] {where}",
        params,
    ).fetchone()[0]

    offset = (page - 1) * page_size
    chunk_rows = conn.execute(
        f"SELECT DISTINCT [chunk_id] FROM [entities] {where} ORDER BY [chunk_id] LIMIT ? OFFSET ?",
        params + [page_size, offset],
    ).fetchall()

    items = []
    for cr in chunk_rows:
        chunk_id = cr["chunk_id"]
        ent_rows = conn.execute(
            "SELECT [name], [entity_type] FROM [entities] WHERE [chunk_id] = ?",
            (chunk_id,),
        ).fetchall()

        # Get full text from chunks table (no truncation — data validation use case)
        text = ""
        chunk_row = conn.execute(
            "SELECT [text] FROM [chunks] WHERE [chunk_id] = ? LIMIT 1",
            (chunk_id,),
        ).fetchone()
        if chunk_row:
            text = chunk_row["text"] or ""

        items.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "entities": [{"name": e["name"], "entity_type": e["entity_type"]} for e in ent_rows],
                "entity_count": len(ent_rows),
            }
        )

    return {"items": items, "total": total, "page": page, "page_size": page_size}


class KGQueryRequest(BaseModel):
    """Request body for KG search query."""

    query: str
    k: int = 10


@router.post("/query")
def kg_query(request: KGQueryRequest, conn: sqlite3.Connection = Depends(db_session)) -> dict[str, Any]:
    """Execute a KG search with server-side embedding, FTS, VSS + UMAP, and CTE graph."""

    try:
        return run_kg_search(conn, request.query, k=request.k)
    except Exception as e:
        log.error("KG search failed: %s", e)
        raise HTTPException(status_code=500, detail=f"KG search failed: {e}") from e
