"""Knowledge Graph search endpoint."""

import logging
import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from server.services.db import db_session
from server.services.kg import run_kg_search

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kg", tags=["kg"])


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
