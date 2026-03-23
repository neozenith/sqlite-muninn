"""Health check endpoint."""

import logging
import sqlite3
from typing import Any

from fastapi import APIRouter, Depends

from server.services.db import (
    db_session,
    discover_edge_tables,
    discover_hnsw_indexes,
    get_active_db_id,
    get_active_db_path,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health(conn: sqlite3.Connection = Depends(db_session)) -> dict[str, Any]:
    """Health check with database and extension status."""
    db_path = get_active_db_path()
    status: dict[str, Any] = {
        "status": "ok",
        "db_path": db_path,
        "active_database": get_active_db_id(),
    }

    try:
        indexes = discover_hnsw_indexes(conn)
        graphs = discover_edge_tables(conn)
        status["extension_loaded"] = True
        status["hnsw_index_count"] = len(indexes)
        status["edge_table_count"] = len(graphs)
    except Exception as e:
        log.warning("Health check database error: %s", e)
        status["extension_loaded"] = False
        status["error"] = str(e)

    return status
