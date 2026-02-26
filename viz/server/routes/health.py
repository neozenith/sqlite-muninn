"""Health check endpoint."""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from server.config import DB_PATH, EXTENSION_PATH
from server.services.db import discover_edge_tables, discover_hnsw_indexes, get_active_db_id, get_connection

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health() -> dict[str, Any]:
    """Health check with database and extension status."""
    status: dict[str, Any] = {
        "status": "ok",
        "db_path": DB_PATH,
        "db_exists": Path(DB_PATH).exists(),
        "extension_path": EXTENSION_PATH,
        "active_database": get_active_db_id(),
    }

    try:
        conn = get_connection()
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
