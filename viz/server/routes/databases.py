"""Database listing and switching endpoints."""

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from server import config as _config
from server.services.db import db_session, get_active_db_id, reconnect, set_active_db_id
from server.services import kg as _kg

try:
    import pysqlite3 as sqlite3  # type: ignore[import-not-found]
except ImportError:
    import sqlite3

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["databases"])


def _read_manifest() -> list[dict[str, Any]]:
    """Read manifest.json from _config.DEMOS_DIR. Returns empty list if missing."""
    manifest_path = _config.DEMOS_DIR / "manifest.json"
    if not manifest_path.exists():
        return []
    data: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: list[dict[str, Any]] = data.get("databases", [])
    return result


class SelectRequest(BaseModel):
    id: str


@router.get("/databases")
def list_databases() -> dict[str, Any]:
    """List available demo databases from manifest.json."""
    databases = _read_manifest()
    active = get_active_db_id()
    return {"databases": databases, "active": active}


@router.post("/databases/select")
def select_database(
    body: SelectRequest,
    conn: sqlite3.Connection = Depends(db_session),
) -> dict[str, Any]:
    """Switch the active database connection to the selected demo DB."""
    databases = _read_manifest()
    db_map = {db["id"]: db for db in databases}

    if body.id not in db_map:
        raise HTTPException(status_code=404, detail=f"Database not found: {body.id}")

    db_info = db_map[body.id]
    db_path = str(_config.DEMOS_DIR / db_info["file"])

    reconnect(db_path)
    set_active_db_id(body.id)

    # Notify the embedding service so it loads the correct model on next query.
    # The manifest's model field maps to our embedding config keys
    # (e.g. "NomicEmbed" → nomic-ai/nomic-embed-text-v1.5 with query prefix).
    model_slug = db_info.get("model")
    if model_slug:
        _kg.set_active_embedding_model(model_slug)

    log.info("Switched to database: %s (%s)", body.id, db_path)

    return {
        "status": "ok",
        "active": body.id,
        "db_path": db_path,
        "model": db_info.get("model"),
        "dim": db_info.get("dim"),
    }
