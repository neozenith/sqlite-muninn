"""Database listing and switching endpoints."""

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server import config as _config
from server.services import kg as _kg
from server.services.db import get_active_db_id, set_active_db

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
def select_database(body: SelectRequest) -> dict[str, Any]:
    """Switch the active database.

    No connection parameter needed — just atomically swaps the path.
    In-flight requests finish on the old DB, new requests use the new path.
    """
    databases = _read_manifest()
    db_map = {db["id"]: db for db in databases}

    if body.id not in db_map:
        raise HTTPException(status_code=404, detail=f"Database not found: {body.id}")

    db_info = db_map[body.id]
    db_path = str(_config.DEMOS_DIR / db_info["file"])

    # Validate the file exists before committing to the switch
    if not Path(db_path).exists():
        raise HTTPException(status_code=404, detail=f"Database file not found: {db_path}")

    # Atomic switch — no connection teardown, no lock contention
    set_active_db(body.id, db_path)

    # Notify the embedding service so it loads the correct model on next query.
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
