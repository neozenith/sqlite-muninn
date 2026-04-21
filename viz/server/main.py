"""FastAPI application entry point for muninn-viz."""

import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

from server.databases import DatabaseInfo, ManifestError, get_database, load_manifest

log = logging.getLogger(__name__)

DEFAULT_DEMOS_DIR = Path(__file__).resolve().parent.parent / "frontend" / "public" / "demos"


def get_demos_dir() -> Path:
    """Resolve the demos directory — env var takes precedence over default."""
    return Path(os.environ.get("MUNINN_DEMOS_DIR", str(DEFAULT_DEMOS_DIR)))


app = FastAPI(
    title="muninn-viz",
    description="Interactive visualization for the muninn SQLite extension",
    version="0.1.0",
)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe — returns {'status': 'ok'} when the server is up."""
    return {"status": "ok"}


@app.get("/api/databases")
def list_databases(demos_dir: Path = Depends(get_demos_dir)) -> dict[str, list[DatabaseInfo]]:
    """Return every database entry from the demos manifest."""
    try:
        databases = load_manifest(demos_dir)
    except ManifestError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"databases": databases}


@app.get("/api/databases/{database_id}")
def get_database_info(database_id: str, demos_dir: Path = Depends(get_demos_dir)) -> DatabaseInfo:
    """Return metadata for a single database, or 404 if unknown."""
    try:
        db = get_database(demos_dir, database_id)
    except ManifestError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    if db is None:
        raise HTTPException(status_code=404, detail=f"Database '{database_id}' not found")
    return db
