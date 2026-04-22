"""FastAPI application entry point for muninn-viz."""

import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from server.databases import DatabaseInfo, ManifestError, get_database, load_manifest
from server.db import DatabaseConnectionError, open_demo_db, table_exists
from server.embed import (
    EMBED_TABLES,
    EmbedDataMissing,
    EmbedPayload,
    UnknownEmbedTable,
    load_embed_points,
)
from server.kg import (
    KG_TABLES,
    KGDataMissing,
    KGPayload,
    UnknownKGTable,
    load_kg_graph,
)

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


# ── Health & manifest ────────────────────────────────────────────────────


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


# ── Per-database discovery + viz endpoints ───────────────────────────────


class TablesResponse(BaseModel):
    """Which embed + kg table variants are available in a given database."""

    database_id: str
    embed_tables: list[str]
    kg_tables: list[str]
    resolutions: list[float]


@app.get("/api/databases/{database_id}/tables")
def get_tables(
    database_id: str, demos_dir: Path = Depends(get_demos_dir)
) -> TablesResponse:
    """Discover which embed + kg tables exist for the given database."""
    try:
        with open_demo_db(demos_dir, database_id) as conn:
            embed_available: list[str] = []
            if table_exists(conn, "chunks_vec_umap") and table_exists(conn, "chunks"):
                embed_available.append("chunks")
            if table_exists(conn, "entities_vec_umap") and table_exists(conn, "entity_vec_map"):
                embed_available.append("entities")

            kg_available: list[str] = []
            if table_exists(conn, "nodes") and table_exists(conn, "edges"):
                kg_available.append("base")
            if table_exists(conn, "entity_clusters") and table_exists(conn, "edges"):
                kg_available.append("er")

            resolutions: list[float] = []
            if table_exists(conn, "leiden_communities"):
                resolutions = [
                    float(r[0])
                    for r in conn.execute(
                        "SELECT DISTINCT resolution FROM leiden_communities ORDER BY resolution"
                    )
                ]

        return TablesResponse(
            database_id=database_id,
            embed_tables=embed_available,
            kg_tables=kg_available,
            resolutions=resolutions,
        )
    except DatabaseConnectionError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/api/databases/{database_id}/embed/{table_id}")
def get_embed(
    database_id: str,
    table_id: str,
    demos_dir: Path = Depends(get_demos_dir),
) -> EmbedPayload:
    """Return 3D UMAP points for the given database + embed table."""
    if table_id not in EMBED_TABLES:
        raise HTTPException(
            status_code=400,
            detail=f"invalid embed table {table_id!r}; expected one of {list(EMBED_TABLES)}",
        )
    try:
        with open_demo_db(demos_dir, database_id) as conn:
            return load_embed_points(conn, table_id)
    except DatabaseConnectionError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except UnknownEmbedTable as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except EmbedDataMissing as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.get("/api/databases/{database_id}/kg/{table_id}")
def get_kg(
    database_id: str,
    table_id: str,
    resolution: float | None = None,
    top_n: int = 50,
    seed_metric: str = "edge_betweenness",
    max_depth: int = 0,
    min_degree: int = 1,
    demos_dir: Path = Depends(get_demos_dir),
) -> KGPayload:
    """Return the KG payload (nodes + edges + communities).

    `top_n` picks the N highest-scoring seed nodes by `seed_metric`
    (degree / node_betweenness / edge_betweenness). BFS then expands from
    those seeds through the undirected edge view up to `max_depth` hops
    (0 = unlimited, yields every connected component containing a seed).
    The response carries per-node `node_betweenness` and per-edge
    `edge_betweenness` scores computed over the FULL graph, so the client
    can size elements by those metrics without recomputing.
    """
    if table_id not in KG_TABLES:
        raise HTTPException(
            status_code=400,
            detail=f"invalid kg table {table_id!r}; expected one of {list(KG_TABLES)}",
        )
    if seed_metric not in ("degree", "node_betweenness", "edge_betweenness"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"invalid seed_metric {seed_metric!r}; "
                "expected one of ['degree', 'node_betweenness', 'edge_betweenness']"
            ),
        )
    if max_depth < 0:
        raise HTTPException(status_code=400, detail=f"max_depth must be >= 0, got {max_depth}")
    if min_degree < 0:
        raise HTTPException(status_code=400, detail=f"min_degree must be >= 0, got {min_degree}")
    try:
        with open_demo_db(demos_dir, database_id) as conn:
            return load_kg_graph(
                conn,
                table_id,
                resolution,
                top_n=top_n,
                seed_metric=seed_metric,  # type: ignore[arg-type]
                max_depth=max_depth,
                min_degree=min_degree,
            )
    except DatabaseConnectionError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except UnknownKGTable as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KGDataMissing as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
