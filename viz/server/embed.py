"""3D UMAP point loading for the Deck.GL scatter viz.

Two supported tables:
  * chunks   — 3D UMAP coords joined with chunk text (label = first 120 chars)
  * entities — 3D UMAP coords joined with entity name + entity_type (category)
"""

import sqlite3

from pydantic import BaseModel

from server.db import table_exists

LABEL_MAX_CHARS = 120

EMBED_TABLES = ("chunks", "entities")


class EmbedPoint(BaseModel):
    """One point in the 3D UMAP space."""

    id: int
    x: float
    y: float
    z: float
    label: str
    category: str | None = None


class EmbedPayload(BaseModel):
    """Response shape for the /embed/:table_id endpoint."""

    table_id: str
    count: int
    points: list[EmbedPoint]


class UnknownEmbedTable(ValueError):
    """Raised when table_id is not one of EMBED_TABLES."""


class EmbedDataMissing(RuntimeError):
    """Raised when the DB is missing the expected UMAP source tables."""


def _truncate(text: str | None) -> str:
    if not text:
        return ""
    t = text.strip()
    return t[:LABEL_MAX_CHARS] + ("…" if len(t) > LABEL_MAX_CHARS else "")


def load_embed_points(conn: sqlite3.Connection, table_id: str) -> EmbedPayload:
    """Load 3D UMAP points for `table_id` ∈ {'chunks', 'entities'}."""
    if table_id not in EMBED_TABLES:
        raise UnknownEmbedTable(f"unknown embed table: {table_id!r}. Expected one of {EMBED_TABLES}")

    if table_id == "chunks":
        if not table_exists(conn, "chunks_vec_umap") or not table_exists(conn, "chunks"):
            raise EmbedDataMissing("chunks_vec_umap or chunks table missing")
        rows = conn.execute(
            "SELECT u.id, u.x3d AS x, u.y3d AS y, u.z3d AS z, c.text "
            "FROM chunks_vec_umap u "
            "LEFT JOIN chunks c ON c.chunk_id = u.id "
            "ORDER BY u.id"
        ).fetchall()
        points = [
            EmbedPoint(
                id=r["id"],
                x=r["x"],
                y=r["y"],
                z=r["z"],
                label=_truncate(r["text"]) or f"#{r['id']}",
                category=None,
            )
            for r in rows
        ]
    else:  # entities
        if (
            not table_exists(conn, "entities_vec_umap")
            or not table_exists(conn, "entity_vec_map")
        ):
            raise EmbedDataMissing("entities_vec_umap or entity_vec_map table missing")
        rows = conn.execute(
            "SELECT u.id, u.x3d AS x, u.y3d AS y, u.z3d AS z, m.name, n.entity_type "
            "FROM entities_vec_umap u "
            "LEFT JOIN entity_vec_map m ON m.rowid = u.id "
            "LEFT JOIN nodes n ON n.name = m.name "
            "ORDER BY u.id"
        ).fetchall()
        points = [
            EmbedPoint(
                id=r["id"],
                x=r["x"],
                y=r["y"],
                z=r["z"],
                label=r["name"] or f"#{r['id']}",
                category=r["entity_type"],
            )
            for r in rows
        ]

    return EmbedPayload(table_id=table_id, count=len(points), points=points)
