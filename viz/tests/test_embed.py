"""Tests for server.embed — 3D UMAP point loading."""

import sqlite3
from pathlib import Path

import pytest

from server.db import open_demo_db
from server.embed import (
    EMBED_TABLES,
    EmbedDataMissing,
    UnknownEmbedTable,
    load_embed_points,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEMOS_DIR = PROJECT_ROOT / "viz" / "frontend" / "public" / "demos"
SAMPLE_DB_ID = "3300_MiniLM"

HAS_DEMO = (DEMOS_DIR / f"{SAMPLE_DB_ID}.db").exists()


def test_unknown_table_raises() -> None:
    conn = sqlite3.connect(":memory:")
    with pytest.raises(UnknownEmbedTable, match="unknown embed table"):
        load_embed_points(conn, "bogus")
    conn.close()


def test_missing_tables_raises_embed_data_missing() -> None:
    conn = sqlite3.connect(":memory:")
    with pytest.raises(EmbedDataMissing, match="chunks_vec_umap"):
        load_embed_points(conn, "chunks")
    with pytest.raises(EmbedDataMissing, match="entities_vec_umap"):
        load_embed_points(conn, "entities")
    conn.close()


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_load_chunks_returns_points() -> None:
    with open_demo_db(DEMOS_DIR, SAMPLE_DB_ID) as conn:
        payload = load_embed_points(conn, "chunks")
    assert payload.table_id == "chunks"
    assert payload.count == len(payload.points)
    assert payload.count > 0
    first = payload.points[0]
    assert isinstance(first.id, int)
    assert isinstance(first.x, float)
    assert isinstance(first.y, float)
    assert isinstance(first.z, float)
    assert len(first.label) > 0


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_load_entities_returns_points_with_category() -> None:
    with open_demo_db(DEMOS_DIR, SAMPLE_DB_ID) as conn:
        payload = load_embed_points(conn, "entities")
    assert payload.table_id == "entities"
    assert payload.count > 0
    # Entities have entity_type populated for most rows
    with_category = [p for p in payload.points if p.category is not None]
    assert len(with_category) > 0


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_all_embed_tables_enumerate() -> None:
    with open_demo_db(DEMOS_DIR, SAMPLE_DB_ID) as conn:
        for table_id in EMBED_TABLES:
            payload = load_embed_points(conn, table_id)
            assert payload.count > 0


def test_label_truncation() -> None:
    """Server-side truncation keeps long chunks within a sane payload size."""
    from server.embed import LABEL_MAX_CHARS, _truncate

    long = "x" * (LABEL_MAX_CHARS + 50)
    truncated = _truncate(long)
    assert len(truncated) <= LABEL_MAX_CHARS + 1  # +1 for the ellipsis
    assert truncated.endswith("…")


def test_truncate_empty_values() -> None:
    from server.embed import _truncate

    assert _truncate(None) == ""
    assert _truncate("") == ""
    assert _truncate("  short  ") == "short"
