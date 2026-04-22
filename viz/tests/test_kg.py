"""Tests for server.kg — KG payload assembly."""

import sqlite3
from pathlib import Path

import pytest

from server.db import open_demo_db
from server.kg import (
    KG_TABLES,
    DEFAULT_RESOLUTION,
    KGDataMissing,
    UnknownKGTable,
    load_kg_graph,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEMOS_DIR = PROJECT_ROOT / "viz" / "frontend" / "public" / "demos"
SAMPLE_DB_ID = "3300_MiniLM"

HAS_DEMO = (DEMOS_DIR / f"{SAMPLE_DB_ID}.db").exists()


def test_unknown_kg_table() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE leiden_communities (node TEXT, resolution REAL, community_id INTEGER, modularity REAL)")
    with pytest.raises(UnknownKGTable):
        load_kg_graph(conn, "bogus")
    conn.close()


def test_missing_leiden_raises() -> None:
    conn = sqlite3.connect(":memory:")
    with pytest.raises(KGDataMissing, match="leiden_communities"):
        load_kg_graph(conn, "base")
    conn.close()


def test_missing_base_tables_raises() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE leiden_communities (node TEXT, resolution REAL, community_id INTEGER, modularity REAL)")
    conn.execute("INSERT INTO leiden_communities VALUES ('n', 0.25, 1, 0.0)")
    with pytest.raises(KGDataMissing, match="nodes.*edges"):
        load_kg_graph(conn, "base")
    conn.close()


def test_invalid_resolution_raises() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE leiden_communities (node TEXT, resolution REAL, community_id INTEGER, modularity REAL)")
    conn.execute("INSERT INTO leiden_communities VALUES ('n', 0.25, 1, 0.0)")
    conn.execute("CREATE TABLE nodes (node_id INTEGER, name TEXT, entity_type TEXT, mention_count INTEGER)")
    conn.execute("CREATE TABLE edges (src TEXT, dst TEXT, rel_type TEXT, weight REAL)")
    with pytest.raises(ValueError, match="not in"):
        load_kg_graph(conn, "base", resolution=999.0)
    conn.close()


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_load_base_kg() -> None:
    with open_demo_db(DEMOS_DIR, SAMPLE_DB_ID) as conn:
        payload = load_kg_graph(conn, "base")

    assert payload.table_id == "base"
    assert payload.resolution == DEFAULT_RESOLUTION
    assert payload.node_count > 0
    assert payload.edge_count > 0
    assert payload.community_count > 0
    assert len(payload.nodes) == payload.node_count
    assert len(payload.edges) == payload.edge_count
    assert len(payload.communities) == payload.community_count
    # Every node is a string id
    assert all(isinstance(n.id, str) for n in payload.nodes)
    # Every community has 1+ members and references actual node ids
    node_ids = {n.id for n in payload.nodes}
    for c in payload.communities:
        assert c.member_count == len(c.node_ids)
        assert set(c.node_ids).issubset(node_ids)


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_load_er_kg_collapses_duplicates() -> None:
    with open_demo_db(DEMOS_DIR, SAMPLE_DB_ID) as conn:
        base = load_kg_graph(conn, "base")
        er = load_kg_graph(conn, "er")

    # ER should never have MORE nodes than base; equal is OK if ER is trivial
    assert er.node_count <= base.node_count
    # ER should not contain self-loops
    assert all(e.source != e.target for e in er.edges)


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_all_kg_tables_enumerate() -> None:
    with open_demo_db(DEMOS_DIR, SAMPLE_DB_ID) as conn:
        for table_id in KG_TABLES:
            payload = load_kg_graph(conn, table_id)
            assert payload.node_count > 0


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_resolution_override() -> None:
    with open_demo_db(DEMOS_DIR, SAMPLE_DB_ID) as conn:
        payload = load_kg_graph(conn, "base", resolution=1.0)
    assert payload.resolution == 1.0


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_seed_metric_invalid_raises() -> None:
    with open_demo_db(DEMOS_DIR, SAMPLE_DB_ID) as conn:
        with pytest.raises(ValueError, match="invalid seed_metric"):
            load_kg_graph(conn, "base", seed_metric="bogus")  # type: ignore[arg-type]


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_max_depth_negative_raises() -> None:
    with open_demo_db(DEMOS_DIR, SAMPLE_DB_ID) as conn:
        with pytest.raises(ValueError, match="max_depth"):
            load_kg_graph(conn, "base", max_depth=-2)


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_betweenness_metrics_attached() -> None:
    """Every returned node/edge carries a full-graph BC score (or None for isolates)."""
    with open_demo_db(DEMOS_DIR, SAMPLE_DB_ID) as conn:
        payload = load_kg_graph(conn, "base", top_n=20, max_depth=1)
    # Nodes that participated in any edge have a numeric BC value
    node_bc = [n.node_betweenness for n in payload.nodes]
    assert any(bc is not None for bc in node_bc)
    # Edges that survive expansion carry their BC score
    edge_bc = [e.edge_betweenness for e in payload.edges]
    assert any(bc is not None for bc in edge_bc)


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_max_depth_limits_expansion() -> None:
    """Depth-0 is unlimited; depth=1 should never exceed depth=0."""
    with open_demo_db(DEMOS_DIR, SAMPLE_DB_ID) as conn:
        unlimited = load_kg_graph(conn, "base", top_n=5, max_depth=0)
        depth_one = load_kg_graph(conn, "base", top_n=5, max_depth=1)
    assert depth_one.node_count <= unlimited.node_count
