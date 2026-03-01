"""Tests for KG pipeline service functions."""

import pathlib
import struct

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

from unittest.mock import MagicMock, patch

import numpy as np

from server.services.kg import (
    _build_n2v_rowid_map,
    get_pipeline_summary,
    get_stage_detail,
    run_graphrag_query,
    run_kg_search,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")


def _make_kg_conn(tmp_path: pathlib.Path) -> sqlite3.Connection:
    """Create a connection with minimal KG tables for testing."""
    db_path = str(tmp_path / "kg_test.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)

    # Stage 1: chunks
    conn.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT)")
    conn.executemany(
        "INSERT INTO chunks VALUES (?, ?)",
        [
            (1, "Adam Smith wrote about the division of labor."),
            (2, "The wealth of nations depends on productive labor."),
            (3, "Free trade promotes economic growth."),
        ],
    )

    # Stage 3: entities
    conn.execute("""
        CREATE TABLE entities (
            name TEXT, entity_type TEXT, strategy TEXT, chunk_id INTEGER
        )
    """)
    conn.executemany(
        "INSERT INTO entities VALUES (?, ?, ?, ?)",
        [
            ("Adam Smith", "PERSON", "llm", 1),
            ("division of labor", "CONCEPT", "llm", 1),
            ("wealth of nations", "WORK", "llm", 2),
        ],
    )

    # Stage 4: relations
    conn.execute("""
        CREATE TABLE relations (
            src TEXT, dst TEXT, rel_type TEXT, weight REAL
        )
    """)
    conn.executemany(
        "INSERT INTO relations VALUES (?, ?, ?, ?)",
        [
            ("Adam Smith", "division of labor", "wrote_about", 1.0),
            ("Adam Smith", "wealth of nations", "authored", 1.0),
        ],
    )

    # Stage 5: entity_clusters
    conn.execute("CREATE TABLE entity_clusters (name TEXT PRIMARY KEY, canonical TEXT)")
    conn.executemany(
        "INSERT INTO entity_clusters VALUES (?, ?)",
        [
            ("Adam Smith", "Adam Smith"),
            ("division of labor", "division of labor"),
            ("wealth of nations", "wealth of nations"),
        ],
    )

    # Stage 6: nodes + edges
    conn.execute("""
        CREATE TABLE nodes (
            node_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE, entity_type TEXT, mention_count INTEGER
        )
    """)
    conn.executemany(
        "INSERT INTO nodes (name, entity_type, mention_count) VALUES (?, ?, ?)",
        [
            ("Adam Smith", "PERSON", 5),
            ("division of labor", "CONCEPT", 3),
        ],
    )
    conn.execute("""
        CREATE TABLE edges (src TEXT, dst TEXT, rel_type TEXT, weight REAL)
    """)
    conn.execute("INSERT INTO edges VALUES ('Adam Smith', 'division of labor', 'wrote_about', 1.0)")

    conn.commit()
    return conn


def _make_full_kg_conn(tmp_path: pathlib.Path) -> sqlite3.Connection:
    """Create a connection with all KG tables including HNSW, FTS, and Node2Vec."""
    conn = _make_kg_conn(tmp_path)

    # Stage 2: HNSW index for chunk embeddings
    conn.execute("""
        CREATE VIRTUAL TABLE chunks_vec USING hnsw_index(
            dimensions=4, metric='cosine', m=8, ef_construction=50
        )
    """)
    for i in range(1, 4):
        vec = struct.pack("4f", float(i), float(i + 1), float(i + 2), float(i + 3))
        conn.execute("INSERT INTO chunks_vec (rowid, vector) VALUES (?, ?)", (i, vec))
    conn.commit()

    # Stage 7: Node2Vec embeddings
    conn.execute("""
        CREATE VIRTUAL TABLE node2vec_emb USING hnsw_index(
            dimensions=4, metric='cosine', m=8, ef_construction=50
        )
    """)
    # Node2Vec assigns rowids by first-seen order in edges table
    # edges: ("Adam Smith" -> "division of labor")
    # So: Adam Smith = rowid 1, division of labor = rowid 2
    for i in range(1, 3):
        vec = struct.pack("4f", float(i * 0.1), float(i * 0.2), float(i * 0.3), float(i * 0.4))
        conn.execute("INSERT INTO node2vec_emb (rowid, vector) VALUES (?, ?)", (i, vec))
    conn.commit()

    # FTS for full-text search on chunks
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id, text)")
    conn.executemany(
        "INSERT INTO chunks_fts VALUES (?, ?)",
        [
            (1, "Adam Smith wrote about the division of labor."),
            (2, "The wealth of nations depends on productive labor."),
            (3, "Free trade promotes economic growth."),
        ],
    )
    conn.commit()

    # entity_vec_map for VSS metadata
    conn.execute("CREATE TABLE entity_vec_map (rowid INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany(
        "INSERT INTO entity_vec_map VALUES (?, ?)",
        [
            (1, "Adam Smith"),
            (2, "division of labor"),
        ],
    )
    conn.commit()

    return conn


def test_pipeline_summary_counts(tmp_path: pathlib.Path) -> None:
    """Pipeline summary returns correct counts for present tables."""
    conn = _make_kg_conn(tmp_path)
    summary = get_pipeline_summary(conn)
    stages = {s["stage"]: s for s in summary["stages"]}

    assert stages[1]["count"] == 3  # chunks
    assert stages[1]["available"] is True
    assert stages[3]["count"] == 3  # entities
    assert stages[4]["count"] == 2  # relations
    assert stages[5]["count"] == 3  # entity_clusters
    assert stages[6]["count"] == 2  # nodes
    conn.close()


def test_pipeline_summary_with_hnsw(tmp_path: pathlib.Path) -> None:
    """Pipeline summary reports HNSW indexes when present."""
    conn = _make_full_kg_conn(tmp_path)
    summary = get_pipeline_summary(conn)
    stages = {s["stage"]: s for s in summary["stages"]}

    assert stages[2]["count"] == 3  # chunks_vec_nodes
    assert stages[2]["available"] is True
    assert stages[7]["count"] == 2  # node2vec_emb_nodes
    assert stages[7]["available"] is True
    conn.close()


def test_stage_chunks_detail(tmp_path: pathlib.Path) -> None:
    """Stage 1 returns chunk stats."""
    conn = _make_kg_conn(tmp_path)
    detail = get_stage_detail(conn, 1)
    assert detail["stage"] == 1
    assert detail["count"] == 3
    assert "text_lengths" in detail
    assert "samples" in detail
    assert len(detail["samples"]) <= 5
    conn.close()


def test_stage_embeddings_detail(tmp_path: pathlib.Path) -> None:
    """Stage 2 returns embedding config when HNSW index exists."""
    conn = _make_full_kg_conn(tmp_path)
    detail = get_stage_detail(conn, 2)
    assert detail["stage"] == 2
    assert detail["name"] == "Embedding"
    assert detail["count"] == 3
    assert "config" in detail
    assert "dimensions" in detail["config"]
    conn.close()


def test_stage_entities_detail(tmp_path: pathlib.Path) -> None:
    """Stage 3 returns entity breakdown."""
    conn = _make_kg_conn(tmp_path)
    detail = get_stage_detail(conn, 3)
    assert detail["count"] == 3
    assert len(detail["by_type"]) > 0
    conn.close()


def test_stage_relations_detail(tmp_path: pathlib.Path) -> None:
    """Stage 4 returns relation breakdown."""
    conn = _make_kg_conn(tmp_path)
    detail = get_stage_detail(conn, 4)
    assert detail["count"] == 2
    assert len(detail["by_type"]) > 0
    assert len(detail["samples"]) > 0
    conn.close()


def test_stage_resolution_detail(tmp_path: pathlib.Path) -> None:
    """Stage 5 returns entity resolution stats."""
    conn = _make_kg_conn(tmp_path)
    detail = get_stage_detail(conn, 5)
    assert detail["total_entities"] == 3
    assert detail["canonical_count"] == 3
    # All entities map to themselves, so merged_count=0
    assert detail["merged_count"] == 0
    conn.close()


def test_stage_resolution_with_merges(tmp_path: pathlib.Path) -> None:
    """Stage 5 reports merges when entities share a canonical name."""
    conn = _make_kg_conn(tmp_path)
    # Add a synonym that maps to an existing canonical
    conn.execute("INSERT INTO entity_clusters VALUES ('A. Smith', 'Adam Smith')")
    conn.commit()

    detail = get_stage_detail(conn, 5)
    assert detail["total_entities"] == 4
    assert detail["canonical_count"] == 3
    assert detail["merged_count"] == 1  # A. Smith != Adam Smith
    assert detail["merge_ratio"] == round(1 / 4, 3)
    # Should have at least one cluster with size > 1
    assert len(detail["largest_clusters"]) >= 1
    cluster = detail["largest_clusters"][0]
    assert cluster["canonical"] == "Adam Smith"
    assert cluster["size"] == 2
    assert set(cluster["members"]) == {"Adam Smith", "A. Smith"}
    conn.close()


def test_stage_graph_detail(tmp_path: pathlib.Path) -> None:
    """Stage 6 returns graph stats."""
    conn = _make_kg_conn(tmp_path)
    detail = get_stage_detail(conn, 6)
    assert detail["node_count"] == 2
    assert detail["edge_count"] == 1
    assert "degree_distribution" in detail
    assert len(detail["degree_distribution"]) > 0
    assert "weight_stats" in detail
    assert detail["weight_stats"]["min"] == 1.0
    assert detail["weight_stats"]["max"] == 1.0
    conn.close()


def test_stage_graph_empty(tmp_path: pathlib.Path) -> None:
    """Stage 6 returns available=False when nodes table is empty."""
    db_path = str(tmp_path / "empty_graph.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE nodes (node_id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE edges (src TEXT, dst TEXT, weight REAL)")
    conn.commit()

    detail = get_stage_detail(conn, 6)
    assert detail.get("available") is False
    conn.close()


def test_stage_node2vec_detail(tmp_path: pathlib.Path) -> None:
    """Stage 7 returns node2vec config when HNSW index exists."""
    conn = _make_full_kg_conn(tmp_path)
    detail = get_stage_detail(conn, 7)
    assert detail["stage"] == 7
    assert detail["name"] == "Node2Vec"
    assert detail["count"] == 2
    assert "config" in detail
    assert "dimensions" in detail["config"]
    conn.close()


def test_stage_missing_table(tmp_path: pathlib.Path) -> None:
    """Stage for a missing table returns available=False."""
    db_path = str(tmp_path / "empty.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    detail = get_stage_detail(conn, 1)
    assert detail.get("available") is False
    conn.close()


def test_stage_embeddings_missing(tmp_path: pathlib.Path) -> None:
    """Stage 2 returns available=False when no HNSW config."""
    conn = _make_kg_conn(tmp_path)
    detail = get_stage_detail(conn, 2)
    assert detail.get("available") is False
    conn.close()


def test_stage_node2vec_missing(tmp_path: pathlib.Path) -> None:
    """Stage 7 returns available=False when no node2vec."""
    conn = _make_kg_conn(tmp_path)
    detail = get_stage_detail(conn, 7)
    assert detail.get("available") is False
    conn.close()


def test_stage_unknown_number(tmp_path: pathlib.Path) -> None:
    """Unknown stage number returns error dict."""
    db_path = str(tmp_path / "empty.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    detail = get_stage_detail(conn, 99)
    assert "error" in detail
    assert "99" in detail["error"]
    conn.close()


def test_stage_entities_missing(tmp_path: pathlib.Path) -> None:
    """Stage 3 returns available=False when entities table is absent."""
    db_path = str(tmp_path / "no_entities.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    detail = get_stage_detail(conn, 3)
    assert detail.get("available") is False
    conn.close()


def test_stage_relations_missing(tmp_path: pathlib.Path) -> None:
    """Stage 4 returns available=False when relations table is absent."""
    db_path = str(tmp_path / "no_relations.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    detail = get_stage_detail(conn, 4)
    assert detail.get("available") is False
    conn.close()


def test_stage_resolution_missing(tmp_path: pathlib.Path) -> None:
    """Stage 5 returns available=False when entity_clusters table is absent."""
    db_path = str(tmp_path / "no_clusters.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    detail = get_stage_detail(conn, 5)
    assert detail.get("available") is False
    conn.close()


# --- _build_n2v_rowid_map ---


def test_build_n2v_rowid_map(tmp_path: pathlib.Path) -> None:
    """_build_n2v_rowid_map replicates first-seen ordering from edges."""
    conn = _make_kg_conn(tmp_path)
    mapping = _build_n2v_rowid_map(conn)
    # edges: ("Adam Smith" -> "division of labor")
    # First seen: Adam Smith (idx=0 -> rowid=1), division of labor (idx=1 -> rowid=2)
    assert mapping["Adam Smith"] == 1
    assert mapping["division of labor"] == 2
    conn.close()


def test_build_n2v_rowid_map_no_edges(tmp_path: pathlib.Path) -> None:
    """_build_n2v_rowid_map returns empty dict when edges table is missing."""
    db_path = str(tmp_path / "no_edges.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    mapping = _build_n2v_rowid_map(conn)
    assert mapping == {}
    conn.close()


# --- run_graphrag_query ---


def test_graphrag_query_full_pipeline(tmp_path: pathlib.Path) -> None:
    """run_graphrag_query exercises all 7 stages with full KG data."""
    conn = _make_full_kg_conn(tmp_path)
    result = run_graphrag_query(conn, "Adam Smith labor", k=10, max_depth=2)

    assert result["query"] == "Adam Smith labor"
    assert "stages" in result

    # Stage 1: FTS should find chunks mentioning Adam Smith and labor
    stage1 = result["stages"]["1_fts_chunks"]
    assert stage1["count"] > 0

    # Stage 2: Seed entities from matching chunks
    stage2 = result["stages"]["2_seed_entities"]
    assert "count" in stage2

    # Stage 3: BFS expansion
    stage3 = result["stages"]["3_bfs_expansion"]
    assert "seed_count" in stage3
    assert "expanded_count" in stage3

    # Stage 4: Centrality ranking
    stage4 = result["stages"]["4_centrality"]
    assert "count" in stage4

    # Stage 5: Leiden communities
    stage5 = result["stages"]["5_leiden_communities"]
    assert "community_entities" in stage5

    # Stage 6: Node2Vec structural similarity
    stage6 = result["stages"]["6_node2vec"]
    assert "structurally_similar" in stage6

    # Stage 7: Assembly
    stage7 = result["stages"]["7_assembly"]
    assert "total_entities" in stage7
    assert "total_passages" in stage7
    conn.close()


def test_graphrag_query_no_fts(tmp_path: pathlib.Path) -> None:
    """run_graphrag_query handles missing FTS table gracefully."""
    conn = _make_kg_conn(tmp_path)
    # No chunks_fts table created
    result = run_graphrag_query(conn, "test query")
    assert result["stages"]["1_fts_chunks"]["count"] == 0
    conn.close()


def test_graphrag_query_no_entity_clusters(tmp_path: pathlib.Path) -> None:
    """run_graphrag_query handles FTS results but no entity_clusters."""
    db_path = str(tmp_path / "fts_only.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    conn.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT)")
    conn.execute("INSERT INTO chunks VALUES (1, 'test content here')")
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id, text)")
    conn.execute("INSERT INTO chunks_fts VALUES (1, 'test content here')")
    conn.commit()

    result = run_graphrag_query(conn, "test")
    assert result["stages"]["1_fts_chunks"]["count"] > 0
    assert result["stages"]["2_seed_entities"]["count"] == 0
    conn.close()


# --- run_kg_search ---


def _make_kg_search_conn(tmp_path: pathlib.Path) -> sqlite3.Connection:
    """Create a connection with entities_vec, entity_vec_map, relations for KG search tests."""
    db_path = str(tmp_path / "kg_search.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)

    # Chunks + FTS
    conn.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT)")
    conn.executemany(
        "INSERT INTO chunks VALUES (?, ?)",
        [(1, "Adam Smith wrote about the division of labor."), (2, "Free trade promotes economic growth.")],
    )
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id, text)")
    conn.executemany(
        "INSERT INTO chunks_fts VALUES (?, ?)",
        [(1, "Adam Smith wrote about the division of labor."), (2, "Free trade promotes economic growth.")],
    )

    # HNSW entity embeddings (4-dim for test simplicity)
    conn.execute("""
        CREATE VIRTUAL TABLE entities_vec USING hnsw_index(
            dimensions=4, metric='cosine', m=8, ef_construction=50
        )
    """)
    vecs = [
        (1, struct.pack("4f", 1.0, 0.0, 0.0, 0.0)),
        (2, struct.pack("4f", 0.9, 0.1, 0.0, 0.0)),
        (3, struct.pack("4f", 0.0, 1.0, 0.0, 0.0)),
    ]
    for rowid, vec in vecs:
        conn.execute("INSERT INTO entities_vec (rowid, vector) VALUES (?, ?)", (rowid, vec))

    # Entity vec map
    conn.execute("CREATE TABLE entity_vec_map (rowid INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany(
        "INSERT INTO entity_vec_map VALUES (?, ?)",
        [(1, "Adam Smith"), (2, "division of labor"), (3, "free trade")],
    )

    # Relations
    conn.execute("CREATE TABLE relations (src TEXT, dst TEXT, rel_type TEXT, weight REAL)")
    conn.executemany(
        "INSERT INTO relations VALUES (?, ?, ?, ?)",
        [
            ("Adam Smith", "division of labor", "wrote_about", 1.0),
            ("Adam Smith", "free trade", "promoted", 1.0),
            ("division of labor", "free trade", "related_to", 1.0),
        ],
    )
    conn.commit()
    return conn


def test_kg_search_returns_community_data(tmp_path: pathlib.Path) -> None:
    """run_kg_search returns node_community and community_count with graph results."""
    conn = _make_kg_search_conn(tmp_path)

    # Mock embedding model to return a 4-dim vector close to "Adam Smith"
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    with patch("server.services.kg._get_embedding_model", return_value=(mock_model, "")):
        result = run_kg_search(conn, "Adam Smith")

    assert "node_community" in result
    assert "community_count" in result
    assert isinstance(result["node_community"], dict)
    assert isinstance(result["community_count"], int)

    # With 3 connected nodes and depth 3, BFS should find all of them
    assert len(result["graph_nodes"]) > 0
    # Leiden should assign communities to BFS nodes
    if result["graph_nodes"]:
        assert result["community_count"] >= 1
        for name in result["node_community"]:
            assert isinstance(result["node_community"][name], int)
    conn.close()


def test_kg_search_community_empty_without_graph(tmp_path: pathlib.Path) -> None:
    """run_kg_search returns empty community data when graph search has no results."""
    db_path = str(tmp_path / "no_graph.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)
    conn.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT)")
    conn.commit()

    # No embedding model → no graph search → no communities
    with patch("server.services.kg._get_embedding_model", side_effect=RuntimeError("no model")):
        result = run_kg_search(conn, "test query")

    assert result["node_community"] == {}
    assert result["community_count"] == 0
    conn.close()
