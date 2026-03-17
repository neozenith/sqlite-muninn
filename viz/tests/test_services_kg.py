"""Tests for KG search service function."""

import pathlib
import struct

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

from unittest.mock import MagicMock, patch

import numpy as np

from server.services.kg import (
    _detect_model_from_db,
    run_kg_search,
    set_active_embedding_model,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")


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

    # HNSW chunk embeddings (4-dim for test simplicity)
    conn.execute("""
        CREATE VIRTUAL TABLE chunks_vec USING hnsw_index(
            dimensions=4, metric='cosine', m=8, ef_construction=50
        )
    """)
    for i in range(1, 3):
        vec = struct.pack("4f", float(i), float(i + 1), float(i + 2), float(i + 3))
        conn.execute("INSERT INTO chunks_vec (rowid, vector) VALUES (?, ?)", (i, vec))

    # Pre-computed UMAP for chunks
    conn.execute("""
        CREATE TABLE chunks_vec_umap (
            id INTEGER PRIMARY KEY, x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL
        )
    """)
    conn.executemany(
        "INSERT INTO chunks_vec_umap VALUES (?, ?, ?, ?, ?, ?)",
        [(1, 0.1, 0.2, 0.3, 0.4, 0.5), (2, 1.1, 1.2, 1.3, 1.4, 1.5)],
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


def test_kg_search_fts_only(tmp_path: pathlib.Path) -> None:
    """run_kg_search returns FTS results even without embedding model."""
    conn = _make_kg_search_conn(tmp_path)

    with patch("server.services.kg._get_embedding_model", side_effect=RuntimeError("no model")):
        result = run_kg_search(conn, "Adam Smith")

    assert result["query"] == "Adam Smith"
    assert len(result["fts_results"]) > 0
    assert result["vss_results"] == []
    assert result["graph_nodes"] == []
    assert result["graph_edges"] == []
    conn.close()


def test_kg_search_with_embedding(tmp_path: pathlib.Path) -> None:
    """run_kg_search returns VSS + graph results when embedding model is available."""
    conn = _make_kg_search_conn(tmp_path)

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    with patch("server.services.kg._get_embedding_model", return_value=(mock_model, "")):
        result = run_kg_search(conn, "Adam Smith")

    assert len(result["fts_results"]) > 0
    assert len(result["vss_results"]) > 0
    # VSS results should have UMAP coords
    vss0 = result["vss_results"][0]
    assert "x3d" in vss0
    assert "y3d" in vss0
    assert "z3d" in vss0
    assert "similarity" in vss0
    assert "text" in vss0

    assert len(result["graph_nodes"]) > 0
    assert len(result["graph_edges"]) > 0
    conn.close()


def test_kg_search_returns_community_data(tmp_path: pathlib.Path) -> None:
    """run_kg_search returns node_community, community_count, labels, and resolutions."""
    conn = _make_kg_search_conn(tmp_path)

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    with patch("server.services.kg._get_embedding_model", return_value=(mock_model, "")):
        result = run_kg_search(conn, "Adam Smith")

    assert "node_community" in result
    assert "community_count" in result
    assert "community_labels" in result
    assert "available_resolutions" in result
    assert isinstance(result["node_community"], dict)
    assert isinstance(result["community_count"], int)
    assert isinstance(result["community_labels"], dict)
    assert isinstance(result["available_resolutions"], list)

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

    with patch("server.services.kg._get_embedding_model", side_effect=RuntimeError("no model")):
        result = run_kg_search(conn, "test query")

    assert result["node_community"] == {}
    assert result["community_count"] == 0
    assert result["community_labels"] == {}
    assert result["available_resolutions"] == []
    conn.close()


def test_kg_search_empty_query(tmp_path: pathlib.Path) -> None:
    """run_kg_search with empty query returns empty results."""
    conn = _make_kg_search_conn(tmp_path)

    with patch("server.services.kg._get_embedding_model", side_effect=RuntimeError("no model")):
        result = run_kg_search(conn, "")

    assert result["query"] == ""
    assert result["fts_results"] == []
    conn.close()


def test_kg_search_no_chunks_fts(tmp_path: pathlib.Path) -> None:
    """run_kg_search handles missing FTS table."""
    db_path = str(tmp_path / "no_fts.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT)")
    conn.commit()

    with patch("server.services.kg._get_embedding_model", side_effect=RuntimeError("no model")):
        result = run_kg_search(conn, "test")

    assert result["fts_results"] == []
    conn.close()


def test_set_active_embedding_model() -> None:
    """set_active_embedding_model updates the active config key."""
    set_active_embedding_model("NomicEmbed")
    from server.services.kg import _active_config_key

    assert _active_config_key == "NomicEmbed"

    set_active_embedding_model("MiniLM")
    from server.services.kg import _active_config_key as key2

    assert key2 == "MiniLM"


def test_set_active_embedding_model_unknown_slug() -> None:
    """set_active_embedding_model falls back to MiniLM for unknown slugs."""
    set_active_embedding_model("unknown_model_slug")
    from server.services.kg import _active_config_key

    assert _active_config_key == "MiniLM"


def test_kg_search_graph_node_fields(tmp_path: pathlib.Path) -> None:
    """Graph nodes in KG search have required fields: name, depth, similarity, is_anchor."""
    conn = _make_kg_search_conn(tmp_path)

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    with patch("server.services.kg._get_embedding_model", return_value=(mock_model, "")):
        result = run_kg_search(conn, "Adam Smith")

    for node in result["graph_nodes"]:
        assert "name" in node
        assert "depth" in node
        assert "similarity" in node
        assert "is_anchor" in node
    conn.close()


def test_kg_search_graph_edge_fields(tmp_path: pathlib.Path) -> None:
    """Graph edges in KG search have required fields: src, rel, dst."""
    conn = _make_kg_search_conn(tmp_path)

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    with patch("server.services.kg._get_embedding_model", return_value=(mock_model, "")):
        result = run_kg_search(conn, "Adam Smith")

    for edge in result["graph_edges"]:
        assert "src" in edge
        assert "rel" in edge
        assert "dst" in edge
    conn.close()


def test_kg_search_vss_with_umap(tmp_path: pathlib.Path) -> None:
    """VSS results include 3D UMAP coordinates from pre-computed table."""
    conn = _make_kg_search_conn(tmp_path)

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)

    with patch("server.services.kg._get_embedding_model", return_value=(mock_model, "")):
        result = run_kg_search(conn, "division of labor")

    vss_with_umap = [r for r in result["vss_results"] if r["x3d"] is not None]
    assert len(vss_with_umap) > 0, "At least one VSS result should have UMAP coords"
    conn.close()


def test_detect_model_from_db_meta(tmp_path: pathlib.Path) -> None:
    """_detect_model_from_db reads embedding_model from meta table."""
    db_path = str(tmp_path / "with_meta.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO meta VALUES ('embedding_model', 'NomicEmbed')")
    conn.commit()

    assert _detect_model_from_db(conn) == "NomicEmbed"
    conn.close()


def test_detect_model_from_db_no_meta(tmp_path: pathlib.Path) -> None:
    """_detect_model_from_db falls back to MiniLM when no meta table exists."""
    db_path = str(tmp_path / "no_meta.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.commit()

    assert _detect_model_from_db(conn) == "MiniLM"
    conn.close()


def test_kg_search_precomputed_communities(tmp_path: pathlib.Path) -> None:
    """run_kg_search reads precomputed leiden_communities and community_labels."""
    conn = _make_kg_search_conn(tmp_path)

    # Add precomputed Leiden communities
    conn.execute("""
        CREATE TABLE leiden_communities (
            node TEXT NOT NULL, resolution REAL NOT NULL,
            community_id INTEGER NOT NULL, modularity REAL,
            PRIMARY KEY (node, resolution)
        )
    """)
    conn.executemany(
        "INSERT INTO leiden_communities VALUES (?, ?, ?, ?)",
        [
            ("Adam Smith", 1.0, 0, 0.45),
            ("division of labor", 1.0, 0, 0.45),
            ("free trade", 1.0, 1, 0.45),
        ],
    )

    # Add precomputed community labels
    conn.execute("""
        CREATE TABLE community_labels (
            resolution REAL NOT NULL, community_id INTEGER NOT NULL,
            label TEXT NOT NULL, member_count INTEGER NOT NULL,
            model TEXT NOT NULL, generated_at TEXT NOT NULL,
            PRIMARY KEY (resolution, community_id)
        )
    """)
    conn.executemany(
        "INSERT INTO community_labels VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1.0, 0, "Classical Economics", 2, "test", "2026-01-01"),
            (1.0, 1, "Trade Theory", 1, "test", "2026-01-01"),
        ],
    )
    conn.commit()

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    with patch("server.services.kg._get_embedding_model", return_value=(mock_model, "")):
        result = run_kg_search(conn, "Adam Smith")

    # Should read precomputed communities
    assert len(result["available_resolutions"]) >= 1
    assert 1.0 in result["available_resolutions"]

    # Should include community labels for communities present in BFS subgraph
    if result["community_count"] > 0:
        assert isinstance(result["community_labels"], dict)
        # At least one label should be present
        labels = result["community_labels"]
        assert any(v in ("Classical Economics", "Trade Theory") for v in labels.values())
    conn.close()


def test_kg_search_resolution_switching(tmp_path: pathlib.Path) -> None:
    """run_kg_search returns different communities at different resolutions."""
    conn = _make_kg_search_conn(tmp_path)

    # Add precomputed Leiden communities at two resolutions
    conn.execute("""
        CREATE TABLE leiden_communities (
            node TEXT NOT NULL, resolution REAL NOT NULL,
            community_id INTEGER NOT NULL, modularity REAL,
            PRIMARY KEY (node, resolution)
        )
    """)
    # Coarse: all in one community
    conn.executemany(
        "INSERT INTO leiden_communities VALUES (?, ?, ?, ?)",
        [
            ("Adam Smith", 0.25, 0, 0.3),
            ("division of labor", 0.25, 0, 0.3),
            ("free trade", 0.25, 0, 0.3),
        ],
    )
    # Fine: each in separate communities
    conn.executemany(
        "INSERT INTO leiden_communities VALUES (?, ?, ?, ?)",
        [
            ("Adam Smith", 3.0, 0, 0.6),
            ("division of labor", 3.0, 1, 0.6),
            ("free trade", 3.0, 2, 0.6),
        ],
    )
    conn.commit()

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    with patch("server.services.kg._get_embedding_model", return_value=(mock_model, "")):
        coarse = run_kg_search(conn, "Adam Smith", resolution=0.25)
        fine = run_kg_search(conn, "Adam Smith", resolution=3.0)

    # Coarse should have fewer communities than fine
    coarse_communities = set(coarse["node_community"].values())
    fine_communities = set(fine["node_community"].values())
    assert len(coarse_communities) <= len(fine_communities)

    # Both should report available resolutions
    assert 0.25 in coarse["available_resolutions"]
    assert 3.0 in coarse["available_resolutions"]
    conn.close()


def test_kg_search_asymmetric_prefix(tmp_path: pathlib.Path) -> None:
    """run_kg_search prepends query prefix for asymmetric models (NomicEmbed)."""
    conn = _make_kg_search_conn(tmp_path)

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    with patch("server.services.kg._get_embedding_model", return_value=(mock_model, "search_query: ")):
        run_kg_search(conn, "test query")

    # Verify the model was called with the prefixed text
    mock_model.encode.assert_called_once()
    call_args = mock_model.encode.call_args[0][0]
    assert call_args == ["search_query: test query"]
    conn.close()
