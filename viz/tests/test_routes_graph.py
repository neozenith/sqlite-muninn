"""Tests for graph endpoints."""

import pathlib

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

from fastapi.testclient import TestClient

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")


def test_list_graphs(client: TestClient) -> None:
    """GET /api/graphs returns discovered edge tables."""
    resp = client.get("/api/graphs")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    edges = next(t for t in data if t["table_name"] == "test_edges")
    assert edges["src_col"] == "src"
    assert edges["dst_col"] == "dst"
    assert edges["weight_col"] == "weight"
    assert edges["edge_count"] == 5


def test_get_subgraph(client: TestClient) -> None:
    """GET /api/graph/{table}/subgraph returns nodes and edges."""
    resp = client.get("/api/graph/test_edges/subgraph?limit=100")
    assert resp.status_code == 200
    data = resp.json()
    assert data["edge_table"] == "test_edges"
    assert data["node_count"] == 4  # alice, bob, carol, dave
    assert data["edge_count"] == 5
    # All nodes should have labels
    for node in data["nodes"]:
        assert "id" in node
        assert "label" in node


def test_get_subgraph_invalid_table(client: TestClient) -> None:
    """GET /api/graph/{bad}/subgraph returns 404."""
    resp = client.get("/api/graph/nonexistent/subgraph")
    assert resp.status_code == 404


def test_get_subgraph_invalid_identifier(client: TestClient) -> None:
    """GET /api/graph/{injection}/subgraph returns 400."""
    resp = client.get("/api/graph/drop;--/subgraph")
    assert resp.status_code == 400


def test_get_subgraph_no_nodes_table(tmp_path: pathlib.Path) -> None:
    """Subgraph endpoint handles missing nodes table gracefully (except pass branch)."""
    db_path = str(tmp_path / "no_nodes.db")
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)

    # Create edge table but NO nodes table for metadata lookup
    conn.execute("CREATE TABLE test_edges (src TEXT, dst TEXT, weight REAL)")
    conn.executemany(
        "INSERT INTO test_edges VALUES (?, ?, ?)",
        [
            ("alice", "bob", 1.0),
            ("bob", "carol", 2.0),
        ],
    )
    conn.commit()
    conn.close()

    from server import config
    from server.services import db

    original_db_path = config.DB_PATH
    config.DB_PATH = db_path
    db.reset_state(db_path=db_path)

    from server.main import app

    test_client = TestClient(app)
    try:
        resp = test_client.get("/api/graph/test_edges/subgraph?limit=100")
        assert resp.status_code == 200
        data = resp.json()
        # Nodes should still be returned, just without metadata
        assert data["node_count"] == 3
        assert data["edge_count"] == 2
        # Nodes should have id and label but no mention_count or entity_type
        for node in data["nodes"]:
            assert "id" in node
            assert "mention_count" not in node
    finally:
        config.DB_PATH = original_db_path
        db.reset_state()


def test_bfs_traversal(client: TestClient) -> None:
    """GET /api/graph/{table}/bfs returns BFS results from a start node."""
    resp = client.get("/api/graph/test_edges/bfs?start=alice&max_depth=2&direction=both")
    assert resp.status_code == 200
    data = resp.json()
    assert data["start_node"] == "alice"
    assert data["max_depth"] == 2
    assert data["count"] > 0
    # All results should have node and depth
    for node in data["nodes"]:
        assert "node" in node
        assert "depth" in node
        assert node["depth"] <= 2


def test_bfs_invalid_table(client: TestClient) -> None:
    """GET /api/graph/{bad}/bfs returns 404."""
    resp = client.get("/api/graph/nonexistent/bfs?start=alice")
    assert resp.status_code == 404


def test_communities(client: TestClient) -> None:
    """GET /api/graph/{table}/communities returns Leiden results."""
    resp = client.get("/api/graph/test_edges/communities?resolution=1.0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["edge_table"] == "test_edges"
    assert data["community_count"] >= 1
    assert data["node_count"] >= 1
    # Every node should be assigned a community
    assert set(data["node_community"].keys()) == {n for nodes in data["communities"].values() for n in nodes}


def test_centrality_degree(client: TestClient) -> None:
    """GET /api/graph/{table}/centrality?measure=degree returns scores."""
    resp = client.get("/api/graph/test_edges/centrality?measure=degree")
    assert resp.status_code == 200
    data = resp.json()
    assert data["measure"] == "degree"
    assert data["count"] > 0
    # Scores should be sorted descending
    scores = [s["centrality"] for s in data["scores"]]
    assert scores == sorted(scores, reverse=True)


def test_centrality_betweenness(client: TestClient) -> None:
    """GET /api/graph/{table}/centrality?measure=betweenness returns scores."""
    resp = client.get("/api/graph/test_edges/centrality?measure=betweenness&direction=both")
    assert resp.status_code == 200
    data = resp.json()
    assert data["measure"] == "betweenness"
    assert data["count"] > 0


def test_centrality_closeness(client: TestClient) -> None:
    """GET /api/graph/{table}/centrality?measure=closeness returns scores."""
    resp = client.get("/api/graph/test_edges/centrality?measure=closeness&direction=both")
    assert resp.status_code == 200
    data = resp.json()
    assert data["measure"] == "closeness"
    assert data["count"] > 0


def test_centrality_invalid_table(client: TestClient) -> None:
    """GET /api/graph/{bad}/centrality returns 404."""
    resp = client.get("/api/graph/nonexistent/centrality?measure=degree")
    assert resp.status_code == 404


def test_subgraph_with_rel_type(tmp_path: pathlib.Path) -> None:
    """Subgraph endpoint includes rel_type when column exists."""
    db_path = str(tmp_path / "rel_type.db")
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)

    conn.execute("CREATE TABLE typed_edges (src TEXT, dst TEXT, weight REAL, rel_type TEXT)")
    conn.executemany(
        "INSERT INTO typed_edges VALUES (?, ?, ?, ?)",
        [
            ("alice", "bob", 1.0, "KNOWS"),
            ("bob", "carol", 2.0, "WORKS_WITH"),
        ],
    )
    conn.commit()
    conn.close()

    from server import config
    from server.services import db

    original_db_path = config.DB_PATH
    config.DB_PATH = db_path
    db.reset_state(db_path=db_path)

    from server.main import app

    test_client = TestClient(app)
    try:
        resp = test_client.get("/api/graph/typed_edges/subgraph?limit=100")
        assert resp.status_code == 200
        data = resp.json()
        assert data["edge_count"] == 2
        # All edges should include rel_type
        for edge in data["edges"]:
            assert "rel_type" in edge
        assert data["edges"][0]["rel_type"] in ("KNOWS", "WORKS_WITH")
    finally:
        config.DB_PATH = original_db_path
        db.reset_state()


def test_subgraph_without_rel_type(client: TestClient) -> None:
    """Subgraph endpoint omits rel_type when column doesn't exist."""
    resp = client.get("/api/graph/test_edges/subgraph?limit=100")
    assert resp.status_code == 200
    data = resp.json()
    for edge in data["edges"]:
        assert "rel_type" not in edge
