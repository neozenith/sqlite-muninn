"""Tests for the health endpoint."""

import pathlib

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    """Health endpoint returns status ok."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "db_path" in data
    assert data["extension_loaded"] is True


def test_health_reports_index_count(client: TestClient) -> None:
    """Health endpoint reports HNSW index count."""
    resp = client.get("/api/health")
    data = resp.json()
    assert "hnsw_index_count" in data
    assert data["hnsw_index_count"] >= 1  # test_vec


def test_health_reports_edge_table_count(client: TestClient) -> None:
    """Health endpoint reports edge table count."""
    resp = client.get("/api/health")
    data = resp.json()
    assert "edge_table_count" in data
    assert data["edge_table_count"] >= 1  # test_edges


def test_health_error_branch_when_db_unavailable(tmp_path: pathlib.Path) -> None:
    """Health endpoint returns 500 when the database file does not exist.

    With per-request connections, db_session() raises FileNotFoundError
    before the route handler runs, resulting in a 500 Internal Server Error.
    """
    from server import config
    from server.services import db

    original_db_path = config.DB_PATH
    bad_path = str(tmp_path / "nonexistent.db")
    config.DB_PATH = bad_path
    db.reset_state(db_path=bad_path)

    from server.main import app

    test_client = TestClient(app, raise_server_exceptions=False)
    try:
        resp = test_client.get("/api/health")
        assert resp.status_code == 500
    finally:
        config.DB_PATH = original_db_path
        db.reset_state()
