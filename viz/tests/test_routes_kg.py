"""Tests for KG search endpoint."""

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_kg_search_query(client: TestClient) -> None:
    """POST /api/kg/query returns a result dict with search results."""
    resp = client.post("/api/kg/query", json={"query": "test query"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "test query"
    assert "fts_results" in data
    assert "vss_results" in data
    assert "graph_nodes" in data
    assert "graph_edges" in data


def test_kg_search_error_handling(client: TestClient) -> None:
    """POST /api/kg/query returns 500 when KG search fails."""
    with patch("server.routes.kg.run_kg_search", side_effect=RuntimeError("forced error")):
        resp = client.post("/api/kg/query", json={"query": "test query"})
        assert resp.status_code == 500
        assert "KG search failed" in resp.json()["detail"]


def test_kg_search_empty_query(client: TestClient) -> None:
    """POST /api/kg/query with empty query returns empty results."""
    resp = client.post("/api/kg/query", json={"query": ""})
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == ""
    assert data["fts_results"] == []


def test_kg_search_with_k_param(client: TestClient) -> None:
    """POST /api/kg/query respects k parameter."""
    resp = client.post("/api/kg/query", json={"query": "test", "k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "test"


def test_kg_search_validation(client: TestClient) -> None:
    """POST /api/kg/query requires query field."""
    resp = client.post("/api/kg/query", json={})
    assert resp.status_code == 422


def test_removed_endpoints_return_404(client: TestClient) -> None:
    """Previously existing pipeline/stage endpoints are now gone."""
    assert client.get("/api/kg/pipeline").status_code == 404
    assert client.get("/api/kg/stage/1").status_code == 404
    assert client.get("/api/kg/stage/1/items").status_code == 404
    assert client.get("/api/kg/stage/3/entities-grouped").status_code == 404
    assert client.get("/api/kg/stage/3/entities-by-chunk").status_code == 404
