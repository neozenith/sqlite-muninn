"""HTTP-level tests for the FastAPI app."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.main import DEFAULT_DEMOS_DIR, app, get_demos_dir


def test_health_returns_ok() -> None:
    """The /api/health endpoint has no dependencies and must always return ok."""
    response = TestClient(app).get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_demos_dir_uses_default_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MUNINN_DEMOS_DIR", raising=False)
    assert get_demos_dir() == DEFAULT_DEMOS_DIR


def test_get_demos_dir_honours_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MUNINN_DEMOS_DIR", str(tmp_path))
    assert get_demos_dir() == tmp_path


def test_list_databases_returns_500_on_broken_manifest(tmp_path: Path) -> None:
    """If the manifest is unreadable, the API surfaces a 500 with the reason — not a generic crash."""
    app.dependency_overrides[get_demos_dir] = lambda: tmp_path  # tmp_path has no manifest.json
    try:
        response = TestClient(app).get("/api/databases")
        assert response.status_code == 500
        assert "manifest not found" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_get_database_returns_500_on_broken_manifest(tmp_path: Path) -> None:
    app.dependency_overrides[get_demos_dir] = lambda: tmp_path
    try:
        response = TestClient(app).get("/api/databases/any_id")
        assert response.status_code == 500
        assert "manifest not found" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_list_databases_returns_all_entries(client: TestClient) -> None:
    response = client.get("/api/databases")
    assert response.status_code == 200
    body = response.json()
    assert "databases" in body
    assert len(body["databases"]) == 2
    assert {db["id"] for db in body["databases"]} == {"3300_MiniLM", "39653_NomicEmbed"}


def test_list_databases_preserves_manifest_order(client: TestClient) -> None:
    """The order in manifest.json matters for UI grouping — never sort silently."""
    response = client.get("/api/databases")
    ids = [db["id"] for db in response.json()["databases"]]
    assert ids == ["3300_MiniLM", "39653_NomicEmbed"]


def test_list_databases_exposes_expected_fields(client: TestClient) -> None:
    response = client.get("/api/databases")
    entry = response.json()["databases"][0]
    assert set(entry.keys()) == {"id", "book_id", "model", "dim", "file", "size_bytes", "label"}
    assert entry["dim"] == 384
    assert entry["book_id"] == 3300


def test_get_database_returns_single_entry(client: TestClient) -> None:
    response = client.get("/api/databases/3300_MiniLM")
    assert response.status_code == 200
    assert response.json()["id"] == "3300_MiniLM"
    assert response.json()["dim"] == 384


def test_get_database_unknown_id_returns_404(client: TestClient) -> None:
    response = client.get("/api/databases/does_not_exist")
    assert response.status_code == 404
    assert "does_not_exist" in response.json()["detail"]


def test_get_database_with_slash_in_id_is_not_matched(client: TestClient) -> None:
    """FastAPI's path resolver shouldn't match `/databases/a/b` to /databases/{id}."""
    response = client.get("/api/databases/3300_MiniLM/extra")
    assert response.status_code == 404
