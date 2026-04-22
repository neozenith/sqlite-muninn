"""HTTP-level tests for the FastAPI app."""

from collections.abc import Generator
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
    assert len(body["databases"]) == 3
    assert {db["id"] for db in body["databases"]} == {
        "3300_MiniLM",
        "39653_NomicEmbed",
        "sessions_demo",
    }


def test_list_databases_preserves_manifest_order(client: TestClient) -> None:
    """The order in manifest.json matters for UI grouping — never sort silently."""
    response = client.get("/api/databases")
    ids = [db["id"] for db in response.json()["databases"]]
    assert ids == ["3300_MiniLM", "39653_NomicEmbed", "sessions_demo"]


def test_list_databases_exposes_expected_fields(client: TestClient) -> None:
    response = client.get("/api/databases")
    entry = response.json()["databases"][0]
    assert set(entry.keys()) == {"id", "book_id", "model", "dim", "file", "size_bytes", "label"}
    assert entry["dim"] == 384
    assert entry["book_id"] == 3300


def test_list_databases_session_entry_has_null_book_id(client: TestClient) -> None:
    """Session-log demos omit book_id in the manifest; the API must surface it as null."""
    response = client.get("/api/databases")
    entries = {db["id"]: db for db in response.json()["databases"]}
    assert entries["sessions_demo"]["book_id"] is None
    assert entries["3300_MiniLM"]["book_id"] == 3300


def test_get_database_session_entry(client: TestClient) -> None:
    response = client.get("/api/databases/sessions_demo")
    assert response.status_code == 200
    assert response.json()["book_id"] is None
    assert response.json()["id"] == "sessions_demo"


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


# ── Tables discovery, embed, kg (real demo DB integration) ──────────────


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEMOS_DIR = PROJECT_ROOT / "viz" / "frontend" / "public" / "demos"
SAMPLE_DB_ID = "3300_MiniLM"

HAS_DEMO = (DEMOS_DIR / f"{SAMPLE_DB_ID}.db").exists()


@pytest.fixture
def real_client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """TestClient pointed at the real demos directory (needs the muninn extension)."""
    monkeypatch.setenv("MUNINN_DEMOS_DIR", str(DEMOS_DIR))
    app.dependency_overrides[get_demos_dir] = lambda: DEMOS_DIR
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_tables_discovery(real_client: TestClient) -> None:
    response = real_client.get(f"/api/databases/{SAMPLE_DB_ID}/tables")
    assert response.status_code == 200
    body = response.json()
    assert body["database_id"] == SAMPLE_DB_ID
    assert set(body["embed_tables"]) == {"chunks", "entities"}
    assert set(body["kg_tables"]) == {"base", "er"}
    assert 0.25 in body["resolutions"]


def test_tables_discovery_missing_db_is_404(real_client: TestClient) -> None:
    response = real_client.get("/api/databases/not_a_real_db/tables")
    assert response.status_code == 404


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_embed_endpoint_chunks(real_client: TestClient) -> None:
    response = real_client.get(f"/api/databases/{SAMPLE_DB_ID}/embed/chunks")
    assert response.status_code == 200
    body = response.json()
    assert body["table_id"] == "chunks"
    assert body["count"] > 0
    first = body["points"][0]
    assert set(first.keys()) >= {"id", "x", "y", "z", "label", "category"}


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_embed_endpoint_entities(real_client: TestClient) -> None:
    response = real_client.get(f"/api/databases/{SAMPLE_DB_ID}/embed/entities")
    assert response.status_code == 200
    assert response.json()["count"] > 0


def test_embed_endpoint_invalid_table_400(real_client: TestClient) -> None:
    response = real_client.get(f"/api/databases/{SAMPLE_DB_ID}/embed/bogus")
    assert response.status_code == 400


def test_embed_endpoint_unknown_db_404(real_client: TestClient) -> None:
    response = real_client.get("/api/databases/not_a_real_db/embed/chunks")
    assert response.status_code == 404


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_kg_endpoint_base(real_client: TestClient) -> None:
    response = real_client.get(f"/api/databases/{SAMPLE_DB_ID}/kg/base")
    assert response.status_code == 200
    body = response.json()
    assert body["table_id"] == "base"
    assert body["resolution"] == 0.25
    assert body["node_count"] > 0
    assert body["edge_count"] > 0
    assert body["community_count"] > 0
    # New fields in the enriched payload
    assert body["seed_metric"] == "edge_betweenness"
    assert body["max_depth"] == 0
    assert "node_betweenness" in body["nodes"][0]
    assert "edge_betweenness" in body["edges"][0]


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_kg_endpoint_seed_metric_degree(real_client: TestClient) -> None:
    response = real_client.get(
        f"/api/databases/{SAMPLE_DB_ID}/kg/base?seed_metric=degree&top_n=10&max_depth=1"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["seed_metric"] == "degree"
    assert body["max_depth"] == 1
    assert body["node_count"] >= 10  # at least the seeds, plus 1-hop neighbors


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_kg_endpoint_seed_metric_node_betweenness(real_client: TestClient) -> None:
    response = real_client.get(
        f"/api/databases/{SAMPLE_DB_ID}/kg/base?seed_metric=node_betweenness&top_n=20&max_depth=0"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["seed_metric"] == "node_betweenness"


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_kg_endpoint_max_depth_expands(real_client: TestClient) -> None:
    """Increasing max_depth from 1 to 2 should never shrink the result."""
    r1 = real_client.get(
        f"/api/databases/{SAMPLE_DB_ID}/kg/base?seed_metric=degree&top_n=5&max_depth=1"
    )
    r2 = real_client.get(
        f"/api/databases/{SAMPLE_DB_ID}/kg/base?seed_metric=degree&top_n=5&max_depth=2"
    )
    assert r1.status_code == 200 and r2.status_code == 200
    assert r2.json()["node_count"] >= r1.json()["node_count"]


def test_kg_endpoint_invalid_seed_metric_400(real_client: TestClient) -> None:
    response = real_client.get(
        f"/api/databases/{SAMPLE_DB_ID}/kg/base?seed_metric=bogus"
    )
    assert response.status_code == 400
    assert "seed_metric" in response.json()["detail"]


def test_kg_endpoint_negative_max_depth_400(real_client: TestClient) -> None:
    response = real_client.get(
        f"/api/databases/{SAMPLE_DB_ID}/kg/base?max_depth=-1"
    )
    assert response.status_code == 400
    assert "max_depth" in response.json()["detail"]


def test_kg_endpoint_negative_min_degree_400(real_client: TestClient) -> None:
    response = real_client.get(
        f"/api/databases/{SAMPLE_DB_ID}/kg/base?min_degree=-2"
    )
    assert response.status_code == 400
    assert "min_degree" in response.json()["detail"]


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_kg_endpoint_echoes_min_degree(real_client: TestClient) -> None:
    response = real_client.get(
        f"/api/databases/{SAMPLE_DB_ID}/kg/base?min_degree=3"
    )
    assert response.status_code == 200
    assert response.json()["min_degree"] == 3


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_kg_endpoint_er(real_client: TestClient) -> None:
    response = real_client.get(f"/api/databases/{SAMPLE_DB_ID}/kg/er")
    assert response.status_code == 200
    body = response.json()
    assert body["table_id"] == "er"
    assert body["node_count"] > 0


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_kg_endpoint_resolution_override(real_client: TestClient) -> None:
    response = real_client.get(f"/api/databases/{SAMPLE_DB_ID}/kg/base?resolution=1.0")
    assert response.status_code == 200
    assert response.json()["resolution"] == 1.0


def test_kg_endpoint_invalid_table_400(real_client: TestClient) -> None:
    response = real_client.get(f"/api/databases/{SAMPLE_DB_ID}/kg/bogus")
    assert response.status_code == 400


def test_kg_endpoint_unknown_db_404(real_client: TestClient) -> None:
    response = real_client.get("/api/databases/not_a_real_db/kg/base")
    assert response.status_code == 404


@pytest.mark.skipif(not HAS_DEMO, reason="sample demo db not available")
def test_kg_endpoint_invalid_resolution_400(real_client: TestClient) -> None:
    response = real_client.get(f"/api/databases/{SAMPLE_DB_ID}/kg/base?resolution=42.0")
    assert response.status_code == 400
