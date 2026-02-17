"""Tests for KG pipeline endpoints.

These tests use the test_db which has minimal data -- no KG pipeline tables.
The endpoints should return gracefully with available=False for missing stages.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_pipeline_summary(client: TestClient) -> None:
    """GET /api/kg/pipeline returns all 7 stages."""
    resp = client.get("/api/kg/pipeline")
    assert resp.status_code == 200
    data = resp.json()
    assert "stages" in data
    assert len(data["stages"]) == 7
    # Each stage should have required fields
    for stage in data["stages"]:
        assert "stage" in stage
        assert "name" in stage
        assert "count" in stage
        assert "available" in stage


def test_pipeline_stages_numbered(client: TestClient) -> None:
    """Pipeline stages are numbered 1-7."""
    resp = client.get("/api/kg/pipeline")
    data = resp.json()
    stage_nums = [s["stage"] for s in data["stages"]]
    assert stage_nums == [1, 2, 3, 4, 5, 6, 7]


def test_stage_detail_valid(client: TestClient) -> None:
    """GET /api/kg/stage/{n} returns stage detail for valid numbers."""
    for n in range(1, 8):
        resp = client.get(f"/api/kg/stage/{n}")
        assert resp.status_code == 200


def test_stage_detail_invalid_number(client: TestClient) -> None:
    """GET /api/kg/stage/{n} returns 400 for out-of-range numbers."""
    resp = client.get("/api/kg/stage/0")
    assert resp.status_code == 400
    resp = client.get("/api/kg/stage/8")
    assert resp.status_code == 400


def test_stage_unavailable_returns_gracefully(client: TestClient) -> None:
    """Stages with missing tables return available=False or count=0."""
    # The test_db doesn't have chunks, entities, etc.
    # Stage 2 (embeddings) should be unavailable -- no chunks_vec_config
    resp = client.get("/api/kg/stage/2")
    data = resp.json()
    assert data.get("available") is False


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


def test_stage_items_invalid_stage(client: TestClient) -> None:
    """GET /api/kg/stage/{n}/items returns 400 for unsupported stages."""
    resp = client.get("/api/kg/stage/2/items")
    assert resp.status_code == 400


def test_stage_items_missing_table(client: TestClient) -> None:
    """GET /api/kg/stage/{n}/items returns empty for missing tables."""
    # The test_db doesn't have a chunks table, but stage 1 is valid
    resp = client.get("/api/kg/stage/1/items")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["items"] == []
    assert data["page"] == 1


def _kg_test_client(tmp_path, db_name: str = "pagination.db", setup_cb=None):
    """Create a test client with a temporary KG database."""
    import pathlib

    try:
        import pysqlite3 as sqlite3
    except ImportError:
        import sqlite3

    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
    EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")

    db_path = str(tmp_path / db_name)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)

    if setup_cb:
        setup_cb(conn)

    conn.commit()
    conn.close()

    from server import config
    from server.services import db

    original_db_path = config.DB_PATH
    config.DB_PATH = db_path
    db.close_connection()
    db.reset_connection()

    from server.main import app

    test_client = TestClient(app)
    return test_client, config, db, original_db_path


def test_stage_items_pagination(tmp_path) -> None:
    """GET /api/kg/stage/{n}/items returns paginated data."""

    def setup(conn):
        conn.execute("CREATE TABLE entities (name TEXT, entity_type TEXT, chunk_id INTEGER)")
        for i in range(50):
            conn.execute("INSERT INTO entities VALUES (?, ?, ?)", (f"entity_{i}", "PERSON", i))

    test_client, config, db, original_db_path = _kg_test_client(tmp_path, setup_cb=setup)
    try:
        # Page 1
        resp = test_client.get("/api/kg/stage/3/items?page=1&page_size=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 50
        assert len(data["items"]) == 10
        assert data["page"] == 1

        # Page 2
        resp = test_client.get("/api/kg/stage/3/items?page=2&page_size=10")
        data = resp.json()
        assert len(data["items"]) == 10
        assert data["page"] == 2

        # With filter
        resp = test_client.get("/api/kg/stage/3/items?q=entity_1")
        data = resp.json()
        assert data["total"] >= 1
        assert all("entity_1" in item["name"] for item in data["items"])
    finally:
        config.DB_PATH = original_db_path
        db.close_connection()
        db.reset_connection()


def test_entities_grouped(tmp_path) -> None:
    """GET /api/kg/stage/3/entities-grouped returns grouped entities."""

    def setup(conn):
        conn.execute("CREATE TABLE entities (name TEXT, entity_type TEXT, chunk_id INTEGER)")
        # "Alice" appears in 3 chunks, "Bob" in 1
        for cid in [1, 2, 3]:
            conn.execute("INSERT INTO entities VALUES ('Alice', 'PERSON', ?)", (cid,))
        conn.execute("INSERT INTO entities VALUES ('Bob', 'PERSON', 10)")

    test_client, config, db, original_db_path = _kg_test_client(tmp_path, "grouped.db", setup)
    try:
        resp = test_client.get("/api/kg/stage/3/entities-grouped")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        # Sorted by count DESC so Alice first
        alice = data["items"][0]
        assert alice["name"] == "Alice"
        assert alice["mention_count"] == 3
        assert len(alice["chunk_ids"]) == 3

        # Filter
        resp = test_client.get("/api/kg/stage/3/entities-grouped?q=Bob")
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["name"] == "Bob"
    finally:
        config.DB_PATH = original_db_path
        db.close_connection()
        db.reset_connection()


def test_entities_grouped_missing_table(client: TestClient) -> None:
    """GET /api/kg/stage/3/entities-grouped with no entities table returns empty."""
    resp = client.get("/api/kg/stage/3/entities-grouped")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["items"] == []


def test_entities_by_chunk(tmp_path) -> None:
    """GET /api/kg/stage/3/entities-by-chunk returns per-chunk entities with full text."""

    def setup(conn):
        conn.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT)")
        conn.execute("INSERT INTO chunks VALUES (1, 'The quick brown fox jumped over the lazy dog.')")
        conn.execute("INSERT INTO chunks VALUES (2, 'Another chunk of text about markets.')")
        conn.execute("CREATE TABLE entities (name TEXT, entity_type TEXT, chunk_id INTEGER)")
        conn.execute("INSERT INTO entities VALUES ('fox', 'ANIMAL', 1)")
        conn.execute("INSERT INTO entities VALUES ('dog', 'ANIMAL', 1)")
        conn.execute("INSERT INTO entities VALUES ('markets', 'CONCEPT', 2)")

    test_client, config, db, original_db_path = _kg_test_client(tmp_path, "bychunk.db", setup)
    try:
        resp = test_client.get("/api/kg/stage/3/entities-by-chunk")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

        chunk1 = next(c for c in data["items"] if c["chunk_id"] == 1)
        assert chunk1["entity_count"] == 2
        assert "fox" in chunk1["text"]  # Full text, not truncated
        assert len(chunk1["entities"]) == 2

        chunk2 = next(c for c in data["items"] if c["chunk_id"] == 2)
        assert chunk2["entity_count"] == 1
    finally:
        config.DB_PATH = original_db_path
        db.close_connection()
        db.reset_connection()


def test_entities_by_chunk_missing_table(client: TestClient) -> None:
    """GET /api/kg/stage/3/entities-by-chunk with no entities table returns empty."""
    resp = client.get("/api/kg/stage/3/entities-by-chunk")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["items"] == []
