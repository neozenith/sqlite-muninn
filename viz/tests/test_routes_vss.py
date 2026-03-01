"""Tests for VSS endpoints."""

import pathlib
import struct

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

from fastapi.testclient import TestClient

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")


def test_list_indexes(client: TestClient) -> None:
    """GET /api/indexes returns discovered HNSW indexes."""
    resp = client.get("/api/indexes")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    idx = next(i for i in data if i["name"] == "test_vec")
    assert idx["dimensions"] == 4
    assert idx["metric"] == "cosine"
    assert idx["node_count"] == 10


def test_get_embeddings(client: TestClient) -> None:
    """GET /api/vss/{name}/embeddings returns UMAP-projected points."""
    resp = client.get("/api/vss/test_vec/embeddings?dimensions=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["index"] == "test_vec"
    assert data["count"] == 10
    assert data["projected_dimensions"] == 2
    for point in data["points"]:
        assert "x" in point
        assert "y" in point
        assert "id" in point


def test_get_embeddings_3d(client: TestClient) -> None:
    """GET /api/vss/{name}/embeddings?dimensions=3 returns 3D points."""
    resp = client.get("/api/vss/test_vec/embeddings?dimensions=3")
    assert resp.status_code == 200
    data = resp.json()
    assert data["projected_dimensions"] == 3
    for point in data["points"]:
        assert "z" in point


def test_get_embeddings_invalid_index(client: TestClient) -> None:
    """GET /api/vss/{bad}/embeddings returns 404."""
    resp = client.get("/api/vss/nonexistent/embeddings")
    assert resp.status_code == 404


def test_get_embeddings_invalid_identifier(client: TestClient) -> None:
    """GET /api/vss/{injection}/embeddings returns 400."""
    resp = client.get("/api/vss/drop;--/embeddings")
    assert resp.status_code == 400


def test_search_vss(client: TestClient) -> None:
    """GET /api/vss/{name}/search returns KNN results ordered by distance."""
    resp = client.get("/api/vss/test_vec/search?query_id=1&k=5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["index"] == "test_vec"
    assert data["query_id"] == 1
    assert data["count"] <= 5
    # Results should be ordered by distance
    distances = [n["distance"] for n in data["neighbors"]]
    assert distances == sorted(distances)


def test_search_vss_invalid_node(client: TestClient) -> None:
    """GET /api/vss/{name}/search with nonexistent node returns 404."""
    resp = client.get("/api/vss/test_vec/search?query_id=99999")
    assert resp.status_code == 404


def test_search_vss_invalid_index(client: TestClient) -> None:
    """GET /api/vss/{bad}/search returns 404."""
    resp = client.get("/api/vss/nonexistent/search?query_id=1")
    assert resp.status_code == 404


def test_get_embeddings_empty_index(tmp_path: pathlib.Path) -> None:
    """GET /api/vss/{name}/embeddings returns empty result for index with no vectors."""
    # Create a database with an empty HNSW index
    db_path = str(tmp_path / "empty_vec.db")
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)
    conn.execute("""
        CREATE VIRTUAL TABLE empty_vec USING hnsw_index(
            dimensions=4, metric='cosine', m=8, ef_construction=50
        )
    """)
    # Pre-computed UMAP table (empty — no vectors inserted)
    conn.execute("""
        CREATE TABLE empty_vec_umap (
            id INTEGER PRIMARY KEY,
            x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL
        )
    """)
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
    try:
        resp = test_client.get("/api/vss/empty_vec/embeddings?dimensions=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["index"] == "empty_vec"
        assert data["count"] == 0
        assert data["points"] == []
    finally:
        config.DB_PATH = original_db_path
        db.close_connection()
        db.reset_connection()


def test_get_metadata_entities_vec(tmp_path: pathlib.Path) -> None:
    """_get_metadata returns entity name for entities_vec index."""
    db_path = str(tmp_path / "entity_vec.db")
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)
    conn.row_factory = sqlite3.Row

    # Create entities_vec HNSW index with vectors
    conn.execute("""
        CREATE VIRTUAL TABLE entities_vec USING hnsw_index(
            dimensions=4, metric='cosine', m=8, ef_construction=50
        )
    """)
    for i in range(1, 6):
        vec = struct.pack("4f", float(i), float(i + 1), float(i + 2), float(i + 3))
        conn.execute("INSERT INTO entities_vec (rowid, vector) VALUES (?, ?)", (i, vec))

    # Create entity_vec_map for metadata lookups
    conn.execute("CREATE TABLE entity_vec_map (rowid INTEGER PRIMARY KEY, name TEXT)")
    conn.executemany(
        "INSERT INTO entity_vec_map VALUES (?, ?)",
        [
            (1, "Alice"),
            (2, "Bob"),
            (3, "Carol"),
            (4, "Dave"),
            (5, "Eve"),
        ],
    )
    # Pre-computed UMAP table required by /vss/entities_vec/embeddings
    conn.execute("""
        CREATE TABLE entities_vec_umap (
            id INTEGER PRIMARY KEY,
            x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL
        )
    """)
    conn.executemany(
        "INSERT INTO entities_vec_umap VALUES (?, ?, ?, ?, ?, ?)",
        [(i, i * 0.1, i * 0.2, i * 0.05, i * 0.1, i * 0.15) for i in range(1, 6)],
    )
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
    try:
        resp = test_client.get("/api/vss/entities_vec/embeddings?dimensions=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["index"] == "entities_vec"
        assert data["count"] == 5
        # Check that metadata includes entity names
        names = {p["metadata"].get("name") for p in data["points"]}
        assert "Alice" in names
        assert "Bob" in names
        # Check metadata type
        for point in data["points"]:
            assert point["metadata"]["type"] == "entity"
    finally:
        config.DB_PATH = original_db_path
        db.close_connection()
        db.reset_connection()


def test_get_metadata_node2vec_emb(tmp_path: pathlib.Path) -> None:
    """node2vec_emb embeddings resolve node names from edges table."""
    db_path = str(tmp_path / "n2v.db")
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)
    conn.row_factory = sqlite3.Row

    # Create node2vec_emb HNSW index
    conn.execute("""
        CREATE VIRTUAL TABLE node2vec_emb USING hnsw_index(
            dimensions=4, metric='cosine', m=8, ef_construction=50
        )
    """)
    # Edge table defines first-seen ordering: Alice(1), Bob(2), Carol(3), Dave(4), Eve(5), Frank(6)
    conn.execute("CREATE TABLE edges (src TEXT, dst TEXT, weight REAL)")
    conn.execute("INSERT INTO edges VALUES ('Alice', 'Bob', 1.0)")
    conn.execute("INSERT INTO edges VALUES ('Bob', 'Carol', 1.0)")
    conn.execute("INSERT INTO edges VALUES ('Carol', 'Dave', 1.0)")
    conn.execute("INSERT INTO edges VALUES ('Dave', 'Eve', 1.0)")
    conn.execute("INSERT INTO edges VALUES ('Eve', 'Frank', 1.0)")

    # 6 vectors matching the 6 nodes (UMAP needs >= 4 samples)
    for i in range(1, 7):
        vec = struct.pack("4f", float(i * 0.1), float(i * 0.2), float(i * 0.3), float(i * 0.4))
        conn.execute("INSERT INTO node2vec_emb (rowid, vector) VALUES (?, ?)", (i, vec))
    # Pre-computed UMAP table required by /vss/node2vec_emb/embeddings
    conn.execute("""
        CREATE TABLE node2vec_emb_umap (
            id INTEGER PRIMARY KEY,
            x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL
        )
    """)
    conn.executemany(
        "INSERT INTO node2vec_emb_umap VALUES (?, ?, ?, ?, ?, ?)",
        [(i, i * 0.1, i * 0.2, i * 0.05, i * 0.1, i * 0.15) for i in range(1, 7)],
    )
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
    try:
        resp = test_client.get("/api/vss/node2vec_emb/embeddings?dimensions=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["index"] == "node2vec_emb"
        assert data["count"] == 6
        # Check metadata type is node2vec
        for point in data["points"]:
            assert point["metadata"]["type"] == "node2vec"
        # Check that node names are resolved from edges
        labels = {p["label"] for p in data["points"]}
        assert "Alice" in labels
        assert "Bob" in labels
        assert "Carol" in labels
        assert "Frank" in labels
        # Names also appear in metadata
        names = {p["metadata"].get("name") for p in data["points"]}
        assert "Alice" in names
    finally:
        config.DB_PATH = original_db_path
        db.close_connection()
        db.reset_connection()


def test_search_text_no_fts_table(client: TestClient) -> None:
    """GET /api/vss/{name}/search_text returns 400 when no FTS5 companion table."""
    resp = client.get("/api/vss/test_vec/search_text?q=hello")
    assert resp.status_code == 400
    assert "No FTS5 companion table" in resp.json()["detail"]


def test_search_text_with_fts(tmp_path: pathlib.Path) -> None:
    """GET /api/vss/{name}/search_text returns KNN results from FTS5 centroid."""
    db_path = str(tmp_path / "fts_test.db")
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)
    conn.row_factory = sqlite3.Row

    # Create chunks_vec HNSW index
    conn.execute("""
        CREATE VIRTUAL TABLE chunks_vec USING hnsw_index(
            dimensions=4, metric='cosine', m=8, ef_construction=50
        )
    """)
    for i in range(1, 6):
        vec = struct.pack("4f", float(i), float(i + 1), float(i + 2), float(i + 3))
        conn.execute("INSERT INTO chunks_vec (rowid, vector) VALUES (?, ?)", (i, vec))

    # Create chunks table + FTS5 companion
    conn.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT)")
    conn.executemany(
        "INSERT INTO chunks VALUES (?, ?)",
        [
            (1, "The quick brown fox"),
            (2, "jumped over the lazy dog"),
            (3, "hello world example text"),
            (4, "another sample chunk"),
            (5, "the fox runs fast"),
        ],
    )
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content=chunks, content_rowid=chunk_id)")
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
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
    try:
        resp = test_client.get("/api/vss/chunks_vec/search_text?q=fox&k=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["index"] == "chunks_vec"
        assert data["query"] == "fox"
        assert data["count"] <= 3
        assert data["count"] > 0
    finally:
        config.DB_PATH = original_db_path
        db.close_connection()
        db.reset_connection()


def test_search_text_no_matches(tmp_path: pathlib.Path) -> None:
    """GET /api/vss/{name}/search_text returns empty when FTS5 has no matches."""
    db_path = str(tmp_path / "fts_empty.db")
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)

    conn.execute("""
        CREATE VIRTUAL TABLE chunks_vec USING hnsw_index(
            dimensions=4, metric='cosine', m=8, ef_construction=50
        )
    """)
    vec = struct.pack("4f", 1.0, 2.0, 3.0, 4.0)
    conn.execute("INSERT INTO chunks_vec (rowid, vector) VALUES (1, ?)", (vec,))

    conn.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT)")
    conn.execute("INSERT INTO chunks VALUES (1, 'some text')")
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content=chunks, content_rowid=chunk_id)")
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
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
    try:
        resp = test_client.get("/api/vss/chunks_vec/search_text?q=nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["neighbors"] == []
    finally:
        config.DB_PATH = original_db_path
        db.close_connection()
        db.reset_connection()


def test_get_metadata_chunks_vec(tmp_path: pathlib.Path) -> None:
    """_get_metadata returns text preview for chunks_vec index."""
    db_path = str(tmp_path / "chunks_meta.db")
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)
    conn.row_factory = sqlite3.Row

    # Create chunks_vec HNSW index
    conn.execute("""
        CREATE VIRTUAL TABLE chunks_vec USING hnsw_index(
            dimensions=4, metric='cosine', m=8, ef_construction=50
        )
    """)
    for i in range(1, 6):
        vec = struct.pack("4f", float(i), float(i + 1), float(i + 2), float(i + 3))
        conn.execute("INSERT INTO chunks_vec (rowid, vector) VALUES (?, ?)", (i, vec))

    # Create chunks table for metadata
    conn.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT)")
    conn.executemany(
        "INSERT INTO chunks VALUES (?, ?)",
        [
            (1, "First chunk of text"),
            (2, "Second chunk of text"),
            (3, "Third chunk of text"),
            (4, "Fourth chunk of text"),
            (5, "Fifth chunk of text"),
        ],
    )
    # Pre-computed UMAP table required by /vss/chunks_vec/embeddings
    conn.execute("""
        CREATE TABLE chunks_vec_umap (
            id INTEGER PRIMARY KEY,
            x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL
        )
    """)
    conn.executemany(
        "INSERT INTO chunks_vec_umap VALUES (?, ?, ?, ?, ?, ?)",
        [(i, i * 0.1, i * 0.2, i * 0.05, i * 0.1, i * 0.15) for i in range(1, 6)],
    )
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
    try:
        resp = test_client.get("/api/vss/chunks_vec/embeddings?dimensions=2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["index"] == "chunks_vec"
        assert data["count"] == 5
        # Check metadata type is chunk
        for point in data["points"]:
            assert point["metadata"]["type"] == "chunk"
            assert "text" in point["metadata"]
    finally:
        config.DB_PATH = original_db_path
        db.close_connection()
        db.reset_connection()
