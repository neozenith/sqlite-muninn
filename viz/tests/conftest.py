"""Test fixtures for muninn-viz API tests."""

import math
import pathlib
import sqlite3
import struct

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")


def _create_test_db(tmp_path: pathlib.Path) -> str:
    """Create a test database with synthetic HNSW index + edge table."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)

    # Create a small HNSW index (4 dimensions, cosine)
    conn.execute("""
        CREATE VIRTUAL TABLE test_vec USING hnsw_index(
            dimensions=4, metric='cosine', m=8, ef_construction=50
        )
    """)

    # Insert some test vectors
    for i in range(10):
        vec = struct.pack("4f", float(i), float(i + 1), float(i + 2), float(i + 3))
        conn.execute("INSERT INTO test_vec (rowid, vector) VALUES (?, ?)", (i + 1, vec))
    conn.commit()

    # Pre-computed UMAP table (required by /vss/{name}/embeddings endpoint)
    conn.execute("""
        CREATE TABLE test_vec_umap (
            id INTEGER PRIMARY KEY,
            x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL
        )
    """)
    for i in range(10):
        angle = i * 2 * math.pi / 10
        conn.execute(
            "INSERT INTO test_vec_umap VALUES (?, ?, ?, ?, ?, ?)",
            (
                i + 1,
                round(math.cos(angle), 4),
                round(math.sin(angle), 4),
                round(math.cos(angle) * 0.5, 4),
                round(math.sin(angle) * 0.5, 4),
                round(i * 0.1, 4),
            ),
        )
    conn.commit()

    # Create edge table
    conn.execute("""
        CREATE TABLE test_edges (
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            weight REAL DEFAULT 1.0
        )
    """)
    edges = [
        ("alice", "bob", 1.0),
        ("bob", "carol", 2.0),
        ("carol", "dave", 1.5),
        ("alice", "carol", 0.5),
        ("dave", "alice", 1.0),
    ]
    conn.executemany("INSERT INTO test_edges VALUES (?, ?, ?)", edges)
    conn.commit()

    # Create nodes table for metadata
    conn.execute("""
        CREATE TABLE nodes (
            node_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            entity_type TEXT,
            mention_count INTEGER DEFAULT 0
        )
    """)
    for name in ("alice", "bob", "carol", "dave"):
        conn.execute("INSERT INTO nodes (name, mention_count) VALUES (?, ?)", (name, 3))
    conn.commit()

    conn.close()
    return db_path


@pytest.fixture
def test_db(tmp_path: pathlib.Path) -> str:
    """Create a test database and return its path."""
    return _create_test_db(tmp_path)


@pytest.fixture
def client(test_db: str) -> TestClient:
    """Create a TestClient with the server configured to use the test database."""
    from server import config
    from server.services import db

    original_db_path = config.DB_PATH
    config.DB_PATH = test_db
    # Point the per-request connection factory at the test DB
    db.reset_state(db_path=test_db)

    from server.main import app

    test_client = TestClient(app)
    yield test_client

    # Restore
    config.DB_PATH = original_db_path
    db.reset_state()
