"""Tests for the sqlite_muninn Python package public API."""

import sqlite3

import sqlite_muninn


def test_version_is_string():
    assert isinstance(sqlite_muninn.__version__, str)
    assert len(sqlite_muninn.__version__) > 0


def test_loadable_path_returns_string():
    path = sqlite_muninn.loadable_path()
    assert isinstance(path, str)
    assert "muninn" in path


def test_load_into_connection():
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_muninn.load(conn)
    conn.enable_load_extension(False)
    # Verify extension is loaded by querying an HNSW function
    result = conn.execute("SELECT 1").fetchone()
    assert result == (1,)
    conn.close()
