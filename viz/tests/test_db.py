"""Tests for server.db — connection helper + extension loading."""

import sqlite3
from pathlib import Path

import pytest

from server.db import (
    DEFAULT_EXTENSION_PATH,
    DatabaseConnectionError,
    get_extension_path,
    open_demo_db,
    resolve_db_path,
    table_exists,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEMOS_DIR = PROJECT_ROOT / "viz" / "frontend" / "public" / "demos"
SAMPLE_DB_ID = "3300_MiniLM"


def test_resolve_db_path_maps_id_to_file(tmp_path: Path) -> None:
    assert resolve_db_path(tmp_path, "foo") == tmp_path / "foo.db"


def test_get_extension_path_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MUNINN_EXTENSION_PATH", raising=False)
    assert get_extension_path() == DEFAULT_EXTENSION_PATH


def test_get_extension_path_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "my_extension"
    monkeypatch.setenv("MUNINN_EXTENSION_PATH", str(fake))
    assert get_extension_path() == fake


def test_open_demo_db_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatabaseConnectionError, match="database file not found"):
        with open_demo_db(tmp_path, "does_not_exist"):
            pass


def test_open_demo_db_missing_extension_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "empty.db").write_bytes(b"")
    monkeypatch.setenv("MUNINN_EXTENSION_PATH", str(tmp_path / "no_such_lib"))
    with pytest.raises(DatabaseConnectionError, match="failed to load muninn extension"):
        with open_demo_db(tmp_path, "empty"):
            pass


@pytest.mark.skipif(not (DEMOS_DIR / f"{SAMPLE_DB_ID}.db").exists(), reason="sample demo db not available")
def test_open_demo_db_with_real_database() -> None:
    """Sanity-check the context manager against a real demo DB."""
    with open_demo_db(DEMOS_DIR, SAMPLE_DB_ID) as conn:
        assert isinstance(conn, sqlite3.Connection)
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        assert count > 0


def test_table_exists(tmp_path: Path) -> None:
    db_path = tmp_path / "t.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE foo (x INTEGER)")
    assert table_exists(conn, "foo")
    assert not table_exists(conn, "bar")
    conn.close()
