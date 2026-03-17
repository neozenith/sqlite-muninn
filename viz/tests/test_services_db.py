"""Tests for database service (discovery, connection)."""

import pathlib
import struct

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

import pytest

from server.services.db import (
    discover_edge_tables,
    discover_hnsw_indexes,
    get_active_db_id,
    get_active_db_path,
    reset_state,
    sanitize_fts_query,
    set_active_db,
    validate_startup,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")


def _make_conn(tmp_path: pathlib.Path) -> sqlite3.Connection:
    """Create an in-memory connection with muninn loaded and test data."""
    db_path = str(tmp_path / "disc.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)

    # HNSW index
    conn.execute("""
        CREATE VIRTUAL TABLE my_vec USING hnsw_index(
            dimensions=8, metric='l2', m=16, ef_construction=100
        )
    """)
    for i in range(5):
        vec = struct.pack("8f", *[float(i + j) for j in range(8)])
        conn.execute("INSERT INTO my_vec (rowid, vector) VALUES (?, ?)", (i + 1, vec))
    conn.commit()

    # Edge table
    conn.execute("CREATE TABLE links (source TEXT, target TEXT, score REAL)")
    conn.execute("INSERT INTO links VALUES ('a', 'b', 0.5)")
    conn.commit()

    return conn


def test_discover_hnsw_finds_index(tmp_path: pathlib.Path) -> None:
    """discover_hnsw_indexes finds HNSW virtual tables."""
    conn = _make_conn(tmp_path)
    indexes = discover_hnsw_indexes(conn)
    assert len(indexes) >= 1
    idx = next(i for i in indexes if i["name"] == "my_vec")
    assert idx["dimensions"] == 8
    assert idx["metric"] == "l2"
    assert idx["node_count"] == 5
    conn.close()


def test_discover_hnsw_metric_mapping(tmp_path: pathlib.Path) -> None:
    """discover_hnsw_indexes maps metric integers to names."""
    conn = _make_conn(tmp_path)
    # Create a cosine index
    conn.execute("""
        CREATE VIRTUAL TABLE cos_vec USING hnsw_index(
            dimensions=4, metric='cosine'
        )
    """)
    conn.commit()

    indexes = discover_hnsw_indexes(conn)
    cos = next(i for i in indexes if i["name"] == "cos_vec")
    assert cos["metric"] == "cosine"
    conn.close()


def test_discover_edge_tables(tmp_path: pathlib.Path) -> None:
    """discover_edge_tables finds tables with src/dst-like columns."""
    conn = _make_conn(tmp_path)
    tables = discover_edge_tables(conn)
    assert len(tables) >= 1
    links = next(t for t in tables if t["table_name"] == "links")
    assert links["src_col"] == "source"
    assert links["dst_col"] == "target"
    assert links["weight_col"] == "score"
    assert links["edge_count"] == 1
    conn.close()


def test_discover_edge_tables_column_priority(tmp_path: pathlib.Path) -> None:
    """discover_edge_tables prefers 'src' over 'source' when both present."""
    conn = _make_conn(tmp_path)
    conn.execute("CREATE TABLE relations (src TEXT, dst TEXT, weight REAL, source TEXT)")
    conn.execute("INSERT INTO relations VALUES ('a', 'b', 1.0, 'llm')")
    conn.commit()

    tables = discover_edge_tables(conn)
    rel = next(t for t in tables if t["table_name"] == "relations")
    assert rel["src_col"] == "src"
    assert rel["dst_col"] == "dst"
    assert rel["weight_col"] == "weight"
    conn.close()


def test_discover_edge_tables_no_weight(tmp_path: pathlib.Path) -> None:
    """discover_edge_tables handles tables without weight column."""
    conn = _make_conn(tmp_path)
    conn.execute("CREATE TABLE simple_edges (src TEXT, dst TEXT)")
    conn.execute("INSERT INTO simple_edges VALUES ('x', 'y')")
    conn.commit()

    tables = discover_edge_tables(conn)
    simple = next(t for t in tables if t["table_name"] == "simple_edges")
    assert simple["weight_col"] is None
    assert simple["edge_count"] == 1
    conn.close()


def test_discover_skips_shadow_tables(tmp_path: pathlib.Path) -> None:
    """discover_edge_tables skips HNSW shadow tables."""
    conn = _make_conn(tmp_path)
    tables = discover_edge_tables(conn)
    table_names = {t["table_name"] for t in tables}
    # Shadow tables like my_vec_config, my_vec_nodes should be excluded
    assert "my_vec_config" not in table_names
    assert "my_vec_nodes" not in table_names
    conn.close()


def test_validate_startup_file_not_found(tmp_path: pathlib.Path) -> None:
    """validate_startup raises FileNotFoundError for missing database."""
    from server import config
    from server.services import db

    original_db_path = config.DB_PATH
    config.DB_PATH = str(tmp_path / "nonexistent.db")
    db.reset_state(db_path=config.DB_PATH)

    try:
        with pytest.raises(FileNotFoundError, match="Database not found"):
            validate_startup()
    finally:
        config.DB_PATH = original_db_path
        db.reset_state()


def test_per_request_connections_are_independent(tmp_path: pathlib.Path) -> None:
    """db_session yields a new connection each time — no singleton."""
    from server import config
    from server.services import db

    # Create a valid test database
    db_path = str(tmp_path / "per_req.db")
    conn_setup = sqlite3.connect(db_path)
    conn_setup.enable_load_extension(True)
    conn_setup.load_extension(EXTENSION_PATH)
    conn_setup.close()

    original_db_path = config.DB_PATH
    config.DB_PATH = db_path
    db.reset_state(db_path=db_path)

    try:
        gen1 = db.db_session()
        conn1 = next(gen1)
        gen2 = db.db_session()
        conn2 = next(gen2)
        # Per-request: different connection objects
        assert conn1 is not conn2
        # Both work
        conn1.execute("SELECT 1").fetchone()
        conn2.execute("SELECT 1").fetchone()
        # Clean up generators
        try:
            next(gen1)
        except StopIteration:
            pass
        try:
            next(gen2)
        except StopIteration:
            pass
    finally:
        config.DB_PATH = original_db_path
        db.reset_state()


def test_validate_startup_extension_load_failure(tmp_path: pathlib.Path) -> None:
    """validate_startup raises when extension cannot be loaded."""
    from server import config
    from server.services import db

    # Create a valid test database
    db_path = str(tmp_path / "ext_fail.db")
    conn_setup = sqlite3.connect(db_path)
    conn_setup.close()

    original_db_path = config.DB_PATH
    original_ext_path = config.EXTENSION_PATH
    config.DB_PATH = db_path
    config.EXTENSION_PATH = str(tmp_path / "nonexistent_extension")
    db.reset_state(db_path=db_path)

    try:
        with pytest.raises(sqlite3.OperationalError):
            validate_startup()
    finally:
        config.DB_PATH = original_db_path
        config.EXTENSION_PATH = original_ext_path
        db.reset_state()


def test_set_active_db() -> None:
    """set_active_db updates both path and id atomically."""
    from server.services import db

    db.reset_state()
    assert get_active_db_id() is None

    set_active_db("test_db", "/tmp/test.db")
    assert get_active_db_id() == "test_db"
    assert get_active_db_path() == "/tmp/test.db"

    db.reset_state()
    assert get_active_db_id() is None


def test_discover_hnsw_skips_non_hnsw_config(tmp_path: pathlib.Path) -> None:
    """discover_hnsw_indexes skips _config tables that aren't HNSW indexes."""
    conn = _make_conn(tmp_path)
    # Create a non-HNSW _config table (no 'dimensions' key)
    conn.execute("CREATE TABLE foo_config (key TEXT, value TEXT)")
    conn.execute("INSERT INTO foo_config VALUES ('setting', 'bar')")
    conn.commit()

    indexes = discover_hnsw_indexes(conn)
    names = {i["name"] for i in indexes}
    assert "foo" not in names
    conn.close()


def test_discover_hnsw_handles_config_query_error(tmp_path: pathlib.Path) -> None:
    """discover_hnsw_indexes skips tables where config query fails."""
    conn = _make_conn(tmp_path)
    # Create a _config table with wrong schema (no key/value columns)
    conn.execute("CREATE TABLE bad_config (x INTEGER, y INTEGER)")
    conn.execute("INSERT INTO bad_config VALUES (1, 2)")
    conn.commit()

    # Should not raise -- just skip bad_config
    indexes = discover_hnsw_indexes(conn)
    names = {i["name"] for i in indexes}
    assert "bad" not in names
    conn.close()


def test_discover_edge_tables_skips_internal_tables(tmp_path: pathlib.Path) -> None:
    """discover_edge_tables skips tables starting with _ or sqlite_."""
    conn = _make_conn(tmp_path)
    # Create tables that should be skipped
    conn.execute("CREATE TABLE _internal (src TEXT, dst TEXT)")
    conn.execute("INSERT INTO _internal VALUES ('a', 'b')")
    conn.commit()

    tables = discover_edge_tables(conn)
    table_names = {t["table_name"] for t in tables}
    assert "_internal" not in table_names
    conn.close()


def test_discover_edge_tables_skips_fts_internals(tmp_path: pathlib.Path) -> None:
    """discover_edge_tables skips FTS internal tables."""
    conn = _make_conn(tmp_path)
    # FTS tables create _content, _data, etc. suffix tables
    # These should be excluded
    conn.execute("CREATE TABLE my_content (src TEXT, dst TEXT)")
    conn.execute("INSERT INTO my_content VALUES ('a', 'b')")
    conn.commit()

    tables = discover_edge_tables(conn)
    table_names = {t["table_name"] for t in tables}
    assert "my_content" not in table_names
    conn.close()


def test_discover_hnsw_missing_nodes_table(tmp_path: pathlib.Path) -> None:
    """discover_hnsw_indexes returns node_count=0 when nodes table is missing."""
    db_path = str(tmp_path / "broken_hnsw.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Manually create only a _config table without the _nodes table
    conn.execute("CREATE TABLE orphan_config (key TEXT, value TEXT)")
    conn.executemany(
        "INSERT INTO orphan_config VALUES (?, ?)",
        [
            ("dimensions", "4"),
            ("metric", "1"),
            ("m", "16"),
            ("ef_construction", "200"),
        ],
    )
    conn.commit()

    indexes = discover_hnsw_indexes(conn)
    orphan = next((i for i in indexes if i["name"] == "orphan"), None)
    assert orphan is not None
    assert orphan["node_count"] == 0
    conn.close()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hello world", "hello world"),
        ("Adam Smith's theory", "Adam Smith s theory"),
        ("labor (1776)", "labor 1776"),
        ('the "invisible hand"', "the invisible hand"),
        ("cost: $5.00", "cost 5 00"),
        ("A + B - C", "A B C"),
        ("  lots   of   spaces  ", "lots of spaces"),
        ("!!!", ""),
        ("", ""),
    ],
)
def test_sanitize_fts_query(raw: str, expected: str) -> None:
    """sanitize_fts_query strips FTS5-breaking punctuation."""
    assert sanitize_fts_query(raw) == expected
