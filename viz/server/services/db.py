"""Database connection and discovery services."""

import logging
import re
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

try:
    import pysqlite3 as sqlite3  # type: ignore[import-not-found]
except ImportError:
    import sqlite3

from server import config as _config

log = logging.getLogger(__name__)

# Module-level connection singleton
_connection: sqlite3.Connection | None = None

# Lock to serialize database access across FastAPI's thread pool.
# muninn TVFs (Leiden, centrality) create internal sub-queries that
# cause re-entrancy issues when multiple requests use the same connection.
_db_lock = threading.Lock()


def sanitize_fts_query(text: str) -> str:
    """Strip punctuation that breaks FTS5 MATCH syntax.

    FTS5 treats characters like (, ), *, ", ^, :, + as query operators.
    This strips everything except word characters and whitespace,
    then collapses runs of spaces.
    """
    cleaned = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def get_connection(db_path: str | None = None, extension_path: str | None = None) -> sqlite3.Connection:
    """Get or create a singleton database connection with the muninn extension loaded."""
    global _connection
    if _connection is not None:
        return _connection

    path = db_path or _config.DB_PATH
    ext = extension_path or _config.EXTENSION_PATH

    log.info("Connecting to database: %s", path)

    if not Path(path).exists():
        msg = f"Database not found: {path}"
        raise FileNotFoundError(msg)

    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Load muninn extension
    try:
        conn.enable_load_extension(True)
        conn.load_extension(ext)
        log.info("Loaded muninn extension from: %s", ext)
    except sqlite3.OperationalError:
        log.warning("Could not load muninn extension from: %s", ext)

    _connection = conn
    return conn


def close_connection() -> None:
    """Close the singleton connection."""
    global _connection
    if _connection is not None:
        _connection.close()
        _connection = None


def reset_connection() -> None:
    """Reset singleton for testing."""
    global _connection
    _connection = None


def db_session() -> Generator[sqlite3.Connection, None, None]:
    """FastAPI dependency that serializes database access.

    Acquires a threading lock before yielding the connection,
    ensuring only one route handler accesses the database at a time.
    """
    with _db_lock:
        yield get_connection()


def discover_hnsw_indexes(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Discover all HNSW virtual tables by finding their _config shadow tables.

    Each HNSW index creates shadow tables: {name}_config, {name}_nodes, {name}_edges.
    The _config table stores: dimensions, metric (0=l2, 1=cosine, 2=inner_product), m, ef_construction.
    """
    metric_names = {0: "l2", 1: "cosine", 2: "inner_product"}

    # Find all tables ending in _config that have the HNSW schema
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_config'").fetchall()

    indexes = []
    for (table_name,) in tables:
        index_name = table_name.rsplit("_config", 1)[0]
        if not index_name:
            continue

        # Verify it's an HNSW config table by checking columns
        try:
            row = conn.execute(f"SELECT key, value FROM [{index_name}_config]").fetchall()
            config = {r["key"]: r["value"] for r in row}
        except sqlite3.OperationalError:
            continue

        if "dimensions" not in config:
            continue

        # Count nodes
        try:
            node_count = conn.execute(f"SELECT count(*) FROM [{index_name}_nodes]").fetchone()[0]
        except sqlite3.OperationalError:
            node_count = 0

        metric_int = int(config.get("metric", 0))
        indexes.append(
            {
                "name": index_name,
                "dimensions": int(config["dimensions"]),
                "metric": metric_names.get(metric_int, f"unknown({metric_int})"),
                "m": int(config.get("m", 16)),
                "ef_construction": int(config.get("ef_construction", 200)),
                "node_count": node_count,
            }
        )

    return indexes


def discover_edge_tables(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Discover tables that look like graph edge tables.

    Heuristic: tables with at least two text columns that could be src/dst.
    We specifically look for tables with known patterns (src/dst, source/target).
    """
    # First, find HNSW shadow table names to exclude
    hnsw_indexes = discover_hnsw_indexes(conn)
    shadow_tables: set[str] = set()
    for idx in hnsw_indexes:
        name = idx["name"]
        shadow_tables.update({f"{name}_config", f"{name}_nodes", f"{name}_edges"})

    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()

    edge_tables = []
    # Column name patterns in priority order (most specific first).
    # Ordered lists ensure deterministic matching when a table has
    # multiple columns matching the same pattern (e.g. "src" and "source").
    src_priority = ["src", "source", "from_node", "subject"]
    dst_priority = ["dst", "dest", "destination", "target", "to_node", "object"]
    weight_patterns = {"weight", "score", "value"}

    for (table_name,) in tables:
        # Skip internal tables
        if table_name.startswith("_") or table_name.startswith("sqlite_"):
            continue
        # Skip HNSW shadow tables and FTS internals
        if table_name in shadow_tables:
            continue
        if table_name.endswith(("_content", "_data", "_idx", "_docsize", "_config")):
            continue

        try:
            columns_info = conn.execute(f"PRAGMA table_info([{table_name}])").fetchall()
            col_names = {row["name"].lower() for row in columns_info}
        except sqlite3.OperationalError:
            continue

        src_col = next((p for p in src_priority if p in col_names), None)
        dst_col = next((p for p in dst_priority if p in col_names), None)

        if src_col and dst_col:
            weight_col = next((c for c in col_names if c in weight_patterns), None)

            try:
                edge_count = conn.execute(f"SELECT count(*) FROM [{table_name}]").fetchone()[0]
            except sqlite3.OperationalError:
                edge_count = 0

            edge_tables.append(
                {
                    "table_name": table_name,
                    "src_col": src_col,
                    "dst_col": dst_col,
                    "weight_col": weight_col,
                    "edge_count": edge_count,
                }
            )

    return edge_tables
