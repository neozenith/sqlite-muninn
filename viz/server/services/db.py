"""Database connection management — per-request connections.

Each request opens its own sqlite3.Connection, uses it, and closes it.
No global lock. SQLite WAL mode handles concurrent readers natively.
Database switching is an atomic path swap — in-flight requests finish
on the old DB, new requests use the new path.
"""

import logging
import re
import sqlite3
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

from server import config as _config

log = logging.getLogger(__name__)

# ── Shared state (protected by _state_lock) ────────────────────────
_state_lock = threading.Lock()
_active_db_path: str = _config.DB_PATH
_active_db_id: str | None = None


def sanitize_fts_query(text: str) -> str:
    """Strip punctuation that breaks FTS5 MATCH syntax.

    FTS5 treats characters like (, ), *, ", ^, :, + as query operators.
    This strips everything except word characters and whitespace,
    then collapses runs of spaces.
    """
    cleaned = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


# ── Per-request connection factory ─────────────────────────────────


def _open_connection(db_path: str, extension_path: str | None = None) -> sqlite3.Connection:
    """Open a new connection with muninn loaded and pragmas set."""
    ext = extension_path or _config.EXTENSION_PATH

    if not Path(db_path).exists():
        msg = f"Database not found: {db_path}"
        raise FileNotFoundError(msg)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA query_only = ON")

    # Load muninn extension — fail loudly if missing
    conn.enable_load_extension(True)
    conn.load_extension(ext)
    return conn


def db_session() -> Generator[sqlite3.Connection, None, None]:
    """FastAPI dependency: one connection per request, closed after response.

    No lock — SQLite WAL mode allows concurrent readers. Each request
    gets its own connection object so there is no shared mutable state.
    """
    db_path = get_active_db_path()
    conn = _open_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


# ── Active database state ──────────────────────────────────────────


def get_active_db_path() -> str:
    """Return the currently active database path (thread-safe)."""
    with _state_lock:
        return _active_db_path


def set_active_db(db_id: str, db_path: str) -> None:
    """Atomically switch the active database.

    Only updates the path/id. Each subsequent request will open a fresh
    connection to the new path. In-flight requests finish on the old DB.
    """
    global _active_db_path, _active_db_id
    with _state_lock:
        _active_db_path = db_path
        _active_db_id = db_id
    log.info("Active database set to: %s (%s)", db_id, db_path)


def get_active_db_id() -> str | None:
    """Return the currently active demo database id."""
    with _state_lock:
        return _active_db_id


def reset_state(db_path: str | None = None) -> None:
    """Reset state for testing or reinitialisation."""
    global _active_db_path, _active_db_id
    with _state_lock:
        _active_db_path = db_path or _config.DB_PATH
        _active_db_id = None


# ── Startup validation ─────────────────────────────────────────────


def validate_startup(db_path: str | None = None, extension_path: str | None = None) -> sqlite3.Connection:
    """Open a test connection at startup to validate the DB and extension load.

    Returns the connection (caller can use it for warm-up, then close it).
    Raises immediately if the DB or extension can't be loaded.
    """
    path = db_path or get_active_db_path()
    conn = _open_connection(path, extension_path)
    log.info("Startup validation OK: %s", path)
    return conn


# ── Discovery helpers (take conn as parameter) ────────────────────


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
