"""Per-request SQLite connection opener for demo databases.

Each demo database is a self-contained file at `{demos_dir}/{db_id}.db`
with muninn-specific virtual tables (hnsw_index) declared in its schema.
Opening the DB without the muninn extension works for most reads but
fails the moment a virtual table is touched — so we load the extension
up front.
"""

import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_EXTENSION_PATH = Path(__file__).resolve().parent.parent.parent / "build" / "muninn"


class DatabaseConnectionError(RuntimeError):
    """Raised when a demo database can't be opened or its extension refuses to load."""


def get_extension_path() -> Path:
    """Resolve the muninn shared library path (without extension suffix).

    Python's `sqlite3.load_extension` adds `.so`/`.dylib`/`.dll` itself —
    we pass the path without a suffix and let SQLite pick the right one.
    """
    return Path(os.environ.get("MUNINN_EXTENSION_PATH", str(DEFAULT_EXTENSION_PATH)))


def resolve_db_path(demos_dir: Path, db_id: str) -> Path:
    """Map a database id to its SQLite file path."""
    return demos_dir / f"{db_id}.db"


@contextmanager
def open_demo_db(demos_dir: Path, db_id: str) -> Iterator[sqlite3.Connection]:
    """Open a demo database with the muninn extension loaded.

    Raises DatabaseConnectionError if the file is missing or the extension
    cannot be loaded.
    """
    db_path = resolve_db_path(demos_dir, db_id)
    if not db_path.exists():
        raise DatabaseConnectionError(f"database file not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        conn.enable_load_extension(True)
        conn.load_extension(str(get_extension_path()))
    except sqlite3.OperationalError as e:
        conn.close()
        raise DatabaseConnectionError(f"failed to load muninn extension from {get_extension_path()}: {e}") from e

    try:
        yield conn
    finally:
        conn.close()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Return True iff `name` is a table or view in the current database."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (name,),
    ).fetchone()
    return row is not None
