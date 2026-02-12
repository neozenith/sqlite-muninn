"""
sqlite-muninn: HNSW vector search + graph traversal + Node2Vec for SQLite.

Zero-dependency C11 SQLite extension. Three subsystems in one .load:
HNSW approximate nearest neighbor search, graph traversal TVFs, and Node2Vec.
"""
import pathlib
import sqlite3

_PKG_DIR = pathlib.Path(__file__).parent
_REPO_ROOT = _PKG_DIR.parent

__version__ = (_REPO_ROOT / "VERSION").read_text().strip()


def loadable_path() -> str:
    """Return path to the muninn loadable extension (without file extension).

    SQLite's load_extension() automatically appends .so/.dylib/.dll.
    Searches in package directory first (wheel install), then repo root (dev / git install).
    """
    # Wheel install: binary is inside the package directory
    pkg_path = _PKG_DIR / "muninn"
    if any(_PKG_DIR.glob("muninn.*")):
        return str(pkg_path)

    # Development / git install: binary is at the repo root
    if any(_REPO_ROOT.glob("muninn.*")):
        return str(_REPO_ROOT / "muninn")

    raise FileNotFoundError(
        "muninn extension not found. Build it with: make all"
    )


def load(conn: sqlite3.Connection) -> None:
    """Load muninn into the given SQLite connection.

    The connection must have load_extension enabled:
        conn.enable_load_extension(True)
        sqlite_muninn.load(conn)
        conn.enable_load_extension(False)
    """
    conn.load_extension(loadable_path())
