"""
sqlite-muninn: HNSW vector search + graph traversal + Node2Vec for SQLite.

Zero-dependency C11 SQLite extension. Three subsystems in one .load:
HNSW approximate nearest neighbor search, graph traversal TVFs, and Node2Vec.
"""
import pathlib
import sqlite3

__version__ = (pathlib.Path(__file__).parent.parent / "VERSION").read_text().strip()


def loadable_path() -> str:
    """Return path to the muninn loadable extension (without file extension).

    SQLite's load_extension() automatically appends .so/.dylib/.dll.
    """
    return str(pathlib.Path(__file__).parent / "muninn")


def load(conn: sqlite3.Connection) -> None:
    """Load muninn into the given SQLite connection.

    The connection must have load_extension enabled:
        conn.enable_load_extension(True)
        sqlite_muninn.load(conn)
        conn.enable_load_extension(False)
    """
    conn.load_extension(loadable_path())
