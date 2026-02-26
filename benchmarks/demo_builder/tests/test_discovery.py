"""Tests for the discovery module."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from benchmarks.demo_builder.discovery import _chunk_count, _fmt_size, discover_book_ids


def test_fmt_size_bytes() -> None:
    assert _fmt_size(500) == "500.0 B"


def test_fmt_size_kb() -> None:
    assert _fmt_size(2048) == "2.0 KB"


def test_fmt_size_mb() -> None:
    assert _fmt_size(1024 * 1024 * 3) == "3.0 MB"


def test_discover_book_ids_returns_sorted_list() -> None:
    ids = discover_book_ids()
    assert isinstance(ids, list)
    assert ids == sorted(ids)


def test_chunk_count_nonexistent_book() -> None:
    assert _chunk_count(999999) == 0


def test_chunk_count_no_output_folder() -> None:
    """_chunk_count returns 0 when output_folder is None."""
    assert _chunk_count(3300, None) == 0


def test_chunk_count_empty_output_folder(tmp_path: Path) -> None:
    """_chunk_count returns 0 when output folder has no matching DBs."""
    assert _chunk_count(3300, tmp_path) == 0


def test_chunk_count_from_built_db(tmp_path: Path) -> None:
    """_chunk_count reads from a built DB's chunks table."""
    db_path = tmp_path / "3300_MiniLM.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
    conn.executemany("INSERT INTO chunks (chunk_id, text) VALUES (?, ?)", [(i, f"chunk {i}") for i in range(42)])
    conn.commit()
    conn.close()

    assert _chunk_count(3300, tmp_path) == 42
