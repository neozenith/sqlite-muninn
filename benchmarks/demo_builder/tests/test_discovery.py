"""Tests for the discovery module."""

from __future__ import annotations

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


def test_chunk_count_existing_book() -> None:
    ids = discover_book_ids()
    if ids:
        count = _chunk_count(ids[0])
        assert count > 0
