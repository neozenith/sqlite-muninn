"""Filesystem scanning for available books and models.

This module has no ML dependencies — it only reads the filesystem and
SQLite chunk databases to discover what inputs are available.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from benchmarks.demo_builder.constants import (
    BOOK_ID_TO_DATASET,
    EMBEDDING_MODELS,
    KG_DIR,
    TEXTS_DIR,
    VECTORS_DIR,
)

log = logging.getLogger(__name__)


def _fmt_size(size: int | float) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def discover_book_ids() -> list[int]:
    """Discover book IDs that have both text and chunks available.

    Scans benchmarks/texts/gutenberg_{id}.txt and benchmarks/kg/{id}_chunks.db,
    returns sorted IDs present in both directories.
    """
    text_ids: set[int] = set()
    if TEXTS_DIR.exists():
        for path in TEXTS_DIR.glob("gutenberg_*.txt"):
            parts = path.stem.split("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                text_ids.add(int(parts[1]))

    chunk_ids: set[int] = set()
    if KG_DIR.exists():
        for path in KG_DIR.glob("*_chunks.db"):
            stem_id = path.stem.split("_")[0]
            if stem_id.isdigit():
                chunk_ids.add(int(stem_id))

    available = sorted(text_ids & chunk_ids)
    log.info(
        "Discovered %d book(s) with text + chunks: %s",
        len(available),
        available,
    )
    return available


def _chunk_count(book_id: int) -> int:
    """Get chunk count for a book from its chunks DB."""
    chunks_db = KG_DIR / f"{book_id}_chunks.db"
    if not chunks_db.exists():
        return 0
    conn = sqlite3.connect(str(chunks_db))
    count: int = conn.execute("SELECT COUNT(*) FROM text_chunks").fetchone()[0]
    conn.close()
    return count


def print_books() -> None:
    """Print discovered books with text size and chunk count."""
    print("\n=== Available Books ===\n")
    print(f"  {'BOOK_ID':>7s}   {'TEXT_FILE':<28s}   {'TEXT_SIZE':>10s}   {'CHUNKS':>8s}   {'VECTORS'}")
    print(f"  {'-' * 7}   {'-' * 28}   {'-' * 10}   {'-' * 8}   {'-' * 30}")

    # Scan texts
    text_paths: dict[int, Path] = {}
    if TEXTS_DIR.exists():
        for path in TEXTS_DIR.glob("gutenberg_*.txt"):
            parts = path.stem.split("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                text_paths[int(parts[1])] = path

    # Scan chunks
    chunk_paths: dict[int, Path] = {}
    if KG_DIR.exists():
        for path in KG_DIR.glob("*_chunks.db"):
            stem_id = path.stem.split("_")[0]
            if stem_id.isdigit():
                chunk_paths[int(stem_id)] = path

    all_ids = sorted(text_paths.keys() | chunk_paths.keys())

    for book_id in all_ids:
        text_path = text_paths.get(book_id)
        chunk_path = chunk_paths.get(book_id)

        text_name = text_path.name if text_path else "(missing)"
        text_size = _fmt_size(text_path.stat().st_size) if text_path else ""

        # Count chunks from DB
        chunk_str = ""
        if chunk_path:
            conn = sqlite3.connect(str(chunk_path))
            chunk_str = str(conn.execute("SELECT COUNT(*) FROM text_chunks").fetchone()[0])
            conn.close()
        else:
            chunk_str = "(missing)"

        # Check which models have cached vectors
        dataset_name = BOOK_ID_TO_DATASET.get(book_id)
        cached_models = []
        if dataset_name:
            for mname in EMBEDDING_MODELS:
                npy_path = VECTORS_DIR / f"{mname}_{dataset_name}_docs.npy"
                if npy_path.exists():
                    cached_models.append(mname)

        vec_str = ", ".join(cached_models) if cached_models else "(compute on-the-fly)"

        ready = text_path and chunk_path
        marker = "" if ready else "  <-- incomplete"
        print(f"  {book_id:>7d}   {text_name:<28s}   {text_size:>10s}   {chunk_str:>8s}   {vec_str}{marker}")

    print(f"\n  Text dir:   {TEXTS_DIR}")
    print(f"  Chunks dir: {KG_DIR}")
    print()


def print_models() -> None:
    """Print available embedding models."""
    print("\n=== Embedding Models ===\n")
    print(f"  {'NAME':<14s}   {'DIM':>5s}   {'SENTENCE-TRANSFORMERS ID'}")
    print(f"  {'-' * 14}   {'-' * 5}   {'-' * 40}")

    for name, info in EMBEDDING_MODELS.items():
        print(f"  {name:<14s}   {info['dim']:>5d}   {info['st_name']}")

    print()
