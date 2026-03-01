"""Filesystem scanning for available books and models.

This module has no ML dependencies — it only reads the filesystem and
SQLite databases to discover what inputs are available.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from benchmarks.demo_builder.constants import (
    BOOK_ID_TO_DATASET,
    EMBEDDING_MODELS,
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
    """Discover book IDs that have text files available.

    Scans benchmarks/texts/gutenberg_{id}.txt and returns sorted IDs.
    Chunks are now computed inline during the build (model-dependent sizing),
    so only text files are required.
    """
    text_ids: set[int] = set()
    if TEXTS_DIR.exists():
        for path in TEXTS_DIR.glob("gutenberg_*.txt"):
            parts = path.stem.split("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                text_ids.add(int(parts[1]))

    available = sorted(text_ids)
    log.info(
        "Discovered %d book(s) with text files: %s",
        len(available),
        available,
    )
    return available


def _chunk_count(book_id: int, output_folder: Path | None = None) -> int:
    """Get chunk count for a book from a built demo DB.

    Since chunks are now model-dependent and computed at build time,
    we check for an existing built DB in the output folder.
    Returns 0 if no DB is found or output_folder is None.
    """
    if output_folder is None:
        return 0
    # Check all built DBs for this book_id (could be multiple models)
    for db_path in output_folder.glob(f"{book_id}_*.db"):
        conn = sqlite3.connect(str(db_path))
        try:
            count: int = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            conn.close()
            return count
        except sqlite3.OperationalError:
            conn.close()
    return 0


def print_books() -> None:
    """Print discovered books with text size and cached vector info."""
    print("\n=== Available Books ===\n")
    print(f"  {'BOOK_ID':>7s}   {'TEXT_FILE':<28s}   {'TEXT_SIZE':>10s}   {'VECTORS'}")
    print(f"  {'-' * 7}   {'-' * 28}   {'-' * 10}   {'-' * 30}")

    # Scan texts
    text_paths: dict[int, Path] = {}
    if TEXTS_DIR.exists():
        for path in TEXTS_DIR.glob("gutenberg_*.txt"):
            parts = path.stem.split("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                text_paths[int(parts[1])] = path

    all_ids = sorted(text_paths.keys())

    for book_id in all_ids:
        text_path = text_paths[book_id]
        text_size = _fmt_size(text_path.stat().st_size)

        # Check which models have cached vectors
        dataset_name = BOOK_ID_TO_DATASET.get(book_id)
        cached_models = []
        if dataset_name:
            for mname in EMBEDDING_MODELS:
                npy_path = VECTORS_DIR / f"{mname}_{dataset_name}_docs.npy"
                if npy_path.exists():
                    cached_models.append(mname)

        vec_str = ", ".join(cached_models) if cached_models else "(compute on-the-fly)"

        print(f"  {book_id:>7d}   {text_path.name:<28s}   {text_size:>10s}   {vec_str}")

    print(f"\n  Text dir: {TEXTS_DIR}")
    print()


def print_models() -> None:
    """Print available embedding models."""
    print("\n=== Embedding Models ===\n")
    print(f"  {'NAME':<14s}   {'DIM':>5s}   {'MAX_TOK':>7s}   {'CHUNK_CH':>8s}   {'SENTENCE-TRANSFORMERS ID'}")
    print(f"  {'-' * 14}   {'-' * 5}   {'-' * 7}   {'-' * 8}   {'-' * 40}")

    for name, info in EMBEDDING_MODELS.items():
        print(
            f"  {name:<14s}   {info['dim']:>5d}   {info['max_tokens']:>7d}"
            f"   {info['chunk_chars']:>8d}   {info['st_name']}"
        )

    print()
