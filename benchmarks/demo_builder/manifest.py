"""Permutation manifest: listing, filtering, and command generation.

No ML dependencies — works with just sqlite3 and filesystem access.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from benchmarks.demo_builder.constants import (
    EMBEDDING_MODELS,
    PHASE_NAMES,
    PROJECT_ROOT,
)
from benchmarks.demo_builder.discovery import _chunk_count, discover_book_ids

log = logging.getLogger(__name__)


def _fmt_size(size: int | float) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _read_build_progress(staging_db: Path) -> tuple[int, str] | None:
    """Read the last completed phase from a staging DB's _build_progress table.

    Returns (phase_num, phase_name) or None if the table doesn't exist or is empty.
    """
    try:
        conn = sqlite3.connect(str(staging_db))
        row = conn.execute("SELECT phase, name FROM _build_progress ORDER BY phase DESC LIMIT 1").fetchone()
        conn.close()
        if row:
            return (row[0], row[1])
    except sqlite3.OperationalError:
        pass
    return None


def permutation_manifest(output_folder: Path) -> list[dict[str, Any]]:
    """Build manifest of all (book_id, model) permutations with build status.

    Each entry contains permutation_id, done/size info, and a sort_key
    ordered by estimated build cost (chunk count, then dimension).

    Status is one of:
    - "DONE": final DB exists in output_folder
    - "BUILD [N/8] phase_name": in-progress build in _build/ staging area
    - "MISS": no build exists
    """
    available_books = discover_book_ids()
    model_names = list(EMBEDDING_MODELS.keys())

    entries = []
    for book_id in available_books:
        chunks = _chunk_count(book_id)
        for model_name in model_names:
            dim = EMBEDDING_MODELS[model_name]["dim"]
            perm_id = f"{book_id}_{model_name}"
            db_path = output_folder / f"{perm_id}.db"
            done = db_path.exists()
            db_size = db_path.stat().st_size if done else None

            # Check for in-progress build in staging area
            status = "DONE" if done else "MISS"
            staging_db = output_folder / "_build" / perm_id / f"{perm_id}.db"
            if not done and staging_db.exists():
                progress = _read_build_progress(staging_db)
                if progress:
                    phase_num, phase_name = progress
                    status = f"BUILD [{phase_num}/{len(PHASE_NAMES)}] {phase_name}"
                else:
                    status = "BUILD [0/8] starting"

            entries.append(
                {
                    "permutation_id": perm_id,
                    "book_id": book_id,
                    "model_name": model_name,
                    "dim": dim,
                    "chunks": chunks,
                    "done": done,
                    "status": status,
                    "db_size": db_size,
                    "output_path": db_path,
                    "sort_key": (chunks, dim, book_id, model_name),
                    "label": f"Book {book_id} + {model_name} ({dim}d, {chunks} chunks)",
                }
            )

    return entries


def print_manifest(
    entries: list[dict[str, Any]],
    output_folder: Path,
    *,
    missing: bool = False,
    done: bool = False,
    sort: str = "size",
    limit: int | None = None,
    commands: bool = False,
    force: bool = False,
) -> None:
    """Print manifest of permutations with filtering, sorting, and command generation."""
    # Filter
    if missing:
        entries = [e for e in entries if not e["done"]]
    if done:
        entries = [e for e in entries if e["done"]]

    # Sort
    if sort == "name":
        entries = sorted(entries, key=lambda e: e["permutation_id"])
    else:  # "size" -- smallest/cheapest first
        entries = sorted(entries, key=lambda e: e["sort_key"])

    # Limit
    if limit is not None:
        entries = entries[:limit]

    # Commands mode -- print runnable self-commands
    if commands:
        force_suffix = " --force" if force else ""
        try:
            output_rel = str(output_folder.relative_to(PROJECT_ROOT))
        except ValueError:
            output_rel = str(output_folder)
        for e in entries:
            print(
                f"uv run -m benchmarks.demo_builder build"
                f" --output-folder {output_rel}"
                f" --book-id {e['book_id']}"
                f" --embedding-model {e['model_name']}"
                f"{force_suffix}"
            )
        return

    # Display mode
    total_done = sum(1 for e in entries if e["done"])
    total_building = sum(1 for e in entries if e["status"].startswith("BUILD"))

    print(f"\n=== Demo DB Manifest ({total_done}/{len(entries)}) ===\n")

    for e in entries:
        size_str = f" ({_fmt_size(e['db_size'])})" if e["db_size"] else ""
        print(f"  [{e['status']:<30s}] {e['permutation_id']:<25s} {e['label']}{size_str}")

    if total_building:
        print(f"\n  In-progress builds: {total_building}")
    print(f"\n  Output folder: {output_folder}")
    print()
