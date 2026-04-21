"""Manifest writer for sessions_demo — scans output folder and writes manifest.json.

Independent of demo_builder. Reads the meta table from each .db file and
produces a manifest.json for the viz frontend.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _read_meta(db_path: Path) -> dict[str, str]:
    """Read key/value pairs from a built DB's meta table."""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
        return dict(rows)
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()


def write_manifest_json(output_folder: Path) -> Path:
    """Scan output_folder for built .db files and write manifest.json.

    Reads each DB's meta table for book_id, embedding_model, embedding_dim.
    Writes atomically via .tmp rename.

    Returns the path to the written manifest.json.
    """
    databases: list[dict[str, Any]] = []

    for db_path in sorted(output_folder.glob("*.db")):
        meta = _read_meta(db_path)
        if not meta:
            log.warning("  Skipping %s (no meta table)", db_path.name)
            continue

        book_id_str = meta.get("book_id", "")
        db_id_str = meta.get("db_id", "")
        model = meta.get("embedding_model", "")
        dim_str = meta.get("embedding_dim", "0")

        if not model:
            log.warning("  Skipping %s (missing embedding_model in meta)", db_path.name)
            continue

        dim = int(dim_str)

        if book_id_str:
            book_id = int(book_id_str)
            perm_id = f"{book_id_str}_{model}"
            text_file = meta.get("text_file", f"book_{book_id}")
            book_label = text_file.replace("gutenberg_", "").replace(".txt", "")
            label = f"Book {book_label} + {model} ({dim}d)"
            databases.append(
                {
                    "id": perm_id,
                    "book_id": book_id,
                    "model": model,
                    "dim": dim,
                    "file": db_path.name,
                    "size_bytes": db_path.stat().st_size,
                    "label": label,
                }
            )
        elif db_id_str:
            source = meta.get("source", db_id_str)
            label = f"{source.replace('_', ' ').title()} ({dim}d)"
            databases.append(
                {
                    "id": db_id_str,
                    "model": model,
                    "dim": dim,
                    "file": db_path.name,
                    "size_bytes": db_path.stat().st_size,
                    "label": label,
                }
            )
        else:
            log.warning("  Skipping %s (missing book_id and db_id in meta)", db_path.name)
            continue

    manifest = {"databases": databases}

    manifest_path = output_folder / "manifest.json"
    tmp_path = output_folder / "manifest.json.tmp"
    tmp_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp_path.rename(manifest_path)

    log.info("Wrote manifest.json with %d database(s) to %s", len(databases), manifest_path)
    return manifest_path
