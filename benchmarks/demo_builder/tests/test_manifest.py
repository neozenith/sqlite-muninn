"""Tests for the manifest module."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from benchmarks.demo_builder.constants import DEFAULT_OUTPUT_FOLDER, PROJECT_ROOT
from benchmarks.demo_builder.manifest import _read_build_progress, permutation_manifest, write_manifest_json


def test_manifest_returns_list() -> None:
    output_folder = PROJECT_ROOT / DEFAULT_OUTPUT_FOLDER
    entries = permutation_manifest(output_folder)
    assert isinstance(entries, list)


def test_manifest_entries_have_required_keys() -> None:
    output_folder = PROJECT_ROOT / DEFAULT_OUTPUT_FOLDER
    entries = permutation_manifest(output_folder)
    required = {"permutation_id", "book_id", "model_name", "dim", "chunks", "done", "status", "sort_key", "label"}
    for entry in entries:
        assert required.issubset(entry.keys()), f"Missing keys: {required - entry.keys()}"


def test_manifest_sort_key_is_tuple() -> None:
    output_folder = PROJECT_ROOT / DEFAULT_OUTPUT_FOLDER
    entries = permutation_manifest(output_folder)
    for entry in entries:
        assert isinstance(entry["sort_key"], tuple)


def test_read_build_progress_nonexistent() -> None:
    result = _read_build_progress(Path("/tmp/nonexistent.db"))
    assert result is None


def test_write_manifest_json_empty_folder(tmp_path: Path) -> None:
    """write_manifest_json with no .db files produces empty databases list."""
    result = write_manifest_json(tmp_path)
    assert result == tmp_path / "manifest.json"
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data == {"databases": []}


def test_write_manifest_json_with_db(tmp_path: Path) -> None:
    """write_manifest_json reads meta table from built DBs."""
    db_path = tmp_path / "3300_MiniLM.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.executemany(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        [
            ("book_id", "3300"),
            ("embedding_model", "MiniLM"),
            ("embedding_dim", "384"),
            ("text_file", "gutenberg_3300.txt"),
        ],
    )
    conn.commit()
    conn.close()

    result = write_manifest_json(tmp_path)
    data = json.loads(result.read_text(encoding="utf-8"))

    assert len(data["databases"]) == 1
    db = data["databases"][0]
    assert db["id"] == "3300_MiniLM"
    assert db["book_id"] == 3300
    assert db["model"] == "MiniLM"
    assert db["dim"] == 384
    assert db["file"] == "3300_MiniLM.db"
    assert db["size_bytes"] > 0
    assert "label" in db


def test_write_manifest_json_skips_db_without_meta(tmp_path: Path) -> None:
    """DBs without a meta table are skipped."""
    db_path = tmp_path / "bad.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT)")
    conn.commit()
    conn.close()

    result = write_manifest_json(tmp_path)
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data == {"databases": []}
