"""Unit tests for server.databases — the pure manifest-loading logic."""

import json
from pathlib import Path

import pytest

from server.databases import DatabaseInfo, ManifestError, get_database, load_manifest


def _write_manifest(dir_: Path, payload: dict) -> None:
    (dir_ / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_manifest_returns_typed_entries(demos_dir: Path) -> None:
    entries = load_manifest(demos_dir)
    assert len(entries) == 3
    assert all(isinstance(e, DatabaseInfo) for e in entries)
    assert entries[0].id == "3300_MiniLM"
    assert entries[0].dim == 384
    assert entries[1].id == "39653_NomicEmbed"
    assert entries[2].id == "sessions_demo"


def test_load_manifest_accepts_entries_without_book_id(demos_dir: Path) -> None:
    """Session-log demos omit book_id. The pydantic model must default it to None."""
    entries = load_manifest(demos_dir)
    by_id = {e.id: e for e in entries}
    assert by_id["sessions_demo"].book_id is None
    assert by_id["3300_MiniLM"].book_id == 3300


def test_load_manifest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="manifest not found"):
        load_manifest(tmp_path)


def test_load_manifest_malformed_json_raises(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{not-valid-json", encoding="utf-8")
    with pytest.raises(ManifestError, match="not valid JSON"):
        load_manifest(tmp_path)


def test_load_manifest_missing_databases_key_raises(tmp_path: Path) -> None:
    _write_manifest(tmp_path, {"other_key": []})
    with pytest.raises(ManifestError, match="missing 'databases' list"):
        load_manifest(tmp_path)


def test_load_manifest_databases_not_list_raises(tmp_path: Path) -> None:
    _write_manifest(tmp_path, {"databases": "not-a-list"})
    with pytest.raises(ManifestError, match="missing 'databases' list"):
        load_manifest(tmp_path)


def test_load_manifest_invalid_entry_raises(tmp_path: Path) -> None:
    """Pydantic should reject entries missing required fields."""
    _write_manifest(tmp_path, {"databases": [{"id": "x"}]})
    with pytest.raises(Exception):  # pydantic.ValidationError is fine here
        load_manifest(tmp_path)


def test_get_database_found(demos_dir: Path) -> None:
    db = get_database(demos_dir, "3300_MiniLM")
    assert db is not None
    assert db.id == "3300_MiniLM"
    assert db.model == "MiniLM"


def test_get_database_missing_returns_none(demos_dir: Path) -> None:
    assert get_database(demos_dir, "does_not_exist") is None


def test_get_database_propagates_manifest_errors(tmp_path: Path) -> None:
    """If the manifest is broken, get_database should surface the error — not return None."""
    with pytest.raises(ManifestError):
        get_database(tmp_path, "any_id")
