"""Tests for the manifest module."""

from __future__ import annotations

from pathlib import Path

from benchmarks.demo_builder.constants import PROJECT_ROOT
from benchmarks.demo_builder.manifest import _read_build_progress, permutation_manifest


def test_manifest_returns_list() -> None:
    output_folder = PROJECT_ROOT / "wasm" / "assets"
    entries = permutation_manifest(output_folder)
    assert isinstance(entries, list)


def test_manifest_entries_have_required_keys() -> None:
    output_folder = PROJECT_ROOT / "wasm" / "assets"
    entries = permutation_manifest(output_folder)
    required = {"permutation_id", "book_id", "model_name", "dim", "chunks", "done", "status", "sort_key", "label"}
    for entry in entries:
        assert required.issubset(entry.keys()), f"Missing keys: {required - entry.keys()}"


def test_manifest_sort_key_is_tuple() -> None:
    output_folder = PROJECT_ROOT / "wasm" / "assets"
    entries = permutation_manifest(output_folder)
    for entry in entries:
        assert isinstance(entry["sort_key"], tuple)


def test_read_build_progress_nonexistent() -> None:
    result = _read_build_progress(Path("/tmp/nonexistent.db"))
    assert result is None
