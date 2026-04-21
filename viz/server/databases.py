"""Database manifest loading — reads `frontend/public/demos/manifest.json`."""

import json
from pathlib import Path

from pydantic import BaseModel


class DatabaseInfo(BaseModel):
    """One entry in the demos manifest.

    Mirrors the shape written by `benchmarks/demo_builder/manifest.py`.
    """

    id: str
    book_id: int
    model: str
    dim: int
    file: str
    size_bytes: int
    label: str


class ManifestError(RuntimeError):
    """Raised when the manifest file is missing, unreadable, or malformed."""


def load_manifest(demos_dir: Path) -> list[DatabaseInfo]:
    """Load every database entry from `demos_dir/manifest.json`.

    Raises ManifestError if the manifest is missing or malformed so the
    caller can turn it into a 500 with a useful message rather than a
    generic unhandled-exception page.
    """
    manifest_path = demos_dir / "manifest.json"
    if not manifest_path.exists():
        raise ManifestError(f"manifest not found: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ManifestError(f"manifest is not valid JSON: {e}") from e
    if "databases" not in raw or not isinstance(raw["databases"], list):
        raise ManifestError("manifest missing 'databases' list")
    return [DatabaseInfo(**entry) for entry in raw["databases"]]


def get_database(demos_dir: Path, database_id: str) -> DatabaseInfo | None:
    """Return the manifest entry for `database_id`, or None if absent."""
    for db in load_manifest(demos_dir):
        if db.id == database_id:
            return db
    return None
