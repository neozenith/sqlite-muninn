"""Shared test fixtures for muninn-viz."""

import json
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.main import app, get_demos_dir

SAMPLE_DATABASES = [
    {
        "id": "3300_MiniLM",
        "book_id": 3300,
        "model": "MiniLM",
        "dim": 384,
        "file": "3300_MiniLM.db",
        "size_bytes": 52285440,
        "label": "Book 3300 + MiniLM (384d)",
    },
    {
        "id": "39653_NomicEmbed",
        "book_id": 39653,
        "model": "NomicEmbed",
        "dim": 768,
        "file": "39653_NomicEmbed.db",
        "size_bytes": 8044544,
        "label": "Book 39653 + NomicEmbed (768d)",
    },
]


@pytest.fixture
def demos_dir(tmp_path: Path) -> Path:
    """Create a tmp demos directory with a valid manifest.json."""
    manifest = {"databases": SAMPLE_DATABASES}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(demos_dir: Path) -> Generator[TestClient, None, None]:
    """TestClient with get_demos_dir overridden to point at the fixture dir."""
    app.dependency_overrides[get_demos_dir] = lambda: demos_dir
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
