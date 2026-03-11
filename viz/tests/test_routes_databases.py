"""Tests for the databases API routes."""

import json
import pathlib
import shutil

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def demos_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a temporary demos directory."""
    demos = tmp_path / "demos"
    demos.mkdir()
    return demos


@pytest.fixture
def client_with_demos(test_db: str, demos_dir: pathlib.Path) -> TestClient:
    """Create a TestClient with DEMOS_DIR pointed at a temp directory."""
    from server import config
    from server.services import db

    original_db_path = config.DB_PATH
    original_demos_dir = config.DEMOS_DIR
    config.DB_PATH = test_db
    config.DEMOS_DIR = demos_dir
    db.reset_connection()

    from server.main import app

    test_client = TestClient(app)
    yield test_client

    config.DB_PATH = original_db_path
    config.DEMOS_DIR = original_demos_dir
    db.close_connection()
    db.reset_connection()


def test_list_databases_empty(client_with_demos: TestClient) -> None:
    """GET /api/databases returns empty list when no manifest exists."""
    response = client_with_demos.get("/api/databases")
    assert response.status_code == 200
    data = response.json()
    assert data["databases"] == []
    assert data["active"] is None


def test_list_databases_with_manifest(
    client_with_demos: TestClient,
    demos_dir: pathlib.Path,
) -> None:
    """GET /api/databases returns databases from manifest.json."""
    manifest = {
        "databases": [
            {
                "id": "3300_MiniLM",
                "book_id": 3300,
                "model": "MiniLM",
                "dim": 384,
                "file": "3300_MiniLM.db",
                "size_bytes": 1000,
                "label": "Book 3300 + MiniLM (384d)",
            }
        ]
    }
    (demos_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    response = client_with_demos.get("/api/databases")
    assert response.status_code == 200
    data = response.json()
    assert len(data["databases"]) == 1
    assert data["databases"][0]["id"] == "3300_MiniLM"


def test_select_database_not_found(client_with_demos: TestClient) -> None:
    """POST /api/databases/select with invalid id returns 404."""
    response = client_with_demos.post(
        "/api/databases/select",
        json={"id": "nonexistent"},
    )
    assert response.status_code == 404


def test_select_database_success(
    client_with_demos: TestClient,
    demos_dir: pathlib.Path,
    test_db: str,
) -> None:
    """POST /api/databases/select with valid id switches the connection."""
    # Copy test DB to demos dir so reconnect() finds it
    db_name = "3300_MiniLM.db"
    shutil.copy2(test_db, str(demos_dir / db_name))

    manifest = {
        "databases": [
            {
                "id": "3300_MiniLM",
                "book_id": 3300,
                "model": "MiniLM",
                "dim": 384,
                "file": db_name,
                "size_bytes": 1000,
                "label": "Book 3300 + MiniLM (384d)",
            }
        ]
    }
    (demos_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    response = client_with_demos.post(
        "/api/databases/select",
        json={"id": "3300_MiniLM"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["active"] == "3300_MiniLM"

    # Verify GET /api/databases now shows active
    response = client_with_demos.get("/api/databases")
    assert response.json()["active"] == "3300_MiniLM"


def test_manifest_database_ids_are_unique() -> None:
    """The real manifest.json must have unique database IDs.

    Duplicate IDs cause React to generate duplicate <option key={db.id}> elements,
    which fires console.error and makes all Playwright checkpoint assertions fail.
    """
    from server import config

    manifest_path = config.DEMOS_DIR / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("manifest.json not present — demo databases not built yet")

    data: dict = json.loads(manifest_path.read_text(encoding="utf-8"))
    db_ids = [db["id"] for db in data.get("databases", [])]
    duplicates = [db_id for db_id in set(db_ids) if db_ids.count(db_id) > 1]
    assert not duplicates, (
        f"Duplicate database IDs in manifest.json: {duplicates!r}. "
        "Each database entry must have a unique id to avoid React key conflicts."
    )
