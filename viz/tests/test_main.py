"""Tests for the main application module (lifespan, app creation)."""

from fastapi.testclient import TestClient

from server.main import app


def test_lifespan_startup_and_shutdown(test_db: str) -> None:
    """TestClient context manager triggers lifespan startup and shutdown."""
    from server import config
    from server.services import db

    original_db_path = config.DB_PATH
    config.DB_PATH = test_db
    db.reset_state(db_path=test_db)

    try:
        # Using TestClient as context manager exercises lifespan
        with TestClient(app) as client:
            resp = client.get("/api/health")
            assert resp.status_code == 200
        # After exiting context, shutdown has been called
    finally:
        config.DB_PATH = original_db_path
        db.reset_state()


def test_app_has_correct_title() -> None:
    """App metadata is set correctly."""
    assert app.title == "muninn-viz"
    assert app.version == "0.1.0"
