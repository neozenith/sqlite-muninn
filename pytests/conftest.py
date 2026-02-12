"""Fixtures for sqlite-muninn Python integration tests.

Compiles the extension (if needed) and provides a fresh sqlite3 connection
with the extension loaded.
"""

import pathlib
import sqlite3
import subprocess
from collections.abc import Generator

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")


@pytest.fixture(scope="session", autouse=True)
def build_extension() -> None:
    """Build the extension before running any tests."""
    result = subprocess.run(
        ["make", "all"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to build extension:\n{result.stderr}")


@pytest.fixture
def conn() -> Generator[sqlite3.Connection, None, None]:
    """Provide a fresh in-memory SQLite connection with the extension loaded."""
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXTENSION_PATH)
    yield db
    db.close()
