"""
Fixtures for sqlite-vec-graph Python integration tests.

Compiles the extension (if needed) and provides a fresh sqlite3 connection
with the extension loaded.
"""
import os
import sqlite3
import subprocess
import sys

import pytest

# Extension path relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTENSION_PATH = os.path.join(PROJECT_ROOT, "vec_graph")


@pytest.fixture(scope="session", autouse=True)
def build_extension():
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
def conn():
    """Provide a fresh in-memory SQLite connection with the extension loaded."""
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXTENSION_PATH)
    yield db
    db.close()
