"""Shared fixtures for benchmark harness tests."""

import sqlite3

import numpy as np
import pytest


@pytest.fixture
def tmp_results_dir(tmp_path):
    """Temporary results directory for test isolation."""
    results = tmp_path / "results"
    results.mkdir()
    return results


@pytest.fixture
def tmp_output_root(tmp_path):
    """Temporary output root mimicking benchmarks/refactored_outputs/."""
    for subdir in ("results", "charts", "vectors", "texts", "kg"):
        (tmp_path / subdir).mkdir()
    return tmp_path


@pytest.fixture
def tiny_vectors():
    """Small numpy array of random float32 vectors for testing."""
    rng = np.random.default_rng(42)
    return rng.random((50, 8), dtype=np.float32)


@pytest.fixture
def tiny_npy(tmp_path, tiny_vectors):
    """Tiny .npy file on disk."""
    npy_path = tmp_path / "tiny_test.npy"
    np.save(npy_path, tiny_vectors)
    return npy_path


@pytest.fixture
def mock_conn():
    """In-memory SQLite connection (no extensions loaded)."""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()
