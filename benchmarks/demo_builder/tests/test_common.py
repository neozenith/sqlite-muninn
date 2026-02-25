"""Tests for common utility functions.

Note: These tests import common.py which has numpy/spacy at top level.
They require ML dependencies to be installed.
"""

from __future__ import annotations

import struct

import numpy as np

from benchmarks.demo_builder.common import fmt_size, jaro_winkler, pack_vector


class TestPackVector:
    def test_from_list(self) -> None:
        v = [1.0, 2.0, 3.0]
        blob = pack_vector(v)
        assert len(blob) == 3 * 4  # 3 floats * 4 bytes
        unpacked = struct.unpack("3f", blob)
        assert unpacked == (1.0, 2.0, 3.0)

    def test_from_numpy(self) -> None:
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        blob = pack_vector(v)
        assert len(blob) == 3 * 4
        unpacked = struct.unpack("3f", blob)
        assert abs(unpacked[0] - 1.0) < 1e-6

    def test_roundtrip(self) -> None:
        original = np.random.randn(384).astype(np.float32)
        blob = pack_vector(original)
        restored = np.frombuffer(blob, dtype=np.float32)
        np.testing.assert_array_equal(original, restored)


class TestJaroWinkler:
    def test_identical(self) -> None:
        assert jaro_winkler("hello", "hello") == 1.0

    def test_empty(self) -> None:
        assert jaro_winkler("", "hello") == 0.0
        assert jaro_winkler("hello", "") == 0.0

    def test_similar(self) -> None:
        score = jaro_winkler("martha", "marhta")
        assert 0.9 < score < 1.0

    def test_different(self) -> None:
        score = jaro_winkler("abc", "xyz")
        assert score < 0.5

    def test_prefix_bonus(self) -> None:
        # Winkler prefix bonus should make "johnson" / "jonhson" higher
        # than strings that differ at the start
        score_prefix = jaro_winkler("johnson", "jonhson")
        score_noprefix = jaro_winkler("xohnson", "johnson")
        assert score_prefix > score_noprefix


class TestFmtSize:
    def test_bytes(self) -> None:
        assert fmt_size(100) == "100.0 B"

    def test_kilobytes(self) -> None:
        assert fmt_size(2048) == "2.0 KB"

    def test_megabytes(self) -> None:
        assert fmt_size(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self) -> None:
        assert fmt_size(2 * 1024**3) == "2.0 GB"
