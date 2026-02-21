"""Tests for benchmarks/harness/common.py — shared utilities."""

import numpy as np

from benchmarks.harness.common import (
    BENCHMARKS_ROOT,
    RESULTS_DIR,
    fmt_bytes,
    generate_barabasi_albert,
    generate_erdos_renyi,
    pack_vector,
    peak_rss_mb,
    platform_info,
    read_jsonl,
    unpack_vector,
    write_jsonl,
)


class TestPackVector:
    def test_numpy_array_roundtrip(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        blob = pack_vector(arr)
        assert isinstance(blob, bytes)
        assert len(blob) == 12  # 3 * 4 bytes
        result = unpack_vector(blob, 3)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_list_roundtrip(self):
        values = [0.5, -1.5, 3.14]
        blob = pack_vector(values)
        assert isinstance(blob, bytes)
        assert len(blob) == 12
        result = unpack_vector(blob, 3)
        assert len(result) == 3
        assert abs(result[0] - 0.5) < 1e-6

    def test_numpy_float64_converts_to_float32(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        blob = pack_vector(arr)
        assert len(blob) == 8  # 2 * 4 bytes (float32, not float64)

    def test_empty_vector(self):
        blob = pack_vector([])
        assert blob == b""


class TestPeakRssMb:
    def test_returns_positive_float(self):
        rss = peak_rss_mb()
        assert isinstance(rss, float)
        assert rss > 0


class TestFmtBytes:
    def test_bytes(self):
        assert fmt_bytes(500) == "500.0 B"

    def test_kilobytes(self):
        assert fmt_bytes(2048) == "2.0 KB"

    def test_megabytes(self):
        assert fmt_bytes(4 * 1024 * 1024) == "4.0 MB"

    def test_none(self):
        assert fmt_bytes(None) == "n/a"


class TestPlatformInfo:
    def test_has_required_keys(self):
        info = platform_info()
        assert "platform" in info
        assert "python_version" in info
        assert "timestamp" in info

    def test_platform_format(self):
        info = platform_info()
        # Should be like "darwin-arm64" or "linux-x86_64"
        assert "-" in info["platform"]

    def test_timestamp_is_iso(self):
        info = platform_info()
        # Should parse without error
        from datetime import datetime

        datetime.fromisoformat(info["timestamp"])


class TestJsonl:
    def test_write_and_read(self, tmp_path):
        path = tmp_path / "test.jsonl"
        write_jsonl(path, {"a": 1, "b": "hello"})
        write_jsonl(path, {"a": 2, "b": "world"})
        records = read_jsonl(path)
        assert len(records) == 2
        assert records[0]["a"] == 1
        assert records[1]["b"] == "world"

    def test_read_nonexistent(self, tmp_path):
        records = read_jsonl(tmp_path / "missing.jsonl")
        assert records == []

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "test.jsonl"
        write_jsonl(path, {"x": 1})
        assert path.exists()


class TestGenerateErdosRenyi:
    def test_basic_generation(self):
        edges, adj = generate_erdos_renyi(100, 5, seed=42)
        assert len(adj) == 100  # all nodes present
        assert len(edges) > 0  # some edges generated
        # Each edge is (src, dst, weight)
        assert len(edges[0]) == 3

    def test_edges_are_bidirectional(self):
        edges, adj = generate_erdos_renyi(20, 4, seed=42)
        edge_set = {(e[0], e[1]) for e in edges}
        for src, dst, _ in edges:
            assert (dst, src) in edge_set

    def test_weighted(self):
        edges, adj = generate_erdos_renyi(20, 4, weighted=True, seed=42)
        weights = [e[2] for e in edges]
        # Weighted edges should not all be 1.0
        assert not all(w == 1.0 for w in weights)

    def test_deterministic(self):
        e1, _ = generate_erdos_renyi(50, 5, seed=123)
        e2, _ = generate_erdos_renyi(50, 5, seed=123)
        assert e1 == e2


class TestGenerateBarabasiAlbert:
    def test_basic_generation(self):
        edges, adj = generate_barabasi_albert(100, 3, seed=42)
        assert len(adj) == 100
        assert len(edges) > 0

    def test_scale_free_property(self):
        """Barabasi-Albert should produce a power-law-like degree distribution."""
        edges, adj = generate_barabasi_albert(500, 3, seed=42)
        degrees = [len(neighbors) for neighbors in adj.values()]
        max_degree = max(degrees)
        min_degree = min(degrees)
        # Scale-free: max degree should be much larger than min
        assert max_degree > min_degree * 2

    def test_deterministic(self):
        e1, _ = generate_barabasi_albert(50, 3, seed=123)
        e2, _ = generate_barabasi_albert(50, 3, seed=123)
        assert e1 == e2


class TestPathConstants:
    def test_benchmarks_root_exists(self):
        assert BENCHMARKS_ROOT.is_dir()

    def test_results_dir_is_under_output(self):
        assert "refactored_outputs" in str(RESULTS_DIR)
