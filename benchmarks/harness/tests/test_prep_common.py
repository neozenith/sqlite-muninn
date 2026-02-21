"""Tests for prep/common.py shared helpers."""

from benchmarks.harness.prep.common import count_jsonl_lines, fmt_size, write_jsonl


class TestFmtSize:
    def test_bytes(self):
        assert fmt_size(500) == "500 B"

    def test_zero(self):
        assert fmt_size(0) == "0 B"

    def test_kilobytes(self):
        assert fmt_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert fmt_size(3 * 1024 * 1024) == "3.0 MB"

    def test_fractional_kb(self):
        assert fmt_size(1536) == "1.5 KB"


class TestWriteJsonl:
    def test_writes_records(self, tmp_path):
        path = tmp_path / "test.jsonl"
        records = [{"a": 1}, {"b": 2}]
        write_jsonl(path, records)

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert '"a": 1' in lines[0]
        assert '"b": 2' in lines[1]

    def test_empty_records(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        write_jsonl(path, [])
        assert path.read_text(encoding="utf-8") == "\n"

    def test_unicode(self, tmp_path):
        path = tmp_path / "unicode.jsonl"
        write_jsonl(path, [{"text": "caf\u00e9"}])
        content = path.read_text(encoding="utf-8")
        assert "caf\u00e9" in content


class TestCountJsonlLines:
    def test_counts_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
        assert count_jsonl_lines(path) == 2

    def test_ignores_blank_lines(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        path.write_text('{"a": 1}\n\n{"b": 2}\n\n', encoding="utf-8")
        assert count_jsonl_lines(path) == 2

    def test_single_line(self, tmp_path):
        path = tmp_path / "single.jsonl"
        path.write_text('{"a": 1}\n', encoding="utf-8")
        assert count_jsonl_lines(path) == 1
