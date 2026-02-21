"""Tests for prep/common.py shared helpers."""

from benchmarks.harness.prep.common import bio_to_spans, count_jsonl_lines, fmt_size, io_to_spans, write_jsonl


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


class TestBioToSpans:
    def test_single_entity(self):
        tokens = ["Alice", "went", "to", "London"]
        tags = ["B-PER", "O", "O", "B-LOC"]
        spans = bio_to_spans(tokens, tags)
        assert len(spans) == 2
        assert spans[0] == {"start": 0, "end": 5, "label": "PER", "surface": "Alice"}
        assert spans[1] == {"start": 14, "end": 20, "label": "LOC", "surface": "London"}

    def test_multi_token_entity(self):
        tokens = ["New", "York", "City"]
        tags = ["B-LOC", "I-LOC", "I-LOC"]
        spans = bio_to_spans(tokens, tags)
        assert len(spans) == 1
        assert spans[0]["surface"] == "New York City"
        assert spans[0]["label"] == "LOC"

    def test_no_entities(self):
        tokens = ["the", "cat", "sat"]
        tags = ["O", "O", "O"]
        assert bio_to_spans(tokens, tags) == []

    def test_adjacent_entities(self):
        tokens = ["Alice", "Bob"]
        tags = ["B-PER", "B-PER"]
        spans = bio_to_spans(tokens, tags)
        assert len(spans) == 2
        assert spans[0]["surface"] == "Alice"
        assert spans[1]["surface"] == "Bob"

    def test_malformed_i_without_b(self):
        """I- without preceding B- should be treated as B-."""
        tokens = ["Alice", "went"]
        tags = ["I-PER", "O"]
        spans = bio_to_spans(tokens, tags)
        assert len(spans) == 1
        assert spans[0]["surface"] == "Alice"
        assert spans[0]["label"] == "PER"

    def test_empty_input(self):
        assert bio_to_spans([], []) == []


class TestIoToSpans:
    def test_single_entity(self):
        tokens = ["Alice", "went", "to", "London"]
        tags = ["person-actor", "O", "O", "location-city"]
        spans = io_to_spans(tokens, tags)
        assert len(spans) == 2
        assert spans[0] == {"start": 0, "end": 5, "label": "person-actor", "surface": "Alice"}
        assert spans[1] == {"start": 14, "end": 20, "label": "location-city", "surface": "London"}

    def test_multi_token_entity(self):
        tokens = ["New", "York", "City"]
        tags = ["location-city", "location-city", "location-city"]
        spans = io_to_spans(tokens, tags)
        assert len(spans) == 1
        assert spans[0]["surface"] == "New York City"

    def test_no_entities(self):
        tokens = ["the", "cat", "sat"]
        tags = ["O", "O", "O"]
        assert io_to_spans(tokens, tags) == []

    def test_adjacent_different_entities(self):
        tokens = ["Alice", "London"]
        tags = ["person-actor", "location-city"]
        spans = io_to_spans(tokens, tags)
        assert len(spans) == 2
        assert spans[0]["label"] == "person-actor"
        assert spans[1]["label"] == "location-city"

    def test_empty_input(self):
        assert io_to_spans([], []) == []

    def test_trailing_entity(self):
        tokens = ["go", "to", "London"]
        tags = ["O", "O", "location-city"]
        spans = io_to_spans(tokens, tags)
        assert len(spans) == 1
        assert spans[0]["surface"] == "London"
