"""Tests for prep subcommands."""

import sqlite3
from unittest.mock import patch

import numpy as np
import pytest

from benchmarks.harness.prep.kg_chunks import KGChunksPrepTask, _chunk_text, chunks_prep_tasks, create_chunks_db, prep_kg_chunks
from benchmarks.harness.prep.kg_chunks import print_status as kg_status
from benchmarks.harness.prep.texts import (
    GutenbergTextPrepTask,
    TEXT_PREP_TASKS,
    format_book_info,
    get_cached_book_ids,
    list_cached_texts,
    prep_texts,
    print_cached_list,
)
from benchmarks.harness.prep.texts import (
    print_status as texts_status,
)
from benchmarks.harness.prep.vectors import VECTOR_PREP_TASKS, VectorPrepTask


class TestChunkText:
    def test_basic_chunking(self):
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = _chunk_text(text, window=20, overlap=5)
        assert len(chunks) > 0
        # Each chunk should have roughly 20 words
        for chunk in chunks:
            words = chunk.split()
            assert len(words) <= 20

    def test_empty_text(self):
        assert _chunk_text("") == []

    def test_overlap(self):
        text = " ".join([f"word{i}" for i in range(50)])
        chunks = _chunk_text(text, window=10, overlap=3)
        assert len(chunks) > 1
        # Overlapping words should appear in consecutive chunks
        words_0 = set(chunks[0].split())
        words_1 = set(chunks[1].split())
        assert len(words_0 & words_1) > 0  # Some overlap

    def test_small_trailing_chunks_skipped(self):
        text = " ".join([f"word{i}" for i in range(25)])
        chunks = _chunk_text(text, window=20, overlap=5)
        # The trailing chunk (5 words) should be skipped (< window // 4 = 5)
        for chunk in chunks:
            assert len(chunk.split()) >= 5


class TestCreateChunksDb:
    def test_creates_db_from_text_file(self, tmp_path):
        # Create a fake text file
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        text_file = texts_dir / "gutenberg_9999.txt"
        text_file.write_text(" ".join([f"word{i}" for i in range(500)]), encoding="utf-8")

        kg_dir = tmp_path / "kg"
        kg_dir.mkdir()

        with (
            patch("benchmarks.harness.prep.kg_chunks.TEXTS_DIR", texts_dir),
            patch("benchmarks.harness.prep.kg_chunks.KG_DIR", kg_dir),
        ):
            db_path = create_chunks_db(9999)

        assert db_path.exists()
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM text_chunks").fetchone()[0]
        conn.close()
        assert count > 0

    def test_missing_text_raises(self, tmp_path):
        with patch("benchmarks.harness.prep.kg_chunks.TEXTS_DIR", tmp_path):
            with pytest.raises(FileNotFoundError):
                create_chunks_db(99999)


class TestKgChunksStatusOnly:
    def test_status_prints_without_error(self, tmp_path, capsys):
        kg_dir = tmp_path / "kg"
        kg_dir.mkdir()
        with patch("benchmarks.harness.prep.kg_chunks.KG_DIR", kg_dir):
            kg_status()
        captured = capsys.readouterr()
        assert "KG Chunk Database Status" in captured.out
        assert "MISSING" in captured.out

    def test_status_shows_existing_db(self, tmp_path, capsys):
        kg_dir = tmp_path / "kg"
        kg_dir.mkdir()
        db_path = kg_dir / "3300_chunks.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE text_chunks(id INTEGER PRIMARY KEY, text TEXT, token_count INTEGER)")
        conn.execute("INSERT INTO text_chunks VALUES (0, 'hello world', 2)")
        conn.commit()
        conn.close()

        with patch("benchmarks.harness.prep.kg_chunks.KG_DIR", kg_dir):
            kg_status()
        captured = capsys.readouterr()
        assert "READY" in captured.out
        assert "3300" in captured.out


class TestKgChunksForce:
    def test_skip_existing_without_force(self, tmp_path, capsys):
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        text_file = texts_dir / "gutenberg_9999.txt"
        text_file.write_text(" ".join([f"word{i}" for i in range(500)]), encoding="utf-8")

        kg_dir = tmp_path / "kg"
        kg_dir.mkdir()
        # Create existing DB
        db_path = kg_dir / "9999_chunks.db"
        db_path.write_text("placeholder")

        with (
            patch("benchmarks.harness.prep.kg_chunks.TEXTS_DIR", texts_dir),
            patch("benchmarks.harness.prep.kg_chunks.KG_DIR", kg_dir),
        ):
            prep_kg_chunks(book_id=9999, force=False)

        # Should still be the placeholder (not overwritten)
        assert db_path.read_text() == "placeholder"

    def test_force_recreates_existing(self, tmp_path):
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        text_file = texts_dir / "gutenberg_9999.txt"
        text_file.write_text(" ".join([f"word{i}" for i in range(500)]), encoding="utf-8")

        kg_dir = tmp_path / "kg"
        kg_dir.mkdir()
        # Create existing DB
        db_path = kg_dir / "9999_chunks.db"
        db_path.write_text("placeholder")

        with (
            patch("benchmarks.harness.prep.kg_chunks.TEXTS_DIR", texts_dir),
            patch("benchmarks.harness.prep.kg_chunks.KG_DIR", kg_dir),
        ):
            prep_kg_chunks(book_id=9999, force=True)

        # Should be a real SQLite DB now
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM text_chunks").fetchone()[0]
        conn.close()
        assert count > 0


class TestKgChunksPrepTask:
    def test_task_id(self):
        t = KGChunksPrepTask(3300)
        assert t.task_id == "chunks:3300"

    def test_label(self):
        t = KGChunksPrepTask(3300)
        assert t.label == "Chunks for Gutenberg #3300"

    def test_outputs(self, tmp_path):
        with patch("benchmarks.harness.prep.kg_chunks.KG_DIR", tmp_path / "kg"):
            t = KGChunksPrepTask(3300)
            outputs = t.outputs()
        assert len(outputs) == 1
        assert "3300_chunks.db" in outputs[0].name

    def test_chunks_prep_tasks_includes_default(self, tmp_path):
        with patch("benchmarks.harness.prep.kg_chunks.TEXTS_DIR", tmp_path):
            tasks = chunks_prep_tasks()
        assert any(t._book_id == 3300 for t in tasks)

    def test_chunks_prep_tasks_discovers_extra(self, tmp_path):
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        (texts_dir / "gutenberg_1234.txt").write_text("hello", encoding="utf-8")
        (texts_dir / "gutenberg_3300.txt").write_text("world", encoding="utf-8")

        with patch("benchmarks.harness.prep.kg_chunks.TEXTS_DIR", texts_dir):
            tasks = chunks_prep_tasks()
        book_ids = {t._book_id for t in tasks}
        assert 3300 in book_ids
        assert 1234 in book_ids


class TestTextsListCached:
    def test_list_empty_directory(self, tmp_path, capsys):
        with patch("benchmarks.harness.prep.texts.TEXTS_DIR", tmp_path):
            print_cached_list()
        captured = capsys.readouterr()
        assert "No cached texts" in captured.out

    def test_list_with_cached_files(self, tmp_path, capsys):
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        text_file = texts_dir / "gutenberg_3300.txt"
        text_file.write_text("word " * 100, encoding="utf-8")

        with patch("benchmarks.harness.prep.texts.TEXTS_DIR", texts_dir):
            print_cached_list()
        captured = capsys.readouterr()
        assert "Cached Gutenberg Texts" in captured.out
        assert "3300" in captured.out
        assert "100" in captured.out  # word count

    def test_get_cached_book_ids(self, tmp_path):
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        (texts_dir / "gutenberg_3300.txt").write_text("hello", encoding="utf-8")
        (texts_dir / "gutenberg_1234.txt").write_text("world", encoding="utf-8")
        (texts_dir / "not_a_book.txt").write_text("ignored", encoding="utf-8")

        with patch("benchmarks.harness.prep.texts.TEXTS_DIR", texts_dir):
            ids = get_cached_book_ids()
        assert ids == {3300, 1234}

    def test_list_cached_texts_returns_tuples(self, tmp_path):
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        (texts_dir / "gutenberg_42.txt").write_text("hi", encoding="utf-8")

        with patch("benchmarks.harness.prep.texts.TEXTS_DIR", texts_dir):
            result = list_cached_texts()
        assert len(result) == 1
        book_id, path, size = result[0]
        assert book_id == 42
        assert path.name == "gutenberg_42.txt"
        assert size > 0


class TestTextsStatus:
    def test_status_shows_missing(self, tmp_path, capsys):
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()

        with patch("benchmarks.harness.prep.texts.TEXTS_DIR", texts_dir):
            texts_status()
        captured = capsys.readouterr()
        assert "Text Cache Status" in captured.out
        assert "MISSING" in captured.out

    def test_status_shows_cached(self, tmp_path, capsys):
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        (texts_dir / "gutenberg_3300.txt").write_text("hello", encoding="utf-8")

        with patch("benchmarks.harness.prep.texts.TEXTS_DIR", texts_dir):
            texts_status()
        captured = capsys.readouterr()
        assert "CACHED" in captured.out

    def test_status_shows_extra_books(self, tmp_path, capsys):
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()
        (texts_dir / "gutenberg_3300.txt").write_text("hello", encoding="utf-8")
        (texts_dir / "gutenberg_9999.txt").write_text("extra", encoding="utf-8")

        with patch("benchmarks.harness.prep.texts.TEXTS_DIR", texts_dir):
            texts_status()
        captured = capsys.readouterr()
        assert "CACHED" in captured.out
        assert "extra" in captured.out.lower()


class TestTextsForce:
    def test_prep_texts_status_only(self, tmp_path, capsys):
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()

        with patch("benchmarks.harness.prep.texts.TEXTS_DIR", texts_dir):
            prep_texts(status_only=True)
        captured = capsys.readouterr()
        assert "Text Cache Status" in captured.out

    def test_prep_texts_list_cached(self, tmp_path, capsys):
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()

        with patch("benchmarks.harness.prep.texts.TEXTS_DIR", texts_dir):
            prep_texts(list_cached=True)
        captured = capsys.readouterr()
        assert "No cached texts" in captured.out


class TestGutenbergTextPrepTask:
    def test_task_id(self):
        t = GutenbergTextPrepTask(3300)
        assert t.task_id == "text:3300"

    def test_label(self):
        t = GutenbergTextPrepTask(3300)
        assert t.label == "Gutenberg #3300"

    def test_outputs(self, tmp_path):
        with patch("benchmarks.harness.prep.texts.TEXTS_DIR", tmp_path / "texts"):
            t = GutenbergTextPrepTask(3300)
            outputs = t.outputs()
        assert len(outputs) == 1
        assert "gutenberg_3300.txt" in outputs[0].name

    def test_text_prep_tasks_default(self):
        assert len(TEXT_PREP_TASKS) >= 1
        assert any(t.task_id == "text:3300" for t in TEXT_PREP_TASKS)


class TestFormatBookInfo:
    def test_formats_book_dict(self):
        book = {
            "id": 3300,
            "title": "The Wealth of Nations",
            "authors": [{"name": "Adam Smith"}],
            "subjects": ["Economics", "Political science"],
            "languages": ["en"],
        }
        result = format_book_info(book)
        assert "3300" in result
        assert "Wealth of Nations" in result
        assert "Adam Smith" in result

    def test_handles_missing_fields(self):
        book = {"id": 999, "title": "Test Book", "authors": [], "subjects": [], "languages": []}
        result = format_book_info(book)
        assert "999" in result
        assert "Test Book" in result


class TestVectorStatus:
    def test_status_shows_missing(self, tmp_path, capsys):
        vectors_dir = tmp_path / "vectors"
        vectors_dir.mkdir()

        with patch("benchmarks.harness.prep.vectors.VECTORS_DIR", vectors_dir):
            from benchmarks.harness.prep.vectors import _print_vector_status

            _print_vector_status(["ag_news"], {"MiniLM": {"model_id": "test", "dim": 384}})
        captured = capsys.readouterr()
        assert "Vector Cache Status" in captured.out
        assert "MISSING" in captured.out

    def test_status_shows_cached(self, tmp_path, capsys):
        vectors_dir = tmp_path / "vectors"
        vectors_dir.mkdir()
        arr = np.random.rand(100, 384).astype(np.float32)
        np.save(vectors_dir / "MiniLM_ag_news.npy", arr)

        with patch("benchmarks.harness.prep.vectors.VECTORS_DIR", vectors_dir):
            from benchmarks.harness.prep.vectors import _print_vector_status

            _print_vector_status(["ag_news"], {"MiniLM": {"model_id": "test", "dim": 384}})
        captured = capsys.readouterr()
        assert "CACHED" in captured.out
        assert "100" in captured.out


class TestVectorPrepTask:
    def test_task_id(self):
        t = VectorPrepTask("MiniLM", {"model_id": "test", "dim": 384}, "ag_news")
        assert t.task_id == "vector:MiniLM:ag_news"

    def test_label(self):
        t = VectorPrepTask("MiniLM", {"model_id": "test", "dim": 384}, "ag_news")
        assert t.label == "MiniLM / ag_news"

    def test_outputs(self, tmp_path):
        with patch("benchmarks.harness.prep.vectors.VECTORS_DIR", tmp_path / "vectors"):
            t = VectorPrepTask("MiniLM", {"model_id": "test", "dim": 384}, "ag_news")
            outputs = t.outputs()
        assert len(outputs) == 1
        assert "MiniLM_ag_news.npy" in outputs[0].name

    def test_vector_prep_tasks_count(self):
        """Should have 6 tasks: 3 models × 2 datasets."""
        assert len(VECTOR_PREP_TASKS) == 6
