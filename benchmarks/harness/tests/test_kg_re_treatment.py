"""Tests for KG relation extraction treatment."""

import json
import sqlite3
from unittest.mock import patch

from benchmarks.harness.treatments.kg_re import KGRelationExtractionTreatment


class TestKGRETreatment:
    def test_instantiates(self):
        t = KGRelationExtractionTreatment("fts5", "docred")
        assert t.category == "kg-re"
        assert t.permutation_id == "kg-re_fts5_docred"
        assert "fts5" in t.label
        assert "docred" in t.label

    def test_params_dict(self):
        t = KGRelationExtractionTreatment("gliner_small-v2.1", "webnlg")
        params = t.params_dict()
        assert params["model_slug"] == "gliner_small-v2.1"
        assert params["dataset"] == "webnlg"

    def test_sort_key(self):
        t = KGRelationExtractionTreatment("fts5", "docred")
        assert t.sort_key == ("docred", "fts5")

    def test_setup_creates_tables(self):
        t = KGRelationExtractionTreatment("fts5", "docred")
        conn = sqlite3.connect(":memory:")
        t.setup(conn, "/tmp/test.db")

        # Check tables exist
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {row[0] for row in tables}
        assert "predicted_triples" in table_names
        assert "text_timing" in table_names
        conn.close()

    def test_run_with_empty_data(self, tmp_path):
        """Run with missing dataset returns zero metrics gracefully."""
        t = KGRelationExtractionTreatment("fts5", "nonexistent_dataset")
        conn = sqlite3.connect(":memory:")
        t.setup(conn, tmp_path / "db.sqlite")

        with patch("benchmarks.harness.treatments.kg_re.KG_DIR", tmp_path / "kg"):
            metrics = t.run(conn)

        conn.close()

        assert metrics["total_time_s"] == 0
        assert metrics["n_texts"] == 0
        assert metrics["n_triples"] == 0
        assert metrics["triple_f1"] == 0.0

    def test_run_with_mock_data(self, tmp_path):
        """Run with mock RE dataset produces metrics including triple F1."""
        # Set up mock RE dataset
        ds_dir = tmp_path / "kg" / "re" / "test_re"
        ds_dir.mkdir(parents=True)

        texts = [
            {"id": 0, "text": "Alice works at ACME Corp in London"},
        ]
        triples = [
            {"text_id": 0, "subject": "Alice", "predicate": "works_at", "object": "ACME Corp"},
        ]

        (ds_dir / "texts.jsonl").write_text("\n".join(json.dumps(t) for t in texts), encoding="utf-8")
        (ds_dir / "triples.jsonl").write_text("\n".join(json.dumps(t) for t in triples), encoding="utf-8")

        t = KGRelationExtractionTreatment("fts5", "test_re")
        conn = sqlite3.connect(":memory:")
        t.setup(conn, tmp_path / "db.sqlite")

        with patch("benchmarks.harness.treatments.kg_re.KG_DIR", tmp_path / "kg"):
            metrics = t.run(conn)

        conn.close()

        assert "triple_f1" in metrics
        assert "triple_precision" in metrics
        assert "triple_recall" in metrics
        assert metrics["n_texts"] == 1
        assert isinstance(metrics["triple_f1"], float)
