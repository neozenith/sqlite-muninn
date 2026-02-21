"""Tests for expanded KG NER extraction: model slugs, data_source param, F1 evaluation."""

import json
import sqlite3
from unittest.mock import patch

from benchmarks.harness.treatments.kg_extract import (
    NER_ADAPTERS,
    KGNerExtractionTreatment,
    _data_source_slug,
    _parse_data_source,
)


class TestNerAdapters:
    def test_all_expected_model_slugs_present(self):
        expected = {
            "fts5",
            "gliner_small-v2.1",
            "gliner_medium-v2.1",
            "gliner_large-v2.1",
            "numind_NuNerZero",
            "gner-t5-base",
            "gner-t5-large",
            "spacy_en_core_web_lg",
        }
        assert set(NER_ADAPTERS.keys()) == expected

    def test_fts5_adapter_is_concrete(self):
        assert NER_ADAPTERS["fts5"] is not None

    def test_missing_adapters_are_none(self):
        for slug in ["gliner_medium-v2.1", "gliner_large-v2.1", "numind_NuNerZero", "gner-t5-base", "gner-t5-large"]:
            assert NER_ADAPTERS[slug] is None


class TestDataSourceParsing:
    def test_gutenberg_source(self):
        source_type, source_id = _parse_data_source("gutenberg:3300")
        assert source_type == "gutenberg"
        assert source_id == "3300"

    def test_ner_dataset_source(self):
        source_type, source_id = _parse_data_source("crossner_conll2003")
        assert source_type == "ner_dataset"
        assert source_id == "crossner_conll2003"

    def test_crossner_ai_source(self):
        source_type, source_id = _parse_data_source("crossner_ai")
        assert source_type == "ner_dataset"
        assert source_id == "crossner_ai"

    def test_data_source_slug_gutenberg(self):
        assert _data_source_slug("gutenberg:3300") == "gutenberg-3300"

    def test_data_source_slug_dataset(self):
        assert _data_source_slug("crossner_conll2003") == "crossner_conll2003"


class TestPermutationId:
    def test_gutenberg_permutation_id(self):
        t = KGNerExtractionTreatment("fts5", "gutenberg:3300")
        assert t.permutation_id == "kg-extract_fts5_gutenberg-3300"

    def test_crossner_permutation_id(self):
        t = KGNerExtractionTreatment("fts5", "crossner_conll2003")
        assert t.permutation_id == "kg-extract_fts5_crossner_conll2003"

    def test_category(self):
        t = KGNerExtractionTreatment("fts5", "gutenberg:3300")
        assert t.category == "kg-extract"

    def test_label(self):
        t = KGNerExtractionTreatment("gliner_small-v2.1", "crossner_conll2003")
        assert t.label == "KG Extract: gliner_small-v2.1 / crossner_conll2003"

    def test_params_dict_gutenberg(self):
        t = KGNerExtractionTreatment("fts5", "gutenberg:3300")
        params = t.params_dict()
        assert params["model_slug"] == "fts5"
        assert params["data_source"] == "gutenberg:3300"
        assert params["source_type"] == "gutenberg"
        assert params["source_id"] == "3300"

    def test_params_dict_ner_dataset(self):
        t = KGNerExtractionTreatment("fts5", "crossner_conll2003")
        params = t.params_dict()
        assert params["source_type"] == "ner_dataset"
        assert params["source_id"] == "crossner_conll2003"


class TestNerDatasetWithGold:
    def test_fts5_produces_f1_with_gold_labels(self, tmp_path):
        """When running against an NER dataset with gold labels, F1 fields are present."""
        # Set up mock NER dataset
        ds_dir = tmp_path / "kg" / "ner" / "test_ds"
        ds_dir.mkdir(parents=True)

        # Write texts
        texts = [
            {"id": 0, "text": "Alice went to London", "tokens": ["Alice", "went", "to", "London"]},
        ]
        (ds_dir / "texts.jsonl").write_text("\n".join(json.dumps(t) for t in texts), encoding="utf-8")

        # Write gold entities — FTS5 won't match these (span-based), but we still get F1=0
        entities = [
            {"text_id": 0, "start": 0, "end": 5, "label": "PER", "surface": "Alice"},
            {"text_id": 0, "start": 14, "end": 20, "label": "LOC", "surface": "London"},
        ]
        (ds_dir / "entities.jsonl").write_text("\n".join(json.dumps(e) for e in entities), encoding="utf-8")

        t = KGNerExtractionTreatment("fts5", "test_ds")
        conn = sqlite3.connect(":memory:")
        t.setup(conn, tmp_path / "db.sqlite")

        with patch("benchmarks.harness.treatments.kg_extract.KG_DIR", tmp_path / "kg"):
            metrics = t.run(conn)

        conn.close()

        assert "entity_f1" in metrics
        assert "entity_precision" in metrics
        assert "entity_recall" in metrics
        assert isinstance(metrics["entity_f1"], float)

    def test_fts5_no_f1_for_gutenberg(self, tmp_path):
        """Gutenberg source has no gold labels, so F1 fields should be absent."""
        t = KGNerExtractionTreatment("fts5", "gutenberg:99999")
        conn = sqlite3.connect(":memory:")
        t.setup(conn, tmp_path / "db.sqlite")

        with patch("benchmarks.harness.treatments.kg_extract.KG_DIR", tmp_path / "kg"):
            metrics = t.run(conn)

        conn.close()

        # Gutenberg source with missing DB returns basic metrics, no F1
        assert "entity_f1" not in metrics
        assert "n_chunks" in metrics
