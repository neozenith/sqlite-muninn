"""Tests for prep/kg_datasets.py — CrossNERPrepTask, HFDatasetPrepTask, and FebrlPrepTask."""

from unittest.mock import patch

from benchmarks.harness.prep.kg_datasets import (
    KG_PREP_TASKS,
    CrossNERPrepTask,
    FebrlPrepTask,
    HFDatasetPrepTask,
    print_status,
    prep_kg_datasets,
)


class TestCrossNERPrepTask:
    def test_task_id(self):
        t = CrossNERPrepTask("ai")
        assert t.task_id == "crossner:ai"

    def test_label(self):
        t = CrossNERPrepTask("science")
        assert t.label == "CrossNER science"

    def test_outputs(self, tmp_path):
        with patch("benchmarks.harness.prep.kg_datasets.KG_DIR", tmp_path / "kg"):
            t = CrossNERPrepTask("ai")
            outputs = t.outputs()
        assert len(outputs) == 2
        assert outputs[0].name == "texts.jsonl"
        assert outputs[1].name == "entities.jsonl"
        assert "crossner_ai" in str(outputs[0])

    def test_invalid_domain_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown CrossNER domain"):
            CrossNERPrepTask("invalid_domain")

    def test_fetch_and_transform(self, tmp_path):
        """Test fetch (from mock files) and transform with sample BIO data."""
        kg_dir = tmp_path / "kg"

        # Create mock raw data in BIO format
        raw_dir = kg_dir / "ner" / "crossner_ai" / "raw"
        raw_dir.mkdir(parents=True)

        bio_content = (
            "Deep\tB-algorithm\n"
            "learning\tI-algorithm\n"
            "is\tO\n"
            "a\tO\n"
            "technique\tO\n"
            "\n"
            "Google\tB-organisation\n"
            "uses\tO\n"
            "TensorFlow\tB-product\n"
            "\n"
        )
        (raw_dir / "train.txt").write_text(bio_content, encoding="utf-8")
        (raw_dir / "dev.txt").write_text("", encoding="utf-8")
        (raw_dir / "test.txt").write_text("", encoding="utf-8")

        with patch("benchmarks.harness.prep.kg_datasets.KG_DIR", kg_dir):
            t = CrossNERPrepTask("ai")
            # Skip fetch (we already created raw files), go straight to transform
            t.transform()

            outputs = t.outputs()
            assert outputs[0].exists(), "texts.jsonl should exist"
            assert outputs[1].exists(), "entities.jsonl should exist"
            assert t.status() == "READY"

            # Verify content
            import json

            texts = [json.loads(line) for line in outputs[0].read_text(encoding="utf-8").strip().split("\n")]
            entities = [json.loads(line) for line in outputs[1].read_text(encoding="utf-8").strip().split("\n")]

            assert len(texts) == 2
            assert texts[0]["text"] == "Deep learning is a technique"
            assert texts[0]["split"] == "train"
            assert texts[1]["text"] == "Google uses TensorFlow"

            # Should have 3 entities: "Deep learning", "Google", "TensorFlow"
            assert len(entities) == 3
            assert entities[0]["label"] == "algorithm"
            assert entities[0]["surface"] == "Deep learning"
            assert entities[1]["label"] == "organisation"
            assert entities[1]["surface"] == "Google"
            assert entities[2]["label"] == "product"
            assert entities[2]["surface"] == "TensorFlow"

    def test_status_missing_when_no_files(self, tmp_path):
        with patch("benchmarks.harness.prep.kg_datasets.KG_DIR", tmp_path / "kg"):
            t = CrossNERPrepTask("ai")
            assert t.status() == "MISSING"

    def test_all_domains_valid(self):
        """All DOMAINS should create valid tasks."""
        for domain in CrossNERPrepTask.DOMAINS:
            t = CrossNERPrepTask(domain)
            assert t.task_id == f"crossner:{domain}"


class TestHFDatasetPrepTask:
    def test_task_id_with_config(self):
        t = HFDatasetPrepTask("DFKI-SLT/few-nerd", "supervised")
        assert t.task_id == "DFKI-SLT/few-nerd:supervised"

    def test_task_id_without_config(self):
        t = HFDatasetPrepTask("thunlp/docred")
        assert t.task_id == "thunlp/docred"

    def test_label_with_config(self):
        t = HFDatasetPrepTask("DFKI-SLT/few-nerd", "supervised")
        assert t.label == "HF few-nerd (supervised)"

    def test_label_without_config(self):
        t = HFDatasetPrepTask("thunlp/docred")
        assert t.label == "HF docred"

    def test_outputs_empty(self):
        """HF datasets use opaque cache — outputs() returns empty list."""
        t = HFDatasetPrepTask("DFKI-SLT/few-nerd", "supervised")
        assert t.outputs() == []

    def test_status_missing_when_not_cached(self):
        """status() returns MISSING when dataset is not in HF cache."""
        t = HFDatasetPrepTask("nonexistent/dataset_xyz_12345")
        assert t.status() == "MISSING"

    def test_revision_stored(self):
        """revision parameter should be stored for load_dataset calls."""
        t = HFDatasetPrepTask("thunlp/docred", revision="refs/convert/parquet")
        assert t._revision == "refs/convert/parquet"

    def test_revision_default_none(self):
        t = HFDatasetPrepTask("DFKI-SLT/conll04")
        assert t._revision is None


class TestFebrlPrepTask:
    def test_task_id(self):
        t = FebrlPrepTask("febrl1")
        assert t.task_id == "febrl1"

    def test_label(self):
        t = FebrlPrepTask("febrl4")
        assert t.label == "Febrl febrl4"

    def test_outputs_febrl1(self, tmp_path):
        with patch("benchmarks.harness.prep.kg_datasets.KG_DIR", tmp_path / "kg"):
            t = FebrlPrepTask("febrl1")
            outputs = t.outputs()
        assert len(outputs) == 1
        assert outputs[0].name == "febrl1.parquet"

    def test_outputs_febrl4(self, tmp_path):
        with patch("benchmarks.harness.prep.kg_datasets.KG_DIR", tmp_path / "kg"):
            t = FebrlPrepTask("febrl4")
            outputs = t.outputs()
        assert len(outputs) == 2
        assert outputs[0].name == "febrl4_a.parquet"
        assert outputs[1].name == "febrl4_b.parquet"

    def test_status_missing_when_no_files(self, tmp_path):
        with patch("benchmarks.harness.prep.kg_datasets.KG_DIR", tmp_path / "kg"):
            t = FebrlPrepTask("febrl1")
            assert t.status() == "MISSING"

    def test_status_ready_when_files_exist(self, tmp_path):
        kg_dir = tmp_path / "kg"
        out_dir = kg_dir / "er" / "febrl1"
        out_dir.mkdir(parents=True)
        (out_dir / "febrl1.parquet").write_text("data", encoding="utf-8")

        with patch("benchmarks.harness.prep.kg_datasets.KG_DIR", kg_dir):
            t = FebrlPrepTask("febrl1")
            assert t.status() == "READY"


class TestKGPrepTasks:
    def test_registry_count(self):
        """Should have 14 tasks: 6 CrossNER + 3 Few-NERD + 3 RE + 2 Febrl."""
        assert len(KG_PREP_TASKS) == 14

    def test_all_task_ids_unique(self):
        ids = [t.task_id for t in KG_PREP_TASKS]
        assert len(ids) == len(set(ids))

    def test_contains_expected_datasets(self):
        ids = {t.task_id for t in KG_PREP_TASKS}
        # CrossNER (direct GitHub download)
        assert "crossner:ai" in ids
        assert "crossner:conll2003" in ids
        assert "crossner:literature" in ids
        assert "crossner:music" in ids
        assert "crossner:politics" in ids
        assert "crossner:science" in ids
        # Few-NERD
        assert "DFKI-SLT/few-nerd:supervised" in ids
        # RE
        assert "thunlp/docred" in ids
        assert "webnlg-challenge/web_nlg" in ids
        assert "DFKI-SLT/conll04" in ids
        # ER
        assert "febrl1" in ids
        assert "febrl4" in ids
        # TACRED should be gone
        assert "DFKI-SLT/tacred" not in ids

    def test_no_cross_ner_hf_tasks(self):
        """CrossNER should use CrossNERPrepTask, not HFDatasetPrepTask."""
        for task in KG_PREP_TASKS:
            if "cross_ner" in task.task_id:
                assert False, f"Found HF-based cross_ner task: {task.task_id}"


class TestPrintStatus:
    def test_print_status_runs(self, capsys):
        """print_status() should run without error and show header."""
        print_status()
        captured = capsys.readouterr()
        assert "KG Dataset Status" in captured.out
        assert "TASK_ID" in captured.out


class TestPrepKgDatasets:
    def test_status_only_does_not_download(self, capsys):
        """prep_kg_datasets(status_only=True) just prints status."""
        prep_kg_datasets(status_only=True)
        captured = capsys.readouterr()
        assert "KG Dataset Status" in captured.out
