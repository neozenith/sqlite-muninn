"""Tests for the VSS treatment."""

from benchmarks.harness.treatments.vss import ENGINE_CONFIGS, VSSTreatment


class TestVSSTreatmentInstantiation:
    def test_create_muninn(self):
        t = VSSTreatment("muninn-hnsw", "MiniLM", 384, "ag_news", 1000)
        assert t.category == "vss"
        assert "muninn-hnsw" in t.permutation_id
        assert "MiniLM" in t.permutation_id
        assert "ag-news" in t.permutation_id
        assert "n1000" in t.permutation_id

    def test_create_sqlite_vector_quantize(self):
        t = VSSTreatment("sqlite-vector-quantize", "MiniLM", 384, "ag_news", 5000)
        assert t.category == "vss"
        assert "sqlite-vector-quantize" in t.permutation_id
        assert "n5000" in t.permutation_id

    def test_label_is_descriptive(self):
        t = VSSTreatment("muninn-hnsw", "BGE-Large", 1024, "wealth_of_nations", 10000)
        assert "muninn-hnsw" in t.label
        assert "BGE-Large" in t.label
        assert "10000" in t.label

    def test_params_dict(self):
        t = VSSTreatment("muninn-hnsw", "MiniLM", 384, "ag_news", 1000)
        params = t.params_dict()
        assert params["engine"] == "muninn"
        assert params["search_method"] == "hnsw"
        assert params["model_name"] == "MiniLM"
        assert params["dim"] == 384
        assert params["dataset"] == "ag_news"
        assert params["n"] == 1000
        assert params["k"] == 10

    def test_all_engine_slugs_valid(self):
        for slug in ENGINE_CONFIGS:
            t = VSSTreatment(slug, "MiniLM", 384, "ag_news", 100)
            assert t.category == "vss"
            assert slug in t.permutation_id

    def test_dataset_underscores_replaced(self):
        """Permutation IDs should use hyphens for dataset names."""
        t = VSSTreatment("muninn-hnsw", "MiniLM", 384, "wealth_of_nations", 1000)
        assert "wealth-of-nations" in t.permutation_id
