"""Tests for the benchmark execution harness."""

import sqlite3

from benchmarks.harness.harness import _handle_existing_db, run_treatment
from benchmarks.harness.treatments.base import Treatment


class FakeTreatment(Treatment):
    """Minimal treatment for testing the harness execution flow."""

    def __init__(self, perm_id="test_fake_n100"):
        self._perm_id = perm_id

    @property
    def category(self):
        return "test"

    @property
    def permutation_id(self):
        return self._perm_id

    @property
    def label(self):
        return "Test: fake treatment"

    def params_dict(self):
        return {"n": 100, "engine": "fake"}

    def setup(self, conn, db_path):
        conn.execute("CREATE TABLE test_data(id INTEGER PRIMARY KEY, value REAL)")
        conn.executemany("INSERT INTO test_data(id, value) VALUES (?, ?)", [(i, i * 0.1) for i in range(100)])
        conn.commit()
        return {"rows_loaded": 100}

    def run(self, conn):
        count = conn.execute("SELECT COUNT(*) FROM test_data").fetchone()[0]
        return {"row_count": count, "latency_ms": 1.23}

    def teardown(self, conn):
        pass


class TestHarness:
    def test_creates_db_file(self, tmp_path):
        treatment = FakeTreatment()
        run_treatment(treatment, results_dir=tmp_path)

        db_path = tmp_path / "test_fake_n100" / "db.sqlite"
        assert db_path.exists()

    def test_creates_jsonl_file(self, tmp_path):
        treatment = FakeTreatment()
        run_treatment(treatment, results_dir=tmp_path)

        # JSONL file should be named {category}_{variant}.jsonl
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1

    def test_returns_complete_record(self, tmp_path):
        treatment = FakeTreatment()
        record = run_treatment(treatment, results_dir=tmp_path)

        # Common metrics
        assert "permutation_id" in record
        assert "category" in record
        assert "wall_time_setup_ms" in record
        assert "wall_time_run_ms" in record
        assert "peak_rss_mb" in record
        assert "db_size_bytes" in record
        assert "timestamp" in record
        assert "platform" in record

        # Treatment params
        assert record["n"] == 100
        assert record["engine"] == "fake"

        # Setup metrics
        assert record["rows_loaded"] == 100

        # Run metrics
        assert record["row_count"] == 100
        assert record["latency_ms"] == 1.23

    def test_timing_is_positive(self, tmp_path):
        treatment = FakeTreatment()
        record = run_treatment(treatment, results_dir=tmp_path)
        assert record["wall_time_setup_ms"] >= 0
        assert record["wall_time_run_ms"] >= 0

    def test_db_has_data(self, tmp_path):
        treatment = FakeTreatment()
        run_treatment(treatment, results_dir=tmp_path)

        db_path = tmp_path / "test_fake_n100" / "db.sqlite"
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM test_data").fetchone()[0]
        conn.close()
        assert count == 100

    def test_unique_permutation_ids_get_separate_dirs(self, tmp_path):
        t1 = FakeTreatment("test_a")
        t2 = FakeTreatment("test_b")
        run_treatment(t1, results_dir=tmp_path)
        run_treatment(t2, results_dir=tmp_path)

        assert (tmp_path / "test_a" / "db.sqlite").exists()
        assert (tmp_path / "test_b" / "db.sqlite").exists()


class TestHandleExistingDb:
    def test_no_file_is_noop(self, tmp_path):
        """_handle_existing_db does nothing when file doesn't exist."""
        db_path = tmp_path / "nonexistent.sqlite"
        _handle_existing_db(db_path, force=True)
        assert not db_path.exists()

    def test_force_deletes_immediately(self, tmp_path):
        """force=True deletes existing file without countdown."""
        db_path = tmp_path / "db.sqlite"
        db_path.write_text("dummy")
        assert db_path.exists()

        _handle_existing_db(db_path, force=True)
        assert not db_path.exists()

    def test_rerun_with_force_succeeds(self, tmp_path):
        """Running a treatment twice with force=True should work."""
        treatment = FakeTreatment()

        # First run
        record1 = run_treatment(treatment, results_dir=tmp_path)
        assert record1["row_count"] == 100

        # Second run with force — should not fail with "table already exists"
        record2 = run_treatment(treatment, results_dir=tmp_path, force=True)
        assert record2["row_count"] == 100
