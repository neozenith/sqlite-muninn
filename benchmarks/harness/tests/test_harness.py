"""Tests for the benchmark execution harness."""

import logging
import sqlite3

import pytest

from benchmarks.harness.harness import _cleanup_failed_db, _handle_existing_db, run_treatment
from benchmarks.harness.treatments.base import Treatment


class FakeTreatment(Treatment):
    """Minimal treatment for testing the harness execution flow.

    Sets requires_muninn=False because tests don't have the extension built.
    """

    def __init__(self, perm_id="test_fake_n100"):
        self._perm_id = perm_id

    @property
    def requires_muninn(self):
        return False

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


class FailingSetupTreatment(Treatment):
    """Treatment that fails during setup — for testing cleanup."""

    @property
    def requires_muninn(self):
        return False

    @property
    def category(self):
        return "test"

    @property
    def permutation_id(self):
        return "test_fail_setup"

    @property
    def label(self):
        return "Test: failing setup"

    def setup(self, conn, db_path):
        raise RuntimeError("setup exploded")

    def run(self, conn):
        return {}

    def teardown(self, conn):
        pass


class FailingRunTreatment(Treatment):
    """Treatment that fails during run — for testing cleanup."""

    @property
    def requires_muninn(self):
        return False

    @property
    def category(self):
        return "test"

    @property
    def permutation_id(self):
        return "test_fail_run"

    @property
    def label(self):
        return "Test: failing run"

    def setup(self, conn, db_path):
        conn.execute("CREATE TABLE test_data(id INTEGER PRIMARY KEY)")
        conn.commit()
        return {"rows_loaded": 0}

    def run(self, conn):
        raise RuntimeError("run exploded")

    def teardown(self, conn):
        pass


class TestFailureCleanup:
    """Failed treatments must not leave db.sqlite behind (ghost 'DONE' entries)."""

    def test_setup_failure_removes_db(self, tmp_path):
        """If setup() raises, db.sqlite must be deleted so manifest shows MISS."""
        treatment = FailingSetupTreatment()
        db_path = tmp_path / treatment.permutation_id / "db.sqlite"

        with pytest.raises(RuntimeError, match="setup exploded"):
            run_treatment(treatment, results_dir=tmp_path)

        assert not db_path.exists(), "db.sqlite left behind after setup failure"

    def test_run_failure_removes_db(self, tmp_path):
        """If run() raises, db.sqlite must be deleted so manifest shows MISS."""
        treatment = FailingRunTreatment()
        db_path = tmp_path / treatment.permutation_id / "db.sqlite"

        with pytest.raises(RuntimeError, match="run exploded"):
            run_treatment(treatment, results_dir=tmp_path)

        assert not db_path.exists(), "db.sqlite left behind after run failure"

    def test_setup_failure_removes_empty_parent_dir(self, tmp_path):
        """If the permutation directory is empty after cleanup, remove it too."""
        treatment = FailingSetupTreatment()
        perm_dir = tmp_path / treatment.permutation_id

        with pytest.raises(RuntimeError):
            run_treatment(treatment, results_dir=tmp_path)

        assert not perm_dir.exists(), "empty permutation dir left behind"

    def test_no_jsonl_written_on_failure(self, tmp_path):
        """Failed treatments must not write partial JSONL records."""
        treatment = FailingRunTreatment()

        with pytest.raises(RuntimeError):
            run_treatment(treatment, results_dir=tmp_path)

        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 0, "JSONL written despite treatment failure"


class TestCleanupFailedDb:
    def test_removes_existing_file(self, tmp_path):
        db_path = tmp_path / "db.sqlite"
        db_path.write_text("dummy")
        _cleanup_failed_db(db_path)
        assert not db_path.exists()

    def test_noop_when_file_missing(self, tmp_path):
        db_path = tmp_path / "nonexistent.sqlite"
        _cleanup_failed_db(db_path)  # should not raise

    def test_removes_empty_parent_dir(self, tmp_path):
        perm_dir = tmp_path / "some_perm"
        perm_dir.mkdir()
        db_path = perm_dir / "db.sqlite"
        db_path.write_text("dummy")

        _cleanup_failed_db(db_path)
        assert not perm_dir.exists()

    def test_keeps_parent_dir_if_not_empty(self, tmp_path):
        perm_dir = tmp_path / "some_perm"
        perm_dir.mkdir()
        db_path = perm_dir / "db.sqlite"
        db_path.write_text("dummy")
        (perm_dir / "other_file.txt").write_text("keep me")

        _cleanup_failed_db(db_path)
        assert not db_path.exists()
        assert perm_dir.exists(), "parent dir removed despite having other files"


class RequiresMuninnTreatment(Treatment):
    """Treatment that requires muninn — for testing fatal load failure."""

    @property
    def requires_muninn(self):
        return True

    @property
    def category(self):
        return "test"

    @property
    def permutation_id(self):
        return "test_requires_muninn"

    @property
    def label(self):
        return "Test: requires muninn"

    def setup(self, conn, db_path):
        return {}

    def run(self, conn):
        return {}

    def teardown(self, conn):
        pass


class TestRequiresMuninn:
    """Tests for the requires_muninn contract on Treatment."""

    def test_default_is_true(self):
        """Treatment.requires_muninn defaults to True."""
        assert Treatment.requires_muninn.fget is not None  # type: ignore[attr-defined]  # property exists
        # FakeTreatment overrides to False, but the base default is True
        assert RequiresMuninnTreatment().requires_muninn is True

    def test_requires_muninn_false_skips_loading(self, tmp_path):
        """When requires_muninn=False, harness succeeds without muninn."""
        treatment = FakeTreatment()
        assert treatment.requires_muninn is False

        # Should succeed even though muninn extension doesn't exist
        record = run_treatment(treatment, results_dir=tmp_path)
        assert record["row_count"] == 100


class TestLifecycleLogging:
    """Verify that [SETUP], [RUN], [TEARDOWN] markers appear in harness log output."""

    def test_lifecycle_phases_logged(self, tmp_path, caplog):
        treatment = FakeTreatment()
        with caplog.at_level(logging.INFO, logger="benchmarks.harness.harness"):
            run_treatment(treatment, results_dir=tmp_path)

        messages = [r.message for r in caplog.records]

        assert any("[SETUP]" in m and "done" not in m for m in messages)
        assert any("[SETUP] done" in m for m in messages)
        assert any("[RUN]" in m and "done" not in m for m in messages)
        assert any("[RUN] done" in m for m in messages)
        assert any("[TEARDOWN]" in m and "done" not in m for m in messages)
        assert any("[TEARDOWN] done" in m for m in messages)

    def test_lifecycle_order(self, tmp_path, caplog):
        """Lifecycle phases should appear in order: SETUP → RUN → TEARDOWN."""
        treatment = FakeTreatment()
        with caplog.at_level(logging.INFO, logger="benchmarks.harness.harness"):
            run_treatment(treatment, results_dir=tmp_path)

        tags = ["[SETUP]", "[RUN]", "[TEARDOWN]"]
        phase_msgs = [r.message for r in caplog.records if any(tag in r.message for tag in tags)]

        # Find first occurrence indices
        setup_idx = next(i for i, m in enumerate(phase_msgs) if "[SETUP]" in m)
        run_idx = next(i for i, m in enumerate(phase_msgs) if "[RUN]" in m)
        teardown_idx = next(i for i, m in enumerate(phase_msgs) if "[TEARDOWN]" in m)

        assert setup_idx < run_idx < teardown_idx
