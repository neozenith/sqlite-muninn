"""Tests for the Treatment ABC in treatments/base.py."""

import logging

import pytest

from benchmarks.harness.treatments.base import ProgressTracker, Treatment


class TestTreatmentABC:
    def test_cannot_instantiate_directly(self):
        """Treatment is abstract — instantiating it directly must raise TypeError."""
        with pytest.raises(TypeError):
            Treatment()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        """A properly implemented subclass can be instantiated."""

        class FakeTreatment(Treatment):
            @property
            def category(self):
                return "test"

            @property
            def permutation_id(self):
                return "test_fake_n100"

            @property
            def label(self):
                return "Test: fake / N=100"

            def setup(self, conn, db_path):
                return {"rows_loaded": 100}

            def run(self, conn):
                return {"latency_ms": 1.23}

            def teardown(self, conn):
                pass

        t = FakeTreatment()
        assert t.category == "test"
        assert t.permutation_id == "test_fake_n100"
        assert t.label == "Test: fake / N=100"
        assert t.params_dict() == {}  # default returns empty

    def test_missing_method_raises(self):
        """A subclass missing required methods cannot be instantiated."""

        class IncompleteTreatment(Treatment):
            @property
            def category(self):
                return "test"

            # Missing permutation_id, label, setup, run, teardown

        with pytest.raises(TypeError):
            IncompleteTreatment()  # type: ignore[abstract]

    def test_params_dict_overridable(self):
        """Subclasses can override params_dict to include custom parameters."""

        class ParamTreatment(Treatment):
            @property
            def category(self):
                return "test"

            @property
            def permutation_id(self):
                return "test_param"

            @property
            def label(self):
                return "Test: params"

            def setup(self, conn, db_path):
                return {}

            def run(self, conn):
                return {}

            def teardown(self, conn):
                pass

            def params_dict(self):
                return {"n": 1000, "dim": 128, "engine": "test"}

        t = ParamTreatment()
        params = t.params_dict()
        assert params["n"] == 1000
        assert params["dim"] == 128
        assert params["engine"] == "test"


def _make_fake_treatment():
    """Create a minimal concrete Treatment for testing base helpers."""

    class FakeTreatment(Treatment):
        @property
        def category(self):
            return "test"

        @property
        def permutation_id(self):
            return "test_fake"

        @property
        def label(self):
            return "Test: fake"

        def setup(self, conn, db_path):
            return {}

        def run(self, conn):
            return {}

        def teardown(self, conn):
            pass

    return FakeTreatment()


class TestStep:
    """Tests for Treatment.step() context manager."""

    def test_step_logs_start_and_done(self, caplog):
        t = _make_fake_treatment()
        with caplog.at_level(logging.INFO, logger="benchmarks.harness.treatments.base"):
            with t.step("Load data"):
                pass

        messages = [r.message for r in caplog.records]
        assert any("[STEP] Load data ..." in m for m in messages)
        assert any("[STEP] Load data done" in m for m in messages)

    def test_step_done_includes_timing(self, caplog):
        t = _make_fake_treatment()
        with caplog.at_level(logging.INFO, logger="benchmarks.harness.treatments.base"):
            with t.step("Quick op"):
                pass

        done_msgs = [m for m in caplog.messages if "[STEP] Quick op done" in m]
        assert len(done_msgs) == 1
        # Should contain timing like "(0.0s)"
        assert "s)" in done_msgs[0]

    def test_step_logs_failed_on_exception(self, caplog):
        t = _make_fake_treatment()
        with caplog.at_level(logging.INFO, logger="benchmarks.harness.treatments.base"):
            with pytest.raises(ValueError, match="boom"):
                with t.step("Bad op"):
                    raise ValueError("boom")

        messages = [r.message for r in caplog.records]
        assert any("[STEP] Bad op failed" in m for m in messages)
        # Should NOT have "done" message
        assert not any("[STEP] Bad op done" in m for m in messages)


class TestProgressTracker:
    """Tests for ProgressTracker and Treatment.progress()."""

    def test_progress_throttles_by_time(self, caplog):
        """Updates within the interval should be suppressed."""
        t = _make_fake_treatment()
        with caplog.at_level(logging.INFO, logger="benchmarks.harness.treatments.base"):
            tracker = t.progress(100, interval_s=10.0)
            # Rapidly call update — none should log because 10s hasn't passed
            for i in range(1, 101):
                tracker.update(i)

        progress_msgs = [m for m in caplog.messages if "/" in m and "%" in m]
        assert len(progress_msgs) == 0

    def test_progress_logs_when_interval_elapsed(self, caplog):
        """After enough time elapses, progress should log."""
        tracker = ProgressTracker(total=100, interval_s=0.0)
        with caplog.at_level(logging.INFO, logger="benchmarks.harness.treatments.base"):
            tracker.update(50)

        assert len(caplog.records) >= 1
        assert "50/100 (50%)" in caplog.records[0].message

    def test_progress_percentage_format(self, caplog):
        """Progress should show current/total (pct%)."""
        tracker = ProgressTracker(total=200, interval_s=0.0)
        with caplog.at_level(logging.INFO, logger="benchmarks.harness.treatments.base"):
            tracker.update(50)

        assert "50/200 (25%)" in caplog.records[0].message

    def test_progress_factory_returns_tracker(self):
        t = _make_fake_treatment()
        tracker = t.progress(500, interval_s=3.0)
        assert isinstance(tracker, ProgressTracker)
