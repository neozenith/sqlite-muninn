"""Tests for the Treatment ABC in treatments/base.py."""

import pytest

from benchmarks.harness.treatments.base import Treatment


class TestTreatmentABC:
    def test_cannot_instantiate_directly(self):
        """Treatment is abstract — instantiating it directly must raise TypeError."""
        with pytest.raises(TypeError):
            Treatment()

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
            IncompleteTreatment()

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
