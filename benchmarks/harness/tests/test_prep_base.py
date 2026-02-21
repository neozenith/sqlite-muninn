"""Tests for the PrepTask ABC in prep/base.py."""

import pytest

from benchmarks.harness.prep.base import PrepTask


class TestPrepTaskABC:
    def test_cannot_instantiate_directly(self):
        """PrepTask is abstract — instantiating it directly must raise TypeError."""
        with pytest.raises(TypeError):
            PrepTask()

    def test_concrete_subclass_works(self):
        """A properly implemented subclass can be instantiated."""

        class FakePrepTask(PrepTask):
            @property
            def task_id(self):
                return "test:fake"

            @property
            def label(self):
                return "Fake prep task"

            def outputs(self):
                return []

            def fetch(self, force=False):
                pass

        t = FakePrepTask()
        assert t.task_id == "test:fake"
        assert t.label == "Fake prep task"
        assert t.outputs() == []

    def test_missing_method_raises(self):
        """A subclass missing required methods cannot be instantiated."""

        class IncompletePrepTask(PrepTask):
            @property
            def task_id(self):
                return "test:incomplete"

            # Missing label, outputs, fetch

        with pytest.raises(TypeError):
            IncompletePrepTask()

    def test_status_ready_when_all_outputs_exist(self, tmp_path):
        """status() returns 'READY' when all output files exist."""
        out_file = tmp_path / "output.txt"
        out_file.write_text("data", encoding="utf-8")

        class FilePrepTask(PrepTask):
            @property
            def task_id(self):
                return "test:file"

            @property
            def label(self):
                return "File task"

            def outputs(self):
                return [out_file]

            def fetch(self, force=False):
                pass

        t = FilePrepTask()
        assert t.status() == "READY"

    def test_status_missing_when_outputs_absent(self, tmp_path):
        """status() returns 'MISSING' when output files don't exist."""

        class MissingPrepTask(PrepTask):
            @property
            def task_id(self):
                return "test:missing"

            @property
            def label(self):
                return "Missing task"

            def outputs(self):
                return [tmp_path / "nonexistent.txt"]

            def fetch(self, force=False):
                pass

        t = MissingPrepTask()
        assert t.status() == "MISSING"

    def test_run_skips_when_ready(self, tmp_path):
        """run() skips fetch when status is READY."""
        out_file = tmp_path / "output.txt"
        out_file.write_text("data", encoding="utf-8")
        fetch_called = []

        class CachedPrepTask(PrepTask):
            @property
            def task_id(self):
                return "test:cached"

            @property
            def label(self):
                return "Cached task"

            def outputs(self):
                return [out_file]

            def fetch(self, force=False):
                fetch_called.append(True)

        t = CachedPrepTask()
        t.run(force=False)
        assert len(fetch_called) == 0

    def test_run_calls_fetch_when_missing(self, tmp_path):
        """run() calls fetch when status is MISSING."""
        fetch_called = []

        class NewPrepTask(PrepTask):
            @property
            def task_id(self):
                return "test:new"

            @property
            def label(self):
                return "New task"

            def outputs(self):
                return [tmp_path / "nonexistent.txt"]

            def fetch(self, force=False):
                fetch_called.append(True)

        t = NewPrepTask()
        t.run(force=False)
        assert len(fetch_called) == 1

    def test_run_force_calls_fetch_even_when_ready(self, tmp_path):
        """run(force=True) calls fetch even when outputs exist."""
        out_file = tmp_path / "output.txt"
        out_file.write_text("data", encoding="utf-8")
        fetch_called = []

        class ForcePrepTask(PrepTask):
            @property
            def task_id(self):
                return "test:force"

            @property
            def label(self):
                return "Force task"

            def outputs(self):
                return [out_file]

            def fetch(self, force=False):
                fetch_called.append(True)

        t = ForcePrepTask()
        t.run(force=True)
        assert len(fetch_called) == 1

    def test_transform_called_after_fetch(self, tmp_path):
        """run() calls transform() after fetch()."""
        call_order = []

        class TransformPrepTask(PrepTask):
            @property
            def task_id(self):
                return "test:transform"

            @property
            def label(self):
                return "Transform task"

            def outputs(self):
                return [tmp_path / "nonexistent.txt"]

            def fetch(self, force=False):
                call_order.append("fetch")

            def transform(self):
                call_order.append("transform")

        t = TransformPrepTask()
        t.run(force=False)
        assert call_order == ["fetch", "transform"]

    def test_default_transform_is_noop(self):
        """Default transform() does nothing (returns None)."""

        class NoopPrepTask(PrepTask):
            @property
            def task_id(self):
                return "test:noop"

            @property
            def label(self):
                return "Noop task"

            def outputs(self):
                return []

            def fetch(self, force=False):
                pass

        t = NoopPrepTask()
        assert t.transform() is None
