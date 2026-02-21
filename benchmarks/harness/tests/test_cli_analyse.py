"""Tests for the analyse subcommand end-to-end."""

import json
import subprocess
import sys

from benchmarks.harness.analysis.aggregator import ChartSpec, aggregate_chart
from benchmarks.harness.analysis.renderer import render_chart
from benchmarks.harness.common import write_jsonl


class TestAnalyseCLI:
    def test_analyse_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "analyse", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "--category" in result.stdout

    def test_analyse_with_no_data_runs(self):
        """Analyse should succeed even with no data (just logs 'no data')."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "analyse"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0


class TestRendererIntegration:
    def test_render_chart_creates_json(self, tmp_path):
        spec = ChartSpec(
            name="test_output",
            title="Test",
            sources=["test_*.jsonl"],
            filters={},
            x_field="n",
            y_field="latency_ms",
            group_fields=["engine"],
            variant_fields=[],
            repeat_fields=["engine", "n"],
        )

        # Create sample data
        write_jsonl(tmp_path / "data" / "test_a.jsonl", {"engine": "a", "n": 100, "latency_ms": 1.5})
        write_jsonl(tmp_path / "data" / "test_a.jsonl", {"engine": "a", "n": 200, "latency_ms": 3.0})

        from benchmarks.harness.analysis.aggregator import load_jsonl_files

        records = load_jsonl_files(tmp_path / "data", spec.sources)
        series = aggregate_chart(records, spec)

        charts_dir = tmp_path / "charts"
        render_chart(spec, series, charts_dir=charts_dir)

        json_path = charts_dir / "test_output.json"
        assert json_path.exists()

        chart_data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "data" in chart_data
        assert "layout" in chart_data
        assert chart_data["layout"]["title"]["text"] == "Test"
