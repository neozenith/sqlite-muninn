"""Tests for aggregation, color system, and chart generation."""

from benchmarks.harness.analysis.aggregator import (
    ChartSeries,
    ChartSpec,
    aggregate_chart,
    filter_records,
    load_jsonl_files,
)
from benchmarks.harness.analysis.color_system import assign_color, assign_colors
from benchmarks.harness.common import write_jsonl


class TestAggregator:
    def _sample_spec(self):
        return ChartSpec(
            name="test_chart",
            title="Test Chart",
            sources=["test_*.jsonl"],
            filters={},
            x_field="n",
            y_field="latency_ms",
            group_fields=["engine"],
            variant_fields=[],
            repeat_fields=["engine", "n"],
        )

    def test_filter_records(self):
        records = [
            {"engine": "a", "n": 100},
            {"engine": "b", "n": 200},
            {"engine": "a", "n": 300},
        ]
        filtered = filter_records(records, {"engine": "a"})
        assert len(filtered) == 2
        assert all(r["engine"] == "a" for r in filtered)

    def test_filter_empty(self):
        records = [{"engine": "a"}]
        assert filter_records(records, {}) == records

    def test_aggregate_basic(self):
        spec = self._sample_spec()
        records = [
            {"engine": "a", "n": 100, "latency_ms": 1.0},
            {"engine": "a", "n": 100, "latency_ms": 3.0},
            {"engine": "b", "n": 100, "latency_ms": 5.0},
        ]
        series = aggregate_chart(records, spec)
        assert len(series) == 2  # engine a and b

        # Find the "a" series
        a_series = [s for s in series if s.group_key == "a"][0]
        assert len(a_series.points) == 1
        assert a_series.points[0].y_mean == 2.0  # (1.0 + 3.0) / 2
        assert a_series.points[0].y_min == 1.0
        assert a_series.points[0].y_max == 3.0

    def test_aggregate_empty(self):
        spec = self._sample_spec()
        series = aggregate_chart([], spec)
        assert series == []

    def test_load_jsonl_files(self, tmp_path):
        write_jsonl(tmp_path / "test_a.jsonl", {"x": 1})
        write_jsonl(tmp_path / "test_b.jsonl", {"x": 2})
        write_jsonl(tmp_path / "other.jsonl", {"x": 3})

        records = load_jsonl_files(tmp_path, ["test_*.jsonl"])
        assert len(records) == 2
        assert {r["x"] for r in records} == {1, 2}


class TestColorSystem:
    def test_known_engine_gets_color(self):
        color = assign_color("muninn_hnsw")
        assert "hsl(" in color
        assert "270" in color  # purple hue

    def test_unknown_engine_gets_default(self):
        color = assign_color("unknown_engine")
        assert "hsl(" in color
        assert "0," in color  # default hue

    def test_variant_varies_saturation(self):
        c0 = assign_color("muninn_hnsw", variant_idx=0, n_variants=3)
        c2 = assign_color("muninn_hnsw", variant_idx=2, n_variants=3)
        # Colors should be different (different S/L)
        assert c0 != c2

    def test_assign_colors_to_series(self):
        series = [
            ChartSeries(label="a", group_key="muninn_hnsw", variant_key="MiniLM", points=[]),
            ChartSeries(label="b", group_key="muninn_hnsw", variant_key="MPNet", points=[]),
            ChartSeries(label="c", group_key="sqlite_vector_quantize_scan", variant_key="MiniLM", points=[]),
        ]
        assign_colors(series)
        assert all(s.color != "" for s in series)
        # Same group_key should have same hue
        assert "270" in series[0].color  # muninn = purple
        assert "270" in series[1].color
        assert "175" in series[2].color  # sqlite-vector = teal


class TestChartSpecs:
    def test_vss_charts_importable(self):
        from benchmarks.harness.analysis.charts_vss import VSS_CHARTS

        assert len(VSS_CHARTS) > 0
        for spec in VSS_CHARTS:
            assert isinstance(spec, ChartSpec)

    def test_graph_charts_importable(self):
        from benchmarks.harness.analysis.charts_graph import GRAPH_CHARTS

        assert len(GRAPH_CHARTS) > 0

    def test_graph_vt_charts_importable(self):
        from benchmarks.harness.analysis.charts_graph_vt import GRAPH_VT_CHARTS

        assert len(GRAPH_VT_CHARTS) > 0

    def test_kg_charts_importable(self):
        from benchmarks.harness.analysis.charts_kg import KG_CHARTS

        assert len(KG_CHARTS) > 0
