"""Aggregation engine: load JSONL, group, aggregate, and prepare chart data.

Works at the chart level via ChartSpec definitions. Each chart specifies
how to slice the data: sources, filters, axes, grouping, and aggregation.
"""

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.harness.analysis.color_system import DISPLAY_LABELS
from benchmarks.harness.common import CHARTS_DIR, RESULTS_DIR

log = logging.getLogger(__name__)


@dataclass
class ChartSpec:
    """Defines one output chart."""

    name: str  # Output filename (e.g., "tipping_point_MiniLM_ag_news")
    title: str  # Chart title
    sources: list[str]  # JSONL glob patterns to load (e.g., ["vss_*.jsonl"])
    filters: dict[str, Any]  # Record-level filters (e.g., {"dataset": "ag_news"})
    x_field: str  # Field for x-axis
    y_field: str  # Field for y-axis
    group_fields: list[str]  # Fields forming the grouping dimension / hue
    variant_fields: list[str]  # Fields for S/L variant within a hue
    repeat_fields: list[str]  # Fields identifying "same measurement" for aggregation
    y_agg: str = "mean"  # Aggregation: "mean", "median", "min", "max"
    x_label: str | None = None
    y_label: str | None = None
    log_x: bool = False
    log_y: bool = False


@dataclass
class AggregatedPoint:
    """A single aggregated data point."""

    x: float | str
    y_mean: float
    y_min: float
    y_max: float
    y_std: float
    count: int


@dataclass
class ChartSeries:
    """A series of aggregated points with color assignment."""

    label: str
    group_key: str
    variant_key: str
    points: list[AggregatedPoint] = field(default_factory=list)
    color: str = ""


def load_jsonl_files(results_dir: Path, patterns: list[str]) -> list[dict[str, Any]]:
    """Load all JSONL records matching the given glob patterns."""
    records: list[dict[str, Any]] = []
    for pattern in patterns:
        for filepath in sorted(results_dir.glob(pattern)):
            for line in filepath.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    records.append(json.loads(line))
    return records


def filter_records(records: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep only records matching all filter criteria.

    Filter values can be scalars (exact match) or lists (match if field value is in list).
    """
    filtered: list[dict[str, Any]] = []
    for r in records:
        match = True
        for key, value in filters.items():
            if isinstance(value, list):
                if r.get(key) not in value:
                    match = False
                    break
            else:
                if r.get(key) != value:
                    match = False
                    break
        if match:
            filtered.append(r)
    return filtered


def aggregate_chart(records: list[dict[str, Any]], spec: ChartSpec) -> list[ChartSeries]:
    """Aggregate records according to a ChartSpec.

    Returns list of ChartSeries with aggregated points.
    """
    # Group records by repeat_fields
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = tuple(r.get(f) for f in spec.repeat_fields)
        groups[key].append(r)

    # Aggregate each group
    aggregated_rows: list[dict[str, Any]] = []
    for key, recs in groups.items():
        y_values = [r[spec.y_field] for r in recs if r.get(spec.y_field) is not None]
        if not y_values:
            continue

        mean = sum(y_values) / len(y_values)
        if len(y_values) > 1:
            variance = sum((v - mean) ** 2 for v in y_values) / (len(y_values) - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0

        row: dict[str, Any] = dict(zip(spec.repeat_fields, key, strict=False))
        row["_y_mean"] = mean
        row["_y_min"] = min(y_values)
        row["_y_max"] = max(y_values)
        row["_y_std"] = std
        row["_count"] = len(y_values)
        # Copy x_field from first record
        row[spec.x_field] = recs[0].get(spec.x_field)
        aggregated_rows.append(row)

    # Group into series by group_fields + variant_fields
    series_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregated_rows:
        group_key = "_".join(str(row.get(f, "")) for f in spec.group_fields)
        variant_key = "_".join(str(row.get(f, "")) for f in spec.variant_fields) if spec.variant_fields else ""
        series_key = (group_key, variant_key)
        series_map[series_key].append(row)

    # Build ChartSeries
    series_list: list[ChartSeries] = []
    for (group_key, variant_key), rows in sorted(series_map.items()):
        display_group = DISPLAY_LABELS.get(group_key, group_key.replace("_", "-"))
        label = display_group
        if variant_key:
            display_variant = variant_key.replace("_", "-")
            label = f"{display_group} / {display_variant}"

        points: list[AggregatedPoint] = []
        for row in sorted(
            rows,
            key=lambda r: (
                r.get(spec.x_field, 0)
                if isinstance(r.get(spec.x_field), (int, float))
                else str(r.get(spec.x_field, ""))
            ),
        ):
            points.append(
                AggregatedPoint(
                    x=row[spec.x_field],
                    y_mean=row["_y_mean"],
                    y_min=row["_y_min"],
                    y_max=row["_y_max"],
                    y_std=row["_y_std"],
                    count=row["_count"],
                )
            )
        series_list.append(ChartSeries(label=label, group_key=group_key, variant_key=variant_key, points=points))

    return series_list


def run_aggregation(results_dir: Path | None = None, category: str | None = None, render_docs: bool = False) -> None:
    """Run aggregation and chart generation for all chart specs.

    This is the entry point called by the CLI 'analyse' subcommand.
    """
    results_dir = results_dir or RESULTS_DIR

    # Collect chart specs from chart definition modules
    all_specs: list[ChartSpec] = []

    from benchmarks.harness.analysis.charts_graph import GRAPH_CHARTS
    from benchmarks.harness.analysis.charts_graph_vt import GRAPH_VT_CHARTS
    from benchmarks.harness.analysis.charts_kg import KG_CHARTS
    from benchmarks.harness.analysis.charts_vss import VSS_CHARTS

    all_specs.extend(VSS_CHARTS)
    all_specs.extend(GRAPH_CHARTS)
    all_specs.extend(GRAPH_VT_CHARTS)
    all_specs.extend(KG_CHARTS)

    if category:
        # Filter specs by category (based on source patterns)
        category_prefix = f"{category}_"
        all_specs = [s for s in all_specs if any(p.startswith(category_prefix) for p in s.sources)]

    log.info("Running aggregation for %d chart specs...", len(all_specs))

    for spec in all_specs:
        records = load_jsonl_files(results_dir, spec.sources)
        if not records:
            log.info("  %s: no data (0 records)", spec.name)
            continue

        records = filter_records(records, spec.filters)
        if not records:
            log.info("  %s: no data after filtering", spec.name)
            continue

        series = aggregate_chart(records, spec)
        log.info("  %s: %d series, %d total records", spec.name, len(series), len(records))

        # Render chart (Plotly JSON)
        from benchmarks.harness.analysis.renderer import render_chart

        render_chart(spec, series, charts_dir=CHARTS_DIR)

    if render_docs:
        from benchmarks.harness.analysis.renderer import render_docs as _render_docs

        _render_docs()

    log.info("Analysis complete. Charts in %s", CHARTS_DIR)
