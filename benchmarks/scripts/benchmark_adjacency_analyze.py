"""
Adjacency benchmark analysis: line charts comparing four caching strategies.

Directly compares four methods across graph sizes:
    TVF                              — no cache, SQL scan every query
    CSR — full rebuild               — full edge-table re-scan when stale
    CSR — incremental (all blocks)   — delta + merge, spread mutations touch every block
    Blocked CSR — incremental        — delta + merge, concentrated mutations touch 1 block

Charts produced:
    Per-algorithm query time  (one chart per algorithm: degree, betweenness, …)
    Rebuild time              (3 CSR strategies)
    Initial CSR build time
    Shadow table disk usage
    Trigger overhead

All charts are line charts with x=node_count, y=metric, mean ± std error bars.

Usage:
    python benchmarks/scripts/benchmark_adjacency_analyze.py
"""

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
CHARTS_DIR = PROJECT_ROOT / "benchmarks" / "charts"
DOCS_DIR = PROJECT_ROOT / "docs" / "benchmarks"

# ── Four comparison methods ──────────────────────────────────────

METHODS = ["tvf", "csr_full_rebuild", "csr_incremental", "csr_blocked"]

METHOD_COLORS = {
    "tvf": "hsl(0, 70%, 50%)",  # red
    "csr_full_rebuild": "hsl(30, 80%, 50%)",  # orange
    "csr_incremental": "hsl(160, 70%, 40%)",  # teal
    "csr_blocked": "hsl(270, 75%, 45%)",  # purple
}

METHOD_LABELS = {
    "tvf": "TVF (no cache)",
    "csr_full_rebuild": "CSR — full rebuild",
    "csr_incremental": "CSR — incremental (all blocks)",
    "csr_blocked": "Blocked CSR — incremental (affected)",
}

ALGORITHMS = ["degree", "betweenness", "closeness", "leiden"]

LAYOUT_DEFAULTS = {
    "template": "plotly_white",
    "height": 500,
    "width": 900,
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
    },
}


# ── Data loading ─────────────────────────────────────────────────


def load_results():
    """Load all adjacency_*.jsonl files from results directory."""
    records = []
    jsonl_files = sorted(RESULTS_DIR.glob("adjacency_*.jsonl"))

    if not jsonl_files:
        log.error("No adjacency JSONL files found in %s", RESULTS_DIR)
        return records

    log.info("Loading %d adjacency JSONL file(s)", len(jsonl_files))

    for f in jsonl_files:
        for line in f.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                records.append(json.loads(line))

    log.info("Loaded %d records", len(records))
    return records


# ── Data aggregation ─────────────────────────────────────────────

_NUMERIC_FIELDS = ("wall_time_ms", "disk_bytes", "with_trigger_ms", "without_trigger_ms")


def aggregate_records(records):
    """Group by (approach, workload, operation), compute mean ± std.

    Each group becomes a single summary record with the original field names
    set to the mean value, plus ``{field}_std`` and ``sample_count`` fields.
    """
    groups = defaultdict(list)
    for r in records:
        key = (r["approach"], r["workload"], r["operation"])
        groups[key].append(r)

    result = []
    for recs in groups.values():
        agg = dict(recs[0])
        agg["sample_count"] = len(recs)

        # Fill metadata gaps from newer records
        for r in recs[1:]:
            for k, v in r.items():
                if k not in agg or agg[k] is None:
                    agg[k] = v

        for field in _NUMERIC_FIELDS:
            values = [r[field] for r in recs if field in r and r[field] is not None]
            if values:
                mean = sum(values) / len(values)
                agg[field] = mean
                if len(values) > 1:
                    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                    agg[f"{field}_std"] = math.sqrt(variance)
                else:
                    agg[f"{field}_std"] = 0.0

        result.append(agg)

    log.info("Aggregated %d records into %d groups", len(records), len(result))
    return result


# ── Shared line-chart helpers ────────────────────────────────────


def _line_trace(recs, name, color):
    """Build a Scatter trace from records, sorted by node_count, with error bars."""
    recs = sorted(recs, key=lambda r: r["node_count"])
    x = [r["node_count"] for r in recs]
    y = [r["wall_time_ms"] for r in recs]
    stds = [r.get("wall_time_ms_std", 0) for r in recs]
    has_err = any(s > 0 for s in stds)
    return go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name=name,
        line={"color": color, "width": 2.5},
        marker={"size": 8},
        error_y={"type": "data", "array": stds, "visible": True} if has_err else None,
    )


def _save_chart(fig, name):
    """Write chart as .html and .json to CHARTS_DIR."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(CHARTS_DIR / f"{name}.html"), include_plotlyjs=True, full_html=True)
    (CHARTS_DIR / f"{name}.json").write_text(fig.to_json(), encoding="utf-8")
    log.info("  Saved: %s (.html + .json)", name)


# ── Per-algorithm query time charts ──────────────────────────────


def chart_algorithm_query_times(records):
    """One line chart per algorithm: 4 method series, x=node_count, y=query_time."""
    algos_present = sorted(
        {
            r["operation"]
            for r in records
            if r["approach"] in METHODS and r["operation"] in ALGORITHMS and "wall_time_ms" in r
        }
    )

    for algo in algos_present:
        fig = go.Figure()
        for method in METHODS:
            recs = [r for r in records if r["approach"] == method and r["operation"] == algo and "wall_time_ms" in r]
            if not recs:
                continue
            fig.add_trace(_line_trace(recs, METHOD_LABELS[method], METHOD_COLORS[method]))

        fig.update_layout(
            title=f"{algo.title()} Query Time by Graph Size",
            xaxis_title="Nodes",
            yaxis_title="Query Time (ms)",
            xaxis_type="log",
            yaxis_type="log",
            **LAYOUT_DEFAULTS,
        )
        _save_chart(fig, f"adj_{algo}")


# ── Rebuild time chart ───────────────────────────────────────────


def chart_rebuild_time(records):
    """Line chart: rebuild time for the 3 CSR strategies, x=node_count."""
    rebuild_methods = ["csr_full_rebuild", "csr_incremental", "csr_blocked"]
    fig = go.Figure()

    for method in rebuild_methods:
        recs = [r for r in records if r["approach"] == method and r["operation"] == "rebuild"]
        if not recs:
            continue
        fig.add_trace(_line_trace(recs, METHOD_LABELS[method], METHOD_COLORS[method]))

    fig.update_layout(
        title="Rebuild Time by Graph Size",
        xaxis_title="Nodes",
        yaxis_title="Rebuild Time (ms)",
        xaxis_type="log",
        yaxis_type="log",
        **LAYOUT_DEFAULTS,
    )
    _save_chart(fig, "adj_rebuild_time")


# ── Initial build time chart ─────────────────────────────────────


def chart_build_time(records):
    """Line chart: initial CSR build time, x=node_count."""
    recs = sorted(
        [r for r in records if r["approach"] == "csr" and r["operation"] == "build"],
        key=lambda r: r["node_count"],
    )
    if not recs:
        return

    x = [r["node_count"] for r in recs]
    y = [r["wall_time_ms"] for r in recs]
    stds = [r.get("wall_time_ms_std", 0) for r in recs]
    has_err = any(s > 0 for s in stds)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name="Initial CSR build",
            line={"color": "hsl(210, 60%, 50%)", "width": 2.5},
            marker={"size": 8},
            error_y={"type": "data", "array": stds, "visible": True} if has_err else None,
        )
    )
    fig.update_layout(
        title="Initial CSR Build Time by Graph Size",
        xaxis_title="Nodes",
        yaxis_title="Build Time (ms)",
        xaxis_type="log",
        yaxis_type="log",
        **LAYOUT_DEFAULTS,
    )
    _save_chart(fig, "adj_build_time")


# ── Disk usage chart ─────────────────────────────────────────────


def chart_disk_usage(records):
    """Line chart: CSR shadow table disk usage, x=node_count."""
    recs = sorted(
        [r for r in records if r["operation"] == "disk_usage" and "disk_bytes" in r],
        key=lambda r: r["node_count"],
    )
    if not recs:
        return

    x = [r["node_count"] for r in recs]
    y = [r["disk_bytes"] / 1024 for r in recs]
    stds = [r.get("disk_bytes_std", 0) / 1024 for r in recs]
    has_err = any(s > 0 for s in stds)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name="CSR shadow tables",
            line={"color": "hsl(210, 60%, 50%)", "width": 2.5},
            marker={"size": 8},
            error_y={"type": "data", "array": stds, "visible": True} if has_err else None,
        )
    )
    fig.update_layout(
        title="CSR Shadow Table Disk Usage by Graph Size",
        xaxis_title="Nodes",
        yaxis_title="Disk Usage (KB)",
        xaxis_type="log",
        yaxis_type="log",
        **LAYOUT_DEFAULTS,
    )
    _save_chart(fig, "adj_disk_usage")


# ── Trigger overhead chart ───────────────────────────────────────


def chart_trigger_overhead(records):
    """Line chart: insert batch time with vs without triggers, x=node_count."""
    recs = sorted(
        [r for r in records if r["approach"] == "trigger_overhead" and "with_trigger_ms" in r],
        key=lambda r: r["node_count"],
    )
    if not recs:
        return

    x = [r["node_count"] for r in recs]
    fig = go.Figure()

    for field, label, color in [
        ("without_trigger_ms", "Without triggers", "hsl(200, 60%, 50%)"),
        ("with_trigger_ms", "With triggers", "hsl(30, 80%, 50%)"),
    ]:
        y = [r[field] for r in recs]
        stds = [r.get(f"{field}_std", 0) for r in recs]
        has_err = any(s > 0 for s in stds)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=label,
                line={"color": color, "width": 2.5},
                marker={"size": 8},
                error_y={"type": "data", "array": stds, "visible": True} if has_err else None,
            )
        )

    fig.update_layout(
        title="Trigger Overhead by Graph Size",
        xaxis_title="Nodes",
        yaxis_title="Batch Insert Time (ms)",
        xaxis_type="log",
        yaxis_type="log",
        **LAYOUT_DEFAULTS,
    )
    _save_chart(fig, "adj_trigger_overhead")


# ── Chart saving & doc generation ────────────────────────────────


def generate_docs():
    """Generate docs/benchmarks/adjacency.md with Plotly chart includes."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    charts = [
        ("Degree Query Time", "adj_degree"),
        ("Betweenness Query Time", "adj_betweenness"),
        ("Closeness Query Time", "adj_closeness"),
        ("Leiden Query Time", "adj_leiden"),
        ("Rebuild Time", "adj_rebuild_time"),
        ("Initial CSR Build Time", "adj_build_time"),
        ("Shadow Table Disk Usage", "adj_disk_usage"),
        ("Trigger Overhead", "adj_trigger_overhead"),
    ]

    lines = [
        "<!-- AUTO-GENERATED by benchmark_adjacency_analyze.py — DO NOT EDIT -->",
        "<!-- Regenerate: make -C benchmarks analyze-adjacency -->",
        "",
        "# Adjacency Index Benchmarks",
        "",
        "Direct comparison of four graph-read strategies after edge mutations.",
        "Each chart plots performance against graph size (node count) on a log-log scale.",
        "",
        "## Methods",
        "",
        "| Method | Description |",
        "|--------|-------------|",
        "| **TVF** | No cache — scans edge table via SQL on every query |",
        "| **CSR — full rebuild** | Persistent CSR cache; full edge-table re-scan when stale |",
        "| **CSR — incremental** | Delta + merge; rebuilds all blocks (spread mutations) |",
        "| **Blocked CSR — incremental** | Delta + merge; rebuilds only affected blocks (concentrated mutations) |",
        "",
        "## How Blocked CSR Works",
        "",
        "The CSR is partitioned into blocks of 4,096 nodes. Each block is a separate",
        "row in the shadow table. When edges change, only blocks containing affected",
        "nodes are rewritten — unaffected blocks require zero I/O.",
        "",
    ]

    for title, chart_name in charts:
        if (CHARTS_DIR / f"{chart_name}.json").exists():
            lines.extend(
                [
                    f"## {title}",
                    "",
                    "```plotly",
                    f'--8<-- "benchmarks/charts/{chart_name}.json"',
                    "```",
                    "",
                ]
            )

    (DOCS_DIR / "adjacency.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("Generated docs: %s", DOCS_DIR / "adjacency.md")


# ── CLI ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyze adjacency benchmark results")
    parser.add_argument("--no-docs", action="store_true", help="Skip doc generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    records = load_results()
    if not records:
        return

    records = aggregate_records(records)

    log.info("Generating charts...")
    chart_algorithm_query_times(records)
    chart_rebuild_time(records)
    chart_build_time(records)
    chart_disk_usage(records)
    chart_trigger_overhead(records)
    log.info("All charts saved to %s", CHARTS_DIR)

    if not args.no_docs:
        generate_docs()


if __name__ == "__main__":
    main()
