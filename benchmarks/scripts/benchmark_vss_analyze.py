"""
Benchmark analysis: aggregate JSONL results into text tables and Plotly JSON charts.

Reads all benchmarks/results/*.jsonl files (excluding graph_* files), aggregates by
(engine, search_method, vector_source, model_name, dataset, dim, n), and produces:

1. Text tables: model comparison, search latency, insert throughput, recall, storage
2. Plotly JSON charts: per-model tipping point, cross-model comparison, recall, storage

Engines:
    muninn-hnsw              — this project's HNSW index
    sqlite-vector-quantize      — sqliteai/sqlite-vector quantized approximate search
    sqlite-vector-fullscan      — sqliteai/sqlite-vector brute-force exact search
    vectorlite-hnsw             — vectorlite HNSW (hnswlib backend)
    sqlite-vec-brute            — asg017/sqlite-vec brute-force KNN

Usage:
    python python/benchmark_vss_analyze.py
    python python/benchmark_vss_analyze.py --filter-source model
    python python/benchmark_vss_analyze.py --filter-model MiniLM
    python python/benchmark_vss_analyze.py --filter-dataset wealth_of_nations
"""

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

import jinja2
import numpy as np
import plotly.graph_objects as go
from benchmark_vss import (
    ALL_ENGINES,
    EMBEDDING_MODELS,
    MAX_N_BY_DIM,
    PROFILES,
    VECTORS_DIR,
    make_scenario_name,
)
from benchmark_vss import (
    RESULTS_DIR as VSS_RESULTS_DIR,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
CHARTS_DIR = PROJECT_ROOT / "benchmarks" / "charts"


# ── Data loading ──────────────────────────────────────────────────


def load_all_results(filter_source=None, filter_dim=None, filter_model=None, filter_dataset=None):
    """Load all JSONL files from results directory, applying optional filters."""
    records = []
    # Only load vector benchmark files (exclude graph_*.jsonl and kg_*.jsonl)
    jsonl_files = sorted(
        f for f in RESULTS_DIR.glob("*.jsonl") if not f.name.startswith("graph_") and not f.name.startswith("kg_")
    )

    if not jsonl_files:
        log.error("No JSONL files found in %s", RESULTS_DIR)
        log.error("Run 'make benchmark-models' first.")
        return records

    log.info("Loading %d JSONL file(s) from %s", len(jsonl_files), RESULTS_DIR)

    for f in jsonl_files:
        for line in f.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            record = json.loads(line)

            if filter_source and record.get("vector_source") != filter_source:
                continue
            if filter_dim and record.get("dim") != filter_dim:
                continue
            if filter_model and record.get("model_name") != filter_model:
                continue
            if filter_dataset and record.get("dataset") != filter_dataset:
                continue

            records.append(record)

    log.info("Loaded %d records", len(records))
    return records


# ── Aggregation ───────────────────────────────────────────────────


def aggregate(records):
    """Group records and compute mean/stddev for each metric.

    Groups by (engine, search_method, vector_source, model_name, dataset, dim, n).
    Returns dict mapping group key -> aggregated metrics dict.
    """
    groups = defaultdict(list)

    for r in records:
        key = (
            r["engine"],
            r["search_method"],
            r.get("vector_source", "model"),
            r.get("model_name"),
            r.get("dataset"),
            r["dim"],
            r["n"],
        )
        groups[key].append(r)

    agg = {}
    for key, recs in groups.items():
        agg[key] = _aggregate_group(recs)

    return agg


def _aggregate_group(records):
    """Compute aggregated stats for a group of records."""
    metrics = [
        "insert_rate_vps",
        "search_latency_ms",
        "recall_at_k",
        "memory_delta_mb",
        "db_size_bytes",
    ]

    result = {"count": len(records)}

    for metric in metrics:
        values = [r[metric] for r in records if r.get(metric) is not None]
        if values:
            mean = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                stddev = math.sqrt(variance)
            else:
                stddev = 0.0
            result[f"{metric}_mean"] = mean
            result[f"{metric}_std"] = stddev
        else:
            result[f"{metric}_mean"] = None
            result[f"{metric}_std"] = None

    first = records[0]
    result["quantize_s"] = first.get("quantize_s")
    result["model_name"] = first.get("model_name")

    return result


# ── Key accessors ─────────────────────────────────────────────────
# Key format: (engine, method, source, model_name, dataset, dim, n)


def _get_models_by_dim(agg):
    """Get models sorted by dimension (ascending).

    Returns list of (model_name, dim) tuples.
    """
    model_dims = {}
    for k in agg:
        if k[3] is not None:
            model_dims[k[3]] = k[5]
    return sorted(model_dims.items(), key=lambda pair: pair[1])


def _get_models(agg):
    """Get model names sorted by dimension (ascending)."""
    return [m for m, _ in _get_models_by_dim(agg)]


def _get_datasets(agg):
    """Get sorted unique dataset names (excluding None)."""
    return sorted({k[4] for k in agg if k[4] is not None})


def _get_sizes(agg, dim=None, model=None, dataset=None):
    """Get sorted unique dataset sizes, optionally filtered."""
    sizes = set()
    for k in agg:
        if dim is not None and k[5] != dim:
            continue
        if model is not None and k[3] != model:
            continue
        if dataset is not None and k[4] != dataset:
            continue
        sizes.add(k[6])
    return sorted(sizes)


def _get_engine_method_pairs(agg):
    """Get (engine, method) pairs that actually have data."""
    pairs = set()
    for k in agg:
        pairs.add((k[0], k[1]))
    return sorted(pairs, key=lambda p: ENGINE_METHOD_PAIRS.index(p) if p in ENGINE_METHOD_PAIRS else 999)


def _get_val(agg, engine, method, source, model, dataset, dim, n, metric):
    """Safely get a mean metric value from aggregated data."""
    key = (engine, method, source, model, dataset, dim, n)
    entry = agg.get(key)
    if entry is None:
        return None
    return entry.get(f"{metric}_mean")


def _fmt(val, fmt_str=".3f"):
    """Format a value or return 'n/a'."""
    if val is None:
        return "n/a"
    return f"{val:{fmt_str}}"


def _fmt_bytes(size):
    """Format byte count as human-readable string."""
    if size is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ── Plotly chart infrastructure ───────────────────────────────────
#
# Design rules:
#   1. Labels: {library}-{algorithm} or {library}-{algorithm}-{dims}d-{model}
#   2. Hue per library-algorithm, vary S/L per model -> "fiber bundle" effect
#   3. Non-muninn traces at 80% opacity so muninn pops
#   4. Legend ordered: library-algorithm first, then model (by dim ascending)


ENGINE_METHOD_PAIRS = [
    ("muninn", "hnsw"),
    ("sqlite_vector", "quantize_scan"),
    ("sqlite_vector", "full_scan"),
    ("vectorlite", "hnsw"),
    ("sqlite_vec", "brute_force"),
]

# Library-algorithm labels
ENGINE_LABELS = {
    ("muninn", "hnsw"): "muninn-hnsw",
    ("sqlite_vector", "quantize_scan"): "sqlite-vector-quantize",
    ("sqlite_vector", "full_scan"): "sqlite-vector-fullscan",
    ("vectorlite", "hnsw"): "vectorlite-hnsw",
    ("sqlite_vec", "brute_force"): "sqlite-vec-brute",
}

# Base hue per library-algorithm (HSL hue degrees)
ENGINE_HUES = {
    ("muninn", "hnsw"): 270,  # purple
    ("sqlite_vector", "quantize_scan"): 175,  # teal
    ("sqlite_vector", "full_scan"): 18,  # warm orange
    ("vectorlite", "hnsw"): 210,  # blue
    ("sqlite_vec", "brute_force"): 130,  # green
}


def _engine_label(engine, method):
    """Library-algorithm label: muninn-hnsw, sqlite-vector-quantize, etc."""
    return ENGINE_LABELS.get((engine, method), f"{engine}-{method}")


def _trace_label(engine, method, model=None, dim=None):
    """Full trace label for legends.

    Single-model charts: "muninn-hnsw"
    Cross-model charts:  "muninn-hnsw-384d-MiniLM"
    """
    base = _engine_label(engine, method)
    if model is not None and dim is not None:
        return f"{base}-{dim}d-{model}"
    return base


def _make_color(engine, method, model_idx=0, n_models=1):
    """Generate HSL color for a trace.

    Hue is fixed per library-algorithm.
    Saturation and luminance vary per model to create a fiber-bundle effect:
    smaller dim (idx=0) -> lighter/more vivid, larger dim -> deeper/richer.
    """
    hue = ENGINE_HUES.get((engine, method), 0)
    if n_models <= 1:
        sat, lum = 75, 45
    else:
        t = model_idx / (n_models - 1)
        sat = 85 - int(t * 15)  # 85% -> 70%
        lum = 58 - int(t * 23)  # 58% -> 35%
    return f"hsl({hue}, {sat}%, {lum}%)"


def _trace_opacity(engine):
    """muninn at full opacity, everything else softened to 80%."""
    return 1.0 if engine == "muninn" else 0.8


def _save_chart(fig, name):
    """Save a Plotly figure as standalone HTML and JSON for mkdocs embedding."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    html_path = CHARTS_DIR / f"{name}.html"
    fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)

    json_path = CHARTS_DIR / f"{name}.json"
    json_path.write_text(fig.to_json(), encoding="utf-8")

    log.info("  Chart saved: %s (.html + .json)", CHARTS_DIR / name)


# ── Text tables ───────────────────────────────────────────────────


def print_tables(agg):
    """Print all text summary tables."""
    models = _get_models(agg)
    datasets = _get_datasets(agg)
    active_pairs = _get_engine_method_pairs(agg)

    if models:
        # Group tables by dataset if multiple exist
        for ds in datasets or [None]:
            ds_agg = {k: v for k, v in agg.items() if k[4] == ds} if ds else agg
            ds_label = f" [{ds}]" if ds else ""
            if not ds_agg:
                continue
            print_model_overview(ds_agg, models, active_pairs, ds_label)
            print_model_search_table(ds_agg, models, active_pairs, ds_label)
            print_model_insert_table(ds_agg, models, active_pairs, ds_label)
            print_model_recall_table(ds_agg, models, active_pairs, ds_label)
            print_model_storage_table(ds_agg, models, active_pairs, ds_label)


def _col_header(pairs):
    """Build column header abbreviations for engine-method pairs."""
    abbrevs = []
    for engine, method in pairs:
        label = _engine_label(engine, method)
        # Shorten to fit tables
        if len(label) > 16:
            label = label[:16]
        abbrevs.append(label)
    return abbrevs


def print_model_overview(agg, models, active_pairs, ds_label=""):
    """Print a high-level model comparison summary."""
    model_pairs = [(e, m) for e, m in active_pairs if any(k[0] == e and k[1] == m and k[3] is not None for k in agg)]
    if not model_pairs:
        return

    print(f"\n{'=' * 100}")
    print(f"EMBEDDING MODEL OVERVIEW{ds_label}")
    labels = _col_header(model_pairs)
    print(f"  Engines: {' | '.join(labels)}")
    print("=" * 100)
    print(f"  {'Model':>12} | {'Dim':>5} | {'Sizes Tested':>30} | {'vg-hnsw wins at max N?':>22}")
    print(f"  {'-' * 12}-+-{'-' * 5}-+-{'-' * 30}-+-{'-' * 22}")

    for model in models:
        dims = sorted({k[5] for k in agg if k[3] == model})
        sizes = sorted({k[6] for k in agg if k[3] == model})
        dim = dims[0] if dims else 0
        sizes_str = ", ".join(f"{s:,}" for s in sizes)
        ds = next((k[4] for k in agg if k[3] == model), None)

        largest_n = max(sizes) if sizes else 0
        hnsw_lat = _get_val(agg, "muninn", "hnsw", "model", model, ds, dim, largest_n, "search_latency_ms")
        qscan_lat = _get_val(
            agg, "sqlite_vector", "quantize_scan", "model", model, ds, dim, largest_n, "search_latency_ms"
        )
        wins = ""
        if hnsw_lat is not None and qscan_lat is not None:
            speedup = qscan_lat / hnsw_lat
            wins = f"YES ({speedup:.0f}x)" if hnsw_lat < qscan_lat else "no"

        print(f"  {model:>12} | {dim:>5} | {sizes_str:>30} | {wins:>22}")


def print_model_search_table(agg, models, active_pairs, ds_label=""):
    """Print search latency table grouped by model."""
    model_pairs = [(e, m) for e, m in active_pairs if any(k[0] == e and k[1] == m and k[3] is not None for k in agg)]
    if not model_pairs:
        return

    col_labels = _col_header(model_pairs)
    col_w = max(12, max(len(c) for c in col_labels) + 2)

    print(f"\n{'=' * 100}")
    print(f"SEARCH LATENCY BY MODEL (ms/query){ds_label}")
    print("=" * 100)

    for model in models:
        dims = sorted({k[5] for k in agg if k[3] == model})
        for dim in dims:
            ds = next((k[4] for k in agg if k[3] == model and k[5] == dim), None)
            sizes = _get_sizes(agg, dim=dim, model=model, dataset=ds)
            if not sizes:
                continue

            print(f"\n  {model} (dim={dim})")
            header = f"  {'N':>10}"
            for label in col_labels:
                header += f" | {label:>{col_w}}"
            print(header)
            print(f"  {'-' * 10}" + f"-+-{'-' * col_w}" * len(col_labels))

            for n in sizes:
                row = f"  {n:>10,}"
                for engine, method in model_pairs:
                    val = _get_val(agg, engine, method, "model", model, ds, dim, n, "search_latency_ms")
                    row += f" | {_fmt(val):>{col_w}}"
                print(row)


def print_model_insert_table(agg, models, active_pairs, ds_label=""):
    """Print insert throughput table grouped by model."""
    model_pairs = [(e, m) for e, m in active_pairs if any(k[0] == e and k[1] == m and k[3] is not None for k in agg)]
    if not model_pairs:
        return

    col_labels = _col_header(model_pairs)
    col_w = max(14, max(len(c) for c in col_labels) + 2)

    print(f"\n{'=' * 100}")
    print(f"INSERT THROUGHPUT BY MODEL (vectors/sec){ds_label}")
    print("=" * 100)

    for model in models:
        dims = sorted({k[5] for k in agg if k[3] == model})
        for dim in dims:
            ds = next((k[4] for k in agg if k[3] == model and k[5] == dim), None)
            sizes = _get_sizes(agg, dim=dim, model=model, dataset=ds)
            if not sizes:
                continue

            print(f"\n  {model} (dim={dim})")
            header = f"  {'N':>10}"
            for label in col_labels:
                header += f" | {label:>{col_w}}"
            print(header)
            print(f"  {'-' * 10}" + f"-+-{'-' * col_w}" * len(col_labels))

            for n in sizes:
                row = f"  {n:>10,}"
                for engine, method in model_pairs:
                    val = _get_val(agg, engine, method, "model", model, ds, dim, n, "insert_rate_vps")
                    row += f" | {_fmt(val, ',.0f'):>{col_w}}"
                print(row)


def print_model_recall_table(agg, models, active_pairs, ds_label=""):
    """Print recall@k table grouped by model."""
    model_pairs = [(e, m) for e, m in active_pairs if any(k[0] == e and k[1] == m and k[3] is not None for k in agg)]
    if not model_pairs:
        return

    col_labels = _col_header(model_pairs)
    col_w = max(12, max(len(c) for c in col_labels) + 2)

    print(f"\n{'=' * 100}")
    print(f"RECALL@k BY MODEL{ds_label}")
    print("=" * 100)

    for model in models:
        dims = sorted({k[5] for k in agg if k[3] == model})
        for dim in dims:
            ds = next((k[4] for k in agg if k[3] == model and k[5] == dim), None)
            sizes = _get_sizes(agg, dim=dim, model=model, dataset=ds)
            if not sizes:
                continue

            print(f"\n  {model} (dim={dim})")
            header = f"  {'N':>10}"
            for label in col_labels:
                header += f" | {label:>{col_w}}"
            print(header)
            print(f"  {'-' * 10}" + f"-+-{'-' * col_w}" * len(col_labels))

            for n in sizes:
                row = f"  {n:>10,}"
                for engine, method in model_pairs:
                    val = _get_val(agg, engine, method, "model", model, ds, dim, n, "recall_at_k")
                    row += f" | {_fmt(val, '.1%'):>{col_w}}"
                print(row)


def print_model_storage_table(agg, models, active_pairs, ds_label=""):
    """Print database file size table grouped by model."""
    has_size = any(entry.get("db_size_bytes_mean") is not None for entry in agg.values())
    if not has_size:
        return

    model_pairs = [(e, m) for e, m in active_pairs if any(k[0] == e and k[1] == m and k[3] is not None for k in agg)]
    if not model_pairs:
        return

    col_labels = _col_header(model_pairs)
    col_w = max(14, max(len(c) for c in col_labels) + 2)

    print(f"\n{'=' * 100}")
    print(f"DATABASE FILE SIZE BY MODEL (disk storage){ds_label}")
    print("=" * 100)

    for model in models:
        dims = sorted({k[5] for k in agg if k[3] == model})
        for dim in dims:
            ds = next((k[4] for k in agg if k[3] == model and k[5] == dim), None)
            sizes = _get_sizes(agg, dim=dim, model=model, dataset=ds)
            if not sizes:
                continue

            has_any = any(
                _get_val(agg, e, m, "model", model, ds, dim, n, "db_size_bytes") is not None
                for e, m in model_pairs
                for n in sizes
            )
            if not has_any:
                continue

            print(f"\n  {model} (dim={dim})")
            header = f"  {'N':>10}"
            for label in col_labels:
                header += f" | {label:>{col_w}}"
            print(header)
            print(f"  {'-' * 10}" + f"-+-{'-' * col_w}" * len(col_labels))

            for n in sizes:
                row = f"  {n:>10,}"
                for engine, method in model_pairs:
                    val = _get_val(agg, engine, method, "model", model, ds, dim, n, "db_size_bytes")
                    row += f" | {_fmt_bytes(val):>{col_w}}"
                print(row)


# ── Per-model charts ──────────────────────────────────────────────


def chart_model_tipping_point(agg):
    """Search latency vs N for each engine, one chart per model per dataset.

    This is the primary deliverable — shows where HNSW's O(log n) curve
    diverges from quantized scan's O(n) curve for real embeddings.
    Labels use simple {library}-{algorithm} since there's only one model per chart.
    """
    models = _get_models(agg)
    datasets = _get_datasets(agg) or [None]
    if not models:
        log.info("  No model data, skipping model tipping point charts")
        return

    for ds in datasets:
        for model in models:
            dims = sorted({k[5] for k in agg if k[3] == model and k[4] == ds})
            if not dims:
                continue
            dim = dims[0]

            fig = go.Figure()

            for engine, method in ENGINE_METHOD_PAIRS:
                sizes = sorted(
                    k[6]
                    for k in agg
                    if k[0] == engine and k[1] == method and k[3] == model and k[4] == ds and k[5] == dim
                )
                if not sizes:
                    continue

                latencies = [
                    _get_val(agg, engine, method, "model", model, ds, dim, n, "search_latency_ms") for n in sizes
                ]

                color = _make_color(engine, method)
                opacity = _trace_opacity(engine)
                line_width = 3 if engine == "muninn" else 2

                fig.add_trace(
                    go.Scatter(
                        x=sizes,
                        y=latencies,
                        mode="lines+markers",
                        name=_trace_label(engine, method),
                        line={"color": color, "width": line_width},
                        marker={"size": 8},
                        opacity=opacity,
                    )
                )

            ds_label = f" ({ds})" if ds else ""
            fig.update_layout(
                title=f"Search Latency — {dim}d-{model}{ds_label}",
                xaxis_title="Dataset Size (N)",
                yaxis_title="Search Latency (ms/query)",
                xaxis_type="log",
                yaxis_type="log",
                height=500,
                width=900,
                template="plotly_white",
                legend={"orientation": "v", "yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
            )

            ds_suffix = f"_{ds}" if ds else ""
            _save_chart(fig, f"tipping_point_{model}{ds_suffix}")


# ── Cross-model charts ────────────────────────────────────────────


def chart_model_comparison(agg):
    """Cross-model comparison: all models x all search methods on one chart.

    Labels: {library}-{algorithm}-{dims}d-{model}
    Colors: hue from library-algorithm, S/L from model (fiber bundle).
    Traces ordered: library-algorithm first, then model by dim ascending.
    """
    models_by_dim = _get_models_by_dim(agg)
    if len(models_by_dim) < 2:
        log.info("  Need >=2 models for comparison chart, skipping")
        return

    datasets = _get_datasets(agg) or [None]
    n_models = len(models_by_dim)

    for ds in datasets:
        fig = go.Figure()
        has_traces = False

        # Iterate engine first -> legend groups by library-algorithm
        for engine, method in ENGINE_METHOD_PAIRS:
            for model_idx, (model, dim) in enumerate(models_by_dim):
                sizes = sorted(
                    k[6]
                    for k in agg
                    if k[0] == engine and k[1] == method and k[3] == model and k[4] == ds and k[5] == dim
                )
                if not sizes:
                    continue

                latencies = [
                    _get_val(agg, engine, method, "model", model, ds, dim, n, "search_latency_ms") for n in sizes
                ]

                color = _make_color(engine, method, model_idx, n_models)
                opacity = _trace_opacity(engine)
                line_width = 3 if engine == "muninn" else 2

                fig.add_trace(
                    go.Scatter(
                        x=sizes,
                        y=latencies,
                        mode="lines+markers",
                        name=_trace_label(engine, method, model, dim),
                        line={"color": color, "width": line_width},
                        marker={"size": 7},
                        opacity=opacity,
                        legendgroup=_engine_label(engine, method),
                        legendgrouptitle_text=_engine_label(engine, method),
                    )
                )
                has_traces = True

        if not has_traces:
            continue

        ds_label = f" ({ds})" if ds else ""
        fig.update_layout(
            title=f"Search Latency Scaling by Embedding Model{ds_label}",
            xaxis_title="Dataset Size (N)",
            yaxis_title="Search Latency (ms/query)",
            xaxis_type="log",
            yaxis_type="log",
            height=550,
            width=1000,
            template="plotly_white",
            legend={
                "orientation": "v",
                "yanchor": "top",
                "y": 0.99,
                "xanchor": "left",
                "x": 1.02,
                "groupclick": "togglegroup",
            },
        )

        ds_suffix = f"_{ds}" if ds else ""
        _save_chart(fig, f"model_comparison{ds_suffix}")


def chart_model_recall(agg):
    """Recall@k vs N for each model, all approximate engines.

    Fiber bundle: hue from engine, S/L from model.
    """
    models_by_dim = _get_models_by_dim(agg)
    if not models_by_dim:
        return

    datasets = _get_datasets(agg) or [None]
    n_models = len(models_by_dim)

    for ds in datasets:
        fig = go.Figure()
        has_traces = False

        for engine, method in ENGINE_METHOD_PAIRS:
            for model_idx, (model, dim) in enumerate(models_by_dim):
                sizes = sorted(
                    k[6]
                    for k in agg
                    if k[0] == engine and k[1] == method and k[3] == model and k[4] == ds and k[5] == dim
                )
                if not sizes:
                    continue

                recalls = [_get_val(agg, engine, method, "model", model, ds, dim, n, "recall_at_k") for n in sizes]

                color = _make_color(engine, method, model_idx, n_models)
                opacity = _trace_opacity(engine)
                line_width = 3 if engine == "muninn" else 2

                fig.add_trace(
                    go.Scatter(
                        x=sizes,
                        y=recalls,
                        mode="lines+markers",
                        name=_trace_label(engine, method, model, dim),
                        line={"color": color, "width": line_width},
                        marker={"size": 7},
                        opacity=opacity,
                        legendgroup=_engine_label(engine, method),
                        legendgrouptitle_text=_engine_label(engine, method),
                    )
                )
                has_traces = True

        if not has_traces:
            continue

        ds_label = f" ({ds})" if ds else ""
        fig.update_layout(
            title=f"Recall@k vs Dataset Size (Real Embeddings){ds_label}",
            xaxis_title="Dataset Size (N)",
            yaxis_title="Recall@k",
            xaxis_type="log",
            yaxis={"range": [0.9, 1.01]},
            height=500,
            width=1000,
            template="plotly_white",
            legend={"orientation": "v", "yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
        )

        ds_suffix = f"_{ds}" if ds else ""
        _save_chart(fig, f"recall_models{ds_suffix}")


def chart_model_insert_throughput(agg):
    """Insert throughput vs N, fiber-bundle style.

    Legend groups by library-algorithm, with model shading within each bundle.
    """
    models_by_dim = _get_models_by_dim(agg)
    if not models_by_dim:
        return

    datasets = _get_datasets(agg) or [None]
    n_models = len(models_by_dim)

    for ds in datasets:
        fig = go.Figure()
        has_traces = False

        # Engine first -> legend groups semantically
        for engine, method in ENGINE_METHOD_PAIRS:
            for model_idx, (model, dim) in enumerate(models_by_dim):
                sizes = sorted(
                    k[6]
                    for k in agg
                    if k[0] == engine and k[1] == method and k[3] == model and k[4] == ds and k[5] == dim
                )
                if not sizes:
                    continue

                rates = [_get_val(agg, engine, method, "model", model, ds, dim, n, "insert_rate_vps") for n in sizes]

                color = _make_color(engine, method, model_idx, n_models)
                opacity = _trace_opacity(engine)
                line_width = 3 if engine == "muninn" else 2

                fig.add_trace(
                    go.Scatter(
                        x=sizes,
                        y=rates,
                        mode="lines+markers",
                        name=_trace_label(engine, method, model, dim),
                        line={"color": color, "width": line_width},
                        marker={"size": 7},
                        opacity=opacity,
                        legendgroup=_engine_label(engine, method),
                        legendgrouptitle_text=_engine_label(engine, method),
                    )
                )
                has_traces = True

        if not has_traces:
            continue

        ds_label = f" ({ds})" if ds else ""
        fig.update_layout(
            title=f"Insert Throughput vs Dataset Size (Real Embeddings){ds_label}",
            xaxis_title="Dataset Size (N)",
            yaxis_title="Throughput (vectors/sec)",
            xaxis_type="log",
            yaxis_type="log",
            height=500,
            width=1000,
            template="plotly_white",
            legend={"orientation": "v", "yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
        )

        ds_suffix = f"_{ds}" if ds else ""
        _save_chart(fig, f"insert_throughput_models{ds_suffix}")


def chart_model_db_size(agg):
    """Database file size vs N, fiber-bundle style (disk storage only)."""
    models_by_dim = _get_models_by_dim(agg)
    if not models_by_dim:
        return

    has_size = any(entry.get("db_size_bytes_mean") is not None for key, entry in agg.items() if key[3] is not None)
    if not has_size:
        log.info("  No db_size data for models, skipping chart")
        return

    datasets = _get_datasets(agg) or [None]
    n_models = len(models_by_dim)

    for ds in datasets:
        fig = go.Figure()
        has_traces = False

        for engine, method in ENGINE_METHOD_PAIRS:
            for model_idx, (model, dim) in enumerate(models_by_dim):
                sizes = sorted(
                    k[6]
                    for k in agg
                    if k[0] == engine and k[1] == method and k[3] == model and k[4] == ds and k[5] == dim
                )
                if not sizes:
                    continue

                db_sizes = [_get_val(agg, engine, method, "model", model, ds, dim, n, "db_size_bytes") for n in sizes]

                if all(v is None for v in db_sizes):
                    continue

                db_sizes_mb = [v / (1024 * 1024) if v is not None else None for v in db_sizes]

                color = _make_color(engine, method, model_idx, n_models)
                opacity = _trace_opacity(engine)
                line_width = 3 if engine == "muninn" else 2

                fig.add_trace(
                    go.Scatter(
                        x=sizes,
                        y=db_sizes_mb,
                        mode="lines+markers",
                        name=_trace_label(engine, method, model, dim),
                        line={"color": color, "width": line_width},
                        marker={"size": 7},
                        opacity=opacity,
                        legendgroup=_engine_label(engine, method),
                        legendgrouptitle_text=_engine_label(engine, method),
                    )
                )
                has_traces = True

        if not has_traces:
            continue

        ds_label = f" ({ds})" if ds else ""
        fig.update_layout(
            title=f"Database File Size vs Dataset Size (Real Embeddings){ds_label}",
            xaxis_title="Dataset Size (N)",
            yaxis_title="File Size (MB)",
            xaxis_type="log",
            yaxis_type="log",
            height=500,
            width=1000,
            template="plotly_white",
            legend={"orientation": "v", "yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
        )

        ds_suffix = f"_{ds}" if ds else ""
        _save_chart(fig, f"db_size_models{ds_suffix}")


def chart_dataset_comparison(agg):
    """Cross-dataset comparison: overlay AG News vs Wealth of Nations latency curves.

    One chart per model. Shows how embedding topology affects search performance.
    """
    models = _get_models(agg)
    datasets = _get_datasets(agg)
    if len(datasets) < 2 or not models:
        log.info("  Need >=2 datasets for cross-dataset comparison, skipping")
        return

    ds_dash = dict(zip(datasets, ["solid", "dash", "dot", "dashdot"], strict=False))

    for model in models:
        dims = sorted({k[5] for k in agg if k[3] == model})
        if not dims:
            continue
        dim = dims[0]

        fig = go.Figure()
        has_traces = False

        for engine, method in [("muninn", "hnsw"), ("sqlite_vector", "quantize_scan")]:
            for ds in datasets:
                sizes = sorted(
                    k[6]
                    for k in agg
                    if k[0] == engine and k[1] == method and k[3] == model and k[4] == ds and k[5] == dim
                )
                if not sizes:
                    continue

                latencies = [
                    _get_val(agg, engine, method, "model", model, ds, dim, n, "search_latency_ms") for n in sizes
                ]

                color = _make_color(engine, method)
                opacity = _trace_opacity(engine)
                line_width = 3 if engine == "muninn" else 2
                dash = ds_dash.get(ds, "solid")

                fig.add_trace(
                    go.Scatter(
                        x=sizes,
                        y=latencies,
                        mode="lines+markers",
                        name=f"{_engine_label(engine, method)} ({ds})",
                        line={"color": color, "width": line_width, "dash": dash},
                        marker={"size": 7},
                        opacity=opacity,
                    )
                )
                has_traces = True

        if not has_traces:
            continue

        fig.update_layout(
            title=f"Dataset Comparison — {dim}d-{model}",
            xaxis_title="Dataset Size (N)",
            yaxis_title="Search Latency (ms/query)",
            xaxis_type="log",
            yaxis_type="log",
            height=500,
            width=900,
            template="plotly_white",
            legend={"orientation": "v", "yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
        )

        _save_chart(fig, f"dataset_comparison_{model}")


def generate_all_charts(agg):
    """Generate all Plotly HTML + JSON charts."""
    log.info("Generating charts...")

    # Model-centric charts (primary)
    chart_model_tipping_point(agg)
    chart_model_comparison(agg)
    chart_model_recall(agg)
    chart_model_insert_throughput(agg)
    chart_model_db_size(agg)
    chart_dataset_comparison(agg)

    log.info("All charts generated in %s", CHARTS_DIR)


# ── Doc generation ───────────────────────────────────────────────

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
DOCS_DIR = PROJECT_ROOT / "docs" / "benchmarks"

MODEL_USE_CASES = {
    "MiniLM": "Fast, lightweight semantic search",
    "MPNet": "Balanced quality/speed",
    "BGE-Large": "High-quality retrieval",
}

MODEL_URLS = {
    "MiniLM": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
    "BGE-Large": "https://huggingface.co/BAAI/bge-large-en-v1.5",
}


def generate_docs():
    """Render vss.md.j2 template to docs/benchmarks/vss.md."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        keep_trailing_newline=True,
    )
    template = env.get_template("vss.md.j2")

    models = [
        {
            "label": label,
            "model_id": info["model_id"],
            "dim": info["dim"],
            "use_case": MODEL_USE_CASES.get(label, ""),
            "url": MODEL_URLS.get(label, ""),
        }
        for label, info in EMBEDDING_MODELS.items()
    ]

    datasets = [
        {
            "key": "ag_news",
            "label": "AG News",
            "url": "https://huggingface.co/datasets/fancyzhx/ag_news",
            "source": "HuggingFace",
            "passages": "~120K",
            "topology": "4 discrete clusters",
        },
        {
            "key": "wealth_of_nations",
            "label": "Wealth of Nations",
            "url": "https://www.gutenberg.org/ebooks/3300",
            "source": "Project Gutenberg",
            "passages": "~2,500 (256-token windows, 50-token overlap)",
            "topology": "Continuous conceptual gradient",
        },
    ]

    output = template.render(models=models, datasets=datasets)
    out_path = DOCS_DIR / "vss.md"
    out_path.write_text(output, encoding="utf-8")
    log.info("Generated docs: %s", out_path)


# ── Manifest ──────────────────────────────────────────────────────


def _npy_cache_path(model_label, dataset):
    """Return .npy cache path for a model+dataset."""
    return VECTORS_DIR / f"{model_label}_{dataset}.npy"


def _npy_info(model_label, dataset):
    """Check .npy cache. Returns (n_vectors, file_size_bytes) or None."""
    path = _npy_cache_path(model_label, dataset)
    if not path.exists():
        return None
    arr = np.load(str(path), mmap_mode="r")
    return len(arr), path.stat().st_size


def _clamp_sizes(sizes, max_n):
    """Clamp sizes to max_n and deduplicate, preserving order."""
    seen = set()
    result = []
    for s in sizes:
        clamped = min(s, max_n)
        if clamped not in seen:
            seen.add(clamped)
            result.append(clamped)
    return result


def vss_manifest():
    """Enumerate all achievable VSS scenarios from PROFILES.

    Sizes are clamped to dataset limits (from .npy cache) and memory
    limits (MAX_N_BY_DIM), so only feasible permutations are listed.

    Returns dict mapping profile_name -> list of dicts with:
        scenario, engine, pattern, source, sizes, and optionally dim, dataset.
    """
    manifest = {}
    for profile_name, profile in PROFILES.items():
        scenarios = []
        datasets = profile.get("datasets", ["ag_news"])
        for dataset in datasets:
            for model_label, model_info in EMBEDDING_MODELS.items():
                dim = model_info["dim"]
                source_str = f"model:{model_info['model_id']}"
                # Clamp by dataset size (from cache) and memory budget
                info = _npy_info(model_label, dataset)
                max_n_data = info[0] if info else max(profile["sizes"])
                max_n_mem = MAX_N_BY_DIM.get(dim, 100_000)
                max_n = min(max_n_data, max_n_mem)
                valid_sizes = _clamp_sizes(profile["sizes"], max_n)
                for n in valid_sizes:
                    scenario = make_scenario_name("model", model_label, dataset, dim, n)
                    for engine in ALL_ENGINES:
                        pattern = f"benchmarks/results/{scenario}/*_{engine}.sqlite"
                        scenarios.append(
                            {
                                "scenario": scenario,
                                "engine": engine,
                                "pattern": pattern,
                                "source": source_str,
                                "sizes": n,
                                "dataset": dataset,
                            }
                        )
        manifest[profile_name] = scenarios
    return manifest


def check_vss_completeness():
    """Check which VSS scenarios have SQLite result files.

    Returns dict mapping (scenario_name, engine) -> list of discovered filenames.
    """
    manifest = vss_manifest()
    status = {}
    for scenarios in manifest.values():
        for entry in scenarios:
            scenario, engine = entry["scenario"], entry["engine"]
            key = (scenario, engine)
            if key in status:
                continue
            scenario_dir = VSS_RESULTS_DIR / scenario
            if scenario_dir.exists():
                matches = sorted(scenario_dir.glob(f"*_{engine}.sqlite"))
                status[key] = [m.name for m in matches]
            else:
                status[key] = []
    return status


def _print_prep_manifest():
    """Print models-prep cache status. Returns (done_count, total_count)."""
    datasets = PROFILES["models"].get("datasets", ["ag_news"])

    print("\n--- Models Prep Cache ---")
    done = 0
    total = 0
    for dataset in datasets:
        for model_label in EMBEDDING_MODELS:
            total += 1
            path = _npy_cache_path(model_label, dataset)
            info = _npy_info(model_label, dataset)
            rel_path = path.relative_to(PROJECT_ROOT)
            if info:
                done += 1
                n_vectors, file_size = info
                print(f"  [DONE] {model_label} / {dataset}")
                print(f"         target: {rel_path}")
                print(f"         found:  {n_vectors} vectors, {_fmt_bytes(file_size)}")
            else:
                print(f"  [MISS] {model_label} / {dataset}")
                print(f"         target: {rel_path}")
    return done, total


def print_prep_manifest_report():
    """Print standalone models-prep completeness report."""
    print("\n=== Models Prep Manifest ===")
    done, total = _print_prep_manifest()
    print(f"\nSummary: {done}/{total} caches ready, {total - done} missing")


def _build_vss_cmd(entry, storage=None):
    """Build a project-root-relative CLI command for a single VSS benchmark."""
    parts = [
        ".venv/bin/python benchmarks/scripts/benchmark_vss.py",
        f"--source {entry['source']}",
        f"--sizes {entry['sizes']}",
    ]
    if entry.get("dataset"):
        parts.append(f"--dataset {entry['dataset']}")
    parts.append(f"--engine {entry['engine']}")
    if storage:
        parts.append(f"--storage {storage}")
    return " ".join(parts)


def print_manifest_report(mode="all", storage=None, limit=0):
    """Print JSONL manifest to stdout, summary to stderr.

    Each line is a self-contained JSON object with status, parameters, and cmd.
    Summary counts go to stderr so stdout stays clean for piping to jq.
    """
    manifest = vss_manifest()
    completeness = check_vss_completeness()

    total_done = 0
    total_missing = 0
    emitted = 0

    for profile_name, scenarios in manifest.items():
        for entry in scenarios:
            scenario, engine = entry["scenario"], entry["engine"]
            found = completeness.get((scenario, engine), [])
            is_done = bool(found)

            if is_done:
                total_done += 1
            else:
                total_missing += 1

            # Filter by mode
            if mode == "missing" and is_done:
                continue
            if mode == "done" and not is_done:
                continue

            obj = {
                "status": "done" if is_done else "missing",
                "profile": profile_name,
                "scenario": scenario,
                "engine": engine,
                "source": entry["source"],
                "sizes": entry["sizes"],
            }
            if entry.get("dataset"):
                obj["dataset"] = entry["dataset"]
            if is_done:
                obj["found"] = found
            obj["cmd"] = _build_vss_cmd(entry, storage)

            print(json.dumps(obj))
            emitted += 1

            if limit > 0 and emitted >= limit:
                break
        if limit > 0 and emitted >= limit:
            break

    total = total_done + total_missing
    print(f"VSS manifest: {total_done}/{total} done, {total_missing} missing (emitted {emitted})", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze benchmark results and generate charts")
    manifest_group = parser.add_mutually_exclusive_group()
    manifest_group.add_argument("--manifest-missing", action="store_true", help="Show only missing benchmarks (JSONL)")
    manifest_group.add_argument("--manifest-done", action="store_true", help="Show only completed benchmarks (JSONL)")
    manifest_group.add_argument("--manifest-all", action="store_true", help="Show all benchmarks (JSONL)")
    manifest_group.add_argument("--manifest", action="store_true", help="Alias for --manifest-all")
    parser.add_argument("--limit", type=int, default=0, help="Limit output to first N entries (0 = unlimited)")
    parser.add_argument("--storage", choices=["memory", "disk"], default=None, help="Filter by storage mode")
    parser.add_argument("--prep-manifest", action="store_true", help="Show models-prep cache completeness")
    parser.add_argument("--filter-source", help="Filter by vector source (e.g., 'model')")
    parser.add_argument("--filter-dim", type=int, help="Filter by dimension")
    parser.add_argument("--filter-model", help="Filter by model name (e.g., 'MiniLM', 'MPNet')")
    parser.add_argument("--filter-dataset", help="Filter by dataset (e.g., 'ag_news', 'wealth_of_nations')")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    args = parse_args()

    if args.prep_manifest:
        print_prep_manifest_report()
        return

    manifest_mode = None
    if args.manifest_all or args.manifest:
        manifest_mode = "all"
    elif args.manifest_missing:
        manifest_mode = "missing"
    elif args.manifest_done:
        manifest_mode = "done"

    if manifest_mode:
        print_manifest_report(mode=manifest_mode, storage=args.storage, limit=args.limit)
        return

    records = load_all_results(
        filter_source=args.filter_source,
        filter_dim=args.filter_dim,
        filter_model=args.filter_model,
        filter_dataset=args.filter_dataset,
    )

    if not records:
        return

    agg = aggregate(records)
    log.info("Aggregated into %d groups", len(agg))

    print_tables(agg)
    generate_all_charts(agg)
    generate_docs()


if __name__ == "__main__":
    main()
