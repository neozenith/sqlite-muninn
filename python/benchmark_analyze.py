"""
Benchmark analysis: aggregate JSONL results into text tables and Plotly JSON charts.

Reads all benchmarks/results/*.jsonl files, aggregates by
(engine, search_method, vector_source, model_name, dim, n), and produces:

1. Text tables: model comparison, search latency, insert throughput, recall, storage
2. Plotly JSON charts: per-model tipping point, cross-model comparison, recall, storage

Naming convention for chart labels:
    {library}-{algorithm}                        — per-model charts (single model)
    {library}-{algorithm}-{dims}d-{model}        — cross-model charts (multiple models)

Libraries:
    vec_graph-hnsw              — this project's HNSW index
    sqlite-vector-quantize      — sqliteai/sqlite-vector quantized approximate search
    sqlite-vector-fullscan      — sqliteai/sqlite-vector brute-force exact search

Usage:
    python python/benchmark_analyze.py
    python python/benchmark_analyze.py --filter-source model
    python python/benchmark_analyze.py --filter-model MiniLM
"""
import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
CHARTS_DIR = PROJECT_ROOT / "benchmarks" / "charts"


# ── Data loading ──────────────────────────────────────────────────


def load_all_results(filter_source=None, filter_dim=None, filter_model=None):
    """Load all JSONL files from results directory, applying optional filters."""
    records = []
    jsonl_files = sorted(RESULTS_DIR.glob("*.jsonl"))

    if not jsonl_files:
        log.error("No JSONL files found in %s", RESULTS_DIR)
        log.error("Run 'make benchmark-models' or 'make benchmark-small' first.")
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

            records.append(record)

    log.info("Loaded %d records", len(records))
    return records


# ── Aggregation ───────────────────────────────────────────────────


def aggregate(records):
    """Group records and compute mean/stddev for each metric.

    Groups by (engine, search_method, vector_source, model_name, dim, n).
    Returns dict mapping group key -> aggregated metrics dict.
    """
    groups = defaultdict(list)

    for r in records:
        key = (
            r["engine"],
            r["search_method"],
            r.get("vector_source", "random"),
            r.get("model_name"),
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
        "relative_contrast",
        "distance_cv",
        "nearest_farthest_ratio",
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
# Key format: (engine, method, source, model_name, dim, n)


def _get_models_by_dim(agg):
    """Get models sorted by dimension (ascending).

    Returns list of (model_name, dim) tuples.
    """
    model_dims = {}
    for k in agg:
        if k[3] is not None:
            model_dims[k[3]] = k[4]
    return sorted(model_dims.items(), key=lambda pair: pair[1])


def _get_models(agg):
    """Get model names sorted by dimension (ascending)."""
    return [m for m, _ in _get_models_by_dim(agg)]


def _get_dims(agg):
    """Get sorted unique dimensions."""
    return sorted(set(k[4] for k in agg))


def _get_sizes(agg, dim=None, model=None):
    """Get sorted unique dataset sizes, optionally filtered."""
    sizes = set()
    for k in agg:
        if dim is not None and k[4] != dim:
            continue
        if model is not None and k[3] != model:
            continue
        sizes.add(k[5])
    return sorted(sizes)


def _get_val(agg, engine, method, source, model, dim, n, metric):
    """Safely get a mean metric value from aggregated data."""
    key = (engine, method, source, model, dim, n)
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


# ── Text tables ───────────────────────────────────────────────────


def print_tables(agg):
    """Print all text summary tables."""
    models = _get_models(agg)
    has_random = any(k[2] == "random" for k in agg)

    if models:
        print_model_overview(agg, models)
        print_model_search_table(agg, models)
        print_model_insert_table(agg, models)
        print_model_recall_table(agg, models)
        print_model_storage_table(agg, models)

    if has_random:
        print_random_search_table(agg)

    print_saturation_table(agg)


def print_model_overview(agg, models):
    """Print a high-level model comparison summary."""
    print("\n" + "=" * 100)
    print("EMBEDDING MODEL OVERVIEW")
    print("  vg = vec_graph-hnsw | sv-q = sqlite-vector-quantize | sv-f = sqlite-vector-fullscan")
    print("=" * 100)
    print(f"  {'Model':>12} | {'Dim':>5} | {'Sizes Tested':>30} | {'vg-hnsw wins at max N?':>22}")
    print(f"  {'-'*12}-+-{'-'*5}-+-{'-'*30}-+-{'-'*22}")

    for model in models:
        dims = sorted(set(k[4] for k in agg if k[3] == model))
        sizes = sorted(set(k[5] for k in agg if k[3] == model))
        dim = dims[0] if dims else 0
        sizes_str = ", ".join(f"{s:,}" for s in sizes)

        largest_n = max(sizes) if sizes else 0
        hnsw_lat = _get_val(agg, "vec_graph", "hnsw", "model", model, dim, largest_n, "search_latency_ms")
        qscan_lat = _get_val(agg, "sqlite_vector", "quantize_scan", "model", model, dim, largest_n, "search_latency_ms")
        wins = ""
        if hnsw_lat is not None and qscan_lat is not None:
            speedup = qscan_lat / hnsw_lat
            wins = f"YES ({speedup:.0f}x)" if hnsw_lat < qscan_lat else "no"

        print(f"  {model:>12} | {dim:>5} | {sizes_str:>30} | {wins:>22}")


def print_model_search_table(agg, models):
    """Print search latency table grouped by model."""
    print("\n" + "=" * 100)
    print("SEARCH LATENCY BY MODEL (ms/query)")
    print("  vg-hnsw = vec_graph-hnsw | sv-quantize = sqlite-vector-quantize | sv-fullscan = sqlite-vector-fullscan")
    print("=" * 100)

    for model in models:
        dims = sorted(set(k[4] for k in agg if k[3] == model))
        for dim in dims:
            sizes = _get_sizes(agg, dim=dim, model=model)
            if not sizes:
                continue

            print(f"\n  {model} (dim={dim})")
            print(f"  {'N':>10} | {'vg-hnsw':>10} | {'sv-quantize':>12} | {'sv-fullscan':>12} | {'vg speedup':>10}")
            print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

            for n in sizes:
                hnsw = _get_val(agg, "vec_graph", "hnsw", "model", model, dim, n, "search_latency_ms")
                qscan = _get_val(agg, "sqlite_vector", "quantize_scan", "model", model, dim, n, "search_latency_ms")
                fscan = _get_val(agg, "sqlite_vector", "full_scan", "model", model, dim, n, "search_latency_ms")

                speedup = ""
                if hnsw is not None and qscan is not None and hnsw > 0:
                    speedup = f"{qscan / hnsw:.1f}x"

                print(
                    f"  {n:>10,} | {_fmt(hnsw):>10} | {_fmt(qscan):>12} | {_fmt(fscan):>12} | {speedup:>10}"
                )


def print_model_insert_table(agg, models):
    """Print insert throughput table grouped by model."""
    print("\n" + "=" * 100)
    print("INSERT THROUGHPUT BY MODEL (vectors/sec)")
    print("=" * 100)

    for model in models:
        dims = sorted(set(k[4] for k in agg if k[3] == model))
        for dim in dims:
            sizes = _get_sizes(agg, dim=dim, model=model)
            if not sizes:
                continue

            print(f"\n  {model} (dim={dim})")
            print(f"  {'N':>10} | {'vec_graph':>12} | {'sqlite-vector':>14}")
            print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*14}")

            for n in sizes:
                vg = _get_val(agg, "vec_graph", "hnsw", "model", model, dim, n, "insert_rate_vps")
                sv = _get_val(agg, "sqlite_vector", "quantize_scan", "model", model, dim, n, "insert_rate_vps")

                print(f"  {n:>10,} | {_fmt(vg, ',.0f'):>12} | {_fmt(sv, ',.0f'):>14}")


def print_model_recall_table(agg, models):
    """Print recall@k table grouped by model."""
    print("\n" + "=" * 100)
    print("RECALL@k BY MODEL")
    print("=" * 100)

    for model in models:
        dims = sorted(set(k[4] for k in agg if k[3] == model))
        for dim in dims:
            sizes = _get_sizes(agg, dim=dim, model=model)
            if not sizes:
                continue

            print(f"\n  {model} (dim={dim})")
            print(f"  {'N':>10} | {'vg-hnsw':>10} | {'sv-quantize':>12} | {'sv-fullscan':>12}")
            print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")

            for n in sizes:
                hnsw = _get_val(agg, "vec_graph", "hnsw", "model", model, dim, n, "recall_at_k")
                qscan = _get_val(agg, "sqlite_vector", "quantize_scan", "model", model, dim, n, "recall_at_k")
                fscan = _get_val(agg, "sqlite_vector", "full_scan", "model", model, dim, n, "recall_at_k")

                print(
                    f"  {n:>10,} | {_fmt(hnsw, '.1%'):>10} | {_fmt(qscan, '.1%'):>12} | {_fmt(fscan, '.1%'):>12}"
                )


def print_model_storage_table(agg, models):
    """Print database file size table grouped by model."""
    has_size = any(entry.get("db_size_bytes_mean") is not None for entry in agg.values())
    if not has_size:
        return

    print("\n" + "=" * 100)
    print("DATABASE FILE SIZE BY MODEL (disk storage)")
    print("=" * 100)

    for model in models:
        dims = sorted(set(k[4] for k in agg if k[3] == model))
        for dim in dims:
            sizes = _get_sizes(agg, dim=dim, model=model)
            if not sizes:
                continue

            vg_any = any(
                _get_val(agg, "vec_graph", "hnsw", "model", model, dim, n, "db_size_bytes") is not None
                for n in sizes
            )
            sv_any = any(
                _get_val(agg, "sqlite_vector", "quantize_scan", "model", model, dim, n, "db_size_bytes") is not None
                for n in sizes
            )
            if not vg_any and not sv_any:
                continue

            print(f"\n  {model} (dim={dim})")
            print(f"  {'N':>10} | {'vec_graph':>12} | {'sqlite-vector':>14} | {'ratio':>8}")
            print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*14}-+-{'-'*8}")

            for n in sizes:
                vg = _get_val(agg, "vec_graph", "hnsw", "model", model, dim, n, "db_size_bytes")
                sv = _get_val(agg, "sqlite_vector", "quantize_scan", "model", model, dim, n, "db_size_bytes")

                ratio = ""
                if vg is not None and sv is not None and sv > 0:
                    ratio = f"{vg / sv:.1f}x"

                print(
                    f"  {n:>10,} | {_fmt_bytes(vg):>12} | {_fmt_bytes(sv):>14} | {ratio:>8}"
                )


def print_random_search_table(agg):
    """Print search latency table for random vectors (if present)."""
    random_keys = [k for k in agg if k[2] == "random"]
    if not random_keys:
        return

    print("\n" + "=" * 100)
    print("SEARCH LATENCY — RANDOM VECTORS (ms/query)")
    print("=" * 100)

    dims = sorted(set(k[4] for k in random_keys))
    for dim in dims:
        sizes = sorted(set(k[5] for k in random_keys if k[4] == dim))
        if not sizes:
            continue

        print(f"\n  dim={dim}")
        print(f"  {'N':>10} | {'vg-hnsw':>10} | {'sv-quantize':>12} | {'sv-fullscan':>12} | {'vg wins?':>10}")
        print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

        for n in sizes:
            hnsw = _get_val(agg, "vec_graph", "hnsw", "random", None, dim, n, "search_latency_ms")
            qscan = _get_val(agg, "sqlite_vector", "quantize_scan", "random", None, dim, n, "search_latency_ms")
            fscan = _get_val(agg, "sqlite_vector", "full_scan", "random", None, dim, n, "search_latency_ms")

            winner = ""
            if hnsw is not None and qscan is not None:
                winner = "YES" if hnsw < qscan else "no"

            print(
                f"  {n:>10,} | {_fmt(hnsw):>10} | {_fmt(qscan):>12} | {_fmt(fscan):>12} | {winner:>10}"
            )


def print_saturation_table(agg):
    """Print saturation analysis table."""
    sat_data = {}
    for key, entry in agg.items():
        dim = key[4]
        model = key[3]
        rc = entry.get("relative_contrast_mean")
        cv = entry.get("distance_cv_mean")
        nf = entry.get("nearest_farthest_ratio_mean")
        if rc is not None:
            label = f"{dim}d-{model}" if model else f"{dim}d-random"
            if label not in sat_data:
                sat_data[label] = {"rc": [], "cv": [], "nf": [], "dim": dim}
            sat_data[label]["rc"].append(rc)
            if cv is not None:
                sat_data[label]["cv"].append(cv)
            if nf is not None:
                sat_data[label]["nf"].append(nf)

    if not sat_data:
        return

    print("\n" + "=" * 100)
    print("SATURATION ANALYSIS (curse of dimensionality)")
    print("  RC → 1.0 = saturated | CV → 0 = saturated | NF → 1.0 = saturated")
    print("=" * 100)
    print(f"  {'Source':>25} | {'Relative Contrast':>18} | {'Distance CV':>12} | {'Near/Far Ratio':>15}")
    print(f"  {'-'*25}-+-{'-'*18}-+-{'-'*12}-+-{'-'*15}")

    for label in sorted(sat_data.keys(), key=lambda l: sat_data[l]["dim"]):
        d = sat_data[label]
        rc_mean = sum(d["rc"]) / len(d["rc"]) if d["rc"] else None
        cv_mean = sum(d["cv"]) / len(d["cv"]) if d["cv"] else None
        nf_mean = sum(d["nf"]) / len(d["nf"]) if d["nf"] else None

        print(
            f"  {label:>25} | {_fmt(rc_mean, '.4f'):>18} | "
            f"{_fmt(cv_mean, '.4f'):>12} | {_fmt(nf_mean, '.4f'):>15}"
        )


# ── Plotly chart infrastructure ───────────────────────────────────
#
# Design rules:
#   1. Labels: {library}-{algorithm} or {library}-{algorithm}-{dims}d-{model}
#   2. Hue per library-algorithm, vary S/L per model → "fiber bundle" effect
#   3. Non-vec_graph traces at 80% opacity so vec_graph pops
#   4. Legend ordered: library-algorithm first, then model (by dim ascending)


ENGINE_METHOD_PAIRS = [
    ("vec_graph", "hnsw"),
    ("sqlite_vector", "quantize_scan"),
    ("sqlite_vector", "full_scan"),
]

# Library-algorithm labels
ENGINE_LABELS = {
    ("vec_graph", "hnsw"): "vec_graph-hnsw",
    ("sqlite_vector", "quantize_scan"): "sqlite-vector-quantize",
    ("sqlite_vector", "full_scan"): "sqlite-vector-fullscan",
}

# Base hue per library-algorithm (HSL hue degrees)
ENGINE_HUES = {
    ("vec_graph", "hnsw"): 270,            # purple
    ("sqlite_vector", "quantize_scan"): 175,  # teal
    ("sqlite_vector", "full_scan"): 18,       # warm orange
}


def _engine_label(engine, method):
    """Library-algorithm label: vec_graph-hnsw, sqlite-vector-quantize, etc."""
    return ENGINE_LABELS.get((engine, method), f"{engine}-{method}")


def _trace_label(engine, method, model=None, dim=None):
    """Full trace label for legends.

    Single-model charts: "vec_graph-hnsw"
    Cross-model charts:  "vec_graph-hnsw-384d-MiniLM"
    """
    base = _engine_label(engine, method)
    if model is not None and dim is not None:
        return f"{base}-{dim}d-{model}"
    return base


def _make_color(engine, method, model_idx=0, n_models=1):
    """Generate HSL color for a trace.

    Hue is fixed per library-algorithm.
    Saturation and luminance vary per model to create a fiber-bundle effect:
    smaller dim (idx=0) → lighter/more vivid, larger dim → deeper/richer.
    """
    hue = ENGINE_HUES.get((engine, method), 0)
    if n_models <= 1:
        sat, lum = 75, 45
    else:
        t = model_idx / (n_models - 1)
        sat = 85 - int(t * 15)   # 85% → 70%
        lum = 58 - int(t * 23)   # 58% → 35%
    return f"hsl({hue}, {sat}%, {lum}%)"


def _trace_opacity(engine):
    """vec_graph at full opacity, everything else softened to 80%."""
    return 1.0 if engine == "vec_graph" else 0.8


def _save_chart(fig, name):
    """Save a Plotly figure as standalone HTML and JSON for mkdocs embedding."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    html_path = CHARTS_DIR / f"{name}.html"
    fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)

    json_path = CHARTS_DIR / f"{name}.json"
    json_path.write_text(fig.to_json(), encoding="utf-8")

    log.info("  Chart saved: %s (.html + .json)", CHARTS_DIR / name)


# ── Per-model charts ──────────────────────────────────────────────


def chart_model_tipping_point(agg):
    """Search latency vs N for each engine, one chart per model.

    This is the primary deliverable — shows where HNSW's O(log n) curve
    diverges from quantized scan's O(n) curve for real embeddings.
    Labels use simple {library}-{algorithm} since there's only one model per chart.
    """
    models = _get_models(agg)
    if not models:
        log.info("  No model data, skipping model tipping point charts")
        return

    for model in models:
        dims = sorted(set(k[4] for k in agg if k[3] == model))
        if not dims:
            continue
        dim = dims[0]

        fig = go.Figure()

        for engine, method in ENGINE_METHOD_PAIRS:
            sizes = sorted(
                k[5] for k in agg
                if k[0] == engine and k[1] == method and k[3] == model and k[4] == dim
            )
            if not sizes:
                continue

            latencies = [
                _get_val(agg, engine, method, "model", model, dim, n, "search_latency_ms")
                for n in sizes
            ]

            color = _make_color(engine, method)
            opacity = _trace_opacity(engine)
            line_width = 3 if engine == "vec_graph" else 2

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

        fig.update_layout(
            title=f"Search Latency — {dim}d-{model}",
            xaxis_title="Dataset Size (N)",
            yaxis_title="Search Latency (ms/query)",
            xaxis_type="log",
            yaxis_type="log",
            height=500,
            width=900,
            template="plotly_white",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5},
        )

        _save_chart(fig, f"tipping_point_{model}")


# ── Cross-model charts ────────────────────────────────────────────


def chart_model_comparison(agg):
    """Cross-model comparison: all models × all search methods on one chart.

    Labels: {library}-{algorithm}-{dims}d-{model}
    Colors: hue from library-algorithm, S/L from model (fiber bundle).
    Traces ordered: library-algorithm first, then model by dim ascending.
    """
    models_by_dim = _get_models_by_dim(agg)
    if len(models_by_dim) < 2:
        log.info("  Need >=2 models for comparison chart, skipping")
        return

    n_models = len(models_by_dim)
    fig = go.Figure()

    # Iterate engine first → legend groups by library-algorithm
    for engine, method in [("vec_graph", "hnsw"), ("sqlite_vector", "quantize_scan")]:
        for model_idx, (model, dim) in enumerate(models_by_dim):
            sizes = sorted(
                k[5] for k in agg
                if k[0] == engine and k[1] == method and k[3] == model and k[4] == dim
            )
            if not sizes:
                continue

            latencies = [
                _get_val(agg, engine, method, "model", model, dim, n, "search_latency_ms")
                for n in sizes
            ]

            color = _make_color(engine, method, model_idx, n_models)
            opacity = _trace_opacity(engine)
            line_width = 3 if engine == "vec_graph" else 2

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

    fig.update_layout(
        title="Search Latency Scaling by Embedding Model",
        xaxis_title="Dataset Size (N)",
        yaxis_title="Search Latency (ms/query)",
        xaxis_type="log",
        yaxis_type="log",
        height=550,
        width=1000,
        template="plotly_white",
        legend={"orientation": "v", "yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02,
                "groupclick": "togglegroup"},
    )

    _save_chart(fig, "model_comparison")


def chart_model_recall(agg):
    """Recall@k vs N for each model, HNSW vs quantize_scan.

    Fiber bundle: hue from engine, S/L from model.
    """
    models_by_dim = _get_models_by_dim(agg)
    if not models_by_dim:
        return

    n_models = len(models_by_dim)
    fig = go.Figure()

    for engine, method in [("vec_graph", "hnsw"), ("sqlite_vector", "quantize_scan")]:
        for model_idx, (model, dim) in enumerate(models_by_dim):
            sizes = sorted(
                k[5] for k in agg
                if k[0] == engine and k[1] == method and k[3] == model and k[4] == dim
            )
            if not sizes:
                continue

            recalls = [
                _get_val(agg, engine, method, "model", model, dim, n, "recall_at_k")
                for n in sizes
            ]

            color = _make_color(engine, method, model_idx, n_models)
            opacity = _trace_opacity(engine)
            line_width = 3 if engine == "vec_graph" else 2

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

    fig.update_layout(
        title="Recall@k vs Dataset Size (Real Embeddings)",
        xaxis_title="Dataset Size (N)",
        yaxis_title="Recall@k",
        xaxis_type="log",
        yaxis={"range": [0.9, 1.01]},
        height=500,
        width=1000,
        template="plotly_white",
        legend={"orientation": "v", "yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
    )

    _save_chart(fig, "recall_models")


def chart_model_insert_throughput(agg):
    """Insert throughput vs N, fiber-bundle style.

    Legend groups by library-algorithm, with model shading within each bundle.
    """
    models_by_dim = _get_models_by_dim(agg)
    if not models_by_dim:
        return

    n_models = len(models_by_dim)
    fig = go.Figure()

    # Engine first → legend groups semantically
    for engine, method in [("vec_graph", "hnsw"), ("sqlite_vector", "quantize_scan")]:
        for model_idx, (model, dim) in enumerate(models_by_dim):
            sizes = sorted(
                k[5] for k in agg
                if k[0] == engine and k[1] == method and k[3] == model and k[4] == dim
            )
            if not sizes:
                continue

            rates = [
                _get_val(agg, engine, method, "model", model, dim, n, "insert_rate_vps")
                for n in sizes
            ]

            color = _make_color(engine, method, model_idx, n_models)
            opacity = _trace_opacity(engine)
            line_width = 3 if engine == "vec_graph" else 2

            fig.add_trace(
                go.Scatter(
                    x=sizes, y=rates,
                    mode="lines+markers",
                    name=_trace_label(engine, method, model, dim),
                    line={"color": color, "width": line_width},
                    marker={"size": 7},
                    opacity=opacity,
                    legendgroup=_engine_label(engine, method),
                    legendgrouptitle_text=_engine_label(engine, method),
                )
            )

    fig.update_layout(
        title="Insert Throughput vs Dataset Size (Real Embeddings)",
        xaxis_title="Dataset Size (N)",
        yaxis_title="Throughput (vectors/sec)",
        xaxis_type="log",
        yaxis_type="log",
        height=500,
        width=1000,
        template="plotly_white",
        legend={"orientation": "v", "yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
    )

    _save_chart(fig, "insert_throughput_models")


def chart_model_db_size(agg):
    """Database file size vs N, fiber-bundle style (disk storage only)."""
    models_by_dim = _get_models_by_dim(agg)
    if not models_by_dim:
        return

    has_size = any(
        entry.get("db_size_bytes_mean") is not None
        for key, entry in agg.items() if key[3] is not None
    )
    if not has_size:
        log.info("  No db_size data for models, skipping chart")
        return

    n_models = len(models_by_dim)
    fig = go.Figure()
    has_traces = False

    for engine, method in [("vec_graph", "hnsw"), ("sqlite_vector", "quantize_scan")]:
        for model_idx, (model, dim) in enumerate(models_by_dim):
            sizes = sorted(
                k[5] for k in agg
                if k[0] == engine and k[1] == method and k[3] == model and k[4] == dim
            )
            if not sizes:
                continue

            db_sizes = [
                _get_val(agg, engine, method, "model", model, dim, n, "db_size_bytes")
                for n in sizes
            ]

            if all(v is None for v in db_sizes):
                continue

            db_sizes_mb = [v / (1024 * 1024) if v is not None else None for v in db_sizes]

            color = _make_color(engine, method, model_idx, n_models)
            opacity = _trace_opacity(engine)
            line_width = 3 if engine == "vec_graph" else 2

            fig.add_trace(
                go.Scatter(
                    x=sizes, y=db_sizes_mb,
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
        return

    fig.update_layout(
        title="Database File Size vs Dataset Size (Real Embeddings)",
        xaxis_title="Dataset Size (N)",
        yaxis_title="File Size (MB)",
        xaxis_type="log",
        yaxis_type="log",
        height=500,
        width=1000,
        template="plotly_white",
        legend={"orientation": "v", "yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02},
    )

    _save_chart(fig, "db_size_models")


def chart_saturation(agg):
    """Saturation metrics by model/source as bar chart.

    X-axis labels use {dims}d-{model} format, colors from model's
    primary engine hue (purple for models, grey for random).
    """
    sat_by_label = defaultdict(lambda: {"rc": [], "cv": [], "nf": [], "dim": 0})

    for key, entry in agg.items():
        model = key[3]
        dim = key[4]
        rc = entry.get("relative_contrast_mean")
        cv = entry.get("distance_cv_mean")
        nf = entry.get("nearest_farthest_ratio_mean")
        if rc is None:
            continue

        label = f"{dim}d-{model}" if model else f"{dim}d-random"
        sat_by_label[label]["rc"].append(rc)
        sat_by_label[label]["dim"] = dim
        if cv is not None:
            sat_by_label[label]["cv"].append(cv)
        if nf is not None:
            sat_by_label[label]["nf"].append(nf)

    if not sat_by_label:
        log.info("  No saturation data available, skipping chart")
        return

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Relative Contrast (lower = less saturated)", "Distance CV (higher = less saturated)"],
        horizontal_spacing=0.1,
    )

    sorted_labels = sorted(sat_by_label.keys(), key=lambda l: sat_by_label[l]["dim"])
    n_bars = len(sorted_labels)

    for bar_idx, label in enumerate(sorted_labels):
        data = sat_by_label[label]
        rc_mean = sum(data["rc"]) / len(data["rc"])
        cv_mean = sum(data["cv"]) / len(data["cv"]) if data["cv"] else None

        # Use vec_graph purple hue for model bars, grey for random
        is_model = "random" not in label
        hue = 270 if is_model else 0
        sat = 75 if is_model else 0
        if n_bars <= 1:
            lum = 45
        else:
            t = bar_idx / (n_bars - 1)
            lum = 58 - int(t * 23)
        color = f"hsl({hue}, {sat}%, {lum}%)"

        fig.add_trace(
            go.Bar(x=[label], y=[rc_mean], name=label, marker_color=color, showlegend=False),
            row=1, col=1,
        )

        if cv_mean is not None:
            fig.add_trace(
                go.Bar(x=[label], y=[cv_mean], name=label, marker_color=color, showlegend=False),
                row=1, col=2,
            )

    fig.add_hline(y=1.0, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1,
                  annotation_text="saturated")
    fig.add_hline(y=0.0, line_dash="dash", line_color="red", opacity=0.5, row=1, col=2,
                  annotation_text="saturated")

    fig.update_layout(
        title="Vector Space Saturation by Embedding Model",
        height=450,
        width=900,
        template="plotly_white",
    )

    _save_chart(fig, "saturation")


# ── Random vector charts (secondary) ─────────────────────────────


def chart_random_tipping_point(agg):
    """Search latency vs N for random vectors (if data exists)."""
    random_keys = [k for k in agg if k[2] == "random"]
    if not random_keys:
        return

    dims = sorted(set(k[4] for k in random_keys))
    dim_dash = {32: "solid", 64: "dot", 128: "dash", 256: "dashdot",
                384: "solid", 512: "dot", 768: "dash", 1024: "dashdot", 1536: "longdash"}

    fig = go.Figure()

    for dim in dims:
        for engine, method in ENGINE_METHOD_PAIRS:
            sizes = sorted(
                k[5] for k in random_keys
                if k[0] == engine and k[1] == method and k[4] == dim
            )
            if not sizes:
                continue

            latencies = [
                _get_val(agg, engine, method, "random", None, dim, n, "search_latency_ms")
                for n in sizes
            ]

            color = _make_color(engine, method)
            opacity = _trace_opacity(engine)
            dash = dim_dash.get(dim, "solid")
            label = f"{_engine_label(engine, method)} d={dim}"

            fig.add_trace(
                go.Scatter(
                    x=sizes, y=latencies,
                    mode="lines+markers",
                    name=label,
                    line={"color": color, "dash": dash, "width": 2},
                    marker={"size": 6},
                    opacity=opacity,
                )
            )

    fig.update_layout(
        title="Search Latency vs Dataset Size (Random Vectors)",
        xaxis_title="Dataset Size (N)",
        yaxis_title="Search Latency (ms/query)",
        xaxis_type="log",
        yaxis_type="log",
        height=600,
        width=1000,
        template="plotly_white",
        legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.02},
    )

    _save_chart(fig, "tipping_point_random")


def generate_all_charts(agg):
    """Generate all Plotly HTML + JSON charts."""
    log.info("Generating charts...")

    # Model-centric charts (primary)
    chart_model_tipping_point(agg)
    chart_model_comparison(agg)
    chart_model_recall(agg)
    chart_model_insert_throughput(agg)
    chart_model_db_size(agg)
    chart_saturation(agg)

    # Random vector charts (secondary, if data exists)
    chart_random_tipping_point(agg)

    log.info("All charts generated in %s", CHARTS_DIR)


# ── CLI ───────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze benchmark results and generate charts")
    parser.add_argument("--filter-source", help="Filter by vector source (e.g., 'random', 'model')")
    parser.add_argument("--filter-dim", type=int, help="Filter by dimension")
    parser.add_argument("--filter-model", help="Filter by model name (e.g., 'MiniLM', 'MPNet')")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    args = parse_args()
    records = load_all_results(
        filter_source=args.filter_source,
        filter_dim=args.filter_dim,
        filter_model=args.filter_model,
    )

    if not records:
        return

    agg = aggregate(records)
    log.info("Aggregated into %d groups", len(agg))

    print_tables(agg)
    generate_all_charts(agg)


if __name__ == "__main__":
    main()
