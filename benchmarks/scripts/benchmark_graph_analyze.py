"""
Graph benchmark analysis: aggregate graph JSONL results into text tables and Plotly charts.

Reads all benchmarks/results/graph_*.jsonl files, aggregates by
(engine, operation, graph_model, n_nodes, avg_degree), and produces:

1. Text tables: per-operation engine comparison
2. Plotly JSON charts: query time vs n_nodes, scalability, setup time

Usage:
    python python/benchmark_graph_analyze.py
    python python/benchmark_graph_analyze.py --filter-operation bfs
    python python/benchmark_graph_analyze.py --filter-engine muninn
"""
import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import jinja2
import plotly.graph_objects as go

from benchmark_graph import ALL_GRAPH_ENGINES, ALL_OPERATIONS, GRAPH_PROFILES

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
CHARTS_DIR = PROJECT_ROOT / "benchmarks" / "charts"


# ── Engine registry ───────────────────────────────────────────────

ENGINE_LABELS = {
    "muninn": "muninn-tvf",
    "graphqlite": "graphqlite",
}

ENGINE_HUES = {
    "muninn": 270,      # purple
    "graphqlite": 340,     # pink
}

ENGINE_ORDER = ["muninn", "graphqlite"]

GM_SHORT = {
    "erdos_renyi": "ER",
    "barabasi_albert": "BA",
}

OPERATION_LABELS = {
    "bfs": "BFS",
    "dfs": "DFS",
    "shortest_path": "Shortest Path",
    "components": "Connected Components",
    "pagerank": "PageRank",
}


def _engine_label(engine):
    return ENGINE_LABELS.get(engine, engine)


def _trace_label(engine, gm=None, deg=None):
    """Full trace label for legends.

    Single variant: "muninn-tvf"
    With variant:   "muninn-tvf (ER, deg=5)"
    """
    base = _engine_label(engine)
    if gm is not None and deg is not None:
        return f"{base} ({GM_SHORT.get(gm, gm)}, deg={deg})"
    return base


def _make_color(engine, variant_idx=0, n_variants=1):
    """Generate HSL color with fiber-bundle effect.

    Hue is fixed per engine. Saturation and luminance vary per
    (graph_model, degree) variant index.
    """
    hue = ENGINE_HUES.get(engine, 0)
    if n_variants <= 1:
        sat, lum = 75, 45
    else:
        t = variant_idx / (n_variants - 1)
        sat = 85 - int(t * 15)   # 85% -> 70%
        lum = 58 - int(t * 23)   # 58% -> 35%
    return f"hsl({hue}, {sat}%, {lum}%)"


def _trace_opacity(engine):
    return 1.0 if engine == "muninn" else 0.8


# ── Data loading ──────────────────────────────────────────────────


def load_graph_results(filter_engine=None, filter_operation=None, filter_graph_model=None):
    """Load all graph_*.jsonl files from results directory."""
    records = []
    jsonl_files = sorted(RESULTS_DIR.glob("graph_*.jsonl"))

    if not jsonl_files:
        log.error("No graph JSONL files found in %s", RESULTS_DIR)
        log.error("Run 'make benchmark-graph-small' first.")
        return records

    log.info("Loading %d graph JSONL file(s) from %s", len(jsonl_files), RESULTS_DIR)

    for f in jsonl_files:
        for line in f.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            record = json.loads(line)

            # Skip CTE engine records (disabled benchmark)
            if record.get("engine") == "cte":
                continue
            if filter_engine and record.get("engine") != filter_engine:
                continue
            if filter_operation and record.get("operation") != filter_operation:
                continue
            if filter_graph_model and record.get("graph_model") != filter_graph_model:
                continue

            records.append(record)

    log.info("Loaded %d graph records", len(records))
    return records


# ── Aggregation ───────────────────────────────────────────────────


def aggregate(records):
    """Group records by (engine, operation, graph_model, n_nodes, avg_degree).

    Returns dict mapping key -> aggregated metrics.
    """
    groups = defaultdict(list)
    for r in records:
        key = (
            r["engine"],
            r["operation"],
            r["graph_model"],
            r["n_nodes"],
            round(r["avg_degree"]),
        )
        groups[key].append(r)

    agg = {}
    for key, recs in groups.items():
        agg[key] = _aggregate_group(recs)
    return agg


def _aggregate_group(records):
    """Compute mean/std for graph metrics."""
    metrics = ["query_time_ms", "setup_time_s", "nodes_visited_mean"]
    result = {"count": len(records)}

    for metric in metrics:
        values = [r[metric] for r in records if r.get(metric) is not None]
        if values:
            mean = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
                result[f"{metric}_std"] = math.sqrt(variance)
            else:
                result[f"{metric}_std"] = 0.0
            result[f"{metric}_mean"] = mean
        else:
            result[f"{metric}_mean"] = None
            result[f"{metric}_std"] = None

    # Correctness: all records must be correct
    result["correct"] = all(r.get("correct", False) for r in records)
    result["n_queries"] = records[0].get("n_queries", 0)
    result["n_edges"] = records[0].get("n_edges", 0)

    return result


# ── Key accessors ─────────────────────────────────────────────────
# Key: (engine, operation, graph_model, n_nodes, avg_degree)


def _get_operations(agg):
    return sorted(set(k[1] for k in agg))


def _get_engines(agg):
    engines = set(k[0] for k in agg)
    return sorted(engines, key=lambda e: ENGINE_ORDER.index(e) if e in ENGINE_ORDER else 999)


def _get_node_counts(agg, operation=None, graph_model=None, avg_degree=None):
    sizes = set()
    for k in agg:
        if operation is not None and k[1] != operation:
            continue
        if graph_model is not None and k[2] != graph_model:
            continue
        if avg_degree is not None and k[4] != avg_degree:
            continue
        sizes.add(k[3])
    return sorted(sizes)


def _get_avg_degrees(agg, graph_model=None):
    degrees = set()
    for k in agg:
        if graph_model is not None and k[2] != graph_model:
            continue
        degrees.add(k[4])
    return sorted(degrees)


def _get_variants(agg, operation=None):
    """Get sorted (graph_model, avg_degree) combinations for fiber-bundle indexing."""
    variants = set()
    for k in agg:
        if operation is not None and k[1] != operation:
            continue
        variants.add((k[2], k[4]))
    return sorted(variants)


def _get_val(agg, engine, operation, graph_model, n_nodes, avg_degree, metric):
    key = (engine, operation, graph_model, n_nodes, avg_degree)
    entry = agg.get(key)
    if entry is None:
        return None
    return entry.get(f"{metric}_mean")


def _fmt(val, fmt_str=".3f"):
    if val is None:
        return "n/a"
    return f"{val:{fmt_str}}"


# ── Text tables ───────────────────────────────────────────────────


def print_tables(agg):
    """Print all text summary tables."""
    operations = _get_operations(agg)
    engines = _get_engines(agg)
    graph_models = sorted(set(k[2] for k in agg))

    for gm in graph_models:
        for op in operations:
            print_operation_table(agg, op, gm, engines)

    print_setup_time_table(agg, engines)


def print_operation_table(agg, operation, graph_model, engines):
    """Print query time table for a single operation."""
    avg_degrees = _get_avg_degrees(agg, graph_model)
    if not avg_degrees:
        return

    col_w = max(14, max(len(_engine_label(e)) for e in engines) + 2)

    print(f"\n{'=' * 100}")
    print(f"QUERY TIME: {operation.upper()} — {graph_model} (ms)")
    print("=" * 100)

    for deg in avg_degrees:
        node_counts = _get_node_counts(agg, operation=operation, graph_model=graph_model, avg_degree=deg)
        if not node_counts:
            continue

        print(f"\n  avg_degree={deg}")
        header = f"  {'N':>10}"
        for engine in engines:
            header += f" | {_engine_label(engine):>{col_w}}"
        header += f" | {'correct':>8}"
        print(header)
        print(f"  {'-'*10}" + f"-+-{'-'*col_w}" * len(engines) + f"-+-{'-'*8}")

        for n in node_counts:
            row = f"  {n:>10,}"
            all_correct = True
            for engine in engines:
                val = _get_val(agg, engine, operation, graph_model, n, deg, "query_time_ms")
                row += f" | {_fmt(val):>{col_w}}"
                key = (engine, operation, graph_model, n, deg)
                entry = agg.get(key)
                if entry and not entry.get("correct", True):
                    all_correct = False
            row += f" | {'yes' if all_correct else 'NO':>8}"
            print(row)


def print_setup_time_table(agg, engines):
    """Print edge insertion (setup) time table."""
    print(f"\n{'=' * 100}")
    print("SETUP TIME (edge insertion, seconds)")
    print("=" * 100)

    col_w = max(14, max(len(_engine_label(e)) for e in engines) + 2)

    # Group by graph_model and avg_degree, take the first operation's setup_time
    seen = set()
    rows = []
    for k, v in sorted(agg.items(), key=lambda x: (x[0][2], x[0][4], x[0][3])):
        group_key = (k[2], k[3], k[4])
        if group_key in seen:
            continue
        seen.add(group_key)
        rows.append((k[2], k[3], k[4]))

    if not rows:
        return

    header = f"  {'Graph':>15} {'N':>8} {'Deg':>5}"
    for engine in engines:
        header += f" | {_engine_label(engine):>{col_w}}"
    print(header)
    print(f"  {'-'*15} {'-'*8} {'-'*5}" + f"-+-{'-'*col_w}" * len(engines))

    for gm, n, deg in rows:
        row = f"  {gm:>15} {n:>8,} {deg:>5}"
        for engine in engines:
            val = _get_val(agg, engine, "bfs", gm, n, deg, "setup_time_s")
            if val is None:
                # Try other operations
                for op in _get_operations(agg):
                    val = _get_val(agg, engine, op, gm, n, deg, "setup_time_s")
                    if val is not None:
                        break
            row += f" | {_fmt(val, '.4f'):>{col_w}}"
        print(row)


# ── Charts ────────────────────────────────────────────────────────


def _save_chart(fig, name):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    html_path = CHARTS_DIR / f"{name}.html"
    fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)

    json_path = CHARTS_DIR / f"{name}.json"
    json_path.write_text(fig.to_json(), encoding="utf-8")

    log.info("  Chart saved: %s (.html + .json)", CHARTS_DIR / name)


def chart_query_time_by_operation(agg):
    """Query time vs n_nodes for each operation, all graph topologies combined.

    Fiber bundle: engine hue, (graph_model, degree) as S/L variants.
    One chart per operation with all topologies overlaid.
    """
    operations = _get_operations(agg)
    engines = _get_engines(agg)

    for op in operations:
        variants = _get_variants(agg, operation=op)
        n_variants = len(variants)
        if not variants:
            continue

        fig = go.Figure()
        has_traces = False

        for engine in engines:
            for var_idx, (gm, deg) in enumerate(variants):
                nodes = _get_node_counts(agg, operation=op, graph_model=gm, avg_degree=deg)
                if not nodes:
                    continue

                times = [_get_val(agg, engine, op, gm, n, deg, "query_time_ms") for n in nodes]

                color = _make_color(engine, var_idx, n_variants)
                opacity = _trace_opacity(engine)
                line_width = 3 if engine == "muninn" else 2

                fig.add_trace(
                    go.Scatter(
                        x=nodes, y=times,
                        mode="lines+markers",
                        name=_trace_label(engine, gm, deg),
                        line={"color": color, "width": line_width},
                        marker={"size": 7},
                        opacity=opacity,
                        legendgroup=_engine_label(engine),
                        legendgrouptitle_text=_engine_label(engine),
                    )
                )
                has_traces = True

        if not has_traces:
            continue

        op_title = OPERATION_LABELS.get(op, op.replace("_", " ").title())
        fig.update_layout(
            title=f"Query Time — {op_title}",
            xaxis_title="Number of Nodes",
            yaxis_title="Query Time (ms)",
            xaxis_type="log",
            yaxis_type="log",
            height=500,
            width=900,
            template="plotly_white",
            legend={"orientation": "v", "yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02,
                    "groupclick": "togglegroup"},
        )

        _save_chart(fig, f"graph_query_time_{op}")


def chart_setup_time(agg):
    """Edge insertion throughput (edges/sec) vs graph size, fiber bundle."""
    engines = _get_engines(agg)
    variants = _get_variants(agg)
    n_variants = len(variants)

    fig = go.Figure()
    has_traces = False

    for engine in engines:
        for var_idx, (gm, deg) in enumerate(variants):
            nodes = _get_node_counts(agg, graph_model=gm, avg_degree=deg)
            if not nodes:
                continue

            edge_counts = []
            throughputs = []
            for n in nodes:
                setup_s = _get_val(agg, engine, "bfs", gm, n, deg, "setup_time_s")
                if setup_s is None:
                    for op in _get_operations(agg):
                        setup_s = _get_val(agg, engine, op, gm, n, deg, "setup_time_s")
                        if setup_s is not None:
                            break
                if setup_s is None or setup_s <= 0:
                    continue

                key = (engine, "bfs", gm, n, deg)
                entry = agg.get(key)
                if entry is None:
                    continue
                n_edges = entry.get("n_edges", 0)
                if n_edges > 0:
                    edge_counts.append(n_edges)
                    throughputs.append(n_edges / setup_s)

            if not edge_counts:
                continue

            color = _make_color(engine, var_idx, n_variants)
            opacity = _trace_opacity(engine)
            line_width = 3 if engine == "muninn" else 2

            fig.add_trace(
                go.Scatter(
                    x=edge_counts, y=throughputs,
                    mode="lines+markers",
                    name=_trace_label(engine, gm, deg),
                    line={"color": color, "width": line_width},
                    marker={"size": 7},
                    opacity=opacity,
                    legendgroup=_engine_label(engine),
                    legendgrouptitle_text=_engine_label(engine),
                )
            )
            has_traces = True

    if not has_traces:
        return

    fig.update_layout(
        title="Edge Insertion Throughput",
        xaxis_title="Number of Edges",
        yaxis_title="Throughput (edges/sec)",
        xaxis_type="log",
        yaxis_type="log",
        height=500,
        width=900,
        template="plotly_white",
        legend={"orientation": "v", "yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.02,
                "groupclick": "togglegroup"},
    )

    _save_chart(fig, "graph_setup_time")


def generate_all_charts(agg):
    """Generate all graph benchmark charts."""
    log.info("Generating graph charts...")
    chart_query_time_by_operation(agg)
    chart_setup_time(agg)
    log.info("All graph charts generated in %s", CHARTS_DIR)


# ── Doc generation ───────────────────────────────────────────────

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
DOCS_DIR = PROJECT_ROOT / "docs" / "benchmarks"


def generate_docs():
    """Render graph.md.j2 template to docs/benchmarks/graph.md."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        keep_trailing_newline=True,
    )
    template = env.get_template("graph.md.j2")

    operation_labels = {
        "bfs": "BFS",
        "dfs": "DFS",
        "shortest_path": "Shortest Path",
        "components": "Components",
        "pagerank": "PageRank",
    }
    operations = [
        {"key": op, "label": operation_labels.get(op, op.replace("_", " ").title())}
        for op in ALL_OPERATIONS
    ]
    output = template.render(operations=operations)
    out_path = DOCS_DIR / "graph.md"
    out_path.write_text(output, encoding="utf-8")
    log.info("Generated docs: %s", out_path)


# ── Manifest ──────────────────────────────────────────────────────


def graph_manifest():
    """Enumerate all expected graph scenarios from GRAPH_PROFILES.

    Returns dict mapping profile_name -> list of (graph_model, n_nodes, avg_degree, engine, operation) tuples.
    """
    manifest = {}
    for profile_name, profile in GRAPH_PROFILES.items():
        scenarios = []
        graph_model = profile["graph_model"]
        for n_nodes, avg_degree in profile["configs"]:
            for engine in ALL_GRAPH_ENGINES:
                for operation in ALL_OPERATIONS:
                    scenarios.append((graph_model, n_nodes, avg_degree, engine, operation))
        manifest[profile_name] = scenarios
    return manifest


def check_graph_completeness():
    """Check which graph scenarios have JSONL records.

    Matches on (graph_model, n_nodes, engine, operation) with avg_degree
    rounded to the nearest integer, since the JSONL stores the realized
    degree (e.g. 4.54) rather than the requested target (e.g. 5).

    Returns dict mapping (graph_model, n_nodes, avg_degree, engine, operation)
    -> list of JSONL filenames containing matching records.
    """
    # Build index: scenario key -> set of filenames containing it
    existing = defaultdict(set)
    jsonl_files = sorted(RESULTS_DIR.glob("graph_*.jsonl"))
    for f in jsonl_files:
        for line in f.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            record = json.loads(line)
            key = (
                record.get("graph_model"),
                record.get("n_nodes"),
                round(record.get("avg_degree", 0)),
                record.get("engine"),
                record.get("operation"),
            )
            existing[key].add(f.name)

    manifest = graph_manifest()
    status = {}
    for scenarios in manifest.values():
        for graph_model, n_nodes, avg_degree, engine, operation in scenarios:
            key = (graph_model, n_nodes, round(avg_degree), engine, operation)
            if key in status:
                continue
            status[key] = sorted(existing.get(key, set()))
    return status


def print_manifest_report():
    """Print formatted DONE/MISS completeness report with file patterns."""
    manifest = graph_manifest()
    completeness = check_graph_completeness()

    print("\n=== Graph Benchmark Manifest ===")
    total_done = 0
    total_expected = 0

    for profile_name, scenarios in manifest.items():
        done_count = sum(
            1 for gm, n, d, e, op in scenarios
            if completeness.get((gm, n, round(d), e, op), [])
        )
        total = len(scenarios)
        total_done += done_count
        total_expected += total

        print(f"\n--- Profile: {profile_name} ({done_count}/{total} done) ---")
        for graph_model, n_nodes, avg_degree, engine, operation in scenarios:
            key = (graph_model, n_nodes, round(avg_degree), engine, operation)
            found = completeness.get(key, [])
            target = (f"benchmarks/results/graph_*.jsonl"
                      f" → ({graph_model}, n={n_nodes}, deg≈{avg_degree}, {engine}, {operation})")
            if found:
                print(f"  [DONE] {graph_model} n={n_nodes} deg={avg_degree} / {engine} / {operation}")
                print(f"         target: {target}")
                for f in found:
                    print(f"         found:  {f}")
            else:
                print(f"  [MISS] {graph_model} n={n_nodes} deg={avg_degree} / {engine} / {operation}")
                print(f"         target: {target}")

    print(f"\nSummary: {total_done}/{total_expected} done, {total_expected - total_done} missing")


# ── CLI ───────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze graph benchmark results and generate charts")
    parser.add_argument("--manifest", action="store_true", help="Show benchmark completeness report")
    parser.add_argument("--filter-engine", help="Filter by engine (e.g., 'muninn', 'cte')")
    parser.add_argument("--filter-operation", help="Filter by operation (e.g., 'bfs', 'pagerank')")
    parser.add_argument("--filter-graph-model", help="Filter by graph model (e.g., 'erdos_renyi')")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    args = parse_args()

    if args.manifest:
        print_manifest_report()
        return

    records = load_graph_results(
        filter_engine=args.filter_engine,
        filter_operation=args.filter_operation,
        filter_graph_model=args.filter_graph_model,
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
