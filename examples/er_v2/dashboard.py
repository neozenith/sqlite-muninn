"""Plotly Dash dashboard for ER benchmark results.

Visualises all accumulated results from examples/er_v2/results/ across:
  - Datasets (Abt-Buy, Amazon-Google, DBLP-ACM)
  - Embedding models (MiniLM, NomicEmbed)
  - Pipelines (string-only, llm-cluster)
  - Tuning parameters (dist_threshold, match_threshold/llm_high, llm_low, k)

Usage:
  uv run examples/er_v2/dashboard.py --port 8055
"""

import json
import logging
from pathlib import Path

import dash
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Consistent colour palette: each (dataset, embed_model) gets a unique colour
SERIES_COLORS = {
    "abt-buy / MiniLM": "#2563eb",
    "abt-buy / NomicEmbed": "#60a5fa",
    "amazon-google / MiniLM": "#dc2626",
    "amazon-google / NomicEmbed": "#f87171",
    "dblp-acm / MiniLM": "#059669",
    "dblp-acm / NomicEmbed": "#34d399",
}


def _series_key(r: dict) -> str:
    return f"{r['dataset']} / {r['embed_model']}"


def _series_color(key: str) -> str:
    return SERIES_COLORS.get(key, "#888888")


def load_all_results() -> list[dict]:
    """Load all JSON result files."""
    results = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            r.setdefault("embed_model", "MiniLM")
            r.setdefault("llm_calls", 0)
            r.setdefault("total_pairs", 0)
            r.setdefault("matched_pairs", 0)
            r.setdefault("borderline_pairs", 0)
            params = r.get("params", {})
            r["dist_threshold"] = params.get("dist_threshold", 0.15)
            r["match_threshold"] = params.get("match_threshold", params.get("llm_high", 0.9))
            r["llm_low"] = params.get("llm_low", r["match_threshold"])
            r["llm_high"] = params.get("llm_high", params.get("match_threshold", 0.9))
            r["k"] = params.get("k", 10)
            r["limit_str"] = str(r.get("limit") or "full")
            r["series"] = _series_key(r)
            results.append(r)
        except (json.JSONDecodeError, OSError):
            pass
    return results


RESULTS = load_all_results()

FILTER_INPUTS = [
    Input("dataset-filter", "value"),
    Input("pipeline-filter", "value"),
    Input("embed-filter", "value"),
    Input("limit-filter", "value"),
]

# ── Explanatory text per chart ────────────────────────────────────

EXPLANATIONS = {
    "f1-vs-dist": (
        "Each line is a (dataset, embedding model) combination. The x-axis is the HNSW "
        "cosine distance threshold — how far apart two embeddings can be and still be "
        "considered a candidate pair. Lower distance = tighter blocking = fewer candidates. "
        "Look for the peak of each line: that's where the blocker admits enough true matches "
        "without drowning in false positives. Different embedding models need different "
        "optimal distances because their vector spaces have different densities."
    ),
    "f1-vs-mt": (
        "The match threshold is the minimum combined score (Jaro-Winkler + cosine) for a "
        "candidate pair to become a match edge. Higher threshold = more selective = fewer "
        "false positives but risk missing true matches. Look for where each line peaks — "
        "that's the sweet spot between over-merging (too low) and under-merging (too high)."
    ),
    "pr-scatter": (
        "Each point is a single benchmark run. Precision (y-axis) measures 'of the entities "
        "we merged, how many were correct?' Recall (x-axis) measures 'of the entities that "
        "should have been merged, how many did we find?' The dashed curves are F1 iso-lines — "
        "all points on the same curve have the same F1 score. Points closer to the top-right "
        "corner are better. Bubble size indicates F1 magnitude."
    ),
    "llm-marginal": (
        "Shows the diminishing returns of adding LLM calls. The left panel plots F1 against "
        "number of LLM calls; the right panel plots F1 against wall clock time. Each line is "
        "a dataset, with points representing different llm_low thresholds. The leftmost point "
        "(0 calls) is the string-only baseline. Look for where the curve flattens — that's "
        "where additional LLM calls stop improving quality. Lines that go DOWN mean the LLM "
        "is actively hurting."
    ),
    "pairs-vs-f1": (
        "Shows the relationship between candidate set size (how many pairs the HNSW blocker "
        "generates) and the resulting F1. Too few pairs = missed matches (low recall). "
        "Too many pairs = noise overwhelms the cascade (low precision). Look for the 'sweet "
        "spot' where F1 peaks for each series. Note the log scale on the x-axis."
    ),
    "embed-compare": (
        "Side-by-side comparison of embedding models across datasets. Each subplot is a "
        "dataset, each line is an embedding model. The x-axis is distance threshold. "
        "This shows whether one model consistently outperforms the other, and at what "
        "threshold each model peaks. Full-dataset, string-only runs only."
    ),
    "cost-benefit": (
        "Every full-dataset run plotted by wall clock time (x-axis, log scale) vs F1 quality "
        "(y-axis). Bubble size indicates number of LLM calls. Points in the top-left are the "
        "best: high quality, low cost. The Pareto frontier is the set of points where you "
        "can't improve quality without increasing cost."
    ),
}

# ── App Layout ────────────────────────────────────────────────────

app = dash.Dash(__name__, title="ER Benchmark Dashboard")


def _tab(tab_id: str, label: str) -> dcc.Tab:
    return dcc.Tab(
        label=label,
        children=[
            html.P(
                EXPLANATIONS.get(tab_id, ""),
                style={
                    "color": "#555",
                    "fontSize": "14px",
                    "margin": "12px 0",
                    "lineHeight": "1.5",
                    "maxWidth": "900px",
                },
            ),
            dcc.Graph(id=tab_id),
        ],
    )


app.layout = html.Div(
    style={
        "fontFamily": "system-ui, -apple-system, sans-serif",
        "margin": "20px 40px",
    },
    children=[
        html.H1("Entity Resolution Benchmark Dashboard"),
        html.P(f"{len(RESULTS)} results from {RESULTS_DIR}/"),
        html.Div(
            style={
                "display": "flex",
                "gap": "20px",
                "marginBottom": "20px",
                "flexWrap": "wrap",
            },
            children=[
                html.Div(
                    [
                        html.Label("Dataset"),
                        dcc.Checklist(
                            id="dataset-filter",
                            options=[{"label": d, "value": d} for d in sorted({r["dataset"] for r in RESULTS})],
                            value=sorted({r["dataset"] for r in RESULTS}),
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Pipeline"),
                        dcc.Checklist(
                            id="pipeline-filter",
                            options=[{"label": p, "value": p} for p in sorted({r["pipeline"] for r in RESULTS})],
                            value=sorted({r["pipeline"] for r in RESULTS}),
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Embed Model"),
                        dcc.Checklist(
                            id="embed-filter",
                            options=[{"label": e, "value": e} for e in sorted({r["embed_model"] for r in RESULTS})],
                            value=sorted({r["embed_model"] for r in RESULTS}),
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Limit"),
                        dcc.Checklist(
                            id="limit-filter",
                            options=[
                                {"label": s, "value": s}
                                for s in sorted(
                                    {r["limit_str"] for r in RESULTS},
                                    key=lambda x: (x != "full", x),
                                )
                            ],
                            value=sorted({r["limit_str"] for r in RESULTS}),
                        ),
                    ]
                ),
            ],
        ),
        dcc.Tabs(
            [
                _tab("f1-vs-dist", "F1 vs Distance Threshold"),
                _tab("f1-vs-mt", "F1 vs Match Threshold"),
                _tab("pr-scatter", "Precision vs Recall"),
                _tab("llm-marginal", "LLM Marginal Value"),
                _tab("pairs-vs-f1", "Candidate Pairs vs F1"),
                _tab("embed-compare", "Embed Model Comparison"),
                _tab("cost-benefit", "Cost-Benefit (Time vs F1)"),
            ]
        ),
    ],
)


def _filter(datasets, pipelines, embeds, limits):
    return [
        r
        for r in RESULTS
        if r["dataset"] in datasets
        and r["pipeline"] in pipelines
        and r["embed_model"] in embeds
        and r["limit_str"] in limits
    ]


def _line_chart(filtered, x, y, title, x_label, y_label, hover_extra=None):
    """Build a line+markers chart with one line per (dataset, embed_model) series."""
    fig = go.Figure()
    for series_name in sorted({_series_key(r) for r in filtered}):
        series_data = sorted(
            [r for r in filtered if _series_key(r) == series_name],
            key=lambda r: r[x],
        )
        if not series_data:
            continue
        hover_text = None
        if hover_extra:
            hover_text = ["<br>".join(f"{k}={r.get(k, '?')}" for k in hover_extra) for r in series_data]
        fig.add_trace(
            go.Scatter(
                x=[r[x] for r in series_data],
                y=[r[y] for r in series_data],
                mode="lines+markers",
                name=series_name,
                line={"color": _series_color(series_name)},
                marker={"size": 6},
                text=hover_text,
                hoverinfo="text+name+y" if hover_text else "name+x+y",
            )
        )
    fig.update_layout(
        height=500,
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    return fig


# ── Chart 1: F1 vs dist_threshold ────────────────────────────────


@app.callback(Output("f1-vs-dist", "figure"), FILTER_INPUTS)
def update_f1_vs_dist(datasets, pipelines, embeds, limits):
    filtered = [r for r in _filter(datasets, pipelines, embeds, limits) if r["pipeline"] == "string-only"]
    if not filtered:
        return go.Figure()
    return _line_chart(
        filtered,
        x="dist_threshold",
        y="bcubed_f1",
        title="B-Cubed F1 vs Distance Threshold (string-only)",
        x_label="Distance Threshold",
        y_label="B-Cubed F1",
        hover_extra=["match_threshold", "k", "total_pairs", "limit_str"],
    )


# ── Chart 2: F1 vs match_threshold ───────────────────────────────


@app.callback(Output("f1-vs-mt", "figure"), FILTER_INPUTS)
def update_f1_vs_mt(datasets, pipelines, embeds, limits):
    filtered = [r for r in _filter(datasets, pipelines, embeds, limits) if r["pipeline"] == "string-only"]
    if not filtered:
        return go.Figure()
    return _line_chart(
        filtered,
        x="match_threshold",
        y="bcubed_f1",
        title="B-Cubed F1 vs Match Threshold (string-only)",
        x_label="Match Threshold",
        y_label="B-Cubed F1",
        hover_extra=["dist_threshold", "k", "total_pairs", "limit_str"],
    )


# ── Chart 3: Precision vs Recall ─────────────────────────────────


@app.callback(Output("pr-scatter", "figure"), FILTER_INPUTS)
def update_pr_scatter(datasets, pipelines, embeds, limits):
    filtered = _filter(datasets, pipelines, embeds, limits)
    if not filtered:
        return go.Figure()

    fig = go.Figure()
    for series_name in sorted({_series_key(r) for r in filtered}):
        series_data = [r for r in filtered if _series_key(r) == series_name]
        fig.add_trace(
            go.Scatter(
                x=[r["bcubed_recall"] for r in series_data],
                y=[r["bcubed_precision"] for r in series_data],
                mode="markers",
                name=series_name,
                marker={
                    "color": _series_color(series_name),
                    "size": [max(r["bcubed_f1"] * 18, 4) for r in series_data],
                    "opacity": 0.7,
                },
                text=[
                    f"F1={r['bcubed_f1']:.3f}<br>"
                    f"dist={r['dist_threshold']}<br>"
                    f"mt={r['match_threshold']}<br>"
                    f"LLM={r['llm_calls']}<br>"
                    f"{r['pipeline']}"
                    for r in series_data
                ],
                hoverinfo="text+name",
            )
        )

    # F1 iso-lines
    for f1_val in [0.5, 0.6, 0.7, 0.8, 0.9]:
        r_vals = [i / 100 for i in range(10, 101)]
        p_vals = [(f1_val * r) / (2 * r - f1_val) if (2 * r - f1_val) > 0 else None for r in r_vals]
        fig.add_trace(
            go.Scatter(
                x=r_vals,
                y=p_vals,
                mode="lines",
                line={"dash": "dot", "color": "#ccc", "width": 1},
                name=f"F1={f1_val}",
                showlegend=f1_val == 0.7,
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        height=600,
        title="B-Cubed Precision vs Recall (bubble size = F1)",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_range=[0, 1.05],
        yaxis_range=[0, 1.05],
    )
    return fig


# ── Chart 4: LLM Marginal Value ──────────────────────────────────


@app.callback(Output("llm-marginal", "figure"), FILTER_INPUTS)
def update_llm_marginal(datasets, pipelines, embeds, limits):
    filtered = [
        r
        for r in _filter(datasets, pipelines, embeds, limits)
        if r["pipeline"] == "llm-cluster" and r.get("limit") is None
    ]
    if not filtered:
        return go.Figure()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["F1 vs LLM Calls", "F1 vs Wall Clock Time"],
    )

    for series_name in sorted({_series_key(r) for r in filtered}):
        series_data = sorted(
            [r for r in filtered if _series_key(r) == series_name],
            key=lambda r: r["llm_calls"],
        )
        if not series_data:
            continue
        color = _series_color(series_name)
        labels = [f"lo={r['llm_low']:.2f}" for r in series_data]
        fig.add_trace(
            go.Scatter(
                x=[r["llm_calls"] for r in series_data],
                y=[r["bcubed_f1"] for r in series_data],
                mode="lines+markers",
                name=series_name,
                line={"color": color},
                text=labels,
                hoverinfo="text+name+y",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[r["elapsed_s"] for r in series_data],
                y=[r["bcubed_f1"] for r in series_data],
                mode="lines+markers",
                name=series_name,
                line={"color": color},
                text=labels,
                hoverinfo="text+name+y",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="LLM Calls", row=1, col=1)
    fig.update_xaxes(title_text="Wall Clock (s)", row=1, col=2)
    fig.update_yaxes(title_text="B-Cubed F1", row=1, col=1)
    fig.update_layout(
        height=500,
        title="LLM Marginal Value: F1 vs Cost (full datasets only)",
    )
    return fig


# ── Chart 5: Candidate Pairs vs F1 ───────────────────────────────


@app.callback(Output("pairs-vs-f1", "figure"), FILTER_INPUTS)
def update_pairs_vs_f1(datasets, pipelines, embeds, limits):
    filtered = [r for r in _filter(datasets, pipelines, embeds, limits) if r["total_pairs"] > 0]
    if not filtered:
        return go.Figure()
    return _line_chart(
        filtered,
        x="total_pairs",
        y="bcubed_f1",
        title="B-Cubed F1 vs Candidate Pair Count",
        x_label="Candidate Pairs",
        y_label="B-Cubed F1",
        hover_extra=[
            "dist_threshold",
            "match_threshold",
            "pipeline",
            "limit_str",
        ],
    )


# ── Chart 6: Embed Model Comparison ──────────────────────────────


@app.callback(Output("embed-compare", "figure"), FILTER_INPUTS)
def update_embed_compare(datasets, pipelines, embeds, limits):
    filtered = [
        r
        for r in _filter(datasets, pipelines, embeds, limits)
        if r["pipeline"] == "string-only" and r.get("limit") is None
    ]
    if not filtered:
        return go.Figure()

    ds_list = sorted({r["dataset"] for r in filtered})
    fig = make_subplots(rows=1, cols=len(ds_list), subplot_titles=ds_list)

    for col, ds in enumerate(ds_list, 1):
        for emb in sorted({r["embed_model"] for r in filtered}):
            series_name = f"{ds} / {emb}"
            ds_emb = sorted(
                [r for r in filtered if r["dataset"] == ds and r["embed_model"] == emb],
                key=lambda r: r["dist_threshold"],
            )
            if not ds_emb:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[r["dist_threshold"] for r in ds_emb],
                    y=[r["bcubed_f1"] for r in ds_emb],
                    mode="lines+markers",
                    name=f"{emb}" if col == 1 else None,
                    legendgroup=emb,
                    showlegend=col == 1,
                    line={"color": _series_color(series_name)},
                    text=[f"mt={r['match_threshold']}" for r in ds_emb],
                    hoverinfo="text+y",
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="dist_threshold", row=1, col=col)
        if col == 1:
            fig.update_yaxes(title_text="B-Cubed F1", row=1, col=col)

    fig.update_layout(
        height=450,
        title="Embedding Model Comparison (full datasets, string-only)",
    )
    return fig


# ── Chart 7: Cost-Benefit ─────────────────────────────────────────


@app.callback(Output("cost-benefit", "figure"), FILTER_INPUTS)
def update_cost_benefit(datasets, pipelines, embeds, limits):
    filtered = [r for r in _filter(datasets, pipelines, embeds, limits) if r.get("limit") is None]
    if not filtered:
        return go.Figure()

    fig = go.Figure()
    for series_name in sorted({_series_key(r) for r in filtered}):
        series_data = sorted(
            [r for r in filtered if _series_key(r) == series_name],
            key=lambda r: r["elapsed_s"],
        )
        if not series_data:
            continue
        fig.add_trace(
            go.Scatter(
                x=[r["elapsed_s"] for r in series_data],
                y=[r["bcubed_f1"] for r in series_data],
                mode="markers",
                name=series_name,
                marker={
                    "color": _series_color(series_name),
                    "size": [max(r.get("llm_calls", 0) ** 0.5 * 3 + 5, 5) for r in series_data],
                    "opacity": 0.7,
                },
                text=[
                    f"pipeline={r['pipeline']}<br>"
                    f"dist={r['dist_threshold']}<br>"
                    f"mt={r['match_threshold']}<br>"
                    f"LLM={r['llm_calls']}<br>"
                    f"model={r['model']}"
                    for r in series_data
                ],
                hoverinfo="text+name",
            )
        )

    fig.update_layout(
        height=500,
        title="Cost-Benefit: F1 vs Time (full datasets, size = LLM calls)",
        xaxis_title="Wall Clock (seconds)",
        yaxis_title="B-Cubed F1",
        xaxis_type="log",
    )
    return fig


# ── Run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8055)
    args = p.parse_args()
    print(f"Loaded {len(RESULTS)} results")
    print(f"Dashboard at http://127.0.0.1:{args.port}")
    app.run(debug=True, port=args.port)
