"""Plotly Dash dashboard for ER parameter sweep results.

Focused on the 4-parameter space: dist_threshold, jw_weight, llm_high, borderline_delta.
Each chart shows B-Cubed F1 as a function of one parameter, with the others as filters.

Usage:
  uv run examples/er_v2/dashboard.py --port 8055
"""

import json
import logging
from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_results() -> list[dict]:
    results = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            params = r.get("params", {})
            r["dist"] = params.get("dist_threshold", 0.15)
            r["jw"] = params.get("jw_weight", 0.4)
            r["hi"] = params.get("llm_high", 0.9)
            r["lo"] = params.get("llm_low", 0.9)
            r["delta"] = round(r["hi"] - r["lo"], 4)
            timing = r.get("timing", {})
            r["blocking_s"] = timing.get("blocking_s", 0)
            r["scoring_s"] = timing.get("scoring_s", 0)
            r["llm_s"] = timing.get("llm_s", 0)
            r["leiden_s"] = timing.get("leiden_s", 0)
            r.setdefault("llm_calls", 0)
            r.setdefault("total_pairs", 0)
            r.setdefault("auto_accepted", 0)
            r.setdefault("auto_rejected", 0)
            r.setdefault("borderline_pairs", 0)
            results.append(r)
        except (json.JSONDecodeError, OSError):
            pass
    return results


RESULTS = load_results()

DATASETS = sorted({r["dataset"] for r in RESULTS})
DISTS = sorted({r["dist"] for r in RESULTS})
JWS = sorted({r["jw"] for r in RESULTS})
HIS = sorted({r["hi"] for r in RESULTS})
DELTAS = sorted({r["delta"] for r in RESULTS})

COLORS = {
    "amazon-google": "#dc2626",
    "dblp-acm": "#059669",
    "abt-buy": "#2563eb",
}

# ── App ───────────────────────────────────────────────────────────

app = Dash(__name__, title="ER Parameter Sweep")

app.layout = html.Div(
    style={"fontFamily": "system-ui, sans-serif", "margin": "20px 40px", "maxWidth": "1400px"},
    children=[
        html.H1("ER Parameter Sweep Dashboard"),
        html.P(f"{len(RESULTS)} results across {len(DATASETS)} dataset(s)"),
        # Filters
        html.Div(
            style={"display": "flex", "gap": "30px", "marginBottom": "20px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    [
                        html.Label("Dataset", style={"fontWeight": "bold"}),
                        dcc.Checklist(
                            id="ds",
                            options=[{"label": d, "value": d} for d in DATASETS],
                            value=DATASETS,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("dist_threshold", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="dist-fix",
                            options=[{"label": str(d), "value": d} for d in DISTS],
                            value=None,
                            placeholder="All",
                            clearable=True,
                            style={"width": "120px"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("jw_weight", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="jw-fix",
                            options=[{"label": str(j), "value": j} for j in JWS],
                            value=None,
                            placeholder="All",
                            clearable=True,
                            style={"width": "120px"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("llm_high", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="hi-fix",
                            options=[{"label": str(h), "value": h} for h in HIS],
                            value=None,
                            placeholder="All",
                            clearable=True,
                            style={"width": "120px"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("delta", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="delta-fix",
                            options=[{"label": str(d), "value": d} for d in DELTAS],
                            value=0.0,
                            placeholder="All",
                            clearable=True,
                            style={"width": "120px"},
                        ),
                    ]
                ),
            ],
        ),
        # Charts
        html.Div(
            [
                html.H3("B³ F1 vs Distance Threshold"),
                html.P(
                    "How does the HNSW blocking radius affect quality? Each line is a dataset. "
                    "Fix jw_weight and llm_high above to isolate the dist effect.",
                    style={"color": "#666", "fontSize": "13px"},
                ),
                dcc.Graph(id="f1-vs-dist"),
            ]
        ),
        html.Div(
            [
                html.H3("B³ F1 vs JW Weight"),
                html.P(
                    "How does the balance between string similarity (JW=1.0) and semantic similarity (JW=0.0) "
                    "affect quality? Fix dist and llm_high to isolate.",
                    style={"color": "#666", "fontSize": "13px"},
                ),
                dcc.Graph(id="f1-vs-jw"),
            ]
        ),
        html.Div(
            [
                html.H3("B³ F1 vs Match Threshold (llm_high)"),
                html.P(
                    "How does the auto-accept threshold affect quality? At 1.0, only exact matches are accepted. "
                    "As it decreases, more pairs are auto-accepted. Fix dist and jw_weight to isolate.",
                    style={"color": "#666", "fontSize": "13px"},
                ),
                dcc.Graph(id="f1-vs-hi"),
            ]
        ),
        html.Div(
            [
                html.H3("Precision vs Recall"),
                html.P(
                    "Each point is one run. Dashed curves are F1 iso-lines. Points closer to top-right are better. "
                    "Hover for parameters.",
                    style={"color": "#666", "fontSize": "13px"},
                ),
                dcc.Graph(id="pr-scatter"),
            ]
        ),
    ],
)

FILTER_INPUTS = [
    Input("ds", "value"),
    Input("dist-fix", "value"),
    Input("jw-fix", "value"),
    Input("hi-fix", "value"),
    Input("delta-fix", "value"),
]


def _filter(datasets, dist_fix, jw_fix, hi_fix, delta_fix):
    return [
        r
        for r in RESULTS
        if r["dataset"] in (datasets or DATASETS)
        and (dist_fix is None or r["dist"] == dist_fix)
        and (jw_fix is None or r["jw"] == jw_fix)
        and (hi_fix is None or r["hi"] == hi_fix)
        and (delta_fix is None or r["delta"] == delta_fix)
    ]


def _line_by_dataset(filtered, x_key, title, x_label):
    fig = go.Figure()
    for ds in sorted({r["dataset"] for r in filtered}):
        series = sorted([r for r in filtered if r["dataset"] == ds], key=lambda r: r[x_key])
        if not series:
            continue
        fig.add_trace(
            go.Scatter(
                x=[r[x_key] for r in series],
                y=[r["bcubed_f1"] for r in series],
                mode="lines+markers",
                name=ds,
                line={"color": COLORS.get(ds, "#888")},
                marker={"size": 5},
                text=[
                    f"dist={r['dist']} jw={r['jw']} hi={r['hi']}<br>"
                    f"P={r['bcubed_precision']:.3f} R={r['bcubed_recall']:.3f}<br>"
                    f"pairs={r['total_pairs']} accept={r['auto_accepted']}"
                    for r in series
                ],
                hoverinfo="text+name",
            )
        )
    fig.update_layout(height=400, title=title, xaxis_title=x_label, yaxis_title="B³ F1")
    return fig


@app.callback(Output("f1-vs-dist", "figure"), FILTER_INPUTS)
def update_dist(datasets, dist_fix, jw_fix, hi_fix, delta_fix):
    # For this chart, don't filter by dist — it's the x-axis
    filtered = [
        r
        for r in RESULTS
        if r["dataset"] in (datasets or DATASETS)
        and (jw_fix is None or r["jw"] == jw_fix)
        and (hi_fix is None or r["hi"] == hi_fix)
        and (delta_fix is None or r["delta"] == delta_fix)
    ]
    if not filtered:
        return go.Figure()
    return _line_by_dataset(filtered, "dist", "B³ F1 vs Distance Threshold", "dist_threshold")


@app.callback(Output("f1-vs-jw", "figure"), FILTER_INPUTS)
def update_jw(datasets, dist_fix, jw_fix, hi_fix, delta_fix):
    # Don't filter by jw — it's the x-axis
    filtered = [
        r
        for r in RESULTS
        if r["dataset"] in (datasets or DATASETS)
        and (dist_fix is None or r["dist"] == dist_fix)
        and (hi_fix is None or r["hi"] == hi_fix)
        and (delta_fix is None or r["delta"] == delta_fix)
    ]
    if not filtered:
        return go.Figure()
    return _line_by_dataset(filtered, "jw", "B³ F1 vs JW Weight", "jw_weight (1.0=lexicographic, 0.0=semantic)")


@app.callback(Output("f1-vs-hi", "figure"), FILTER_INPUTS)
def update_hi(datasets, dist_fix, jw_fix, hi_fix, delta_fix):
    # Don't filter by hi — it's the x-axis
    filtered = [
        r
        for r in RESULTS
        if r["dataset"] in (datasets or DATASETS)
        and (dist_fix is None or r["dist"] == dist_fix)
        and (jw_fix is None or r["jw"] == jw_fix)
        and (delta_fix is None or r["delta"] == delta_fix)
    ]
    if not filtered:
        return go.Figure()
    return _line_by_dataset(filtered, "hi", "B³ F1 vs Match Threshold", "llm_high (1.0=strict, 0.80=permissive)")


@app.callback(Output("pr-scatter", "figure"), FILTER_INPUTS)
def update_pr(datasets, dist_fix, jw_fix, hi_fix, delta_fix):
    filtered = _filter(datasets, dist_fix, jw_fix, hi_fix, delta_fix)
    if not filtered:
        return go.Figure()

    fig = go.Figure()
    for ds in sorted({r["dataset"] for r in filtered}):
        series = [r for r in filtered if r["dataset"] == ds]
        fig.add_trace(
            go.Scatter(
                x=[r["bcubed_recall"] for r in series],
                y=[r["bcubed_precision"] for r in series],
                mode="markers",
                name=ds,
                marker={"color": COLORS.get(ds, "#888"), "size": 5, "opacity": 0.6},
                text=[
                    f"F1={r['bcubed_f1']:.3f}<br>dist={r['dist']} jw={r['jw']} hi={r['hi']}<br>"
                    f"pairs={r['total_pairs']} accept={r['auto_accepted']}"
                    for r in series
                ],
                hoverinfo="text+name",
            )
        )

    for f1_val in [0.5, 0.6, 0.7, 0.8, 0.9]:
        r_vals = [i / 100 for i in range(10, 101)]
        p_vals = [(f1_val * r) / (2 * r - f1_val) if (2 * r - f1_val) > 0 else None for r in r_vals]
        fig.add_trace(
            go.Scatter(
                x=r_vals,
                y=p_vals,
                mode="lines",
                line={"dash": "dot", "color": "#ddd", "width": 1},
                name=f"F1={f1_val}",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        height=500,
        title="Precision vs Recall",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_range=[0, 1.05],
        yaxis_range=[0, 1.05],
    )
    return fig


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8055)
    args = p.parse_args()
    print(f"Loaded {len(RESULTS)} results")
    print(f"Dashboard at http://127.0.0.1:{args.port}")
    app.run(debug=True, port=args.port)
