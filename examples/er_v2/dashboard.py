"""Plotly Dash dashboard for ER parameter sweep results.

Uses fibre-bundle colouring: fixed hue per dataset, saturation/luminance
gradient for the swept parameter value within each dataset.

All charts use the Precision-Recall format — position encodes quality,
colour encodes the swept parameter.

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

# ── Colour System (fibre-bundle) ──────────────────────────────────

# Fixed hue per dataset (HSL degrees)
DATASET_HUES = {
    "amazon-google": 0,  # red
    "dblp-acm": 150,  # green
    "abt-buy": 220,  # blue
}


def assign_color(dataset: str, variant_idx: int = 0, n_variants: int = 1) -> str:
    """Assign HSL colour: fixed hue per dataset, S/L gradient per variant."""
    hue = DATASET_HUES.get(dataset, 0)
    if n_variants <= 1:
        return f"hsl({hue}, 75%, 45%)"
    t = variant_idx / (n_variants - 1)
    sat = 85 - int(t * 15)  # 85% → 70%
    lum = 58 - int(t * 23)  # 58% → 35%
    return f"hsl({hue}, {sat}%, {lum}%)"


# ── Data Loading ──────────────────────────────────────────────────


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

# ── App ───────────────────────────────────────────────────────────

app = Dash(__name__, title="ER Parameter Sweep")


def _chart_section(chart_id, title, explanation):
    return html.Div(
        style={"marginBottom": "32px"},
        children=[
            html.H3(title, style={"marginBottom": "4px"}),
            html.P(explanation, style={"color": "#666", "fontSize": "13px", "marginBottom": "8px"}),
            dcc.Graph(id=chart_id),
        ],
    )


FILTER_INPUTS = [
    Input("ds", "value"),
    Input("dist-fix", "value"),
    Input("jw-fix", "value"),
    Input("hi-fix", "value"),
    Input("delta-fix", "value"),
]


def _dropdown(label, id_, options, default=None):
    return html.Div(
        [
            html.Label(label, style={"fontWeight": "bold", "fontSize": "13px"}),
            dcc.Dropdown(
                id=id_,
                options=[{"label": str(v), "value": v} for v in options],
                value=default,
                placeholder="All",
                clearable=True,
                style={"width": "120px"},
            ),
        ]
    )


app.layout = html.Div(
    style={"fontFamily": "system-ui, sans-serif", "margin": "20px 40px", "maxWidth": "1400px"},
    children=[
        html.H1("ER Parameter Sweep"),
        html.P(f"{len(RESULTS)} results across {len(DATASETS)} dataset(s)"),
        # Filters
        html.Div(
            style={
                "display": "flex",
                "gap": "24px",
                "marginBottom": "24px",
                "flexWrap": "wrap",
                "alignItems": "flex-end",
            },
            children=[
                html.Div(
                    [
                        html.Label("Dataset", style={"fontWeight": "bold", "fontSize": "13px"}),
                        dcc.Checklist(id="ds", options=[{"label": d, "value": d} for d in DATASETS], value=DATASETS),
                    ]
                ),
                _dropdown("dist_threshold", "dist-fix", DISTS),
                _dropdown("jw_weight", "jw-fix", JWS),
                _dropdown("llm_high", "hi-fix", HIS),
                _dropdown("delta", "delta-fix", DELTAS, default=0.0),
            ],
        ),
        # Charts — all P/R format with colour-coded parameter sweeps
        _chart_section(
            "pr-by-dist",
            "Precision-Recall coloured by dist_threshold",
            "Each point is one run. Colour gradient: light = small dist (tight blocking), "
            "dark = large dist (wide blocking). Fix jw_weight and llm_high to isolate the dist effect.",
        ),
        _chart_section(
            "pr-by-jw",
            "Precision-Recall coloured by jw_weight",
            "Colour gradient: light = JW=0.0 (pure semantic), dark = JW=1.0 (pure lexicographic). "
            "Fix dist and llm_high to isolate the scoring mix effect.",
        ),
        _chart_section(
            "pr-by-hi",
            "Precision-Recall coloured by llm_high",
            "Colour gradient: light = hi=0.80 (permissive), dark = hi=1.0 (strict). "
            "Fix dist and jw_weight to isolate the acceptance threshold effect.",
        ),
        _chart_section(
            "pr-by-delta",
            "Precision-Recall coloured by borderline_delta",
            "Colour gradient: light = delta=0 (no LLM), dark = delta=0.20 (wide LLM window). "
            "This shows whether the LLM tier shifts points on the PR plane.",
        ),
    ],
)


# ── Chart Builder ─────────────────────────────────────────────────


def _pr_chart(filtered, color_param, color_label, ascending=True):
    """Build a PR scatter where colour encodes `color_param` within each dataset."""
    fig = go.Figure()

    for ds in sorted({r["dataset"] for r in filtered}):
        ds_data = [r for r in filtered if r["dataset"] == ds]
        param_values = sorted({r[color_param] for r in ds_data}, reverse=not ascending)
        n = len(param_values)
        val_to_idx = {v: i for i, v in enumerate(param_values)}

        for val in param_values:
            points = [r for r in ds_data if r[color_param] == val]
            idx = val_to_idx[val]
            color = assign_color(ds, idx, n)
            fig.add_trace(
                go.Scatter(
                    x=[r["bcubed_recall"] for r in points],
                    y=[r["bcubed_precision"] for r in points],
                    mode="markers",
                    name=f"{ds} {color_label}={val}",
                    legendgroup=ds,
                    showlegend=idx == 0,
                    marker={"color": color, "size": 6, "opacity": 0.8},
                    text=[
                        f"F1={r['bcubed_f1']:.3f}<br>"
                        f"dist={r['dist']} jw={r['jw']} hi={r['hi']} δ={r['delta']}<br>"
                        f"pairs={r['total_pairs']} accept={r['auto_accepted']}"
                        for r in points
                    ],
                    hoverinfo="text",
                )
            )

    # F1 iso-lines
    for f1_val in [0.5, 0.6, 0.7, 0.8, 0.9]:
        r_vals = [i / 100 for i in range(10, 101)]
        p_vals = [(f1_val * rv) / (2 * rv - f1_val) if (2 * rv - f1_val) > 0 else None for rv in r_vals]
        fig.add_trace(
            go.Scatter(
                x=r_vals,
                y=p_vals,
                mode="lines",
                line={"dash": "dot", "color": "#e0e0e0", "width": 1},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        height=500,
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_range=[0, 1.05],
        yaxis_range=[0, 1.05],
        legend={"groupclick": "togglegroup"},
    )
    return fig


# ── Callbacks ─────────────────────────────────────────────────────


@app.callback(Output("pr-by-dist", "figure"), FILTER_INPUTS)
def update_pr_dist(datasets, dist_fix, jw_fix, hi_fix, delta_fix):
    # Don't filter by dist — it's the colour axis
    filtered = [
        r
        for r in RESULTS
        if r["dataset"] in (datasets or DATASETS)
        and (jw_fix is None or r["jw"] == jw_fix)
        and (hi_fix is None or r["hi"] == hi_fix)
        and (delta_fix is None or r["delta"] == delta_fix)
    ]
    return _pr_chart(filtered, "dist", "dist", ascending=True) if filtered else go.Figure()


@app.callback(Output("pr-by-jw", "figure"), FILTER_INPUTS)
def update_pr_jw(datasets, dist_fix, jw_fix, hi_fix, delta_fix):
    # Don't filter by jw — it's the colour axis
    filtered = [
        r
        for r in RESULTS
        if r["dataset"] in (datasets or DATASETS)
        and (dist_fix is None or r["dist"] == dist_fix)
        and (hi_fix is None or r["hi"] == hi_fix)
        and (delta_fix is None or r["delta"] == delta_fix)
    ]
    return _pr_chart(filtered, "jw", "jw", ascending=False) if filtered else go.Figure()


@app.callback(Output("pr-by-hi", "figure"), FILTER_INPUTS)
def update_pr_hi(datasets, dist_fix, jw_fix, hi_fix, delta_fix):
    # Don't filter by hi — it's the colour axis
    filtered = [
        r
        for r in RESULTS
        if r["dataset"] in (datasets or DATASETS)
        and (dist_fix is None or r["dist"] == dist_fix)
        and (jw_fix is None or r["jw"] == jw_fix)
        and (delta_fix is None or r["delta"] == delta_fix)
    ]
    return _pr_chart(filtered, "hi", "hi", ascending=False) if filtered else go.Figure()


@app.callback(Output("pr-by-delta", "figure"), FILTER_INPUTS)
def update_pr_delta(datasets, dist_fix, jw_fix, hi_fix, delta_fix):
    # Don't filter by delta — it's the colour axis
    filtered = [
        r
        for r in RESULTS
        if r["dataset"] in (datasets or DATASETS)
        and (dist_fix is None or r["dist"] == dist_fix)
        and (jw_fix is None or r["jw"] == jw_fix)
        and (hi_fix is None or r["hi"] == hi_fix)
    ]
    return _pr_chart(filtered, "delta", "δ", ascending=True) if filtered else go.Figure()


# ── Run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8055)
    args = p.parse_args()
    print(f"Loaded {len(RESULTS)} results")
    print(f"Dashboard at http://127.0.0.1:{args.port}")
    app.run(debug=True, port=args.port)
