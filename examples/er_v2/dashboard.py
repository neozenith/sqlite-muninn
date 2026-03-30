"""Plotly Dash dashboard for ER parameter sweep results.

Colour encoding (HSL):
  Hue:        (dataset, borderline_delta) — 4 distinct hues
  Saturation: 1.0 - jw_weight — desaturated=lexicographic, saturated=semantic
  Luminance:  swept variable (dist or llm_high) per chart
  Size:       inverse relative wall clock — fastest = largest marker

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

# ── Colour System ─────────────────────────────────────────────────

# Hue per (dataset, delta). Adjacent hues for same dataset.
SERIES_HUES: dict[tuple[str, float], int] = {
    ("dblp-acm", 0.0): 0,  # red
    ("dblp-acm", 0.05): 25,  # orange
    ("amazon-google", 0.0): 270,  # purple
    ("amazon-google", 0.05): 310,  # magenta
    ("abt-buy", 0.0): 210,  # blue
    ("abt-buy", 0.05): 180,  # cyan
}


def make_color(dataset: str, delta: float, jw_weight: float, lum_param: float, lum_range: tuple[float, float]) -> str:
    """Build HSL colour from the encoding scheme.

    Args:
        dataset: Dataset name (determines base hue with delta)
        delta: borderline_delta (shifts hue within dataset pair)
        jw_weight: 0.0-1.0, mapped to saturation (inverted: jw=1.0 → sat=0)
        lum_param: The value being mapped to luminance
        lum_range: (min_val, max_val) of the luminance parameter
    """
    hue = SERIES_HUES.get((dataset, delta), 0)

    # Saturation: jw=1.0 (lexicographic) → low sat, jw=0.0 (semantic) → high sat
    sat = int((1.0 - jw_weight) * 80 + 15)  # 15% – 95%

    # Luminance: normalise lum_param within its range
    lo, hi = lum_range
    if hi > lo:
        t = (lum_param - lo) / (hi - lo)
    else:
        t = 0.5
    lum = int(65 - t * 35)  # 65% (light) → 30% (dark)

    return f"hsl({hue}, {sat}%, {lum}%)"


def marker_size(elapsed_s: float, max_elapsed: float) -> float:
    """Inverse relative size: fastest = largest marker."""
    if max_elapsed <= 0 or elapsed_s <= 0:
        return 10
    # Ratio: 1.0 = slowest, approaches 0 = fastest
    ratio = elapsed_s / max_elapsed
    # Invert and scale: fastest → 14, slowest → 3
    return max(3, 14 - ratio * 11)


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
            r.setdefault("elapsed_s", 1.0)
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
            html.P(
                explanation, style={"color": "#666", "fontSize": "13px", "marginBottom": "8px", "maxWidth": "900px"}
            ),
            dcc.Graph(id=chart_id),
        ],
    )


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


FILTER_INPUTS = [
    Input("ds", "value"),
    Input("dist-fix", "value"),
    Input("jw-fix", "value"),
    Input("hi-fix", "value"),
    Input("delta-fix", "value"),
]

app.layout = html.Div(
    style={"fontFamily": "system-ui, sans-serif", "margin": "20px 40px", "maxWidth": "1400px"},
    children=[
        html.H1("ER Parameter Sweep"),
        html.P(
            f"{len(RESULTS)} results across {len(DATASETS)} dataset(s). "
            "Hue = (dataset, delta). Saturation = semantic mix (grey=JW, vivid=cosine). "
            "Luminance = swept variable. Size = speed (bigger = faster).",
            style={"color": "#555"},
        ),
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
                _dropdown("dist", "dist-fix", DISTS),
                _dropdown("jw_weight", "jw-fix", JWS),
                _dropdown("llm_high", "hi-fix", HIS),
                _dropdown("delta", "delta-fix", DELTAS, default=0.0),
            ],
        ),
        _chart_section(
            "pr-by-dist",
            "PR coloured by dist_threshold (luminance)",
            "Luminance encodes dist: light = tight blocking (small dist), dark = wide blocking (large dist). "
            "Fix jw_weight and llm_high to isolate. Saturation shows the JW/semantic mix. Size = speed.",
        ),
        _chart_section(
            "pr-by-hi",
            "PR coloured by llm_high (luminance)",
            "Luminance encodes llm_high: light = permissive (hi=0.80), dark = strict (hi=1.0). "
            "Fix dist and jw_weight to isolate. Saturation shows the JW/semantic mix. Size = speed.",
        ),
        _chart_section(
            "pr-by-jw",
            "PR coloured by jw_weight (saturation)",
            "Saturation IS jw_weight: grey/desaturated = pure lexicographic (jw=1.0), "
            "vivid/saturated = pure semantic (jw=0.0). Luminance encodes dist. Fix llm_high to isolate.",
        ),
        _chart_section(
            "pr-by-delta",
            "PR coloured by borderline_delta (hue shift)",
            "Hue shift: each dataset has two hues — one for delta=0 (no LLM), one for delta=0.05 (with LLM). "
            "Points with LLM should shift up-right vs their no-LLM pair if the LLM helps.",
        ),
    ],
)


# ── Chart Builders ────────────────────────────────────────────────


def _add_iso_lines(fig):
    """Add F1 iso-lines with labels."""
    for f1_val in [0.5, 0.6, 0.7, 0.8, 0.9]:
        r_vals = [i / 100 for i in range(10, 101)]
        p_vals = [(f1_val * rv) / (2 * rv - f1_val) if (2 * rv - f1_val) > 0 else None for rv in r_vals]
        fig.add_trace(
            go.Scatter(
                x=r_vals,
                y=p_vals,
                mode="lines",
                line={"dash": "dot", "color": "#bbb", "width": 1},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        label_r = 0.98
        label_p = (f1_val * label_r) / (2 * label_r - f1_val) if (2 * label_r - f1_val) > 0 else None
        if label_p and 0 < label_p <= 1.0:
            fig.add_annotation(
                x=label_r,
                y=label_p,
                text=f"F1={f1_val}",
                showarrow=False,
                font={"size": 10, "color": "#999"},
                xanchor="left",
            )


def _pr_layout(fig, title=""):
    fig.update_layout(
        height=550,
        title=title,
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis_range=[0, 1.05],
        yaxis_range=[0, 1.05],
        legend={"groupclick": "togglegroup", "font": {"size": 10}},
    )


def _hover_text(r):
    return (
        f"F1={r['bcubed_f1']:.3f} P={r['bcubed_precision']:.3f} R={r['bcubed_recall']:.3f}<br>"
        f"dist={r['dist']} jw={r['jw']} hi={r['hi']} δ={r['delta']}<br>"
        f"pairs={r['total_pairs']} accept={r['auto_accepted']} time={r['elapsed_s']:.1f}s"
    )


def _build_pr_chart(filtered, lum_key, lum_range):
    """Build PR chart. Hue=(dataset,delta), Sat=jw, Lum=lum_key, Size=speed."""
    fig = go.Figure()
    if not filtered:
        return fig

    max_elapsed = max(r["elapsed_s"] for r in filtered) or 1.0

    # Group by (dataset, delta) for legend
    series_keys = sorted({(r["dataset"], r["delta"]) for r in filtered})
    for ds, delta in series_keys:
        series = [r for r in filtered if r["dataset"] == ds and r["delta"] == delta]
        delta_label = "no-LLM" if delta == 0.0 else f"δ={delta}"
        fig.add_trace(
            go.Scatter(
                x=[r["bcubed_recall"] for r in series],
                y=[r["bcubed_precision"] for r in series],
                mode="markers",
                name=f"{ds} ({delta_label})",
                legendgroup=f"{ds}_{delta}",
                marker={
                    "color": [make_color(ds, delta, r["jw"], r[lum_key], lum_range) for r in series],
                    "size": [marker_size(r["elapsed_s"], max_elapsed) for r in series],
                    "opacity": 0.85,
                    "line": {"width": 0.5, "color": "#fff"},
                },
                text=[_hover_text(r) for r in series],
                hoverinfo="text",
            )
        )

    _add_iso_lines(fig)
    _pr_layout(fig)
    return fig


# ── Callbacks ─────────────────────────────────────────────────────


@app.callback(Output("pr-by-dist", "figure"), FILTER_INPUTS)
def update_dist(_datasets, _dist_fix, jw_fix, hi_fix, delta_fix):
    filtered = [
        r
        for r in RESULTS
        if r["dataset"] in (_datasets or DATASETS)
        and (jw_fix is None or r["jw"] == jw_fix)
        and (hi_fix is None or r["hi"] == hi_fix)
        and (delta_fix is None or r["delta"] == delta_fix)
    ]
    dist_vals = sorted({r["dist"] for r in filtered}) if filtered else [0, 1]
    return _build_pr_chart(filtered, "dist", (min(dist_vals), max(dist_vals)))


@app.callback(Output("pr-by-hi", "figure"), FILTER_INPUTS)
def update_hi(_datasets, dist_fix, _jw_fix, _hi_fix, delta_fix):
    filtered = [
        r
        for r in RESULTS
        if r["dataset"] in (_datasets or DATASETS)
        and (dist_fix is None or r["dist"] == dist_fix)
        and (_jw_fix is None or r["jw"] == _jw_fix)
        and (delta_fix is None or r["delta"] == delta_fix)
    ]
    hi_vals = sorted({r["hi"] for r in filtered}) if filtered else [0, 1]
    return _build_pr_chart(filtered, "hi", (min(hi_vals), max(hi_vals)))


@app.callback(Output("pr-by-jw", "figure"), FILTER_INPUTS)
def update_jw(_datasets, dist_fix, _jw_fix, hi_fix, delta_fix):
    # For jw chart, luminance = dist (secondary), saturation = jw (primary visual channel)
    filtered = [
        r
        for r in RESULTS
        if r["dataset"] in (_datasets or DATASETS)
        and (dist_fix is None or r["dist"] == dist_fix)
        and (hi_fix is None or r["hi"] == hi_fix)
        and (delta_fix is None or r["delta"] == delta_fix)
    ]
    dist_vals = sorted({r["dist"] for r in filtered}) if filtered else [0, 1]
    return _build_pr_chart(filtered, "dist", (min(dist_vals), max(dist_vals)))


@app.callback(Output("pr-by-delta", "figure"), FILTER_INPUTS)
def update_delta(_datasets, dist_fix, jw_fix, hi_fix, _delta_fix):
    # Don't filter by delta — it's the hue axis
    filtered = [
        r
        for r in RESULTS
        if r["dataset"] in (_datasets or DATASETS)
        and (dist_fix is None or r["dist"] == dist_fix)
        and (jw_fix is None or r["jw"] == jw_fix)
        and (hi_fix is None or r["hi"] == hi_fix)
    ]
    dist_vals = sorted({r["dist"] for r in filtered}) if filtered else [0, 1]
    return _build_pr_chart(filtered, "dist", (min(dist_vals), max(dist_vals)))


# ── Run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8055)
    args = p.parse_args()
    print(f"Loaded {len(RESULTS)} results")
    print(f"Dashboard at http://127.0.0.1:{args.port}")
    app.run(debug=True, port=args.port)
