"""Chart rendering: Plotly JSON + Jinja2 template rendering.

Generates:
- Plotly JSON files in CHARTS_DIR/
- Optionally renders Jinja2 templates into DOCS_BENCHMARKS_DIR
"""

import json
import logging
from pathlib import Path

from benchmarks.harness.analysis.aggregator import ChartSeries, ChartSpec
from benchmarks.harness.analysis.color_system import DISPLAY_LABELS, assign_colors
from benchmarks.harness.common import CHARTS_DIR, DOCS_BENCHMARKS_DIR

log = logging.getLogger(__name__)


def render_chart(spec: ChartSpec, series_list: list[ChartSeries], charts_dir: Path | None = None) -> None:
    """Render a chart as Plotly JSON from a ChartSpec and series data.

    Args:
        spec: ChartSpec defining the chart.
        series_list: List of ChartSeries with aggregated points.
        charts_dir: Override for output directory.
    """
    charts_dir = charts_dir or CHARTS_DIR
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Assign colors
    assign_colors(series_list)

    # Build Plotly JSON structure
    traces = []
    for s in series_list:
        x_values = [p.x for p in s.points]
        y_values = [p.y_mean for p in s.points]

        is_muninn = s.group_key.startswith("muninn")
        group_display = DISPLAY_LABELS.get(s.group_key, s.group_key.replace("_", "-"))

        trace = {
            "type": "scatter",
            "mode": "lines+markers",
            "name": s.label,
            "x": x_values,
            "y": y_values,
            "marker": {"color": s.color, "size": 8 if is_muninn else 7},
            "line": {"color": s.color, "width": 3 if is_muninn else 2},
            "opacity": 1.0 if is_muninn else 0.8,
            "legendgroup": s.group_key,
            "legendgrouptitle": {"text": group_display},
        }

        # Add error bars if we have min/max spread
        if any(p.count > 1 for p in s.points):
            trace["error_y"] = {
                "type": "data",
                "symmetric": False,
                "array": [p.y_max - p.y_mean for p in s.points],
                "arrayminus": [p.y_mean - p.y_min for p in s.points],
                "visible": True,
            }

        traces.append(trace)

    layout = {
        "title": {"text": spec.title},
        "xaxis": {
            "title": {"text": spec.x_label or spec.x_field},
            "type": "log" if spec.log_x else "linear",
        },
        "yaxis": {
            "title": {"text": spec.y_label or spec.y_field},
            "type": "log" if spec.log_y else "linear",
        },
        "template": "plotly_white",
        "legend": {
            "orientation": "v",
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 1.02,
            "groupclick": "togglegroup",
        },
    }

    chart_json = {"data": traces, "layout": layout}

    # Save Plotly JSON
    json_path = charts_dir / f"{spec.name}.json"
    json_path.write_text(json.dumps(chart_json, indent=2, default=str), encoding="utf-8")

    log.info("  Chart saved: %s", json_path)


def render_docs() -> None:
    """Render Jinja2 templates into DOCS_BENCHMARKS_DIR.

    Templates are in benchmarks/harness/templates/*.md.j2.
    Each template receives the full catalog context from doc_page_context().
    """
    from benchmarks.harness.registry import DOC_PAGES, doc_page_context

    DOCS_BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    templates_dir = Path(__file__).parent.parent / "templates"
    if not templates_dir.exists():
        log.warning("No templates directory found: %s", templates_dir)
        return

    try:
        import jinja2
    except ImportError:
        log.warning("Jinja2 not available — skipping doc rendering")
        return

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(templates_dir)),
        autoescape=False,
    )

    for template_file in sorted(templates_dir.glob("*.md.j2")):
        # Derive page slug from filename: "vss.md.j2" → "vss"
        page_slug = template_file.name.removesuffix(".md.j2")

        if page_slug not in DOC_PAGES:
            log.warning("  No DOC_PAGES entry for template %s — skipping", template_file.name)
            continue

        template = env.get_template(template_file.name)
        output_name = template_file.stem  # Remove .j2 suffix (keeps .md)
        output_path = DOCS_BENCHMARKS_DIR / output_name

        context = doc_page_context(page_slug)
        content = template.render(**context)
        output_path.write_text(content, encoding="utf-8")
        log.info("  Doc rendered: %s", output_path)
