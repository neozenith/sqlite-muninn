"""Graph VT chart definitions — 8 charts matching legacy parity.

Covers per-algorithm query time (degree, betweenness, closeness, leiden),
rebuild time, initial build time, disk usage, and trigger overhead.

Uses wide-format JSONL where each algorithm timing is a separate column
(degree_ms, betweenness_ms, etc.) rather than a row per operation.
"""

from benchmarks.harness.analysis.aggregator import ChartSpec

# Common fields for per-algorithm query time charts
_ALGO_COMMON = {
    "sources": ["graph_vt_*.jsonl"],
    "filters": {},
    "x_field": "n_nodes",
    "group_fields": ["approach"],
    "variant_fields": [],
    "repeat_fields": ["approach", "workload", "graph_model"],
    "x_label": "Graph Size (nodes)",
    "log_x": True,
    "log_y": True,
}


GRAPH_VT_CHARTS = [
    # ── Per-Algorithm Query Time (4 charts) ──
    ChartSpec(
        name="graph_vt_degree",
        title="Degree Query Time",
        y_field="degree_ms",
        y_label="Degree Computation (ms)",
        **_ALGO_COMMON,
    ),
    ChartSpec(
        name="graph_vt_betweenness",
        title="Betweenness Query Time",
        y_field="betweenness_ms",
        y_label="Betweenness Computation (ms)",
        **_ALGO_COMMON,
    ),
    ChartSpec(
        name="graph_vt_closeness",
        title="Closeness Query Time",
        y_field="closeness_ms",
        y_label="Closeness Computation (ms)",
        **_ALGO_COMMON,
    ),
    ChartSpec(
        name="graph_vt_leiden",
        title="Leiden Query Time",
        y_field="leiden_ms",
        y_label="Leiden Computation (ms)",
        **_ALGO_COMMON,
    ),
    # ── Rebuild Time (1 chart) ──
    ChartSpec(
        name="graph_vt_rebuild",
        title="Rebuild Time",
        sources=["graph_vt_*.jsonl"],
        filters={},
        x_field="n_nodes",
        y_field="rebuild_ms",
        group_fields=["approach"],
        variant_fields=[],
        repeat_fields=["approach", "workload", "graph_model"],
        y_label="Rebuild Time (ms)",
        x_label="Graph Size (nodes)",
        log_x=True,
        log_y=True,
    ),
    # ── Initial CSR Build Time (1 chart) — only csr approach has build_ms ──
    ChartSpec(
        name="graph_vt_build",
        title="Initial CSR Build Time",
        sources=["graph_vt_*.jsonl"],
        filters={},
        x_field="n_nodes",
        y_field="build_ms",
        group_fields=["approach"],
        variant_fields=[],
        repeat_fields=["approach", "workload", "graph_model"],
        y_label="Build Time (ms)",
        x_label="Graph Size (nodes)",
        log_x=True,
        log_y=True,
    ),
    # ── Disk Usage (1 chart) — data not yet captured by treatments ──
    ChartSpec(
        name="graph_vt_disk",
        title="Shadow Table Disk Usage",
        sources=["graph_vt_*.jsonl"],
        filters={},
        x_field="n_nodes",
        y_field="disk_bytes",
        group_fields=["approach"],
        variant_fields=[],
        repeat_fields=["approach", "workload", "graph_model"],
        y_label="Disk Usage (bytes)",
        x_label="Graph Size (nodes)",
        log_x=True,
        log_y=True,
    ),
    # ── Trigger Overhead (1 chart) — data not yet captured by treatments ──
    ChartSpec(
        name="graph_vt_trigger",
        title="Trigger Overhead",
        sources=["graph_vt_*.jsonl"],
        filters={},
        x_field="n_nodes",
        y_field="trigger_overhead_ms",
        group_fields=["approach"],
        variant_fields=[],
        repeat_fields=["approach", "workload", "graph_model"],
        y_label="Overhead (ms)",
        x_label="Graph Size (nodes)",
        log_x=True,
    ),
]
