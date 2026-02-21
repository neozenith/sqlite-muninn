"""Graph chart definitions — 10 charts matching legacy parity.

Covers traversal (BFS, DFS, shortest path, components, PageRank),
centrality (degree, betweenness, closeness), community (Leiden),
and graph setup time.
"""

from benchmarks.harness.analysis.aggregator import ChartSpec

# Common field sets for graph traversal charts
_TRAV_COMMON = {
    "sources": ["graph_*.jsonl"],
    "x_field": "n_nodes",
    "y_field": "query_time_ms",
    "group_fields": ["engine"],
    "variant_fields": ["graph_model", "avg_degree"],
    "repeat_fields": ["engine", "operation", "graph_model", "n_nodes", "avg_degree"],
    "x_label": "Number of Nodes",
    "log_x": True,
    "log_y": True,
}

# Common field sets for centrality charts
_CENT_COMMON = {
    "sources": ["centrality_*.jsonl"],
    "x_field": "n_nodes",
    "y_field": "query_time_ms",
    "group_fields": ["engine"],
    "variant_fields": ["graph_model", "avg_degree"],
    "repeat_fields": ["engine", "operation", "graph_model", "n_nodes", "avg_degree"],
    "x_label": "Number of Nodes",
    "log_x": True,
    "log_y": True,
}


GRAPH_CHARTS = [
    # ── Traversal (5 charts) ──
    ChartSpec(
        name="graph_query_time_bfs",
        title="BFS",
        filters={"operation": "bfs"},
        y_label="BFS Query Time (ms)",
        **_TRAV_COMMON,
    ),
    ChartSpec(
        name="graph_query_time_dfs",
        title="DFS",
        filters={"operation": "dfs"},
        y_label="DFS Query Time (ms)",
        **_TRAV_COMMON,
    ),
    ChartSpec(
        name="graph_query_time_shortest_path",
        title="Shortest Path",
        filters={"operation": "shortest_path"},
        y_label="Shortest Path Query Time (ms)",
        **_TRAV_COMMON,
    ),
    ChartSpec(
        name="graph_query_time_components",
        title="Connected Components",
        filters={"operation": "components"},
        y_label="Components Query Time (ms)",
        **_TRAV_COMMON,
    ),
    ChartSpec(
        name="graph_query_time_pagerank",
        title="PageRank",
        filters={"operation": "pagerank"},
        y_label="PageRank Query Time (ms)",
        **_TRAV_COMMON,
    ),
    # ── Centrality (3 charts) ──
    ChartSpec(
        name="graph_query_time_degree",
        title="Degree Centrality",
        filters={"operation": "degree"},
        y_label="Degree Centrality Time (ms)",
        **_CENT_COMMON,
    ),
    ChartSpec(
        name="graph_query_time_betweenness",
        title="Betweenness Centrality",
        filters={"operation": "betweenness"},
        y_label="Betweenness Centrality Time (ms)",
        **_CENT_COMMON,
    ),
    ChartSpec(
        name="graph_query_time_closeness",
        title="Closeness Centrality",
        filters={"operation": "closeness"},
        y_label="Closeness Centrality Time (ms)",
        **_CENT_COMMON,
    ),
    # ── Community Detection (1 chart) ──
    ChartSpec(
        name="graph_query_time_leiden",
        title="Leiden Community Detection",
        sources=["community_*.jsonl"],
        filters={},
        x_field="n_nodes",
        y_field="query_time_ms",
        group_fields=["graph_model"],
        variant_fields=["avg_degree"],
        repeat_fields=["engine", "graph_model", "n_nodes", "avg_degree"],
        y_label="Leiden Query Time (ms)",
        x_label="Number of Nodes",
        log_x=True,
        log_y=True,
    ),
    # ── Setup Time (1 chart) ──
    ChartSpec(
        name="graph_setup_time",
        title="Insertion Throughput",
        sources=["graph_*.jsonl"],
        filters={},
        x_field="n_nodes",
        y_field="wall_time_setup_ms",
        group_fields=["engine"],
        variant_fields=["graph_model", "avg_degree"],
        repeat_fields=["engine", "graph_model", "n_nodes", "avg_degree"],
        y_label="Setup Time (ms)",
        x_label="Number of Nodes",
        log_x=True,
        log_y=True,
    ),
]
