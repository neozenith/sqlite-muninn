"""Embed chart definitions — 12 charts for live GGUF embedding benchmarks."""

from typing import Any

from benchmarks.harness.analysis.aggregator import ChartSpec

# Common field sets for Embed charts
_EMBED_SOURCES = ["embed_*.jsonl"]
_EMBED_GROUP = ["embed_fn", "search_backend"]
_EMBED_REPEAT = ["embed_fn", "search_backend", "model", "dataset", "dim", "n", "k"]

# ── Query+Search Latency charts (4): per model x dataset ──

_QL_COMMON: dict[str, Any] = {
    "sources": _EMBED_SOURCES,
    "x_field": "n",
    "y_field": "query_embed_search_latency_ms",
    "group_fields": _EMBED_GROUP,
    "variant_fields": [],
    "repeat_fields": _EMBED_REPEAT,
    "y_label": "Query Embed+Search Latency (ms)",
    "x_label": "Dataset Size (N vectors)",
    "log_x": True,
    "log_y": True,
}

# ── Cross-Model Comparison charts (2): all models on one chart, per dataset ──

_XMODEL_COMMON: dict[str, Any] = {
    "sources": _EMBED_SOURCES,
    "x_field": "n",
    "y_field": "query_embed_search_latency_ms",
    "group_fields": _EMBED_GROUP,
    "variant_fields": ["model"],
    "repeat_fields": _EMBED_REPEAT,
    "y_label": "Query Embed+Search Latency (ms)",
    "x_label": "Dataset Size (N vectors)",
    "log_x": True,
    "log_y": True,
}

# ── Embedding-Only Latency charts (2): per dataset ──

_EMBED_ONLY_COMMON: dict[str, Any] = {
    "sources": _EMBED_SOURCES,
    "x_field": "n",
    "y_field": "query_embed_only_ms",
    "group_fields": _EMBED_GROUP,
    "variant_fields": ["model"],
    "repeat_fields": _EMBED_REPEAT,
    "y_label": "Embedding-Only Latency (ms)",
    "x_label": "Dataset Size (N vectors)",
    "log_x": True,
    "log_y": True,
}

# ── Insert Throughput charts (2): per dataset ──

_INSERT_COMMON: dict[str, Any] = {
    "sources": _EMBED_SOURCES,
    "x_field": "n",
    "y_field": "bulk_embed_insert_rate_vps",
    "group_fields": _EMBED_GROUP,
    "variant_fields": ["model"],
    "repeat_fields": _EMBED_REPEAT,
    "y_label": "Embed+Insert Rate (vectors/sec)",
    "x_label": "Dataset Size (N vectors)",
    "log_x": True,
}

# ── Recall charts (2): per dataset ──

_RECALL_COMMON: dict[str, Any] = {
    "sources": _EMBED_SOURCES,
    "x_field": "n",
    "y_field": "recall_at_k",
    "group_fields": _EMBED_GROUP,
    "variant_fields": ["model"],
    "repeat_fields": _EMBED_REPEAT,
    "y_label": "Recall@k",
    "x_label": "Dataset Size (N vectors)",
    "log_x": True,
}


EMBED_CHARTS = [
    # ── Query+Search Latency: AG News (2 models) ──
    ChartSpec(
        name="embed_ql_MiniLM_ag_news",
        title="Query Embed+Search Latency (MiniLM / AG News)",
        filters={"model": "MiniLM", "dataset": "ag_news"},
        **_QL_COMMON,
    ),
    ChartSpec(
        name="embed_ql_NomicEmbed_ag_news",
        title="Query Embed+Search Latency (NomicEmbed / AG News)",
        filters={"model": "NomicEmbed", "dataset": "ag_news"},
        **_QL_COMMON,
    ),
    # ── Query+Search Latency: Wealth of Nations (2 models) ──
    ChartSpec(
        name="embed_ql_MiniLM_wealth_of_nations",
        title="Query Embed+Search Latency (MiniLM / Wealth of Nations)",
        filters={"model": "MiniLM", "dataset": "wealth_of_nations"},
        **_QL_COMMON,
    ),
    ChartSpec(
        name="embed_ql_NomicEmbed_wealth_of_nations",
        title="Query Embed+Search Latency (NomicEmbed / Wealth of Nations)",
        filters={"model": "NomicEmbed", "dataset": "wealth_of_nations"},
        **_QL_COMMON,
    ),
    # ── Cross-Model Comparison (2 datasets) ──
    ChartSpec(
        name="embed_xmodel_ag_news",
        title="Cross-Model Comparison (AG News)",
        filters={"dataset": "ag_news"},
        **_XMODEL_COMMON,
    ),
    ChartSpec(
        name="embed_xmodel_wealth_of_nations",
        title="Cross-Model Comparison (Wealth of Nations)",
        filters={"dataset": "wealth_of_nations"},
        **_XMODEL_COMMON,
    ),
    # ── Embedding-Only Latency (2 datasets) ──
    ChartSpec(
        name="embed_only_ag_news",
        title="Embedding-Only Latency (AG News)",
        filters={"dataset": "ag_news"},
        **_EMBED_ONLY_COMMON,
    ),
    ChartSpec(
        name="embed_only_wealth_of_nations",
        title="Embedding-Only Latency (Wealth of Nations)",
        filters={"dataset": "wealth_of_nations"},
        **_EMBED_ONLY_COMMON,
    ),
    # ── Insert Throughput (2 datasets) ──
    ChartSpec(
        name="embed_insert_ag_news",
        title="Embed+Insert Throughput (AG News)",
        filters={"dataset": "ag_news"},
        **_INSERT_COMMON,
    ),
    ChartSpec(
        name="embed_insert_wealth_of_nations",
        title="Embed+Insert Throughput (Wealth of Nations)",
        filters={"dataset": "wealth_of_nations"},
        **_INSERT_COMMON,
    ),
    # ── Recall (2 datasets) ──
    ChartSpec(
        name="embed_recall_ag_news",
        title="Recall@k (AG News)",
        filters={"dataset": "ag_news"},
        **_RECALL_COMMON,
    ),
    ChartSpec(
        name="embed_recall_wealth_of_nations",
        title="Recall@k (Wealth of Nations)",
        filters={"dataset": "wealth_of_nations"},
        **_RECALL_COMMON,
    ),
]
