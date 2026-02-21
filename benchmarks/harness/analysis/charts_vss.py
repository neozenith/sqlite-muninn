"""VSS chart definitions — 17 charts matching legacy parity."""

from benchmarks.harness.analysis.aggregator import ChartSpec

# Common field sets for VSS charts
_VSS_SOURCES = ["vss_*.jsonl"]
_VSS_GROUP = ["engine", "search_method"]
_VSS_REPEAT = ["engine", "search_method", "model_name", "dataset", "dim", "n", "k"]

# ── Tipping Point charts (6): search latency vs dataset size, per model × dataset ──

_TP_COMMON = {
    "sources": _VSS_SOURCES,
    "x_field": "n",
    "y_field": "search_latency_ms",
    "group_fields": _VSS_GROUP,
    "variant_fields": [],
    "repeat_fields": _VSS_REPEAT,
    "y_label": "Search Latency (ms)",
    "x_label": "Dataset Size (N vectors)",
    "log_x": True,
    "log_y": True,
}

# ── Cross-Model Comparison charts (2): all models on one chart, per dataset ──

_XMODEL_COMMON = {
    "sources": _VSS_SOURCES,
    "x_field": "n",
    "y_field": "search_latency_ms",
    "group_fields": _VSS_GROUP,
    "variant_fields": ["model_name"],
    "repeat_fields": _VSS_REPEAT,
    "y_label": "Search Latency (ms)",
    "x_label": "Dataset Size (N vectors)",
    "log_x": True,
    "log_y": True,
}

# ── Dataset Comparison charts (3): overlay datasets, per model ──

_DS_COMMON = {
    "sources": _VSS_SOURCES,
    "x_field": "n",
    "y_field": "search_latency_ms",
    "group_fields": _VSS_GROUP,
    "variant_fields": ["dataset"],
    "repeat_fields": _VSS_REPEAT,
    "y_label": "Search Latency (ms)",
    "x_label": "Dataset Size (N vectors)",
    "log_x": True,
    "log_y": True,
}

# ── Recall charts (2): all models, per dataset ──

_RECALL_COMMON = {
    "sources": _VSS_SOURCES,
    "x_field": "n",
    "y_field": "recall_at_k",
    "group_fields": _VSS_GROUP,
    "variant_fields": ["model_name"],
    "repeat_fields": _VSS_REPEAT,
    "y_label": "Recall@k",
    "x_label": "Dataset Size (N vectors)",
    "log_x": True,
}

# ── Insert Throughput charts (2): all models, per dataset ──

_INSERT_COMMON = {
    "sources": _VSS_SOURCES,
    "x_field": "n",
    "y_field": "insert_rate_vps",
    "group_fields": _VSS_GROUP,
    "variant_fields": ["model_name"],
    "repeat_fields": _VSS_REPEAT,
    "y_label": "Insert Rate (vectors/sec)",
    "x_label": "Dataset Size (N vectors)",
    "log_x": True,
}

# ── Storage charts (2): all models, per dataset ──

_STORAGE_COMMON = {
    "sources": _VSS_SOURCES,
    "x_field": "n",
    "y_field": "db_size_bytes",
    "group_fields": _VSS_GROUP,
    "variant_fields": ["model_name"],
    "repeat_fields": _VSS_REPEAT,
    "y_label": "Database Size (bytes)",
    "x_label": "Dataset Size (N vectors)",
    "log_x": True,
    "log_y": True,
}


VSS_CHARTS = [
    # ── Tipping Point: AG News (3 models) ──
    ChartSpec(
        name="tipping_point_MiniLM_ag_news",
        title="Search Latency vs Dataset Size (MiniLM / AG News)",
        filters={"model_name": "MiniLM", "dataset": "ag_news"},
        **_TP_COMMON,
    ),
    ChartSpec(
        name="tipping_point_MPNet_ag_news",
        title="Search Latency vs Dataset Size (MPNet / AG News)",
        filters={"model_name": "MPNet", "dataset": "ag_news"},
        **_TP_COMMON,
    ),
    ChartSpec(
        name="tipping_point_BGE-Large_ag_news",
        title="Search Latency vs Dataset Size (BGE-Large / AG News)",
        filters={"model_name": "BGE-Large", "dataset": "ag_news"},
        **_TP_COMMON,
    ),
    # ── Cross-Model Comparison: AG News ──
    ChartSpec(
        name="model_comparison_ag_news",
        title="Cross-Model Comparison (AG News)",
        filters={"dataset": "ag_news"},
        **_XMODEL_COMMON,
    ),
    # ── Tipping Point: Wealth of Nations (3 models) ──
    ChartSpec(
        name="tipping_point_MiniLM_wealth_of_nations",
        title="Search Latency vs Dataset Size (MiniLM / Wealth of Nations)",
        filters={"model_name": "MiniLM", "dataset": "wealth_of_nations"},
        **_TP_COMMON,
    ),
    ChartSpec(
        name="tipping_point_MPNet_wealth_of_nations",
        title="Search Latency vs Dataset Size (MPNet / Wealth of Nations)",
        filters={"model_name": "MPNet", "dataset": "wealth_of_nations"},
        **_TP_COMMON,
    ),
    ChartSpec(
        name="tipping_point_BGE-Large_wealth_of_nations",
        title="Search Latency vs Dataset Size (BGE-Large / Wealth of Nations)",
        filters={"model_name": "BGE-Large", "dataset": "wealth_of_nations"},
        **_TP_COMMON,
    ),
    # ── Cross-Model Comparison: Wealth of Nations ──
    ChartSpec(
        name="model_comparison_wealth_of_nations",
        title="Cross-Model Comparison (Wealth of Nations)",
        filters={"dataset": "wealth_of_nations"},
        **_XMODEL_COMMON,
    ),
    # ── Dataset Comparison (3 models) ──
    ChartSpec(
        name="dataset_comparison_MiniLM",
        title="Dataset Comparison (MiniLM)",
        filters={"model_name": "MiniLM", "dataset": ["ag_news", "wealth_of_nations"]},
        **_DS_COMMON,
    ),
    ChartSpec(
        name="dataset_comparison_MPNet",
        title="Dataset Comparison (MPNet)",
        filters={"model_name": "MPNet", "dataset": ["ag_news", "wealth_of_nations"]},
        **_DS_COMMON,
    ),
    ChartSpec(
        name="dataset_comparison_BGE-Large",
        title="Dataset Comparison (BGE-Large)",
        filters={"model_name": "BGE-Large", "dataset": ["ag_news", "wealth_of_nations"]},
        **_DS_COMMON,
    ),
    # ── Recall (2 datasets) ──
    ChartSpec(
        name="recall_models_ag_news",
        title="Recall@k (AG News)",
        filters={"dataset": "ag_news"},
        **_RECALL_COMMON,
    ),
    ChartSpec(
        name="recall_models_wealth_of_nations",
        title="Recall@k (Wealth of Nations)",
        filters={"dataset": "wealth_of_nations"},
        **_RECALL_COMMON,
    ),
    # ── Insert Throughput (2 datasets) ──
    ChartSpec(
        name="insert_throughput_models_ag_news",
        title="Insert Throughput (AG News)",
        filters={"dataset": "ag_news"},
        **_INSERT_COMMON,
    ),
    ChartSpec(
        name="insert_throughput_models_wealth_of_nations",
        title="Insert Throughput (Wealth of Nations)",
        filters={"dataset": "wealth_of_nations"},
        **_INSERT_COMMON,
    ),
    # ── Storage (2 datasets) ──
    ChartSpec(
        name="db_size_models_ag_news",
        title="Storage (AG News)",
        filters={"dataset": "ag_news"},
        **_STORAGE_COMMON,
    ),
    ChartSpec(
        name="db_size_models_wealth_of_nations",
        title="Storage (Wealth of Nations)",
        filters={"dataset": "wealth_of_nations"},
        **_STORAGE_COMMON,
    ),
]
