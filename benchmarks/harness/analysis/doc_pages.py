"""Documentation page catalog: builds DOC_PAGES dicts and doc_page_context() for Jinja2 rendering.

Programmatically constructs page metadata from the annotated dimensional-axis
constants in common.py. Consumed by analysis/renderer.py and tests/test_docs.py.
"""

from typing import Any

from benchmarks.harness.common import (
    CHARTS_DIR,
    DATASETS,
    EMBED_FNS,
    EMBED_SEARCH_BACKENDS,
    EMBEDDING_MODELS,
    GRAPH_MODELS,
    GRAPH_TVF_ENGINES,
    GRAPH_VT_APPROACHES,
    GRAPH_VT_BLOCK_SIZE,
    GRAPH_VT_WORKLOADS,
    KG_NER_DATASETS,
    KG_NER_MODELS,
    KG_RE_DATASETS,
    KG_RESOLUTION_DATASETS,
    PROJECT_ROOT,
    VSS_ENGINES,
)

# ── DOC_PAGES helpers ─────────────────────────────────────────────


def _linked(label: str, url: str | None) -> str:
    """Return a markdown link, or just the label if url is None."""
    if url:
        return f"[{label}]({url})"
    return label


def _build_vss_page() -> dict[str, Any]:
    """Build the VSS doc page from annotated constants."""
    return {
        "title": "Vector Search Benchmarks",
        "description": (
            "Compares vector similarity search engines on insert throughput, "
            "search latency, and recall across multiple embedding models, "
            "dataset sizes, and corpus types."
        ),
        "tables": [
            {
                "title": "Engines",
                "columns": ["Engine", "Library", "Method", "Strategy"],
                "rows": [
                    {
                        "Engine": f"**{e['display']}**",
                        "Library": _linked(e["library_name"], e["library_url"]),
                        "Method": e["method"],
                        "Strategy": e["strategy"],
                    }
                    for e in VSS_ENGINES
                ],
            },
            {
                "title": "Datasets",
                "columns": ["Dataset", "Source", "Passages", "Topology"],
                "rows": [
                    {
                        "Dataset": _linked(d["display_name"], d["url"]),
                        "Source": d["source_label"],
                        "Passages": d["passages_desc"],
                        "Topology": d["topology"],
                    }
                    for d in DATASETS.values()
                ],
            },
            {
                "title": "Embedding Models",
                "columns": ["Model", "Dimension", "Params", "Doc Prefix", "Query Prefix"],
                "rows": [
                    {
                        "Model": f"**{name}**",
                        "Dimension": str(m["dim"]),
                        "Params": m["params"],
                        "Doc Prefix": f'`"{m["doc_prefix"]}"`' if m["doc_prefix"] else "_(none)_",
                        "Query Prefix": f'`"{m["query_prefix"][:30]}..."`'
                        if len(m["query_prefix"]) > 30
                        else (f'`"{m["query_prefix"]}"`' if m["query_prefix"] else "_(none)_"),
                    }
                    for name, m in EMBEDDING_MODELS.items()
                ],
            },
        ],
        "sections": None,
    }


def _build_embed_page() -> dict[str, Any]:
    """Build the Embed doc page from annotated constants."""
    return {
        "title": "Embed Benchmarks",
        "description": (
            "Compares **[`llama.cpp`](https://github.com/ggml-org/llama.cpp)**-based embedding functions on metrics:\n"
            "\n"
            "- end-to-end query latency,\n"
            "- embedding-only latency,\n"
            "- insert throughput,\n"
            "- and recall\n"
            "\n"
            "...across multiple **GGUF** embedding models, search backends, and corpus types."
        ),
        "tables": [
            {
                "title": "Embedding Functions",
                "columns": ["Function", "Description"],
                "rows": [
                    {
                        "Function": f"**{fn['display']}**",
                        "Description": fn["description"],
                    }
                    for fn in EMBED_FNS
                ],
            },
            {
                "title": "Search Backends",
                "columns": ["Backend", "Method", "Strategy"],
                "rows": [
                    {
                        "Backend": f"**{b['display']}**",
                        "Method": b["method"],
                        "Strategy": b["strategy"],
                    }
                    for b in EMBED_SEARCH_BACKENDS
                ],
            },
            {
                "title": "GGUF Embedding Models",
                "columns": ["Model", "Dimension", "Params", "GGUF File"],
                "rows": [
                    {
                        "Model": f"**{name}**",
                        "Dimension": str(m["dim"]),
                        "Params": m["params"],
                        "GGUF File": f"`{m['gguf_filename']}`",
                    }
                    for name, m in EMBEDDING_MODELS.items()
                    if m.get("embed_enabled", True)
                ],
            },
            {
                "title": "Datasets",
                "columns": ["Dataset", "Source", "Passages", "Topology"],
                "rows": [
                    {
                        "Dataset": _linked(d["display_name"], d["url"]),
                        "Source": d["source_label"],
                        "Passages": d["passages_desc"],
                        "Topology": d["topology"],
                    }
                    for d in DATASETS.values()
                ],
            },
        ],
        "sections": None,
    }


def _build_graph_page() -> dict[str, Any]:
    """Build the Graph doc page from annotated constants."""
    return {
        "title": "Graph Benchmarks",
        "description": (
            "Measures graph algorithm performance across traversal, centrality, "
            "community detection, and Node2Vec embedding generation. Benchmarks "
            "compare muninn's built-in TVFs against alternative SQLite graph engines "
            "where available."
        ),
        "tables": [
            {
                "title": "Engines",
                "columns": ["Engine", "Library", "Traversal", "Centrality", "Community", "Node2Vec"],
                "rows": [
                    {
                        "Engine": f"**{e['display']}**",
                        "Library": _linked(e["library_name"], e["library_url"]),
                        "Traversal": e["traversal"],
                        "Centrality": e["centrality"],
                        "Community": e["community"],
                        "Node2Vec": e["node2vec"],
                    }
                    for e in GRAPH_TVF_ENGINES
                ],
            },
            {
                "title": "Graph Models",
                "columns": ["Model", "Description", "Key Property"],
                "rows": [
                    {
                        "Model": _linked(m["display"], m["url"]),
                        "Description": m["description"],
                        "Key Property": m["key_property"],
                    }
                    for m in GRAPH_MODELS
                ],
            },
        ],
        "sections_intro": (
            "Graphs are tested at multiple sizes (100 to 10,000 nodes) and average degrees "
            "(3 to 20 edges per node), producing a range from sparse to dense topologies."
        ),
        "sections": [
            {
                "title": "Traversal",
                "description": "BFS, DFS, shortest path, components, and PageRank. Both engines.",
            },
            {
                "title": "Centrality",
                "description": (
                    "Degree, betweenness (Brandes algorithm), and closeness centrality. "
                    "Muninn-only \u2014 measures how centrality computation scales with graph density."
                ),
            },
            {
                "title": "Community Detection",
                "description": "Leiden algorithm for community detection. Compared across both engines.",
            },
            {
                "title": "Node2Vec",
                "description": (
                    "Random walk generation and Skip-gram training with Negative Sampling. "
                    "Tests vary walk parameters (p, q) and embedding dimensionality (64, 128)."
                ),
            },
        ],
    }


def _build_graph_vt_page() -> dict[str, Any]:
    """Build the Graph VT doc page from annotated constants."""
    return {
        "title": "Graph VT Benchmarks",
        "description": (
            "Compares CSR adjacency caching strategies against direct TVF access "
            "after edge mutations. Each chart plots performance against graph size "
            "(node count) on a log-log scale."
        ),
        "tables": [
            {
                "title": "Methods",
                "columns": ["Method", "Description"],
                "rows": [
                    {
                        "Method": f"**{a['display']}**",
                        "Description": a["description"],
                    }
                    for a in GRAPH_VT_APPROACHES
                ],
            },
            {
                "title": "Workloads",
                "columns": ["Size", "Nodes", "Edges", "Graph Model"],
                "rows": [
                    {
                        "Size": w["name"],
                        "Nodes": f"{w['n_nodes']:,}",
                        "Edges": f"{w['target_edges']:,}",
                        "Graph Model": next(
                            (m["display"] for m in GRAPH_MODELS if m["slug"] == w["graph_model"]),
                            w["graph_model"],
                        ),
                    }
                    for w in GRAPH_VT_WORKLOADS
                ],
            },
        ],
        "sections": [
            {
                "title": "How Blocked CSR Works",
                "description": (
                    f"The CSR is partitioned into blocks of {GRAPH_VT_BLOCK_SIZE:,} nodes. Each block is a "
                    "separate row in the shadow table. When edges change, only blocks "
                    "containing affected nodes are rewritten \u2014 unaffected blocks require zero I/O."
                ),
            },
        ],
    }


def _kg_extraction_section() -> dict[str, Any]:
    """Build the NER Extraction section for the KG doc page."""
    return {
        "title": "NER Extraction",
        "description": (
            "Compares NER models on entity extraction quality (micro F1) and performance. "
            "Evaluated on Gutenberg texts and standard NER benchmark datasets with gold labels."
        ),
        "tables": [
            {
                "title": "NER Models",
                "columns": ["Model", "Type", "Params", "Size", "Description"],
                "rows": [
                    {
                        "Model": f"**{m['display']}**",
                        "Type": m["type"],
                        "Params": m["params"] or "\u2014",
                        "Size": f"{m['size_mb']} MB" if m["size_mb"] else "\u2014",
                        "Description": m["description"],
                    }
                    for m in KG_NER_MODELS
                ],
            },
            {
                "title": "NER Datasets",
                "columns": ["Dataset", "Source", "Description"],
                "rows": [
                    {
                        "Dataset": _linked(d["display"], d.get("url")),
                        "Source": d["source"],
                        "Description": d["description"],
                    }
                    for d in KG_NER_DATASETS
                ],
            },
        ],
    }


def _kg_re_section() -> dict[str, Any]:
    """Build the Relation Extraction section for the KG doc page."""
    return {
        "title": "Relation Extraction",
        "description": (
            "Evaluates relation extraction quality (triple F1) on standard RE benchmark datasets. "
            "Uses NER-based entity pair extraction as a crude relation proxy."
        ),
        "tables": [
            {
                "title": "RE Datasets",
                "columns": ["Dataset", "Source", "Description"],
                "rows": [
                    {
                        "Dataset": _linked(d["display"], d["url"]),
                        "Source": d["source"],
                        "Description": d["description"],
                    }
                    for d in KG_RE_DATASETS
                ],
            },
        ],
    }


def _kg_resolution_section() -> dict[str, Any]:
    """Build the Entity Resolution section for the KG doc page."""
    return {
        "title": "Entity Resolution",
        "description": (
            "Evaluates the HNSW blocking + Jaro-Winkler matching + Leiden clustering "
            "pipeline on standard ER benchmark datasets."
        ),
        "tables": [
            {
                "title": "ER Datasets",
                "columns": ["Dataset", "Source", "Description"],
                "rows": [
                    {
                        "Dataset": d["display"],
                        "Source": d["source"],
                        "Description": d["description"],
                    }
                    for d in KG_RESOLUTION_DATASETS
                ],
            },
        ],
    }


def _kg_graphrag_section() -> dict[str, Any]:
    """Build the GraphRAG Retrieval section for the KG doc page."""
    return {
        "title": "GraphRAG Retrieval",
        "description": "Measures whether graph expansion after a VSS or BM25 entry point improves retrieval quality.",
        "tables": [
            {
                "title": "Retrieval Configurations",
                "columns": ["Entry Point", "Expansion", "Description"],
                "rows": [
                    {
                        "Entry Point": "VSS",
                        "Expansion": "none / BFS-1 / BFS-2",
                        "Description": "Semantic vector search with optional 1-hop or 2-hop graph expansion",
                    },
                    {
                        "Entry Point": "BM25",
                        "Expansion": "none / BFS-1 / BFS-2",
                        "Description": "FTS5 keyword search with optional graph expansion",
                    },
                ],
            },
        ],
    }


def _build_kg_page() -> dict[str, Any]:
    """Build the KG doc page from annotated constants."""
    return {
        "title": "Knowledge Graph Benchmarks",
        "description": (
            "End-to-end benchmarks for the knowledge graph pipeline: entity extraction, "
            "relation extraction, entity resolution, and graph-augmented retrieval (GraphRAG)."
        ),
        "tables": [],
        "sections": [
            _kg_extraction_section(),
            _kg_re_section(),
            _kg_resolution_section(),
            _kg_graphrag_section(),
        ],
    }


# ── Chart group definitions for doc pages ─────────────────────────

_VSS_CHART_GROUPS = [
    {
        "title": "Search Latency — AG News",
        "charts": [
            "tipping_point_MiniLM_ag_news",
            "tipping_point_NomicEmbed_ag_news",
            "tipping_point_BGE-Large_ag_news",
        ],
    },
    {
        "title": "Cross-Model Comparison — AG News",
        "charts": ["model_comparison_ag_news"],
    },
    {
        "title": "Search Latency — Wealth of Nations",
        "charts": [
            "tipping_point_MiniLM_wealth_of_nations",
            "tipping_point_NomicEmbed_wealth_of_nations",
            "tipping_point_BGE-Large_wealth_of_nations",
        ],
    },
    {
        "title": "Cross-Model Comparison — Wealth of Nations",
        "charts": ["model_comparison_wealth_of_nations"],
    },
    {
        "title": "Dataset Comparison",
        "charts": [
            "dataset_comparison_MiniLM",
            "dataset_comparison_NomicEmbed",
            "dataset_comparison_BGE-Large",
        ],
    },
    {
        "title": "Recall",
        "charts": [
            "recall_models_ag_news",
            "recall_models_wealth_of_nations",
        ],
    },
    {
        "title": "Insert Throughput",
        "charts": [
            "insert_throughput_models_ag_news",
            "insert_throughput_models_wealth_of_nations",
        ],
    },
    {
        "title": "Storage",
        "charts": [
            "db_size_models_ag_news",
            "db_size_models_wealth_of_nations",
        ],
    },
]

_EMBED_CHART_GROUPS = [
    {
        "title": "Query+Search Latency — AG News",
        "charts": [
            "embed_ql_MiniLM_ag_news",
            "embed_ql_NomicEmbed_ag_news",
        ],
    },
    {
        "title": "Query+Search Latency — Wealth of Nations",
        "charts": [
            "embed_ql_MiniLM_wealth_of_nations",
            "embed_ql_NomicEmbed_wealth_of_nations",
        ],
    },
    {
        "title": "Cross-Model Comparison — AG News",
        "charts": ["embed_xmodel_ag_news"],
    },
    {
        "title": "Cross-Model Comparison — Wealth of Nations",
        "charts": ["embed_xmodel_wealth_of_nations"],
    },
    {
        "title": "Embedding-Only Latency",
        "charts": [
            "embed_only_ag_news",
            "embed_only_wealth_of_nations",
        ],
    },
    {
        "title": "Insert Throughput",
        "charts": [
            "embed_insert_ag_news",
            "embed_insert_wealth_of_nations",
        ],
    },
    {
        "title": "Recall",
        "charts": [
            "embed_recall_ag_news",
            "embed_recall_wealth_of_nations",
        ],
    },
]

_GRAPH_CHART_GROUPS = [
    {
        "title": "Traversal",
        "charts": [
            "graph_query_time_bfs",
            "graph_query_time_dfs",
            "graph_query_time_shortest_path",
            "graph_query_time_components",
            "graph_query_time_pagerank",
        ],
    },
    {
        "title": "Centrality",
        "charts": [
            "graph_query_time_degree",
            "graph_query_time_betweenness",
            "graph_query_time_closeness",
        ],
    },
    {
        "title": "Community Detection",
        "charts": ["graph_query_time_leiden"],
    },
    {
        "title": "Insertion Throughput",
        "charts": ["graph_setup_time"],
    },
]

_GRAPH_VT_CHART_GROUPS = [
    {
        "title": "Algorithm Query Time",
        "charts": [
            "graph_vt_degree",
            "graph_vt_betweenness",
            "graph_vt_closeness",
            "graph_vt_leiden",
        ],
    },
    {
        "title": "Rebuild Performance",
        "charts": ["graph_vt_rebuild"],
    },
    {
        "title": "Build Performance",
        "charts": ["graph_vt_build"],
    },
    {
        "title": "Storage",
        "charts": ["graph_vt_disk"],
    },
    {
        "title": "Trigger Overhead",
        "charts": ["graph_vt_trigger"],
    },
]

_CHART_GROUPS: dict[str, list[dict[str, Any]]] = {
    "vss": _VSS_CHART_GROUPS,
    "embed": _EMBED_CHART_GROUPS,
    "graph": _GRAPH_CHART_GROUPS,
    "graph_vt": _GRAPH_VT_CHART_GROUPS,
}


# ── Documentation page catalog (built from annotated constants) ───

DOC_PAGES: dict[str, dict[str, Any]] = {
    "vss": _build_vss_page(),
    "embed": _build_embed_page(),
    "graph": _build_graph_page(),
    "graph_vt": _build_graph_vt_page(),
    "kg": _build_kg_page(),
}


def doc_page_context(page_slug: str) -> dict[str, Any]:
    """Return the complete template context dict for a given doc page.

    Combines DOC_PAGES metadata with chart specs from the analysis modules.

    Args:
        page_slug: One of "vss", "graph", "graph_vt", "kg".

    Returns:
        Dict with keys: title, description, tables, sections, charts, charts_dir.

    Raises:
        KeyError: If page_slug is not in DOC_PAGES.
    """
    from benchmarks.harness.analysis.charts_embed import EMBED_CHARTS
    from benchmarks.harness.analysis.charts_graph import GRAPH_CHARTS
    from benchmarks.harness.analysis.charts_graph_vt import GRAPH_VT_CHARTS
    from benchmarks.harness.analysis.charts_kg import KG_CHARTS
    from benchmarks.harness.analysis.charts_vss import VSS_CHARTS

    chart_map: dict[str, list[Any]] = {
        "vss": VSS_CHARTS,
        "embed": EMBED_CHARTS,
        "graph": GRAPH_CHARTS,
        "graph_vt": GRAPH_VT_CHARTS,
        "kg": KG_CHARTS,
    }

    page = DOC_PAGES[page_slug]
    chart_specs = chart_map[page_slug]

    # Relative path from project root for mkdocs snippet includes
    charts_dir_rel = str(CHARTS_DIR.relative_to(PROJECT_ROOT))

    # Spread all page fields, then add charts info
    context: dict[str, Any] = dict(page)
    context["charts"] = [{"name": s.name, "title": s.title} for s in chart_specs]
    context["charts_dir"] = charts_dir_rel

    # Build chart_groups from group definitions (if available)
    group_defs = _CHART_GROUPS.get(page_slug, [])
    if group_defs:
        chart_specs_by_name = {s.name: s for s in chart_specs}
        chart_groups: list[dict[str, Any]] = []
        for group_def in group_defs:
            group_charts = [
                {"name": name, "title": chart_specs_by_name[name].title}
                for name in group_def["charts"]
                if name in chart_specs_by_name
            ]
            if group_charts:
                chart_groups.append({"title": group_def["title"], "charts": group_charts})
        context["chart_groups"] = chart_groups

    return context
