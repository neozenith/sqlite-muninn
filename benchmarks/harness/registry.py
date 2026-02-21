"""Permutation registry: enumerates all benchmark permutations across all treatment categories.

Used by the manifest and benchmark subcommands to list, filter, and execute permutations.
Also provides DOC_PAGES catalog for documentation generation — built programmatically
from the annotated dimensional axis constants in common.py.
"""

import logging
from pathlib import Path
from typing import Any

from benchmarks.harness.common import (
    CHARTS_DIR,
    DATASETS,
    EMBEDDING_MODELS,
    GRAPH_CENTRALITY_OPERATIONS,
    GRAPH_CONFIGS_CENTRALITY,
    GRAPH_CONFIGS_COMMUNITY,
    GRAPH_CONFIGS_NODE2VEC,
    GRAPH_CONFIGS_TRAVERSAL,
    GRAPH_MODELS,
    GRAPH_TRAVERSAL_OPERATIONS,
    GRAPH_TVF_ENGINES,
    GRAPH_VT_APPROACHES,
    GRAPH_VT_BLOCK_SIZE,
    GRAPH_VT_WORKLOADS,
    KG_GRAPHRAG_BOOK_IDS,
    KG_GRAPHRAG_ENTRIES,
    KG_GRAPHRAG_EXPANSIONS,
    KG_NER_DATASETS,
    KG_NER_MODELS,
    KG_RE_DATASETS,
    KG_RE_MODEL_SLUGS,
    KG_RESOLUTION_DATASETS,
    NODE2VEC_DIMS,
    NODE2VEC_P_VALUES,
    NODE2VEC_Q_VALUES,
    PROJECT_ROOT,
    RESULTS_DIR,
    VSS_ENGINES,
    VSS_SIZES,
)
from benchmarks.harness.treatments.base import Treatment

log = logging.getLogger(__name__)


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
                "columns": ["Model", "Dimension", "Params", "Size", "Use Case"],
                "rows": [
                    {
                        "Model": _linked(name, m["url"]),
                        "Dimension": str(m["dim"]),
                        "Params": m["params"],
                        "Size": f"{m['size_mb']} MB",
                        "Use Case": m["use_case"],
                    }
                    for name, m in EMBEDDING_MODELS.items()
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
                "description": "BFS, DFS, shortest path, connected components, and PageRank. Compared across both engines.",
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
            "tipping_point_MPNet_ag_news",
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
            "tipping_point_MPNet_wealth_of_nations",
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
            "dataset_comparison_MPNet",
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
    "graph": _GRAPH_CHART_GROUPS,
    "graph_vt": _GRAPH_VT_CHART_GROUPS,
}


# ── Documentation page catalog (built from annotated constants) ───

DOC_PAGES: dict[str, dict[str, Any]] = {
    "vss": _build_vss_page(),
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
    from benchmarks.harness.analysis.charts_graph import GRAPH_CHARTS
    from benchmarks.harness.analysis.charts_graph_vt import GRAPH_VT_CHARTS
    from benchmarks.harness.analysis.charts_kg import KG_CHARTS
    from benchmarks.harness.analysis.charts_vss import VSS_CHARTS

    chart_map: dict[str, list] = {
        "vss": VSS_CHARTS,
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


# ── VSS permutations ──────────────────────────────────────────────


def _vss_permutations():
    """Generate all VSS treatment permutations."""
    from benchmarks.harness.treatments.vss import VSSTreatment

    perms = []
    all_engine_slugs = [e["slug"] for e in VSS_ENGINES]

    for model_name, model_info in EMBEDDING_MODELS.items():
        dim = model_info["dim"]
        for dataset in DATASETS:
            for n in VSS_SIZES:
                for engine_slug in all_engine_slugs:
                    perms.append(VSSTreatment(engine_slug, model_name, dim, dataset, n))

    return perms


# ── Graph traversal permutations ──────────────────────────────────


def _graph_traversal_permutations():
    """Generate graph traversal treatment permutations."""
    from benchmarks.harness.treatments.graph_traversal import GraphTraversalTreatment

    perms = []
    engine_slugs = [e["slug"] for e in GRAPH_TVF_ENGINES]

    for graph_model, n, deg in GRAPH_CONFIGS_TRAVERSAL:
        for engine in engine_slugs:
            for op in GRAPH_TRAVERSAL_OPERATIONS:
                perms.append(GraphTraversalTreatment(engine, op, graph_model, n, deg))

    return perms


# ── Graph centrality permutations ─────────────────────────────────


def _graph_centrality_permutations():
    """Generate graph centrality treatment permutations."""
    from benchmarks.harness.treatments.graph_centrality import GraphCentralityTreatment

    perms = []

    for graph_model, n, deg in GRAPH_CONFIGS_CENTRALITY:
        for op in GRAPH_CENTRALITY_OPERATIONS:
            perms.append(GraphCentralityTreatment(op, graph_model, n, deg))

    return perms


# ── Graph community permutations ──────────────────────────────────


def _graph_community_permutations():
    """Generate graph community detection treatment permutations."""
    from benchmarks.harness.treatments.graph_community import GraphCommunityTreatment

    perms = []

    for graph_model, n, deg in GRAPH_CONFIGS_COMMUNITY:
        perms.append(GraphCommunityTreatment(graph_model, n, deg))

    return perms


# ── Graph VT permutations ────────────────────────────────────────


def _graph_vt_permutations():
    """Generate graph VT (virtual table) treatment permutations."""
    from benchmarks.harness.treatments.graph_vt import GraphVtTreatment

    perms = []
    approach_slugs = [a["slug"] for a in GRAPH_VT_APPROACHES]

    for approach in approach_slugs:
        for w in GRAPH_VT_WORKLOADS:
            perms.append(GraphVtTreatment(approach, w["name"], w["n_nodes"], w["target_edges"], w["graph_model"]))

    return perms


# ── KG extraction permutations ────────────────────────────────────


def _kg_extraction_permutations():
    """Generate KG NER extraction treatment permutations."""
    from benchmarks.harness.treatments.kg_extract import KGNerExtractionTreatment

    perms = []
    model_slugs = [m["slug"] for m in KG_NER_MODELS]
    data_sources = [d["slug"] for d in KG_NER_DATASETS]

    for model_slug in model_slugs:
        for data_source in data_sources:
            perms.append(KGNerExtractionTreatment(model_slug, data_source))

    return perms


# ── KG resolution permutations ────────────────────────────────────


def _kg_resolution_permutations():
    """Generate KG entity resolution treatment permutations."""
    from benchmarks.harness.treatments.kg_resolve import KGEntityResolutionTreatment

    perms = []

    for d in KG_RESOLUTION_DATASETS:
        perms.append(KGEntityResolutionTreatment(d["slug"]))

    return perms


# ── KG relation extraction permutations ───────────────────────────


def _kg_re_permutations():
    """Generate KG relation extraction treatment permutations."""
    from benchmarks.harness.treatments.kg_re import KGRelationExtractionTreatment

    perms = []
    dataset_slugs = [d["slug"] for d in KG_RE_DATASETS]

    for model_slug in KG_RE_MODEL_SLUGS:
        for dataset in dataset_slugs:
            perms.append(KGRelationExtractionTreatment(model_slug, dataset))

    return perms


# ── KG GraphRAG permutations ──────────────────────────────────────


def _kg_graphrag_permutations():
    """Generate KG GraphRAG retrieval quality treatment permutations."""
    from benchmarks.harness.treatments.kg_graphrag import KGGraphRAGTreatment

    perms = []

    for entry in KG_GRAPHRAG_ENTRIES:
        for expansion in KG_GRAPHRAG_EXPANSIONS:
            for book_id in KG_GRAPHRAG_BOOK_IDS:
                perms.append(KGGraphRAGTreatment(entry, expansion, book_id))

    return perms


# ── Node2Vec permutations ─────────────────────────────────────────


def _node2vec_permutations():
    """Generate Node2Vec training treatment permutations."""
    from benchmarks.harness.treatments.node2vec import Node2VecTreatment

    perms = []

    for graph_model, n, deg in GRAPH_CONFIGS_NODE2VEC:
        for p in NODE2VEC_P_VALUES:
            for q in NODE2VEC_Q_VALUES:
                for dim in NODE2VEC_DIMS:
                    perms.append(Node2VecTreatment(graph_model, n, deg, p, q, dim))

    return perms


# ── Public API ─────────────────────────────────────────────────────


def all_permutations() -> list[Treatment]:
    """Return every registered benchmark permutation."""
    perms = []
    perms.extend(_vss_permutations())
    perms.extend(_graph_traversal_permutations())
    perms.extend(_graph_centrality_permutations())
    perms.extend(_graph_community_permutations())
    perms.extend(_graph_vt_permutations())
    perms.extend(_kg_extraction_permutations())
    perms.extend(_kg_re_permutations())
    perms.extend(_kg_resolution_permutations())
    perms.extend(_kg_graphrag_permutations())
    perms.extend(_node2vec_permutations())
    return perms


def filter_permutations(
    category: str | None = None,
    permutation_id: str | None = None,
) -> list[Treatment]:
    """Filter permutations by category or specific ID."""
    perms = all_permutations()

    if category is not None:
        perms = [p for p in perms if p.category == category]

    if permutation_id is not None:
        perms = [p for p in perms if p.permutation_id == permutation_id]

    return perms


def permutation_status(results_dir: Path | None = None) -> list[dict]:
    """Check which permutations have db.sqlite (done vs missing).

    Returns list of dicts with keys: permutation_id, category, label, done, sort_key.
    """
    results_dir = results_dir or RESULTS_DIR
    status = []
    for perm in all_permutations():
        db_path = results_dir / perm.permutation_id / "db.sqlite"
        status.append(
            {
                "permutation_id": perm.permutation_id,
                "category": perm.category,
                "label": perm.label,
                "done": db_path.exists(),
                "sort_key": perm.sort_key,
            }
        )
    return status
