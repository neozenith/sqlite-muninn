"""Permutation registry: enumerates all benchmark permutations across all treatment categories.

Used by the manifest and benchmark subcommands to list, filter, and execute permutations.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from benchmarks.harness.common import (
    DATASETS,
    EMBED_FNS,
    EMBED_SEARCH_BACKENDS,
    EMBED_SIZES,
    EMBEDDING_MODELS,
    GRAPH_CENTRALITY_OPERATIONS,
    GRAPH_CONFIGS_CENTRALITY,
    GRAPH_CONFIGS_COMMUNITY,
    GRAPH_CONFIGS_NODE2VEC,
    GRAPH_CONFIGS_TRAVERSAL,
    GRAPH_TRAVERSAL_OPERATIONS,
    GRAPH_TVF_ENGINES,
    GRAPH_VT_APPROACHES,
    GRAPH_VT_WORKLOADS,
    HNSW_EF_CONSTRUCTION_VALUES,
    HNSW_EF_SEARCH_VALUES,
    HNSW_M_VALUES,
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
    RESULTS_DIR,
    VSS_ENGINES,
    VSS_SIZES,
)
from benchmarks.harness.treatments.base import Treatment

log = logging.getLogger(__name__)


# ── VSS permutations ──────────────────────────────────────────────


def _vss_permutations():
    """Generate all VSS treatment permutations.

    Non-HNSW engines get one permutation per (model, dataset, N) with default params.
    HNSW engines (muninn-hnsw, vectorlite-hnsw) get a full sweep of
    M x ef_construction x ef_search per (model, dataset, N).
    """
    from benchmarks.harness.treatments.vss import ENGINE_CONFIGS, VSSTreatment

    perms = []
    all_engine_slugs = [e["slug"] for e in VSS_ENGINES]
    hnsw_engines = {slug for slug, cfg in ENGINE_CONFIGS.items() if cfg["method"] == "hnsw"}

    for model_name, model_info in EMBEDDING_MODELS.items():
        dim = model_info["dim"]
        for dataset in DATASETS:
            for n in VSS_SIZES:
                for engine_slug in all_engine_slugs:
                    if engine_slug in hnsw_engines:
                        # HNSW engines: sweep M, ef_construction, ef_search
                        for m in HNSW_M_VALUES:
                            for efc in HNSW_EF_CONSTRUCTION_VALUES:
                                for efs in HNSW_EF_SEARCH_VALUES:
                                    perms.append(VSSTreatment(
                                        engine_slug, model_name, dim, dataset, n,
                                        hnsw_m=m, hnsw_ef_construction=efc, hnsw_ef_search=efs,
                                    ))
                    else:
                        # Non-HNSW engines: single permutation with defaults
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


# ── Embed permutations ───────────────────────────────────────────


def _embed_permutations():
    """Generate embed (text->embedding->search) treatment permutations."""
    from benchmarks.harness.treatments.embed import EmbedTreatment

    perms = []
    embed_fn_slugs = [e["slug"] for e in EMBED_FNS]
    backend_slugs = [b["slug"] for b in EMBED_SEARCH_BACKENDS]

    for model_name, model_info in EMBEDDING_MODELS.items():
        if not model_info.get("embed_enabled", True):
            continue
        dim = model_info["dim"]
        for dataset in DATASETS:
            for n in EMBED_SIZES:
                for embed_fn in embed_fn_slugs:
                    for backend in backend_slugs:
                        perms.append(EmbedTreatment(embed_fn, backend, model_name, dim, dataset, n))

    return perms


# ── Public API ─────────────────────────────────────────────────────


# Categories that are excluded from the registry by default.
# These benchmarks have unresolved dependency issues and should be
# re-enabled once their prep pipeline is validated end-to-end.
# Override with BENCH_EXCLUDE_CATEGORIES env var (comma-separated),
# or set to empty string to include all.
_DEFAULT_EXCLUDE_CATEGORIES = {"kg-extract", "kg-re", "kg-resolve", "kg-graphrag"}


def _get_excluded_categories() -> set[str]:
    """Get the set of excluded categories from env or default."""
    env_val = os.environ.get("BENCH_EXCLUDE_CATEGORIES")
    if env_val is not None:
        if env_val.strip() == "":
            return set()  # empty string = include all
        return {c.strip() for c in env_val.split(",") if c.strip()}
    return _DEFAULT_EXCLUDE_CATEGORIES


def all_permutations() -> list[Treatment]:
    """Return every registered benchmark permutation.

    Categories listed in _DEFAULT_EXCLUDE_CATEGORIES are skipped unless
    overridden via the BENCH_EXCLUDE_CATEGORIES env var.
    """
    excluded = _get_excluded_categories()

    generators = {
        "vss": _vss_permutations,
        "embed": _embed_permutations,
        "graph-traversal": _graph_traversal_permutations,
        "graph-centrality": _graph_centrality_permutations,
        "graph-community": _graph_community_permutations,
        "graph-vt": _graph_vt_permutations,
        "kg-extract": _kg_extraction_permutations,
        "kg-re": _kg_re_permutations,
        "kg-resolve": _kg_resolution_permutations,
        "kg-graphrag": _kg_graphrag_permutations,
        "node2vec": _node2vec_permutations,
    }

    perms = []
    for cat, gen in generators.items():
        if cat in excluded:
            continue
        perms.extend(gen())

    if excluded:
        log.debug("Excluded categories: %s", ", ".join(sorted(excluded)))

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


def _load_completed_permutation_ids(results_dir: Path) -> set[str]:
    """Scan JSONL result files to find which permutation_ids have been run.

    Checks both local and S3 (if --s3-bucket configured) for .jsonl files.
    A permutation is "done" if its permutation_id appears in any JSONL record.
    """
    from benchmarks.harness.s3_mirror import get_s3_mirror

    mirror = get_s3_mirror()
    completed: set[str] = set()

    for filepath in mirror.list_union(results_dir, "*.jsonl"):
        if not mirror.ensure_local(filepath):
            continue
        for line in filepath.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            record = json.loads(line)
            pid = record.get("permutation_id")
            if pid:
                completed.add(pid)

    return completed


def permutation_status(results_dir: Path | None = None) -> list[dict[str, Any]]:
    """Check which permutations have results in JSONL files (done vs missing).

    Scans benchmarks/results/*.jsonl for permutation_id fields rather than
    checking for db.sqlite files on disk. This works correctly when benchmark
    databases have been cleaned up but JSONL results are preserved (locally or in S3).

    Returns list of dicts with keys: permutation_id, category, label, done, sort_key.
    """
    results_dir = results_dir or RESULTS_DIR
    completed = _load_completed_permutation_ids(results_dir)

    status = []
    for perm in all_permutations():
        status.append(
            {
                "permutation_id": perm.permutation_id,
                "category": perm.category,
                "label": perm.label,
                "done": perm.permutation_id in completed,
                "sort_key": perm.sort_key,
            }
        )
    return status
