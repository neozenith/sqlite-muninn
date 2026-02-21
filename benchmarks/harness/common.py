"""Shared utilities for the benchmark harness.

Extracted from legacy scripts to eliminate duplication of pack_vector, peak_rss_mb,
fmt_bytes, write_jsonl, load_muninn, platform_info, and graph generation.
"""

import collections
import datetime
import json
import logging
import platform
import random
import resource
import sqlite3
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# ── Path constants ─────────────────────────────────────────────────



BENCHMARKS_ROOT = Path(__file__).resolve().parent.parent  # benchmarks/
OUTPUT_ROOT = BENCHMARKS_ROOT
RESULTS_DIR = OUTPUT_ROOT / "results"
CHARTS_DIR = OUTPUT_ROOT / "charts"
VECTORS_DIR = OUTPUT_ROOT / "vectors"
TEXTS_DIR = OUTPUT_ROOT / "texts"
KG_DIR = OUTPUT_ROOT / "kg"

PROJECT_ROOT = BENCHMARKS_ROOT.parent  # project root
MUNINN_PATH = str(PROJECT_ROOT / "build" / "muninn")
DOCS_BENCHMARKS_DIR = PROJECT_ROOT / "docs" / "benchmarks"

# ── Benchmark defaults ─────────────────────────────────────────────

K = 10
N_QUERIES = 100
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

# Memory budget per-dimension max N (safe for 8GB total)
MAX_N_BY_DIM: dict[int, int] = {
    32: 500_000,
    64: 500_000,
    128: 500_000,
    256: 500_000,
    384: 500_000,
    512: 350_000,
    768: 250_000,
    1024: 200_000,
    1536: 100_000,
}

# Model definitions — operational + doc annotation fields
EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "MiniLM": {
        "model_id": "all-MiniLM-L6-v2",
        "dim": 384,
        "params": "22M",
        "size_mb": 80,
        "url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
        "use_case": "Fast, lightweight semantic search",
    },
    "MPNet": {
        "model_id": "all-mpnet-base-v2",
        "dim": 768,
        "params": "110M",
        "size_mb": 420,
        "url": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
        "use_case": "Balanced quality/speed",
    },
    "BGE-Large": {
        "model_id": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
        "params": "335M",
        "size_mb": 1340,
        "url": "https://huggingface.co/BAAI/bge-large-en-v1.5",
        "use_case": "High-quality retrieval",
    },
}

# Dataset registry — operational + doc annotation fields
DATASETS: dict[str, dict[str, Any]] = {
    "ag_news": {
        "source_type": "huggingface",
        "hf_name": "ag_news",
        "hf_split": "train",
        "text_field": "text",
        "display_name": "AG News",
        "url": "https://huggingface.co/datasets/fancyzhx/ag_news",
        "source_label": "HuggingFace",
        "passages_desc": "~120K",
        "topology": "4 discrete clusters",
    },
    "wealth_of_nations": {
        "source_type": "gutenberg",
        "gutenberg_id": 3300,
        "chunk_tokens": 256,
        "chunk_overlap": 50,
        "display_name": "Wealth of Nations",
        "url": "https://www.gutenberg.org/ebooks/3300",
        "source_label": "Project Gutenberg",
        "passages_desc": "~2,500 (256-token windows, 50-token overlap)",
        "topology": "Continuous conceptual gradient",
    },
}

# VSS profile sizes
VSS_SIZES = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 250_000]


# ── VSS dimensional axes ──────────────────────────────────────────

VSS_ENGINES: list[dict[str, Any]] = [
    {
        "slug": "muninn-hnsw",
        "display": "muninn-hnsw",
        "library_url": "https://github.com/neozenith/sqlite-muninn",
        "library_name": "muninn",
        "method": "HNSW graph index",
        "strategy": "Approximate, O(log N) search",
        "optional": False,
    },
    {
        "slug": "sqlite-vector-quantize",
        "display": "sqlite-vector-quantize",
        "library_url": "https://github.com/sqliteai/sqlite-vector",
        "library_name": "sqlite-vector",
        "method": "Product Quantization",
        "strategy": "Approximate, O(N) scan",
        "optional": False,
    },
    {
        "slug": "sqlite-vector-fullscan",
        "display": "sqlite-vector-fullscan",
        "library_url": "https://github.com/sqliteai/sqlite-vector",
        "library_name": "sqlite-vector",
        "method": "Brute-force",
        "strategy": "Exact, O(N) scan",
        "optional": False,
    },
    {
        "slug": "vectorlite-hnsw",
        "display": "vectorlite-hnsw",
        "library_url": "https://github.com/1yefuwang1/vectorlite",
        "library_name": "vectorlite",
        "method": "HNSW via hnswlib",
        "strategy": "Approximate, O(log N) search",
        "optional": True,
    },
    {
        "slug": "sqlite-vec-brute",
        "display": "sqlite-vec-brute",
        "library_url": "https://github.com/asg017/sqlite-vec",
        "library_name": "sqlite-vec",
        "method": "Brute-force KNN",
        "strategy": "Exact, O(N) scan",
        "optional": True,
    },
]


# ── Graph TVF dimensional axes ────────────────────────────────────

GRAPH_TVF_ENGINES: list[dict[str, Any]] = [
    {
        "slug": "muninn",
        "display": "muninn",
        "library_url": "https://github.com/neozenith/sqlite-muninn",
        "library_name": "muninn",
        "traversal": "BFS, DFS, shortest path, components, PageRank",
        "centrality": "degree, betweenness, closeness",
        "community": "Leiden",
        "node2vec": "Yes",
    },
    {
        "slug": "graphqlite",
        "display": "graphqlite",
        "library_url": "https://github.com/colliery-io/graphqlite",
        "library_name": "GraphQLite",
        "traversal": "BFS, DFS, shortest path, components, PageRank",
        "centrality": "\u2014",
        "community": "Leiden",
        "node2vec": "\u2014",
    },
]

GRAPH_MODELS: list[dict[str, Any]] = [
    {
        "slug": "erdos_renyi",
        "display": "Erdos-Renyi",
        "url": "https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model",
        "description": "Random edges, uniform probability",
        "key_property": "Uniform degree distribution",
    },
    {
        "slug": "barabasi_albert",
        "display": "Barabasi-Albert",
        "url": "https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model",
        "description": "Preferential attachment",
        "key_property": "Power-law (scale-free) degree distribution",
    },
]

GRAPH_TRAVERSAL_OPERATIONS: list[str] = ["bfs", "dfs", "shortest_path", "components", "pagerank"]
GRAPH_CENTRALITY_OPERATIONS: list[str] = ["degree", "betweenness", "closeness"]

# (graph_model, n_nodes, avg_degree) tuples — largest tier
GRAPH_CONFIGS_TRAVERSAL: list[tuple[str, int, int]] = [
    ("erdos_renyi", 100, 5),
    ("erdos_renyi", 500, 5),
    ("erdos_renyi", 1000, 5),
    ("erdos_renyi", 1000, 20),
    ("erdos_renyi", 5000, 5),
    ("erdos_renyi", 5000, 20),
    ("erdos_renyi", 10000, 5),
    ("erdos_renyi", 10000, 20),
    ("barabasi_albert", 1000, 3),
    ("barabasi_albert", 5000, 5),
    ("barabasi_albert", 10000, 5),
]

# Subset for expensive algorithms (smaller graphs)
GRAPH_CONFIGS_CENTRALITY: list[tuple[str, int, int]] = [
    ("erdos_renyi", 100, 5),
    ("erdos_renyi", 500, 5),
    ("erdos_renyi", 1000, 5),
    ("erdos_renyi", 1000, 20),
    ("erdos_renyi", 5000, 5),
    ("barabasi_albert", 1000, 3),
    ("barabasi_albert", 5000, 5),
]

GRAPH_CONFIGS_COMMUNITY = GRAPH_CONFIGS_CENTRALITY

# Smallest tier for Node2Vec
GRAPH_CONFIGS_NODE2VEC: list[tuple[str, int, int]] = [
    ("erdos_renyi", 500, 5),
    ("erdos_renyi", 1000, 5),
    ("barabasi_albert", 1000, 3),
]

NODE2VEC_P_VALUES: list[float] = [0.5, 1.0, 2.0]
NODE2VEC_Q_VALUES: list[float] = [0.5, 1.0, 2.0]
NODE2VEC_DIMS: list[int] = [64, 128]


# ── Graph VT dimensional axes ─────────────────────────────────────

GRAPH_VT_BLOCK_SIZE: int = 4096

GRAPH_VT_APPROACHES: list[dict[str, str]] = [
    {"slug": "tvf", "display": "TVF", "description": "No cache \u2014 scans edge table via SQL on every query"},
    {"slug": "csr", "display": "CSR", "description": "Persistent CSR cache; initial build from edge table"},
    {
        "slug": "csr_full_rebuild",
        "display": "CSR \u2014 full rebuild",
        "description": "Persistent CSR cache; full edge-table re-scan when stale",
    },
    {
        "slug": "csr_incremental",
        "display": "CSR \u2014 incremental",
        "description": "Delta + merge; rebuilds all blocks (spread mutations)",
    },
    {
        "slug": "csr_blocked",
        "display": "CSR \u2014 blocked incremental",
        "description": "Delta + merge; rebuilds only affected blocks (concentrated mutations)",
    },
]

GRAPH_VT_WORKLOADS: list[dict[str, Any]] = [
    {"name": "xsmall", "n_nodes": 500, "target_edges": 2000, "graph_model": "erdos_renyi"},
    {"name": "small", "n_nodes": 1000, "target_edges": 5000, "graph_model": "erdos_renyi"},
    {"name": "medium", "n_nodes": 5000, "target_edges": 25000, "graph_model": "barabasi_albert"},
    {"name": "large", "n_nodes": 10000, "target_edges": 50000, "graph_model": "barabasi_albert"},
]


# ── KG dimensional axes ───────────────────────────────────────────

KG_NER_MODELS: list[dict[str, Any]] = [
    {
        "slug": "gliner_small-v2.1",
        "display": "GLiNER small-v2.1",
        "type": "Zero-shot NER",
        "description": "Lightweight generalist entity extraction",
        "params": "166M",
        "size_mb": 611,
    },
    {
        "slug": "gliner_medium-v2.1",
        "display": "GLiNER medium-v2.1",
        "type": "Zero-shot NER",
        "description": "Medium-capacity zero-shot NER",
        "params": "209M",
        "size_mb": 781,
    },
    {
        "slug": "gliner_large-v2.1",
        "display": "GLiNER large-v2.1",
        "type": "Zero-shot NER",
        "description": "High-capacity zero-shot NER",
        "params": "459M",
        "size_mb": 1780,
    },
    {
        "slug": "numind_NuNerZero",
        "display": "NuNerZero",
        "type": "Zero-shot NER",
        "description": "NumIND zero-shot NER; labels must be lowercase",
        "params": "~400M",
        "size_mb": 1800,
    },
    {
        "slug": "gner-t5-base",
        "display": "GNER-T5 base",
        "type": "Seq2seq NER",
        "description": "Generative NER via T5-base (slower, higher quality)",
        "params": "248M",
        "size_mb": 990,
    },
    {
        "slug": "gner-t5-large",
        "display": "GNER-T5 large",
        "type": "Seq2seq NER",
        "description": "Generative NER via T5-large",
        "params": "783M",
        "size_mb": 3100,
    },
    {
        "slug": "spacy_en_core_web_lg",
        "display": "spaCy en_core_web_lg",
        "type": "Statistical NER",
        "description": "spaCy's large English pipeline",
        "params": None,
        "size_mb": 560,
    },
    {
        "slug": "fts5",
        "display": "FTS5",
        "type": "Keyword matching",
        "description": "SQLite full-text search as a baseline",
        "params": None,
        "size_mb": 0,
    },
]

KG_NER_DATASETS: list[dict[str, Any]] = [
    {
        "slug": "gutenberg:3300",
        "display": "Wealth of Nations (3300)",
        "source": "Project Gutenberg",
        "url": None,
        "description": "Literary text chunks (no gold labels — speed only)",
    },
    {
        "slug": "crossner_ai",
        "display": "CrossNER (AI)",
        "source": "HuggingFace",
        "url": "https://huggingface.co/datasets/DFKI-SLT/cross_ner",
        "description": "CrossNER AI domain; BIO-tagged entities",
    },
    {
        "slug": "crossner_conll2003",
        "display": "CrossNER (CoNLL-2003)",
        "source": "HuggingFace",
        "url": "https://huggingface.co/datasets/DFKI-SLT/cross_ner",
        "description": "CoNLL-2003 via CrossNER; PER, ORG, LOC, MISC",
    },
    {
        "slug": "crossner_literature",
        "display": "CrossNER (Literature)",
        "source": "HuggingFace",
        "url": "https://huggingface.co/datasets/DFKI-SLT/cross_ner",
        "description": "CrossNER literature domain",
    },
    {
        "slug": "crossner_music",
        "display": "CrossNER (Music)",
        "source": "HuggingFace",
        "url": "https://huggingface.co/datasets/DFKI-SLT/cross_ner",
        "description": "CrossNER music domain",
    },
    {
        "slug": "crossner_politics",
        "display": "CrossNER (Politics)",
        "source": "HuggingFace",
        "url": "https://huggingface.co/datasets/DFKI-SLT/cross_ner",
        "description": "CrossNER politics domain",
    },
    {
        "slug": "crossner_science",
        "display": "CrossNER (Science)",
        "source": "HuggingFace",
        "url": "https://huggingface.co/datasets/DFKI-SLT/cross_ner",
        "description": "CrossNER science domain",
    },
    {
        "slug": "fewnerd_supervised",
        "display": "Few-NERD (supervised)",
        "source": "HuggingFace",
        "url": "https://huggingface.co/datasets/DFKI-SLT/few-nerd",
        "description": "Fine-grained NER with 66 entity types (supervised split)",
    },
    {
        "slug": "fewnerd_inter",
        "display": "Few-NERD (inter)",
        "source": "HuggingFace",
        "url": "https://huggingface.co/datasets/DFKI-SLT/few-nerd",
        "description": "Few-NERD inter-domain split",
    },
    {
        "slug": "fewnerd_intra",
        "display": "Few-NERD (intra)",
        "source": "HuggingFace",
        "url": "https://huggingface.co/datasets/DFKI-SLT/few-nerd",
        "description": "Few-NERD intra-domain split",
    },
]

KG_RE_MODEL_SLUGS: list[str] = ["fts5", "gliner_small-v2.1", "spacy_en_core_web_lg"]

KG_RE_DATASETS: list[dict[str, Any]] = [
    {
        "slug": "docred",
        "display": "DocRED",
        "source": "HuggingFace",
        "url": "https://huggingface.co/datasets/thunlp/docred",
        "description": "Document-level relation extraction",
    },
    {
        "slug": "webnlg",
        "display": "WebNLG",
        "source": "HuggingFace",
        "url": "https://huggingface.co/datasets/webnlg-challenge/web_nlg",
        "description": "RDF triple verbalization and extraction",
    },
    {
        "slug": "conll04",
        "display": "CoNLL-04",
        "source": "HuggingFace",
        "url": "https://huggingface.co/datasets/DFKI-SLT/conll04",
        "description": "Joint entity and relation extraction",
    },
]

KG_RESOLUTION_DATASETS: list[dict[str, Any]] = [
    {
        "slug": "3300",
        "display": "Wealth of Nations (3300)",
        "source": "Project Gutenberg",
        "description": "Literary entity mentions with spelling variations",
    },
    {
        "slug": "febrl1",
        "display": "FEBRL1",
        "source": "Freely Extensible Biomedical Record Linkage",
        "description": "Synthetic person records with controlled duplication",
    },
]

KG_GRAPHRAG_ENTRIES: list[str] = ["vss", "bm25"]
KG_GRAPHRAG_EXPANSIONS: list[str] = ["none", "bfs1", "bfs2"]
KG_GRAPHRAG_BOOK_IDS: list[int] = [3300]


# ── Vector utilities ───────────────────────────────────────────────


def pack_vector(v: np.ndarray | list[float]) -> bytes:
    """Pack a float list/array into a float32 BLOB for SQLite.

    Accepts numpy arrays (fast path via tobytes) or plain Python iterables.
    """
    if isinstance(v, np.ndarray):
        return v.astype(np.float32).tobytes()
    return struct.pack(f"{len(v)}f", *v)


def unpack_vector(blob: bytes, dim: int) -> list[float]:
    """Unpack a float32 BLOB back into a list of floats."""
    return list(struct.unpack(f"{dim}f", blob))


# ── System metrics ─────────────────────────────────────────────────


def peak_rss_mb() -> float:
    """Current peak RSS in MB (macOS returns bytes, Linux returns KB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def fmt_bytes(size: float | int | None) -> str:
    """Format byte count as human-readable string."""
    if size is None:
        return "n/a"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def platform_info() -> dict[str, str]:
    """Return platform identification dict."""
    return {
        "platform": f"{sys.platform}-{platform.machine()}",
        "python_version": platform.python_version(),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }


# ── JSONL I/O ──────────────────────────────────────────────────────


def write_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append a single JSON record as one line to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file. Returns list of dicts."""
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if line.strip():
            records.append(json.loads(line))
    return records


# ── Extension loading ──────────────────────────────────────────────


def load_muninn(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Load the muninn extension into a SQLite connection."""
    conn.enable_load_extension(True)
    conn.load_extension(MUNINN_PATH)
    return conn


def create_benchmark_db(db_path: str | Path) -> sqlite3.Connection:
    """Create a SQLite connection for a benchmark at the given path.

    Creates parent directories and loads muninn extension.
    Returns (conn, db_path).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    load_muninn(conn)
    return conn


# ── Graph generation ───────────────────────────────────────────────


def generate_erdos_renyi(
    n_nodes: int, avg_degree: int | float, weighted: bool = False, seed: int = 42
) -> tuple[list[tuple[int, int, float]], dict[int, list[tuple[int, float]]]]:
    """Generate Erdos-Renyi random graph.

    Returns (edges, adjacency_dict) where edges is a list of (src, dst, weight)
    tuples and adjacency_dict maps node -> [(neighbor, weight)].
    """
    rng = random.Random(seed)
    p = avg_degree / max(1, n_nodes - 1)

    edges: list[tuple[int, int, float]] = []
    adj: dict[int, list[tuple[int, float]]] = collections.defaultdict(list)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < p:
                w = round(rng.uniform(0.1, 10.0), 2) if weighted else 1.0
                edges.append((i, j, w))
                edges.append((j, i, w))
                adj[i].append((j, w))
                adj[j].append((i, w))

    # Ensure all nodes exist in adjacency (even isolates)
    for i in range(n_nodes):
        if i not in adj:
            adj[i] = []

    return edges, dict(adj)


def generate_barabasi_albert(
    n_nodes: int, m: int, weighted: bool = False, seed: int = 42
) -> tuple[list[tuple[int, int, float]], dict[int, list[tuple[int, float]]]]:
    """Generate Barabasi-Albert scale-free graph via preferential attachment.

    Each new node connects to m existing nodes. Returns (edges, adjacency_dict).
    """
    rng = random.Random(seed)
    adj: dict[int, list[tuple[int, float]]] = collections.defaultdict(list)
    edges: list[tuple[int, int, float]] = []
    degree = [0] * n_nodes

    # Start with a complete graph on the first m+1 nodes
    for i in range(min(m + 1, n_nodes)):
        for j in range(i + 1, min(m + 1, n_nodes)):
            w = round(rng.uniform(0.1, 10.0), 2) if weighted else 1.0
            edges.append((i, j, w))
            edges.append((j, i, w))
            adj[i].append((j, w))
            adj[j].append((i, w))
            degree[i] += 1
            degree[j] += 1

    # Preferential attachment for remaining nodes
    for new_node in range(m + 1, n_nodes):
        total_degree = sum(degree[:new_node])
        if total_degree == 0:
            targets: set[int] | list[int] = rng.sample(range(new_node), min(m, new_node))
        else:
            targets = set()
            while len(targets) < min(m, new_node):
                r = rng.random() * total_degree
                cumulative = 0
                for node in range(new_node):
                    cumulative += degree[node]
                    if cumulative >= r:
                        targets.add(node)
                        break

        for target in targets:
            w = round(rng.uniform(0.1, 10.0), 2) if weighted else 1.0
            edges.append((new_node, target, w))
            edges.append((target, new_node, w))
            adj[new_node].append((target, w))
            adj[target].append((new_node, w))
            degree[new_node] += 1
            degree[target] += 1

    for i in range(n_nodes):
        if i not in adj:
            adj[i] = []

    return edges, dict(adj)
