"""
Multi-dimensional benchmark suite for SQLite vector search extensions.

Compares muninn (HNSW), sqliteai/sqlite-vector, vectorlite, and sqlite-vec
across multiple vector dimensions, datasets, and data volumes. Computes
saturation metrics and writes JSONL results for analysis.

Engines:
    muninn           — This project's HNSW index
    sqlite_vector    — sqliteai/sqlite-vector (quantize_scan + full_scan)
    vectorlite       — vectorlite-py (HNSW via hnswlib)
    sqlite_vec       — asg017/sqlite-vec (brute-force KNN)

Datasets:
    ag_news              — 120K short news snippets, 4 categories (HuggingFace)
    wealth_of_nations    — ~2.5K paragraphs from Gutenberg #3300

Profiles:
    small       — 3 dims (384, 768, 1536), N <= 50K, random vectors (~10 min)
    medium      — 2 dims (384, 768), N = 100K-500K, random vectors (~1-2 hrs)
    saturation  — 8 dims (32-1536), N = 50K, random vectors (~20 min)
    models      — 3 real embedding models x 2 datasets (~30 min)

Prerequisites:
    uv sync --all-groups
    make all

Run:
    python python/benchmark_vss.py --profile small
    python python/benchmark_vss.py --source random --dim 384 --sizes 1000,5000
    python python/benchmark_vss.py --source model:all-MiniLM-L6-v2 --sizes 1000 --dataset ag_news
"""
import argparse
import datetime
import importlib.resources
import json
import logging
import math
import platform
import random
import re
import resource
import sqlite3
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

try:
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    HAS_MODEL_DEPS = True
except ImportError:
    HAS_MODEL_DEPS = False

try:
    import sqlite_vec

    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False

try:
    import apsw
    import vectorlite_py

    HAS_VECTORLITE = True
except ImportError:
    HAS_VECTORLITE = False

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MUNINN_PATH = str(PROJECT_ROOT / "muninn")
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
VECTORS_DIR = PROJECT_ROOT / "benchmarks" / "vectors"
TEXTS_DIR = PROJECT_ROOT / "benchmarks" / "texts"

# Benchmark defaults
K = 10
N_QUERIES = 100
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

# Saturation sampling
SATURATION_SAMPLE_PAIRS = 10_000

# Memory budget per-dimension max N (safe for 8GB total)
MAX_N_BY_DIM = {
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

# Model definitions for the 'models' profile
EMBEDDING_MODELS = {
    "MiniLM": {"model_id": "all-MiniLM-L6-v2", "dim": 384},
    "MPNet": {"model_id": "all-mpnet-base-v2", "dim": 768},
    "BGE-Large": {"model_id": "BAAI/bge-large-en-v1.5", "dim": 1024},
}

# Dataset registry
DATASETS = {
    "ag_news": {
        "source_type": "huggingface",
        "hf_name": "ag_news",
        "hf_split": "train",
        "text_field": "text",
    },
    "wealth_of_nations": {
        "source_type": "gutenberg",
        "gutenberg_id": 3300,
        "chunk_tokens": 256,
        "chunk_overlap": 50,
    },
}

# Profile definitions
PROFILES = {
    "saturation": {
        "source": "random",
        "dimensions": [32, 64, 128, 256, 512, 768, 1024, 1536],
        "sizes": [50_000],
    },
    "models": {
        "source": "models",
        "dimensions": None,  # determined by model
        "datasets": ["ag_news", "wealth_of_nations"],
        "sizes": [1_000, 5_000, 10_000, 50_000, 100_000, 250_000],
    },
}


# ── Utilities ──────────────────────────────────────────────────────


def _sqlite_vector_ext_path():
    """Locate the sqliteai-vector binary for load_extension()."""
    return str(importlib.resources.files("sqlite_vector.binaries") / "vector")


def random_vector(dim):
    """Generate a random unit vector."""
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-10:
        v[0] = 1.0
        norm = 1.0
    return [x / norm for x in v]


def pack_vector(v):
    """Pack a float list/array into a float32 BLOB."""
    if isinstance(v, np.ndarray):
        return v.astype(np.float32).tobytes()
    return struct.pack(f"{len(v)}f", *v)


def peak_rss_mb():
    """Current peak RSS in MB (macOS returns bytes, Linux returns KB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def _fmt_bytes(size):
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def generate_dataset(n, dim):
    """Generate n random unit vectors as a dict of {rowid: list[float]}."""
    return {i: random_vector(dim) for i in range(1, n + 1)}


def pick_queries(vectors, n_queries):
    """Pick random query IDs from the dataset."""
    return random.sample(list(vectors.keys()), min(n_queries, len(vectors)))


def enforce_memory_limit(dim, n):
    """Clamp n to the memory-safe maximum for this dimension."""
    max_n = MAX_N_BY_DIM.get(dim, 100_000)
    if n > max_n:
        log.warning("N=%d exceeds memory limit for dim=%d, clamping to %d", n, dim, max_n)
        return max_n
    return n


def platform_info():
    """Return platform identification dict."""
    return {
        "platform": f"{sys.platform}-{platform.machine()}",
        "python_version": platform.python_version(),
    }


def format_time(seconds):
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}hr"


def make_scenario_name(vector_source, model_name, dataset, dim, n):
    """Build a deterministic scenario name from run parameters."""
    if vector_source == "random":
        return f"random_dim{dim}_n{n}"
    ds_suffix = f"_{dataset}" if dataset and dataset != "ag_news" else ""
    return f"model_{model_name}{ds_suffix}_n{n}"


def make_db_path(scenario_name, run_timestamp, engine):
    """Build the SQLite DB path for disk storage."""
    db_dir = RESULTS_DIR / scenario_name
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / f"{run_timestamp}_{engine}.sqlite"


# ── Gutenberg text support ─────────────────────────────────────────


def download_gutenberg(gutenberg_id):
    """Fetch plain text from Project Gutenberg, strip boilerplate, and cache locally.

    Returns the path to the cached text file.
    """
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TEXTS_DIR / f"gutenberg_{gutenberg_id}.txt"

    if cache_path.exists():
        log.info("    Gutenberg #%d: cached at %s", gutenberg_id, cache_path)
        return cache_path

    url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
    log.info("    Downloading Gutenberg #%d from %s...", gutenberg_id, url)

    req = urllib.request.Request(url, headers={"User-Agent": "muninn-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw_text = resp.read().decode("utf-8-sig")

    # Strip Gutenberg boilerplate (header and footer)
    start_markers = ["*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"]
    end_markers = ["*** END OF THE PROJECT GUTENBERG", "*** END OF THIS PROJECT GUTENBERG"]

    start_idx = 0
    for marker in start_markers:
        idx = raw_text.find(marker)
        if idx != -1:
            start_idx = raw_text.index("\n", idx) + 1
            break

    end_idx = len(raw_text)
    for marker in end_markers:
        idx = raw_text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    clean_text = raw_text[start_idx:end_idx].strip()

    cache_path.write_text(clean_text, encoding="utf-8")
    log.info("    Gutenberg #%d: saved %d chars to %s", gutenberg_id, len(clean_text), cache_path)
    return cache_path


def chunk_fixed_tokens(text, window=256, overlap=50):
    """Split text into fixed-size token windows with overlap.

    Uses word-level tokenization as an approximation of sub-word tokens.
    Returns a list of text chunks.
    """
    # Collapse whitespace and split into words
    words = text.split()
    if not words:
        return []

    stride = max(1, window - overlap)
    chunks = []
    for i in range(0, len(words), stride):
        chunk_words = words[i : i + window]
        if len(chunk_words) < window // 4:
            break  # skip tiny trailing chunks
        chunks.append(" ".join(chunk_words))

    return chunks


def load_dataset_texts(dataset_key, max_n):
    """Unified text loading: AG News via HuggingFace, Gutenberg via download+chunk.

    Returns a list of text strings, up to max_n.
    """
    ds_config = DATASETS[dataset_key]

    if ds_config["source_type"] == "huggingface":
        if not HAS_MODEL_DEPS:
            log.error("HuggingFace datasets require: uv sync --all-groups")
            sys.exit(1)
        hf_dataset = load_dataset(ds_config["hf_name"], split=ds_config["hf_split"])
        field = ds_config["text_field"]
        n = min(max_n, len(hf_dataset))
        texts = [row[field] for row in hf_dataset.select(range(n))]
        log.info("    %s: loaded %d texts (HuggingFace)", dataset_key, len(texts))
        return texts

    if ds_config["source_type"] == "gutenberg":
        text_path = download_gutenberg(ds_config["gutenberg_id"])
        raw_text = text_path.read_text(encoding="utf-8")
        # Normalize paragraph breaks for chunking
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
        chunks = chunk_fixed_tokens(raw_text, window=ds_config["chunk_tokens"], overlap=ds_config["chunk_overlap"])
        texts = chunks[:max_n]
        log.info("    %s: %d chunks (window=%d, overlap=%d)", dataset_key, len(texts), ds_config["chunk_tokens"], ds_config["chunk_overlap"])
        return texts

    log.error("Unknown dataset source_type: %s", ds_config["source_type"])
    sys.exit(1)


# ── Saturation metrics ────────────────────────────────────────────


def compute_saturation_metrics(vectors, dim, n_sample_pairs=SATURATION_SAMPLE_PAIRS):
    """Compute vector space saturation metrics from sampled pairwise distances.

    Returns dict with relative_contrast, distance_cv, nearest_farthest_ratio.
    """
    ids = list(vectors.keys())
    n = len(ids)

    if n < 10:
        return {"relative_contrast": None, "distance_cv": None, "nearest_farthest_ratio": None}

    # Sample pairwise distances
    n_pairs = min(n_sample_pairs, n * (n - 1) // 2)
    pairwise_dists = []
    for _ in range(n_pairs):
        i, j = random.sample(ids, 2)
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(vectors[i], vectors[j])))
        pairwise_dists.append(d)

    mean_pairwise = sum(pairwise_dists) / len(pairwise_dists)
    var_pairwise = sum((d - mean_pairwise) ** 2 for d in pairwise_dists) / len(pairwise_dists)
    std_pairwise = math.sqrt(var_pairwise)

    # Sample nearest-neighbor and farthest distances for a subset of points
    n_sample_queries = min(100, n)
    sample_query_ids = random.sample(ids, n_sample_queries)
    nn_dists = []
    nf_ratios = []

    for qid in sample_query_ids:
        q = vectors[qid]
        # Compute distances to a sample of other points
        sample_targets = random.sample(ids, min(500, n))
        dists_to_targets = []
        for tid in sample_targets:
            if tid == qid:
                continue
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(q, vectors[tid])))
            dists_to_targets.append(d)

        if dists_to_targets:
            d_min = min(dists_to_targets)
            d_max = max(dists_to_targets)
            nn_dists.append(d_min)
            if d_max > 1e-10:
                nf_ratios.append(d_min / d_max)

    mean_nn = sum(nn_dists) / len(nn_dists) if nn_dists else 0

    # Relative Contrast: mean(pairwise) / mean(nearest-neighbor) -> 1.0 means saturated
    rc = mean_pairwise / mean_nn if mean_nn > 1e-10 else None

    # Coefficient of Variation: std(pairwise) / mean(pairwise) -> 0 means saturated
    cv = std_pairwise / mean_pairwise if mean_pairwise > 1e-10 else None

    # Nearest/Farthest ratio: mean -> 1.0 means saturated
    nf = sum(nf_ratios) / len(nf_ratios) if nf_ratios else None

    return {
        "relative_contrast": round(rc, 4) if rc is not None else None,
        "distance_cv": round(cv, 4) if cv is not None else None,
        "nearest_farthest_ratio": round(nf, 4) if nf is not None else None,
    }


# ── Ground truth computation ──────────────────────────────────────


def brute_force_knn(query, vectors, k):
    """Brute force KNN by L2 distance. Returns set of rowids."""
    dists = []
    for rowid, v in vectors.items():
        d = sum((a - b) ** 2 for a, b in zip(query, v))
        dists.append((d, rowid))
    dists.sort()
    return set(rowid for _, rowid in dists[:k])


def compute_ground_truth_python(vectors, query_ids, k):
    """Compute ground truth via Python brute-force. Good for N <= 50K."""
    return [brute_force_knn(vectors[qid], vectors, k) for qid in query_ids]


def compute_ground_truth_sqlite_vector(vectors, query_ids, k, dim):
    """Compute ground truth via sqlite-vector full_scan. Better for N > 50K."""
    ext_path = _sqlite_vector_ext_path()
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    conn.load_extension(ext_path)

    conn.execute("CREATE TABLE gt(id INTEGER PRIMARY KEY, embedding BLOB)")
    for rowid, v in vectors.items():
        conn.execute("INSERT INTO gt(id, embedding) VALUES (?, ?)", (rowid, pack_vector(v)))
    conn.execute(f"SELECT vector_init('gt', 'embedding', 'dimension={dim},type=FLOAT32,distance=L2')")

    results = []
    for qid in query_ids:
        rows = conn.execute(
            "SELECT rowid, distance FROM vector_full_scan('gt', 'embedding', ?, ?)",
            (pack_vector(vectors[qid]), k),
        ).fetchall()
        results.append(set(r[0] for r in rows))

    conn.close()
    return results


def compute_ground_truth(vectors, query_ids, k, dim):
    """Choose the best ground truth method based on dataset size."""
    n = len(vectors)
    if n > 50_000:
        log.info("    Using sqlite-vector full_scan for ground truth (N=%d)", n)
        return compute_ground_truth_sqlite_vector(vectors, query_ids, k, dim)
    log.info("    Using Python brute-force for ground truth (N=%d)", n)
    return compute_ground_truth_python(vectors, query_ids, k)


# ── Recall calculation ────────────────────────────────────────────


def compute_recall(search_results, ground_truth):
    """Average recall of search_results vs ground_truth (list of sets)."""
    recalls = []
    for sr, gt in zip(search_results, ground_truth):
        if len(gt) > 0:
            recalls.append(len(sr & gt) / len(gt))
    return sum(recalls) / len(recalls) if recalls else 0.0


# ── muninn (HNSW) runner ──────────────────────────────────────────


def run_muninn(vectors, query_ids, dim, db_path=":memory:"):
    """Benchmark muninn HNSW insert + search. Returns metrics dict."""
    n = len(vectors)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(MUNINN_PATH)

    conn.execute(
        f"CREATE VIRTUAL TABLE bench_vec USING hnsw_index("
        f"dimensions={dim}, metric='l2', m={HNSW_M}, "
        f"ef_construction={HNSW_EF_CONSTRUCTION})"
    )

    # Insert
    rss_before = peak_rss_mb()
    t0 = time.perf_counter()
    for rowid, v in vectors.items():
        conn.execute(
            "INSERT INTO bench_vec (rowid, vector) VALUES (?, ?)",
            (rowid, pack_vector(v)),
        )
    t_insert = time.perf_counter() - t0
    rss_after = peak_rss_mb()

    # Search
    t0 = time.perf_counter()
    results = []
    for qid in query_ids:
        rows = conn.execute(
            "SELECT rowid, distance FROM bench_vec"
            " WHERE vector MATCH ? AND k = ? AND ef_search = ?",
            (pack_vector(vectors[qid]), K, HNSW_EF_SEARCH),
        ).fetchall()
        results.append(set(r[0] for r in rows))
    t_search = time.perf_counter() - t0

    conn.commit()
    conn.close()

    db_size = None
    if db_path != ":memory:":
        db_size = Path(db_path).stat().st_size
        log.info("    DB saved: %s (%s)", db_path, _fmt_bytes(db_size))

    return {
        "insert_rate": n / t_insert if t_insert > 0 else float("inf"),
        "search_ms": (t_search / len(query_ids)) * 1000,
        "results": results,
        "memory_mb": max(0, rss_after - rss_before),
        "db_path": str(db_path) if db_path != ":memory:" else None,
        "db_size_bytes": db_size,
    }


# ── sqliteai-vector runner ────────────────────────────────────────


def run_sqlite_vector(vectors, query_ids, dim, db_path=":memory:"):
    """Benchmark sqliteai-vector insert + quantize + search. Returns metrics dict."""
    n = len(vectors)
    ext_path = _sqlite_vector_ext_path()

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(ext_path)

    conn.execute("CREATE TABLE bench(id INTEGER PRIMARY KEY, embedding BLOB)")

    # Insert
    rss_before = peak_rss_mb()
    t0 = time.perf_counter()
    for rowid, v in vectors.items():
        conn.execute(
            "INSERT INTO bench(id, embedding) VALUES (?, ?)",
            (rowid, pack_vector(v)),
        )
    t_insert = time.perf_counter() - t0

    # Init + quantize
    conn.execute(f"SELECT vector_init('bench', 'embedding', 'dimension={dim},type=FLOAT32,distance=L2')")

    t0_q = time.perf_counter()
    conn.execute("SELECT vector_quantize('bench', 'embedding')")
    t_quantize = time.perf_counter() - t0_q
    rss_after = peak_rss_mb()

    # Approximate search (quantized)
    t0 = time.perf_counter()
    approx_results = []
    for qid in query_ids:
        rows = conn.execute(
            "SELECT rowid, distance FROM vector_quantize_scan('bench', 'embedding', ?, ?)",
            (pack_vector(vectors[qid]), K),
        ).fetchall()
        approx_results.append(set(r[0] for r in rows))
    t_approx = time.perf_counter() - t0

    # Full scan
    t0 = time.perf_counter()
    full_results = []
    for qid in query_ids:
        rows = conn.execute(
            "SELECT rowid, distance FROM vector_full_scan('bench', 'embedding', ?, ?)",
            (pack_vector(vectors[qid]), K),
        ).fetchall()
        full_results.append(set(r[0] for r in rows))
    t_full = time.perf_counter() - t0

    conn.commit()
    conn.close()

    db_size = None
    if db_path != ":memory:":
        db_size = Path(db_path).stat().st_size
        log.info("    DB saved: %s (%s)", db_path, _fmt_bytes(db_size))

    return {
        "insert_rate": n / t_insert if t_insert > 0 else float("inf"),
        "quantize_s": t_quantize,
        "approx_search_ms": (t_approx / len(query_ids)) * 1000,
        "full_search_ms": (t_full / len(query_ids)) * 1000,
        "approx_results": approx_results,
        "full_results": full_results,
        "memory_mb": max(0, rss_after - rss_before),
        "db_path": str(db_path) if db_path != ":memory:" else None,
        "db_size_bytes": db_size,
    }


# ── vectorlite runner ─────────────────────────────────────────────


def run_vectorlite(vectors, query_ids, dim, db_path=":memory:"):
    """Benchmark vectorlite HNSW insert + search. Returns metrics dict.

    Uses apsw instead of sqlite3 because vectorlite requires a newer SQLite
    version than the one bundled with Python's sqlite3 module.
    """
    if not HAS_VECTORLITE:
        log.error("vectorlite not available: pip install vectorlite-py apsw")
        return None

    n = len(vectors)
    conn = apsw.Connection(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(vectorlite_py.vectorlite_path())

    cursor = conn.cursor()
    cursor.execute(
        f"CREATE VIRTUAL TABLE bench_vl USING vectorlite("
        f"embedding float32[{dim}] l2, "
        f"hnsw(max_elements={n}, ef_construction={HNSW_EF_CONSTRUCTION}, M={HNSW_M}))"
    )

    # Insert
    rss_before = peak_rss_mb()
    t0 = time.perf_counter()
    for rowid, v in vectors.items():
        cursor.execute(
            "INSERT INTO bench_vl(rowid, embedding) VALUES (?, ?)",
            (rowid, pack_vector(v)),
        )
    t_insert = time.perf_counter() - t0
    rss_after = peak_rss_mb()

    # Search
    t0 = time.perf_counter()
    results = []
    for qid in query_ids:
        rows = list(cursor.execute(
            "SELECT rowid, distance FROM bench_vl WHERE knn_search(embedding, knn_param(?, ?, ?))",
            (pack_vector(vectors[qid]), K, HNSW_EF_SEARCH),
        ))
        results.append(set(r[0] for r in rows))
    t_search = time.perf_counter() - t0

    conn.close()

    db_size = None
    if db_path != ":memory:":
        db_size = Path(db_path).stat().st_size
        log.info("    DB saved: %s (%s)", db_path, _fmt_bytes(db_size))

    return {
        "insert_rate": n / t_insert if t_insert > 0 else float("inf"),
        "search_ms": (t_search / len(query_ids)) * 1000,
        "results": results,
        "memory_mb": max(0, rss_after - rss_before),
        "db_path": str(db_path) if db_path != ":memory:" else None,
        "db_size_bytes": db_size,
    }


# ── sqlite-vec runner ─────────────────────────────────────────────


def run_sqlite_vec(vectors, query_ids, dim, db_path=":memory:"):
    """Benchmark sqlite-vec brute-force KNN insert + search. Returns metrics dict."""
    if not HAS_SQLITE_VEC:
        log.error("sqlite-vec not available: pip install sqlite-vec")
        return None

    n = len(vectors)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    conn.execute(f"CREATE VIRTUAL TABLE bench_sv USING vec0(embedding float[{dim}])")

    # Insert
    rss_before = peak_rss_mb()
    t0 = time.perf_counter()
    for rowid, v in vectors.items():
        conn.execute(
            "INSERT INTO bench_sv(rowid, embedding) VALUES (?, ?)",
            (rowid, pack_vector(v)),
        )
    t_insert = time.perf_counter() - t0
    rss_after = peak_rss_mb()

    # Search (brute-force KNN)
    t0 = time.perf_counter()
    results = []
    for qid in query_ids:
        rows = conn.execute(
            "SELECT rowid, distance FROM bench_sv WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (pack_vector(vectors[qid]), K),
        ).fetchall()
        results.append(set(r[0] for r in rows))
    t_search = time.perf_counter() - t0

    conn.commit()
    conn.close()

    db_size = None
    if db_path != ":memory:":
        db_size = Path(db_path).stat().st_size
        log.info("    DB saved: %s (%s)", db_path, _fmt_bytes(db_size))

    return {
        "insert_rate": n / t_insert if t_insert > 0 else float("inf"),
        "search_ms": (t_search / len(query_ids)) * 1000,
        "results": results,
        "memory_mb": max(0, rss_after - rss_before),
        "db_path": str(db_path) if db_path != ":memory:" else None,
        "db_size_bytes": db_size,
    }


# ── JSONL output ──────────────────────────────────────────────────


def write_jsonl_record(filepath, record):
    """Append a single JSON record to the JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def make_record(
    engine, search_method, vector_source, model_name, dim, n, metrics,
    saturation, storage="memory", engine_params=None, dataset=None,
):
    """Build a JSONL record from benchmark metrics."""
    info = platform_info()
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "engine": engine,
        "search_method": search_method,
        "vector_source": vector_source,
        "model_name": model_name,
        "dataset": dataset,
        "dim": dim,
        "n": n,
        "k": K,
        "metric": "l2",
        "n_queries": N_QUERIES,
        "storage": storage,
        "insert_rate_vps": round(metrics.get("insert_rate", 0), 1),
        "search_latency_ms": round(metrics.get("search_ms", 0), 3),
        "recall_at_k": round(metrics.get("recall", 0), 4),
        "memory_delta_mb": round(metrics.get("memory_mb", 0), 1),
        "quantize_s": round(metrics["quantize_s"], 3) if metrics.get("quantize_s") is not None else None,
        "db_path": metrics.get("db_path"),
        "db_size_bytes": metrics.get("db_size_bytes"),
        "relative_contrast": saturation.get("relative_contrast"),
        "distance_cv": saturation.get("distance_cv"),
        "nearest_farthest_ratio": saturation.get("nearest_farthest_ratio"),
        "platform": info["platform"],
        "python_version": info["python_version"],
        "engine_params": engine_params or {},
    }


# ── Model embedding support ──────────────────────────────────────


def _model_cache_path(model_label, dataset="ag_news"):
    """Return the .npy cache path for a model+dataset combination."""
    return VECTORS_DIR / f"{model_label}_{dataset}.npy"


def load_or_generate_model_vectors(model_label, model_id, dim, n, dataset="ag_news"):
    """Load cached model embeddings or generate them from the specified dataset.

    Uses a single .npy cache per model+dataset (not per size). If the cache has
    enough vectors it is sliced; otherwise it is regenerated at the
    requested size.
    """
    cache_path = _model_cache_path(model_label, dataset)

    if cache_path.exists():
        arr = np.load(cache_path)
        if len(arr) >= n:
            log.info("    Loading cached embeddings from %s (%d/%d vectors)", cache_path, n, len(arr))
            vectors = {i + 1: arr[i].tolist() for i in range(n)}
            return vectors
        log.info("    Cache has %d vectors, need %d — regenerating", len(arr), n)

    log.info("    Generating %d embeddings with %s (%s) on %s...", n, model_label, model_id, dataset)

    if not HAS_MODEL_DEPS:
        log.error("Model embeddings require: uv sync --all-groups")
        sys.exit(1)

    texts = load_dataset_texts(dataset, max_n=n)

    if len(texts) < n:
        log.warning("    %s has %d texts, requested %d — using available", dataset, len(texts), n)
        n = len(texts)

    model = SentenceTransformer(model_id)
    embeddings = model.encode(texts[:n], show_progress_bar=True, batch_size=256, normalize_embeddings=True)

    # Cache for reuse
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    log.info("    Cached %d embeddings to %s", n, cache_path)

    del model  # free GPU/CPU memory

    vectors = {i + 1: embeddings[i].tolist() for i in range(n)}
    return vectors


def prep_model_vectors(only_model=None, only_dataset=None):
    """Pre-download models, datasets, and generate all .npy cache files.

    Generates one cache file per model per dataset at the maximum size needed
    by the models profile. Subsequent benchmark runs load and slice from these
    caches without touching the network or the GPU.

    Use only_model / only_dataset to limit scope and reduce peak memory.
    Each (model, dataset) pair runs with ~1-2 GB RAM; running all 6 at once
    can spike to 6+ GB.
    """
    if not HAS_MODEL_DEPS:
        log.error("Model prep requires: uv sync --all-groups")
        sys.exit(1)

    max_n = max(PROFILES["models"]["sizes"])
    datasets_to_prep = PROFILES["models"].get("datasets", ["ag_news"])
    models_to_prep = dict(EMBEDDING_MODELS)

    if only_dataset:
        datasets_to_prep = [only_dataset]
    if only_model:
        models_to_prep = {only_model: EMBEDDING_MODELS[only_model]}

    log.info("Pre-building model vector caches (max N=%d, datasets=%s, models=%s)", max_n, datasets_to_prep, list(models_to_prep))

    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_key in datasets_to_prep:
        log.info("\n  Dataset: %s", dataset_key)
        texts = load_dataset_texts(dataset_key, max_n=max_n)
        n = len(texts)
        log.info("  %s: %d texts available", dataset_key, n)

        for model_label, model_info in models_to_prep.items():
            cache_path = _model_cache_path(model_label, dataset_key)

            if cache_path.exists():
                arr = np.load(cache_path)
                if len(arr) >= n:
                    log.info("  %s/%s: cached (%d vectors, %s)", model_label, dataset_key, len(arr), _fmt_bytes(cache_path.stat().st_size))
                    continue
                log.info("  %s/%s: cache has %d vectors, need %d — regenerating", model_label, dataset_key, len(arr), n)

            log.info("  %s: downloading model %s...", model_label, model_info["model_id"])
            model = SentenceTransformer(model_info["model_id"])

            log.info("  %s/%s: encoding %d texts (dim=%d)...", model_label, dataset_key, n, model_info["dim"])
            embeddings = model.encode(texts[:n], show_progress_bar=True, batch_size=256, normalize_embeddings=True)

            np.save(cache_path, embeddings)
            log.info("  %s/%s: cached %d embeddings to %s (%s)", model_label, dataset_key, n, cache_path, _fmt_bytes(cache_path.stat().st_size))

            # Free model memory before loading the next one
            del model

    log.info("Model prep complete. Cached vectors in %s", VECTORS_DIR)


# ── Extension verification ────────────────────────────────────────


def verify_extensions():
    """Verify all extensions are loadable. Returns dict of {engine: bool}."""
    status = {}

    # muninn
    try:
        c = sqlite3.connect(":memory:")
        c.enable_load_extension(True)
        c.load_extension(MUNINN_PATH)
        c.close()
        log.info("  muninn:          OK (%s)", MUNINN_PATH)
        status["muninn"] = True
    except Exception as e:
        log.error("  muninn:          FAILED — %s", e)
        log.error("  Run 'make all' first.")
        status["muninn"] = False

    # sqlite-vector (sqliteai)
    try:
        ext = _sqlite_vector_ext_path()
        c = sqlite3.connect(":memory:")
        c.enable_load_extension(True)
        c.load_extension(ext)
        version = c.execute("SELECT vector_version()").fetchone()[0]
        backend = c.execute("SELECT vector_backend()").fetchone()[0]
        c.close()
        log.info("  sqlite-vector:   OK (v%s, %s)", version, backend)
        status["sqlite_vector"] = True
    except Exception as e:
        log.error("  sqlite-vector:   FAILED — %s", e)
        status["sqlite_vector"] = False

    # vectorlite
    if HAS_VECTORLITE:
        try:
            c = apsw.Connection(":memory:")
            c.enable_load_extension(True)
            c.load_extension(vectorlite_py.vectorlite_path())
            c.close()
            log.info("  vectorlite:      OK (%s)", vectorlite_py.vectorlite_path())
            status["vectorlite"] = True
        except Exception as e:
            log.error("  vectorlite:      FAILED — %s", e)
            status["vectorlite"] = False
    else:
        log.warning("  vectorlite:      not installed (pip install vectorlite-py apsw)")
        status["vectorlite"] = False

    # sqlite-vec
    if HAS_SQLITE_VEC:
        try:
            c = sqlite3.connect(":memory:")
            c.enable_load_extension(True)
            sqlite_vec.load(c)
            version = c.execute("SELECT vec_version()").fetchone()[0]
            c.close()
            log.info("  sqlite-vec:      OK (v%s)", version)
            status["sqlite_vec"] = True
        except Exception as e:
            log.error("  sqlite-vec:      FAILED — %s", e)
            status["sqlite_vec"] = False
    else:
        log.warning("  sqlite-vec:      not installed (pip install sqlite-vec)")
        status["sqlite_vec"] = False

    return status


# ── Main benchmark loop ──────────────────────────────────────────


def run_benchmark(
    vector_source, model_name, dim, sizes, engines, output_path,
    storage="memory", run_timestamp=None, dataset=None,
):
    """Run the benchmark for a single dimension and vector source."""
    total_configs = len(sizes) * len(engines)
    completed = 0
    start_time = time.perf_counter()

    for n in sizes:
        n = enforce_memory_limit(dim, n)

        # Generate or load vectors
        if vector_source == "random":
            log.info("\n  Generating %d random vectors (dim=%d)...", n, dim)
            vectors = generate_dataset(n, dim)
        else:
            model_info = EMBEDDING_MODELS.get(model_name)
            if model_info is None:
                log.error("Unknown model: %s", model_name)
                continue
            vectors = load_or_generate_model_vectors(
                model_name, model_info["model_id"], dim, n, dataset=dataset or "ag_news",
            )
            n = len(vectors)  # may be clamped by dataset size

        query_ids = pick_queries(vectors, N_QUERIES)

        # Compute ground truth
        log.info("  Computing ground truth (N=%d, dim=%d)...", n, dim)
        ground_truth = compute_ground_truth(vectors, query_ids, K, dim)

        # Compute saturation metrics (once per dim/N combo)
        log.info("  Computing saturation metrics...")
        saturation = compute_saturation_metrics(vectors, dim)

        # Determine db paths for disk storage
        scenario = make_scenario_name(vector_source, model_name, dataset, dim, n)

        for engine in engines:
            if storage == "disk":
                db_path = str(make_db_path(scenario, run_timestamp, engine))
            else:
                db_path = ":memory:"

            record_dataset = dataset if vector_source != "random" else None

            if engine == "muninn":
                log.info("  Running muninn HNSW (N=%d, dim=%d, storage=%s)...", n, dim, storage)
                vg = run_muninn(vectors, query_ids, dim, db_path=db_path)
                vg["recall"] = compute_recall(vg.pop("results"), ground_truth)

                record = make_record(
                    engine="muninn",
                    search_method="hnsw",
                    vector_source=vector_source,
                    model_name=model_name if vector_source != "random" else None,
                    dim=dim,
                    n=n,
                    metrics={
                        "insert_rate": vg["insert_rate"],
                        "search_ms": vg["search_ms"],
                        "recall": vg["recall"],
                        "memory_mb": vg["memory_mb"],
                        "db_path": vg.get("db_path"),
                        "db_size_bytes": vg.get("db_size_bytes"),
                    },
                    saturation=saturation,
                    storage=storage,
                    engine_params={"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION, "ef_search": HNSW_EF_SEARCH},
                    dataset=record_dataset,
                )
                write_jsonl_record(output_path, record)

            elif engine == "sqlite_vector":
                log.info("  Running sqlite-vector (N=%d, dim=%d, storage=%s)...", n, dim, storage)
                sv = run_sqlite_vector(vectors, query_ids, dim, db_path=db_path)
                sv["recall_approx"] = compute_recall(sv.pop("approx_results"), ground_truth)
                sv["recall_full"] = compute_recall(sv.pop("full_results"), ground_truth)

                # Write quantize_scan record
                record_q = make_record(
                    engine="sqlite_vector",
                    search_method="quantize_scan",
                    vector_source=vector_source,
                    model_name=model_name if vector_source != "random" else None,
                    dim=dim,
                    n=n,
                    metrics={
                        "insert_rate": sv["insert_rate"],
                        "search_ms": sv["approx_search_ms"],
                        "recall": sv["recall_approx"],
                        "memory_mb": sv["memory_mb"],
                        "quantize_s": sv["quantize_s"],
                        "db_path": sv.get("db_path"),
                        "db_size_bytes": sv.get("db_size_bytes"),
                    },
                    saturation=saturation,
                    storage=storage,
                    dataset=record_dataset,
                )
                write_jsonl_record(output_path, record_q)

                # Write full_scan record
                record_f = make_record(
                    engine="sqlite_vector",
                    search_method="full_scan",
                    vector_source=vector_source,
                    model_name=model_name if vector_source != "random" else None,
                    dim=dim,
                    n=n,
                    metrics={
                        "insert_rate": sv["insert_rate"],
                        "search_ms": sv["full_search_ms"],
                        "recall": sv["recall_full"],
                        "memory_mb": sv["memory_mb"],
                        "quantize_s": sv["quantize_s"],
                        "db_path": sv.get("db_path"),
                        "db_size_bytes": sv.get("db_size_bytes"),
                    },
                    saturation=saturation,
                    storage=storage,
                    dataset=record_dataset,
                )
                write_jsonl_record(output_path, record_f)

            elif engine == "vectorlite":
                log.info("  Running vectorlite HNSW (N=%d, dim=%d, storage=%s)...", n, dim, storage)
                vl = run_vectorlite(vectors, query_ids, dim, db_path=db_path)
                if vl is None:
                    completed += 1
                    continue
                vl["recall"] = compute_recall(vl.pop("results"), ground_truth)

                record = make_record(
                    engine="vectorlite",
                    search_method="hnsw",
                    vector_source=vector_source,
                    model_name=model_name if vector_source != "random" else None,
                    dim=dim,
                    n=n,
                    metrics={
                        "insert_rate": vl["insert_rate"],
                        "search_ms": vl["search_ms"],
                        "recall": vl["recall"],
                        "memory_mb": vl["memory_mb"],
                        "db_path": vl.get("db_path"),
                        "db_size_bytes": vl.get("db_size_bytes"),
                    },
                    saturation=saturation,
                    storage=storage,
                    engine_params={"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION, "ef_search": HNSW_EF_SEARCH},
                    dataset=record_dataset,
                )
                write_jsonl_record(output_path, record)

            elif engine == "sqlite_vec":
                log.info("  Running sqlite-vec brute-force (N=%d, dim=%d, storage=%s)...", n, dim, storage)
                sv = run_sqlite_vec(vectors, query_ids, dim, db_path=db_path)
                if sv is None:
                    completed += 1
                    continue
                sv["recall"] = compute_recall(sv.pop("results"), ground_truth)

                record = make_record(
                    engine="sqlite_vec",
                    search_method="brute_force",
                    vector_source=vector_source,
                    model_name=model_name if vector_source != "random" else None,
                    dim=dim,
                    n=n,
                    metrics={
                        "insert_rate": sv["insert_rate"],
                        "search_ms": sv["search_ms"],
                        "recall": sv["recall"],
                        "memory_mb": sv["memory_mb"],
                        "db_path": sv.get("db_path"),
                        "db_size_bytes": sv.get("db_size_bytes"),
                    },
                    saturation=saturation,
                    storage=storage,
                    dataset=record_dataset,
                )
                write_jsonl_record(output_path, record)

            completed += 1
            elapsed = time.perf_counter() - start_time
            rate = elapsed / completed if completed > 0 else 0
            remaining = rate * (total_configs - completed)
            log.info(
                "  Progress: %d/%d configs — elapsed %s, est. remaining %s",
                completed,
                total_configs,
                format_time(elapsed),
                format_time(remaining),
            )


ALL_ENGINES = ["muninn", "sqlite_vector", "vectorlite", "sqlite_vec"]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-engine benchmark: muninn vs sqlite-vector vs vectorlite vs sqlite-vec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Profiles:
  small       3 dims (384,768,1536), N<=50K, random      (~10 min)
  medium      2 dims (384,768), N=100K-500K, random      (~1-2 hrs)
  saturation  8 dims (32-1536), N=50K, random            (~20 min)
  models      3 models x 2 datasets, N<=250K             (~30 min)

Datasets (for model source):
  ag_news              120K news snippets (HuggingFace)
  wealth_of_nations    ~2.5K paragraphs from Gutenberg #3300

Examples:
  python python/benchmark_vss.py --profile small
  python python/benchmark_vss.py --source random --dim 384 --sizes 1000,5000
  python python/benchmark_vss.py --source model:all-MiniLM-L6-v2 --sizes 1000 --dataset ag_news
  python python/benchmark_vss.py --source model:all-MiniLM-L6-v2 --sizes 1000 --dataset wealth_of_nations
  python python/benchmark_vss.py --engine sqlite_vec --source random --dim 384 --sizes 1000
  python python/benchmark_vss.py --profile small --storage disk
        """,
    )
    parser.add_argument("--profile", choices=PROFILES.keys(), help="Predefined benchmark profile")
    parser.add_argument("--source", default="random", help="Vector source: 'random' or 'model:<model_id>'")
    parser.add_argument("--dim", type=int, help="Vector dimension (for random source)")
    parser.add_argument("--sizes", help="Comma-separated dataset sizes (e.g., 1000,5000,10000)")
    parser.add_argument(
        "--engine",
        choices=["all"] + ALL_ENGINES,
        default="all",
        help="Which engine(s) to benchmark",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="ag_news",
        help="Text dataset for model embeddings (default: ag_news)",
    )
    parser.add_argument(
        "--storage",
        choices=["memory", "disk"],
        default="memory",
        help="Storage backend: 'memory' (default) or 'disk' (persists SQLite files)",
    )
    parser.add_argument(
        "--prep-models",
        action="store_true",
        help="Download models, datasets, and pre-build .npy caches, then exit",
    )
    parser.add_argument(
        "--prep-model",
        choices=list(EMBEDDING_MODELS.keys()),
        help="Prep only this model (use with --prep-models to limit memory)",
    )
    parser.add_argument(
        "--prep-dataset",
        choices=list(DATASETS.keys()),
        help="Prep only this dataset (use with --prep-models to limit memory)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    random.seed(42)
    np.random.seed(42)

    args = parse_args()

    if args.prep_models:
        prep_model_vectors(only_model=args.prep_model, only_dataset=args.prep_dataset)
        return

    # Verify extensions
    log.info("Checking extensions...")
    ext_status = verify_extensions()

    # Determine engines to run
    if args.engine == "all":
        engines = [e for e in ALL_ENGINES if ext_status.get(e)]
    else:
        engines = [args.engine] if ext_status.get(args.engine) else []

    if not engines:
        log.error("No engines available. Exiting.")
        sys.exit(1)

    log.info("Engines: %s", ", ".join(engines))

    # Determine output path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"run_{timestamp}.jsonl"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    storage = args.storage
    log.info("Results will be written to: %s", output_path)
    if storage == "disk":
        log.info("Storage: disk (SQLite files will be saved under benchmarks/results/)")

    if args.profile:
        profile = PROFILES[args.profile]
        log.info("Running profile: %s", args.profile)

        if profile["source"] == "models":
            datasets_to_run = profile.get("datasets", ["ag_news"])
            # Run each model x dataset separately
            for dataset_key in datasets_to_run:
                for model_label, model_info in EMBEDDING_MODELS.items():
                    dim = model_info["dim"]
                    log.info("\n" + "=" * 72)
                    log.info("Model: %s (dim=%d), Dataset: %s", model_label, dim, dataset_key)
                    log.info("=" * 72)
                    run_benchmark(
                        vector_source="model",
                        model_name=model_label,
                        dim=dim,
                        sizes=profile["sizes"],
                        engines=engines,
                        output_path=output_path,
                        storage=storage,
                        run_timestamp=timestamp,
                        dataset=dataset_key,
                    )
        else:
            for dim in profile["dimensions"]:
                log.info("\n" + "=" * 72)
                log.info("Dimension: %d", dim)
                log.info("=" * 72)
                run_benchmark(
                    vector_source=profile["source"],
                    model_name=None,
                    dim=dim,
                    sizes=profile["sizes"],
                    engines=engines,
                    output_path=output_path,
                    storage=storage,
                    run_timestamp=timestamp,
                )
    else:
        # Custom run from individual args
        source = args.source
        model_name = None
        dataset = args.dataset

        if source.startswith("model:"):
            model_id = source.split(":", 1)[1]
            # Find model by ID
            model_name = None
            dim = args.dim
            for label, info in EMBEDDING_MODELS.items():
                if info["model_id"] == model_id:
                    model_name = label
                    dim = info["dim"]
                    break
            if model_name is None:
                model_name = model_id
                if dim is None:
                    log.error("Must specify --dim for unknown model")
                    sys.exit(1)
            source = "model"
        else:
            dim = args.dim or 384

        sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else [1_000, 5_000, 10_000]

        log.info("\n" + "=" * 72)
        log.info("Custom run: source=%s, dim=%d, sizes=%s, dataset=%s", source, dim, sizes, dataset)
        log.info("=" * 72)

        run_benchmark(
            vector_source=source,
            model_name=model_name,
            dim=dim,
            sizes=sizes,
            engines=engines,
            output_path=output_path,
            storage=storage,
            run_timestamp=timestamp,
            dataset=dataset if source != "random" else None,
        )

    log.info("\n" + "=" * 72)
    log.info("Benchmark complete. Results: %s", output_path)
    log.info("Run 'make benchmark-analyze' to generate charts and tables.")


if __name__ == "__main__":
    main()
