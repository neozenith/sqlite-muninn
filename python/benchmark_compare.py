"""
Multi-dimensional benchmark suite: vec_graph (HNSW) vs sqliteai/sqlite-vector.

Runs identical workloads through both extensions across multiple vector dimensions,
computes saturation metrics, and writes JSONL results for analysis.

Profiles:
    small       — 3 dims (384, 768, 1536), N ≤ 50K, random vectors (~10 min)
    medium      — 2 dims (384, 768), N = 100K–500K, random vectors (~1–2 hrs)
    saturation  — 8 dims (32–1536), N = 50K, random vectors (~20 min)
    models      — 3 real embedding models, N ≤ 50K (~30 min)

Prerequisites:
    pip install sqliteai-vector sentence-transformers datasets numpy
    make all

Run:
    python python/benchmark_compare.py --profile small
    python python/benchmark_compare.py --source random --dim 384 --sizes 1000,5000
"""
import argparse
import datetime
import importlib.resources
import json
import logging
import math
import platform
import random
import resource
import sqlite3
import struct
import sys
import time
from pathlib import Path

import numpy as np

try:
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    HAS_MODEL_DEPS = True
except ImportError:
    HAS_MODEL_DEPS = False

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VEC_GRAPH_PATH = str(PROJECT_ROOT / "vec_graph")
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
VECTORS_DIR = PROJECT_ROOT / "benchmarks" / "vectors"

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

# Profile definitions
PROFILES = {
    "small": {
        "source": "random",
        "dimensions": [384, 768, 1536],
        "sizes": [1_000, 5_000, 10_000, 50_000],
    },
    "medium": {
        "source": "random",
        "dimensions": [384, 768],
        "sizes": [100_000, 250_000, 500_000],
    },
    "saturation": {
        "source": "random",
        "dimensions": [32, 64, 128, 256, 512, 768, 1024, 1536],
        "sizes": [50_000],
    },
    "models": {
        "source": "models",
        "dimensions": None,  # determined by model
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


def make_scenario_name(vector_source, model_name, dim, n):
    """Build a deterministic scenario name from run parameters."""
    if vector_source == "random":
        return f"random_dim{dim}_n{n}"
    return f"model_{model_name}_n{n}"


def make_db_path(scenario_name, run_timestamp, engine):
    """Build the SQLite DB path for disk storage."""
    db_dir = RESULTS_DIR / scenario_name
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / f"{run_timestamp}_{engine}.sqlite"


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

    # Relative Contrast: mean(pairwise) / mean(nearest-neighbor) → 1.0 means saturated
    rc = mean_pairwise / mean_nn if mean_nn > 1e-10 else None

    # Coefficient of Variation: std(pairwise) / mean(pairwise) → 0 means saturated
    cv = std_pairwise / mean_pairwise if mean_pairwise > 1e-10 else None

    # Nearest/Farthest ratio: mean → 1.0 means saturated
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
    """Compute ground truth via Python brute-force. Good for N ≤ 50K."""
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


# ── vec_graph (HNSW) runner ───────────────────────────────────────


def run_vec_graph(vectors, query_ids, dim, db_path=":memory:"):
    """Benchmark vec_graph HNSW insert + search. Returns metrics dict."""
    n = len(vectors)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(VEC_GRAPH_PATH)

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


# ── JSONL output ──────────────────────────────────────────────────


def write_jsonl_record(filepath, record):
    """Append a single JSON record to the JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def make_record(
    engine, search_method, vector_source, model_name, dim, n, metrics, saturation, storage="memory", engine_params=None
):
    """Build a JSONL record from benchmark metrics."""
    info = platform_info()
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "engine": engine,
        "search_method": search_method,
        "vector_source": vector_source,
        "model_name": model_name,
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


def _model_cache_path(model_label):
    """Return the .npy cache path for a model (one file per model, not per size)."""
    return VECTORS_DIR / f"{model_label}.npy"


def load_or_generate_model_vectors(model_label, model_id, dim, n):
    """Load cached model embeddings or generate them from AG News dataset.

    Uses a single .npy cache per model (not per size).  If the cache has
    enough vectors it is sliced; otherwise it is regenerated at the
    requested size.
    """
    cache_path = _model_cache_path(model_label)

    if cache_path.exists():
        arr = np.load(cache_path)
        if len(arr) >= n:
            log.info("    Loading cached embeddings from %s (%d/%d vectors)", cache_path, n, len(arr))
            vectors = {i + 1: arr[i].tolist() for i in range(n)}
            return vectors
        log.info("    Cache has %d vectors, need %d — regenerating", len(arr), n)

    log.info("    Generating %d embeddings with %s (%s)...", n, model_label, model_id)

    if not HAS_MODEL_DEPS:
        log.error("Model embeddings require: uv add --group benchmark-models sentence-transformers datasets")
        sys.exit(1)

    model = SentenceTransformer(model_id)
    dataset = load_dataset("ag_news", split="train")
    texts = [row["text"] for row in dataset.select(range(min(n, len(dataset))))]

    if len(texts) < n:
        log.warning("    AG News has %d texts, requested %d — using available", len(texts), n)
        n = len(texts)

    embeddings = model.encode(texts[:n], show_progress_bar=True, batch_size=256, normalize_embeddings=True)

    # Cache for reuse
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    log.info("    Cached %d embeddings to %s", n, cache_path)

    vectors = {i + 1: embeddings[i].tolist() for i in range(n)}
    return vectors


def prep_model_vectors():
    """Pre-download models, dataset, and generate all .npy cache files.

    Generates one cache file per model at the maximum size needed by the
    models profile.  Subsequent benchmark runs load and slice from these
    caches without touching the network or the GPU.
    """
    if not HAS_MODEL_DEPS:
        log.error("Model prep requires: uv add --group benchmark-models sentence-transformers datasets")
        sys.exit(1)

    max_n = max(PROFILES["models"]["sizes"])
    log.info("Pre-building model vector caches (max N=%d)", max_n)

    # Download dataset once (HuggingFace caches it, but first call is slow)
    log.info("  Loading AG News dataset...")
    dataset = load_dataset("ag_news", split="train")
    n_available = len(dataset)
    n = min(max_n, n_available)
    texts = [row["text"] for row in dataset.select(range(n))]
    log.info("  AG News: %d texts available, using %d", n_available, n)

    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    for model_label, model_info in EMBEDDING_MODELS.items():
        cache_path = _model_cache_path(model_label)

        if cache_path.exists():
            arr = np.load(cache_path)
            if len(arr) >= n:
                log.info("  %s: cached (%d vectors, %s)", model_label, len(arr), _fmt_bytes(cache_path.stat().st_size))
                continue
            log.info("  %s: cache has %d vectors, need %d — regenerating", model_label, len(arr), n)

        log.info("  %s: downloading model %s...", model_label, model_info["model_id"])
        model = SentenceTransformer(model_info["model_id"])

        log.info("  %s: encoding %d texts (dim=%d)...", model_label, n, model_info["dim"])
        embeddings = model.encode(texts[:n], show_progress_bar=True, batch_size=256, normalize_embeddings=True)

        np.save(cache_path, embeddings)
        log.info("  %s: cached %d embeddings to %s (%s)", model_label, n, cache_path, _fmt_bytes(cache_path.stat().st_size))

        # Free model memory before loading the next one
        del model

    log.info("Model prep complete. Cached vectors in %s", VECTORS_DIR)


# ── Extension verification ────────────────────────────────────────


def verify_extensions():
    """Verify both extensions are loadable. Returns (vec_graph_ok, sv_ok)."""
    vg_ok = False
    sv_ok = False

    try:
        c = sqlite3.connect(":memory:")
        c.enable_load_extension(True)
        c.load_extension(VEC_GRAPH_PATH)
        c.close()
        log.info("  vec_graph:       OK (%s)", VEC_GRAPH_PATH)
        vg_ok = True
    except Exception as e:
        log.error("  vec_graph:       FAILED — %s", e)
        log.error("  Run 'make all' first.")

    try:
        ext = _sqlite_vector_ext_path()
        c = sqlite3.connect(":memory:")
        c.enable_load_extension(True)
        c.load_extension(ext)
        version = c.execute("SELECT vector_version()").fetchone()[0]
        backend = c.execute("SELECT vector_backend()").fetchone()[0]
        c.close()
        log.info("  sqlite-vector:   OK (v%s, %s)", version, backend)
        sv_ok = True
    except Exception as e:
        log.error("  sqlite-vector:   FAILED — %s", e)
        log.error("  Run 'pip install sqliteai-vector' first.")

    return vg_ok, sv_ok


# ── Main benchmark loop ──────────────────────────────────────────


def run_benchmark(vector_source, model_name, dim, sizes, engines, output_path, storage="memory", run_timestamp=None):
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
            vectors = load_or_generate_model_vectors(model_name, model_info["model_id"], dim, n)
            n = len(vectors)  # may be clamped by dataset size

        query_ids = pick_queries(vectors, N_QUERIES)

        # Compute ground truth
        log.info("  Computing ground truth (N=%d, dim=%d)...", n, dim)
        ground_truth = compute_ground_truth(vectors, query_ids, K, dim)

        # Compute saturation metrics (once per dim/N combo)
        log.info("  Computing saturation metrics...")
        saturation = compute_saturation_metrics(vectors, dim)

        # Determine db paths for disk storage
        scenario = make_scenario_name(vector_source, model_name, dim, n)

        for engine in engines:
            if storage == "disk":
                db_path = str(make_db_path(scenario, run_timestamp, engine))
            else:
                db_path = ":memory:"

            if engine == "vec_graph":
                log.info("  Running vec_graph HNSW (N=%d, dim=%d, storage=%s)...", n, dim, storage)
                vg = run_vec_graph(vectors, query_ids, dim, db_path=db_path)
                vg["recall"] = compute_recall(vg.pop("results"), ground_truth)

                record = make_record(
                    engine="vec_graph",
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
                )
                write_jsonl_record(output_path, record_f)

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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-dimensional benchmark: vec_graph vs sqlite-vector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Profiles:
  small       3 dims (384,768,1536), N≤50K, random      (~10 min)
  medium      2 dims (384,768), N=100K-500K, random      (~1-2 hrs)
  saturation  8 dims (32-1536), N=50K, random            (~20 min)
  models      3 real embedding models, N≤50K             (~30 min)

Examples:
  python python/benchmark_compare.py --profile small
  python python/benchmark_compare.py --source random --dim 384 --sizes 1000,5000
  python python/benchmark_compare.py --source model:all-MiniLM-L6-v2 --sizes 1000,5000
  python python/benchmark_compare.py --profile small --storage disk
        """,
    )
    parser.add_argument("--profile", choices=PROFILES.keys(), help="Predefined benchmark profile")
    parser.add_argument("--source", default="random", help="Vector source: 'random' or 'model:<model_id>'")
    parser.add_argument("--dim", type=int, help="Vector dimension (for random source)")
    parser.add_argument("--sizes", help="Comma-separated dataset sizes (e.g., 1000,5000,10000)")
    parser.add_argument(
        "--engine",
        choices=["all", "vec_graph", "sqlite_vector"],
        default="all",
        help="Which engine(s) to benchmark",
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
        help="Download models, dataset, and pre-build .npy caches, then exit",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    random.seed(42)
    np.random.seed(42)

    args = parse_args()

    if args.prep_models:
        prep_model_vectors()
        return

    # Verify extensions
    log.info("Checking extensions...")
    vg_ok, sv_ok = verify_extensions()

    # Determine engines to run
    if args.engine == "all":
        engines = []
        if vg_ok:
            engines.append("vec_graph")
        if sv_ok:
            engines.append("sqlite_vector")
    elif args.engine == "vec_graph":
        engines = ["vec_graph"] if vg_ok else []
    else:
        engines = ["sqlite_vector"] if sv_ok else []

    if not engines:
        log.error("No engines available. Exiting.")
        sys.exit(1)

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
            # Run each model separately
            for model_label, model_info in EMBEDDING_MODELS.items():
                dim = model_info["dim"]
                log.info("\n" + "=" * 72)
                log.info("Model: %s (dim=%d)", model_label, dim)
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
        log.info("Custom run: source=%s, dim=%d, sizes=%s", source, dim, sizes)
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
        )

    log.info("\n" + "=" * 72)
    log.info("Benchmark complete. Results: %s", output_path)
    log.info("Run 'make benchmark-analyze' to generate charts and tables.")


if __name__ == "__main__":
    main()
