"""VSS (Vector Similarity Search) treatment.

Ports the core VSS benchmark logic from benchmarks/scripts/benchmark_vss.py.
Supports 5 engines: muninn-hnsw, sqlite-vector-quantize, sqlite-vector-fullscan,
vectorlite-hnsw, sqlite-vec-brute.

Each VSSTreatment instance represents one permutation:
    engine x model x dataset x N
"""

import importlib.resources
import logging
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.harness.common import (
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    N_QUERIES,
    VECTORS_DIR,
    K,
    pack_vector,
)
from benchmarks.harness.treatments.base import Treatment

log = logging.getLogger(__name__)

# Engine/method pairs and their human-readable slugs
ENGINE_CONFIGS: dict[str, dict[str, str]] = {
    "muninn-hnsw": {"engine": "muninn", "method": "hnsw"},
    "sqlite-vector-quantize": {"engine": "sqlite_vector", "method": "quantize_scan"},
    "sqlite-vector-fullscan": {"engine": "sqlite_vector", "method": "full_scan"},
    "vectorlite-hnsw": {"engine": "vectorlite", "method": "hnsw"},
    "sqlite-vec-brute": {"engine": "sqlite_vec", "method": "brute_force"},
}


def _sqlite_vector_ext_path() -> str:
    """Locate the sqliteai-vector binary for load_extension()."""
    return str(importlib.resources.files("sqlite_vector.binaries") / "vector")


def _load_vectors(model_name: str, dataset: str, n: int) -> tuple[dict[int, list[float]], int]:
    """Load pre-computed vectors from .npy cache.

    Returns dict mapping rowid -> list[float].
    """
    npy_path = VECTORS_DIR / f"{model_name}_{dataset}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Vector cache not found: {npy_path}. Run 'prep vectors' first.")

    arr = np.load(npy_path)
    actual_n = min(n, len(arr))
    if actual_n < n:
        log.warning("  Cache has %d vectors, need %d — using available", len(arr), n)

    return {i + 1: arr[i].tolist() for i in range(actual_n)}, actual_n


def _brute_force_knn(query: list[float], vectors: dict[int, list[float]], k: int) -> set[int]:
    """Brute force KNN by L2 distance. Returns set of rowids."""
    dists: list[tuple[float, int]] = []
    for rowid, v in vectors.items():
        d = sum((a - b) ** 2 for a, b in zip(query, v, strict=False))
        dists.append((d, rowid))
    dists.sort()
    return {rowid for _, rowid in dists[:k]}


def _compute_ground_truth(vectors: dict[int, list[float]], query_ids: list[int], k: int) -> list[set[int]]:
    """Compute ground truth via Python brute-force."""
    return [_brute_force_knn(vectors[qid], vectors, k) for qid in query_ids]


def _compute_recall(search_results: list[set[int]], ground_truth: list[set[int]]) -> float:
    """Average recall of search_results vs ground_truth (list of sets)."""
    recalls: list[float] = []
    for sr, gt in zip(search_results, ground_truth, strict=False):
        if len(gt) > 0:
            recalls.append(len(sr & gt) / len(gt))
    return sum(recalls) / len(recalls) if recalls else 0.0


class VSSTreatment(Treatment):
    """Single VSS benchmark permutation."""

    def __init__(self, engine_slug: str, model_name: str, dim: int, dataset: str, n: int) -> None:
        self._engine_slug = engine_slug
        self._engine_config = ENGINE_CONFIGS[engine_slug]
        self._model_name = model_name
        self._dim = dim
        self._dataset = dataset
        self._n = n
        self._vectors: dict[int, list[float]] | None = None
        self._query_ids: list[int] | None = None
        self._ground_truth: list[set[int]] | None = None
        self._actual_n = n

    @property
    def category(self) -> str:
        return "vss"

    @property
    def permutation_id(self) -> str:
        ds = self._dataset.replace("_", "-")
        return f"vss_{self._engine_slug}_{self._model_name}_{ds}_n{self._n}"

    @property
    def label(self) -> str:
        return f"VSS: {self._engine_slug} / {self._model_name} / {self._dataset} / N={self._n}"

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (self._n, self._dim, self._model_name, self._engine_slug)

    def params_dict(self) -> dict[str, Any]:
        return {
            "engine": self._engine_config["engine"],
            "search_method": self._engine_config["method"],
            "model_name": self._model_name,
            "dim": self._dim,
            "dataset": self._dataset,
            "n": self._actual_n,
            "k": K,
            "n_queries": N_QUERIES,
        }

    def setup(self, conn: sqlite3.Connection, db_path: Path) -> dict[str, Any]:
        """Load vectors from .npy cache and compute ground truth."""
        self._vectors, self._actual_n = _load_vectors(self._model_name, self._dataset, self._n)

        # Pick query IDs
        rng = random.Random(42)
        self._query_ids = rng.sample(
            list(self._vectors.keys()),
            min(N_QUERIES, len(self._vectors)),
        )

        # Compute ground truth
        self._ground_truth = _compute_ground_truth(self._vectors, self._query_ids, K)

        return {"n_vectors_loaded": self._actual_n, "n_queries": len(self._query_ids)}

    def run(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Run the benchmark for the configured engine."""
        engine = self._engine_config["engine"]
        method = self._engine_config["method"]

        if engine == "muninn":
            return self._run_muninn(conn)
        elif engine == "sqlite_vector" and method == "quantize_scan":
            return self._run_sqlite_vector_quantize(conn)
        elif engine == "sqlite_vector" and method == "full_scan":
            return self._run_sqlite_vector_fullscan(conn)
        elif engine == "vectorlite":
            return self._run_vectorlite(conn)
        elif engine == "sqlite_vec":
            return self._run_sqlite_vec(conn)
        else:
            raise ValueError(f"Unknown engine config: {engine}/{method}")

    def teardown(self, conn: sqlite3.Connection) -> None:
        self._vectors = None
        self._query_ids = None
        self._ground_truth = None

    def _run_muninn(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Benchmark muninn HNSW."""
        assert self._vectors is not None
        assert self._query_ids is not None
        assert self._ground_truth is not None
        dim = self._dim
        conn.execute(
            f"CREATE VIRTUAL TABLE bench_vec USING hnsw_index("
            f"dimensions={dim}, metric='l2', m={HNSW_M}, "
            f"ef_construction={HNSW_EF_CONSTRUCTION})"
        )

        # Insert
        t0 = time.perf_counter()
        for rowid, v in self._vectors.items():
            conn.execute(
                "INSERT INTO bench_vec (rowid, vector) VALUES (?, ?)",
                (rowid, pack_vector(v)),
            )
        insert_time = time.perf_counter() - t0

        # Search
        t0 = time.perf_counter()
        results: list[set[int]] = []
        for qid in self._query_ids:
            rows = conn.execute(
                "SELECT rowid, distance FROM bench_vec WHERE vector MATCH ? AND k = ? AND ef_search = ?",
                (pack_vector(self._vectors[qid]), K, HNSW_EF_SEARCH),
            ).fetchall()
            results.append({r[0] for r in rows})
        search_time = time.perf_counter() - t0

        conn.commit()
        recall = _compute_recall(results, self._ground_truth)

        return {
            "insert_rate_vps": round(self._actual_n / insert_time, 1) if insert_time > 0 else 0,
            "search_latency_ms": round((search_time / len(self._query_ids)) * 1000, 3),
            "recall_at_k": round(recall, 4),
            "engine_params": {"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION, "ef_search": HNSW_EF_SEARCH},
        }

    def _run_sqlite_vector_quantize(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Benchmark sqlite-vector quantize_scan."""
        assert self._vectors is not None
        assert self._query_ids is not None
        assert self._ground_truth is not None
        dim = self._dim
        ext_path = _sqlite_vector_ext_path()
        conn.enable_load_extension(True)
        conn.load_extension(ext_path)

        conn.execute("CREATE TABLE bench(id INTEGER PRIMARY KEY, embedding BLOB)")

        # Insert
        t0 = time.perf_counter()
        for rowid, v in self._vectors.items():
            conn.execute("INSERT INTO bench(id, embedding) VALUES (?, ?)", (rowid, pack_vector(v)))
        insert_time = time.perf_counter() - t0

        conn.execute(f"SELECT vector_init('bench', 'embedding', 'dimension={dim},type=FLOAT32,distance=L2')")

        t0_q = time.perf_counter()
        conn.execute("SELECT vector_quantize('bench', 'embedding')")
        quantize_time = time.perf_counter() - t0_q

        # Search
        t0 = time.perf_counter()
        results: list[set[int]] = []
        for qid in self._query_ids:
            rows = conn.execute(
                "SELECT rowid, distance FROM vector_quantize_scan('bench', 'embedding', ?, ?)",
                (pack_vector(self._vectors[qid]), K),
            ).fetchall()
            results.append({r[0] for r in rows})
        search_time = time.perf_counter() - t0

        conn.commit()
        recall = _compute_recall(results, self._ground_truth)

        return {
            "insert_rate_vps": round(self._actual_n / insert_time, 1) if insert_time > 0 else 0,
            "search_latency_ms": round((search_time / len(self._query_ids)) * 1000, 3),
            "recall_at_k": round(recall, 4),
            "quantize_s": round(quantize_time, 3),
        }

    def _run_sqlite_vector_fullscan(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Benchmark sqlite-vector full_scan."""
        assert self._vectors is not None
        assert self._query_ids is not None
        assert self._ground_truth is not None
        dim = self._dim
        ext_path = _sqlite_vector_ext_path()
        conn.enable_load_extension(True)
        conn.load_extension(ext_path)

        conn.execute("CREATE TABLE bench(id INTEGER PRIMARY KEY, embedding BLOB)")

        t0 = time.perf_counter()
        for rowid, v in self._vectors.items():
            conn.execute("INSERT INTO bench(id, embedding) VALUES (?, ?)", (rowid, pack_vector(v)))
        insert_time = time.perf_counter() - t0

        conn.execute(f"SELECT vector_init('bench', 'embedding', 'dimension={dim},type=FLOAT32,distance=L2')")

        # Search
        t0 = time.perf_counter()
        results: list[set[int]] = []
        for qid in self._query_ids:
            rows = conn.execute(
                "SELECT rowid, distance FROM vector_full_scan('bench', 'embedding', ?, ?)",
                (pack_vector(self._vectors[qid]), K),
            ).fetchall()
            results.append({r[0] for r in rows})
        search_time = time.perf_counter() - t0

        conn.commit()
        recall = _compute_recall(results, self._ground_truth)

        return {
            "insert_rate_vps": round(self._actual_n / insert_time, 1) if insert_time > 0 else 0,
            "search_latency_ms": round((search_time / len(self._query_ids)) * 1000, 3),
            "recall_at_k": round(recall, 4),
        }

    def _run_vectorlite(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Benchmark vectorlite HNSW."""
        import apsw
        import vectorlite_py

        assert self._vectors is not None
        assert self._query_ids is not None
        assert self._ground_truth is not None
        dim = self._dim
        n = self._actual_n
        # vectorlite needs apsw, not sqlite3 — we create our own connection
        db_path = str(conn.execute("PRAGMA database_list").fetchone()[2])
        conn.close()

        vl_conn = apsw.Connection(db_path)
        vl_conn.enable_load_extension(True)
        vl_conn.load_extension(vectorlite_py.vectorlite_path())

        cursor = vl_conn.cursor()
        cursor.execute(
            f"CREATE VIRTUAL TABLE bench_vl USING vectorlite("
            f"embedding float32[{dim}] l2, "
            f"hnsw(max_elements={n}, ef_construction={HNSW_EF_CONSTRUCTION}, M={HNSW_M}))"
        )

        t0 = time.perf_counter()
        for rowid, v in self._vectors.items():
            cursor.execute("INSERT INTO bench_vl(rowid, embedding) VALUES (?, ?)", (rowid, pack_vector(v)))
        insert_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        results: list[set[int]] = []
        for qid in self._query_ids:
            rows = list(
                cursor.execute(
                    "SELECT rowid, distance FROM bench_vl WHERE knn_search(embedding, knn_param(?, ?, ?))",
                    (pack_vector(self._vectors[qid]), K, HNSW_EF_SEARCH),
                )
            )
            results.append({r[0] for r in rows})
        search_time = time.perf_counter() - t0

        vl_conn.close()

        recall = _compute_recall(results, self._ground_truth)

        return {
            "insert_rate_vps": round(n / insert_time, 1) if insert_time > 0 else 0,
            "search_latency_ms": round((search_time / len(self._query_ids)) * 1000, 3),
            "recall_at_k": round(recall, 4),
            "engine_params": {"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION, "ef_search": HNSW_EF_SEARCH},
        }

    def _run_sqlite_vec(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Benchmark sqlite-vec brute-force KNN."""
        import sqlite_vec

        assert self._vectors is not None
        assert self._query_ids is not None
        assert self._ground_truth is not None
        dim = self._dim
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)

        conn.execute(f"CREATE VIRTUAL TABLE bench_sv USING vec0(embedding float[{dim}])")

        t0 = time.perf_counter()
        for rowid, v in self._vectors.items():
            conn.execute("INSERT INTO bench_sv(rowid, embedding) VALUES (?, ?)", (rowid, pack_vector(v)))
        insert_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        results: list[set[int]] = []
        for qid in self._query_ids:
            rows = conn.execute(
                "SELECT rowid, distance FROM bench_sv WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (pack_vector(self._vectors[qid]), K),
            ).fetchall()
            results.append({r[0] for r in rows})
        search_time = time.perf_counter() - t0

        conn.commit()
        recall = _compute_recall(results, self._ground_truth)

        return {
            "insert_rate_vps": round(self._actual_n / insert_time, 1) if insert_time > 0 else 0,
            "search_latency_ms": round((search_time / len(self._query_ids)) * 1000, 3),
            "recall_at_k": round(recall, 4),
        }
