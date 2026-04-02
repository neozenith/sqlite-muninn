"""VSS (Vector Similarity Search) treatment.

Ports the core VSS benchmark logic from benchmarks/scripts/benchmark_vss.py.
Supports 5 engines: muninn-hnsw, sqlite-vector-quantize, sqlite-vector-fullscan,
vectorlite-hnsw, sqlite-vec-brute.

Each VSSTreatment instance represents one permutation:
    engine x model x dataset x N

Uses pool-based random sampling: each benchmark run draws N docs and M queries
from precomputed embedding pools, computes brute-force ground truth on the
specific sample, and measures recall of the approximate index.
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


def _load_pool(model_name: str, dataset: str, kind: str) -> np.ndarray:
    """Load a precomputed embedding pool from .npy cache.

    Args:
        model_name: e.g., "MiniLM", "NomicEmbed"
        dataset: e.g., "ag_news", "wealth_of_nations"
        kind: "docs" or "queries"

    Returns:
        Full np.ndarray pool (float32, shape [pool_size, dim]).
    """
    npy_path = VECTORS_DIR / f"{model_name}_{dataset}_{kind}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Vector cache not found: {npy_path}. Run 'prep vectors' first.")
    return np.load(npy_path)


def _compute_ground_truth(doc_vectors: np.ndarray, query_vectors: np.ndarray, k: int) -> list[set[int]]:
    """Compute brute-force KNN ground truth using vectorized numpy.

    For each query vector, finds the top-K nearest documents by L2 distance.

    Args:
        doc_vectors: shape [N, dim] — the sampled document embeddings
        query_vectors: shape [M, dim] — the sampled query embeddings
        k: number of nearest neighbors

    Returns:
        List of M sets, each containing K rowids (1-based positions in doc set).
    """
    # Vectorized L2 distance: [M, N]
    # dists[i, j] = ||query_i - doc_j||^2
    dists = np.sum((doc_vectors[None, :, :] - query_vectors[:, None, :]) ** 2, axis=2)
    # Top-K indices per query
    top_k_indices = np.argsort(dists, axis=1)[:, :k]
    # Convert to 1-based rowids (matching INSERT positions)
    return [{int(idx) + 1 for idx in row} for row in top_k_indices]


def _compute_recall(search_results: list[set[int]], ground_truth: list[set[int]]) -> float:
    """Average recall of search_results vs ground_truth (list of sets)."""
    recalls: list[float] = []
    for sr, gt in zip(search_results, ground_truth, strict=False):
        if len(gt) > 0:
            recalls.append(len(sr & gt) / len(gt))
    return sum(recalls) / len(recalls) if recalls else 0.0


class VSSTreatment(Treatment):
    """Single VSS benchmark permutation.

    For HNSW engines (muninn-hnsw, vectorlite-hnsw), accepts tunable parameters
    (m, ef_construction, ef_search). Non-HNSW engines ignore these parameters.
    """

    def __init__(
        self,
        engine_slug: str,
        model_name: str,
        dim: int,
        dataset: str,
        n: int,
        hnsw_m: int = HNSW_M,
        hnsw_ef_construction: int = HNSW_EF_CONSTRUCTION,
        hnsw_ef_search: int = HNSW_EF_SEARCH,
    ) -> None:
        self._engine_slug = engine_slug
        self._engine_config = ENGINE_CONFIGS[engine_slug]
        self._model_name = model_name
        self._dim = dim
        self._dataset = dataset
        self._n = n
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search
        # Pool-based state (set during setup)
        self._doc_vectors: np.ndarray | None = None
        self._query_vectors: np.ndarray | None = None
        self._ground_truth: list[set[int]] | None = None
        self._actual_n = n
        self._n_queries = 0
        self._seed: int | None = None

    @property
    def requires_muninn(self) -> bool:
        return self._engine_slug == "muninn-hnsw"

    @property
    def category(self) -> str:
        return "vss"

    @property
    def _is_hnsw_engine(self) -> bool:
        return self._engine_config["method"] == "hnsw"

    @property
    def _has_non_default_hnsw(self) -> bool:
        """True if HNSW params differ from defaults (affects permutation_id)."""
        return self._is_hnsw_engine and (
            self._hnsw_m != HNSW_M
            or self._hnsw_ef_construction != HNSW_EF_CONSTRUCTION
            or self._hnsw_ef_search != HNSW_EF_SEARCH
        )

    @property
    def permutation_id(self) -> str:
        ds = self._dataset.replace("_", "-")
        base = f"vss_{self._engine_slug}_{self._model_name}_{ds}_n{self._n}"
        if self._has_non_default_hnsw:
            base += f"_m{self._hnsw_m}_efc{self._hnsw_ef_construction}_efs{self._hnsw_ef_search}"
        return base

    @property
    def label(self) -> str:
        base = f"VSS: {self._engine_slug} / {self._model_name} / {self._dataset} / N={self._n}"
        if self._has_non_default_hnsw:
            base += f" / M={self._hnsw_m} ef_c={self._hnsw_ef_construction} ef_s={self._hnsw_ef_search}"
        return base

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (self._n, self._dim, self._hnsw_m, self._model_name, self._engine_slug)

    def params_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "engine": self._engine_config["engine"],
            "search_method": self._engine_config["method"],
            "model_name": self._model_name,
            "dim": self._dim,
            "dataset": self._dataset,
            "n": self._actual_n,
            "k": K,
            "n_queries": self._n_queries,
            "seed": self._seed,
        }
        if self._is_hnsw_engine:
            d["hnsw_m"] = self._hnsw_m
            d["hnsw_ef_construction"] = self._hnsw_ef_construction
            d["hnsw_ef_search"] = self._hnsw_ef_search
        return d

    def setup(self, conn: sqlite3.Connection, db_path: Path) -> dict[str, Any]:
        """Load embedding pools, sample docs + queries, compute ground truth."""
        # Generate random seed for this run
        self._seed = random.randint(0, 2**32 - 1)
        rng = np.random.default_rng(seed=self._seed)

        # Load full pools
        doc_pool = _load_pool(self._model_name, self._dataset, "docs")
        query_pool = _load_pool(self._model_name, self._dataset, "queries")

        # Sample N docs from pool
        self._actual_n = min(self._n, len(doc_pool))
        if self._actual_n < self._n:
            log.warning("  Doc pool has %d vectors, need %d — using available", len(doc_pool), self._n)

        if self._actual_n < len(doc_pool):
            doc_indices = rng.choice(len(doc_pool), size=self._actual_n, replace=False)
            self._doc_vectors = doc_pool[doc_indices]
        else:
            # N == pool size: use all docs (no sampling)
            self._doc_vectors = doc_pool

        # Sample M queries from pool
        self._n_queries = min(N_QUERIES, len(query_pool))
        if self._n_queries < N_QUERIES:
            log.warning("  Query pool has %d vectors, need %d — using available", len(query_pool), N_QUERIES)

        query_indices = rng.choice(len(query_pool), size=self._n_queries, replace=False)
        self._query_vectors = query_pool[query_indices]

        # Compute brute-force ground truth on this specific sample
        self._ground_truth = _compute_ground_truth(self._doc_vectors, self._query_vectors, K)

        return {
            "n_vectors_loaded": self._actual_n,
            "n_queries": self._n_queries,
            "seed": self._seed,
            "doc_pool_size": len(doc_pool),
            "query_pool_size": len(query_pool),
        }

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
        self._doc_vectors = None
        self._query_vectors = None
        self._ground_truth = None

    def _insert_doc_vectors(self, conn, table_sql, insert_sql):
        """Insert sampled doc vectors into the benchmark table.

        Returns (insert_time_seconds,).
        """
        assert self._doc_vectors is not None
        conn.execute(table_sql)
        t0 = time.perf_counter()
        for pos in range(len(self._doc_vectors)):
            conn.execute(insert_sql, (pos + 1, pack_vector(self._doc_vectors[pos])))
        return time.perf_counter() - t0

    def _search_queries(self, conn, search_sql, extra_params=()):
        """Search with sampled query vectors. Returns (results, search_time)."""
        assert self._query_vectors is not None
        t0 = time.perf_counter()
        results: list[set[int]] = []
        for q in range(len(self._query_vectors)):
            rows = conn.execute(
                search_sql,
                (pack_vector(self._query_vectors[q]), *extra_params),
            ).fetchall()
            results.append({r[0] for r in rows})
        search_time = time.perf_counter() - t0
        return results, search_time

    def _run_muninn(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Benchmark muninn HNSW."""
        assert self._ground_truth is not None
        dim = self._dim

        insert_time = self._insert_doc_vectors(
            conn,
            f"CREATE VIRTUAL TABLE bench_vec USING hnsw_index("
            f"dimensions={dim}, metric='l2', m={self._hnsw_m}, "
            f"ef_construction={self._hnsw_ef_construction})",
            "INSERT INTO bench_vec (rowid, vector) VALUES (?, ?)",
        )

        results, search_time = self._search_queries(
            conn,
            "SELECT rowid, distance FROM bench_vec WHERE vector MATCH ? AND k = ? AND ef_search = ?",
            extra_params=(K, self._hnsw_ef_search),
        )

        conn.commit()
        recall = _compute_recall(results, self._ground_truth)

        return {
            "insert_rate_vps": round(self._actual_n / insert_time, 1) if insert_time > 0 else 0,
            "search_latency_ms": round((search_time / self._n_queries) * 1000, 3),
            "recall_at_k": round(recall, 4),
            "engine_params": {
                "m": self._hnsw_m,
                "ef_construction": self._hnsw_ef_construction,
                "ef_search": self._hnsw_ef_search,
            },
        }

    def _run_sqlite_vector_quantize(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Benchmark sqlite-vector quantize_scan."""
        assert self._doc_vectors is not None
        assert self._ground_truth is not None
        dim = self._dim
        ext_path = _sqlite_vector_ext_path()
        conn.enable_load_extension(True)
        conn.load_extension(ext_path)

        insert_time = self._insert_doc_vectors(
            conn,
            "CREATE TABLE bench(id INTEGER PRIMARY KEY, embedding BLOB)",
            "INSERT INTO bench(id, embedding) VALUES (?, ?)",
        )

        conn.execute(f"SELECT vector_init('bench', 'embedding', 'dimension={dim},type=FLOAT32,distance=L2')")

        t0_q = time.perf_counter()
        conn.execute("SELECT vector_quantize('bench', 'embedding')")
        quantize_time = time.perf_counter() - t0_q

        results, search_time = self._search_queries(
            conn,
            "SELECT rowid, distance FROM vector_quantize_scan('bench', 'embedding', ?, ?)",
            extra_params=(K,),
        )

        conn.commit()
        recall = _compute_recall(results, self._ground_truth)

        return {
            "insert_rate_vps": round(self._actual_n / insert_time, 1) if insert_time > 0 else 0,
            "search_latency_ms": round((search_time / self._n_queries) * 1000, 3),
            "recall_at_k": round(recall, 4),
            "quantize_s": round(quantize_time, 3),
        }

    def _run_sqlite_vector_fullscan(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Benchmark sqlite-vector full_scan."""
        assert self._ground_truth is not None
        dim = self._dim
        ext_path = _sqlite_vector_ext_path()
        conn.enable_load_extension(True)
        conn.load_extension(ext_path)

        insert_time = self._insert_doc_vectors(
            conn,
            "CREATE TABLE bench(id INTEGER PRIMARY KEY, embedding BLOB)",
            "INSERT INTO bench(id, embedding) VALUES (?, ?)",
        )

        conn.execute(f"SELECT vector_init('bench', 'embedding', 'dimension={dim},type=FLOAT32,distance=L2')")

        results, search_time = self._search_queries(
            conn,
            "SELECT rowid, distance FROM vector_full_scan('bench', 'embedding', ?, ?)",
            extra_params=(K,),
        )

        conn.commit()
        recall = _compute_recall(results, self._ground_truth)

        return {
            "insert_rate_vps": round(self._actual_n / insert_time, 1) if insert_time > 0 else 0,
            "search_latency_ms": round((search_time / self._n_queries) * 1000, 3),
            "recall_at_k": round(recall, 4),
        }

    def _run_vectorlite(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Benchmark vectorlite HNSW."""
        import apsw
        import vectorlite_py

        assert self._doc_vectors is not None
        assert self._query_vectors is not None
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
            f"hnsw(max_elements={n}, ef_construction={self._hnsw_ef_construction}, M={self._hnsw_m}))"
        )

        t0 = time.perf_counter()
        for pos in range(len(self._doc_vectors)):
            cursor.execute(
                "INSERT INTO bench_vl(rowid, embedding) VALUES (?, ?)",
                (pos + 1, pack_vector(self._doc_vectors[pos])),
            )
        insert_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        results: list[set[int]] = []
        for q in range(len(self._query_vectors)):
            rows = list(
                cursor.execute(
                    "SELECT rowid, distance FROM bench_vl WHERE knn_search(embedding, knn_param(?, ?, ?))",
                    (pack_vector(self._query_vectors[q]), K, self._hnsw_ef_search),
                )
            )
            results.append({r[0] for r in rows})
        search_time = time.perf_counter() - t0

        vl_conn.close()

        recall = _compute_recall(results, self._ground_truth)

        return {
            "insert_rate_vps": round(n / insert_time, 1) if insert_time > 0 else 0,
            "search_latency_ms": round((search_time / self._n_queries) * 1000, 3),
            "recall_at_k": round(recall, 4),
            "engine_params": {
                "m": self._hnsw_m,
                "ef_construction": self._hnsw_ef_construction,
                "ef_search": self._hnsw_ef_search,
            },
        }

    def _run_sqlite_vec(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Benchmark sqlite-vec brute-force KNN."""
        import sqlite_vec

        assert self._ground_truth is not None
        dim = self._dim
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)

        insert_time = self._insert_doc_vectors(
            conn,
            f"CREATE VIRTUAL TABLE bench_sv USING vec0(embedding float[{dim}])",
            "INSERT INTO bench_sv(rowid, embedding) VALUES (?, ?)",
        )

        results, search_time = self._search_queries(
            conn,
            "SELECT rowid, distance FROM bench_sv WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            extra_params=(K,),
        )

        conn.commit()
        recall = _compute_recall(results, self._ground_truth)

        return {
            "insert_rate_vps": round(self._actual_n / insert_time, 1) if insert_time > 0 else 0,
            "search_latency_ms": round((search_time / self._n_queries) * 1000, 3),
            "recall_at_k": round(recall, 4),
        }
