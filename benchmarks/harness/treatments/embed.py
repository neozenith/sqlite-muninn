"""Embed (text->embedding->search) benchmark treatment.

Measures the full text-to-search pipeline including GGUF model inference.
Unlike the VSS benchmark (which uses pre-cached vectors), this category embeds
text live through SQL and measures the end-to-end cost.

Two independent axes form a 2x3 permutation matrix:
    embed_fn:        muninn_embed | lembed
    search_backend:  muninn-hnsw  | sqlite-vector-pq | sqlite-vec-brute

Tasks measured per permutation:
    1. Model load time (GGUF -> SQL model registry)
    2. Bulk embed + insert (N texts -> search index)
    3. Trigger-based incremental embedding (per-row latency)
    4. Live query embed + search (end-to-end query latency)
"""

import importlib.resources
import json
import logging
import random
import re
import sqlite3
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.harness.common import (
    DATASETS,
    EMBEDDING_MODELS,
    GGUF_MODELS_DIR,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    N_QUERIES,
    TEXTS_DIR,
    VECTORS_DIR,
    K,
)
from benchmarks.harness.treatments.base import Treatment

log = logging.getLogger(__name__)

# Number of individual trigger inserts to time
TRIGGER_SAMPLE_SIZE = 50


# ── Text loading utilities ───────────────────────────────────────


def _download_gutenberg(gutenberg_id: int) -> Path:
    """Fetch plain text from Project Gutenberg, strip boilerplate, cache locally."""
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TEXTS_DIR / f"gutenberg_{gutenberg_id}.txt"

    if cache_path.exists():
        return cache_path

    url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
    log.info("  Downloading Gutenberg #%d...", gutenberg_id)

    req = urllib.request.Request(url, headers={"User-Agent": "muninn-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw_text = resp.read().decode("utf-8-sig")

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
    return cache_path


def _chunk_fixed_tokens(text: str, window: int = 256, overlap: int = 50) -> list[str]:
    """Split text into fixed-size token windows with overlap."""
    words = text.split()
    if not words:
        return []

    stride = max(1, window - overlap)
    chunks = []
    for i in range(0, len(words), stride):
        chunk_words = words[i : i + window]
        if len(chunk_words) < window // 4:
            break
        chunks.append(" ".join(chunk_words))
    return chunks


def _load_doc_texts(dataset_key: str, max_n: int) -> list[str]:
    """Load document texts from a dataset. Returns list of strings.

    Prefers cached doc texts from the prep vectors step. Falls back to live loading.
    """
    # Try cached doc texts first (from prep vectors step)
    dt_path = VECTORS_DIR / f"{dataset_key}_docs.json"
    if dt_path.exists():
        data = json.loads(dt_path.read_text(encoding="utf-8"))
        docs: list[str] = list(data["docs"])
        if len(docs) >= max_n:
            return docs[:max_n]
        log.warning("  %s: cached docs (%d) < requested (%d), falling back to live load", dataset_key, len(docs), max_n)

    from datasets import load_dataset as hf_load_dataset

    ds_config = DATASETS[dataset_key]

    if ds_config["source_type"] == "huggingface":
        hf_dataset = hf_load_dataset(ds_config["hf_name"], split=ds_config["hf_split"])
        field = ds_config["text_field"]
        n = min(max_n, len(hf_dataset))
        return [row[field] for row in hf_dataset.select(range(n))]

    if ds_config["source_type"] == "gutenberg":
        text_path = _download_gutenberg(ds_config["gutenberg_id"])
        raw_text = text_path.read_text(encoding="utf-8")
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
        chunks = _chunk_fixed_tokens(raw_text, window=ds_config["chunk_tokens"], overlap=ds_config["chunk_overlap"])
        return chunks[:max_n]

    raise ValueError(f"Unknown dataset source_type: {ds_config['source_type']}")


def _load_query_texts(dataset_key: str, doc_texts: list[str] | None = None) -> list[str]:
    """Load or generate query texts for a dataset.

    Prefers cached query texts from the VSS prep step. Falls back to generation.
    """
    # Try cached query texts first (from VSS prep step)
    qt_path = VECTORS_DIR / f"{dataset_key}_queries.json"
    if qt_path.exists():
        data = json.loads(qt_path.read_text(encoding="utf-8"))
        return list(data["queries"])

    from datasets import load_dataset as hf_load_dataset

    ds_config = DATASETS[dataset_key]

    if ds_config["source_type"] == "huggingface":
        hf_dataset = hf_load_dataset(ds_config["hf_name"], split="test")
        field = ds_config["text_field"]
        return [row[field] for row in hf_dataset]

    if ds_config["source_type"] == "gutenberg":
        if doc_texts is None:
            raise ValueError("doc_texts required for gutenberg query generation")
        queries = []
        for chunk in doc_texts:
            match = re.match(r"(.+?[.!?])\s", chunk)
            if match:
                queries.append(match.group(1).strip())
            else:
                words = chunk.split()[:50]
                queries.append(" ".join(words))
        return queries

    raise ValueError(f"Unknown dataset source_type: {ds_config['source_type']}")


# ── Extension loading helpers ────────────────────────────────────


def _sqlite_vector_ext_path() -> str:
    """Locate the sqliteai-vector binary for load_extension()."""
    return str(importlib.resources.files("sqlite_vector.binaries") / "vector")


def _load_lembed(conn: sqlite3.Connection) -> None:
    """Load sqlite-lembed extension into a connection."""
    import sqlite_lembed

    conn.enable_load_extension(True)
    sqlite_lembed.load(conn)


# ── Ground truth for recall computation ──────────────────────────


def _compute_ground_truth_from_pool(
    model_name: str, dataset: str, doc_indices: list[int], query_indices: list[int], k: int
) -> list[set[int]] | None:
    """Try to compute ground truth from pre-cached VSS vector pools.

    Returns None if pools are not available.
    """
    doc_pool_path = VECTORS_DIR / f"{model_name}_{dataset}_docs.npy"
    query_pool_path = VECTORS_DIR / f"{model_name}_{dataset}_queries.npy"

    if not doc_pool_path.exists() or not query_pool_path.exists():
        return None

    doc_pool = np.load(doc_pool_path)
    query_pool = np.load(query_pool_path)

    # Subset to the same indices used in this run
    doc_vectors = doc_pool[doc_indices] if doc_indices is not None else doc_pool
    query_vectors = query_pool[query_indices] if query_indices is not None else query_pool

    # Vectorized L2 distance: [M, N]
    dists = np.sum((doc_vectors[None, :, :] - query_vectors[:, None, :]) ** 2, axis=2)
    top_k_indices = np.argsort(dists, axis=1)[:, :k]
    # Convert to 1-based rowids
    return [{int(idx) + 1 for idx in row} for row in top_k_indices]


def _compute_recall(search_results: list[set[int]], ground_truth: list[set[int]]) -> float:
    """Average recall of search_results vs ground_truth."""
    recalls: list[float] = []
    for sr, gt in zip(search_results, ground_truth, strict=False):
        if len(gt) > 0:
            recalls.append(len(sr & gt) / len(gt))
    return sum(recalls) / len(recalls) if recalls else 0.0


# ── EmbedTreatment ───────────────────────────────────────────────


class EmbedTreatment(Treatment):
    """Single embed benchmark permutation.

    Measures the full text->embedding->search pipeline for one configuration
    of (embed_fn, search_backend, model, dataset, N).
    """

    def __init__(
        self,
        embed_fn_slug: str,
        search_backend_slug: str,
        model_name: str,
        dim: int,
        dataset: str,
        n: int,
    ) -> None:
        self._embed_fn_slug = embed_fn_slug
        self._search_backend_slug = search_backend_slug
        self._model_name = model_name
        self._dim = dim
        self._dataset = dataset
        self._n = n
        # Runtime state
        self._actual_n = n
        self._n_queries = 0
        self._seed: int | None = None
        self._doc_texts: list[str] | None = None
        self._query_texts: list[str] | None = None
        self._doc_indices: list[int] | None = None
        self._query_indices: list[int] | None = None

    @property
    def requires_muninn(self) -> bool:
        return self._embed_fn_slug == "muninn_embed" or self._search_backend_slug == "muninn-hnsw"

    @property
    def category(self) -> str:
        return "embed"

    @property
    def permutation_id(self) -> str:
        ds = self._dataset.replace("_", "-")
        return f"embed_{self._embed_fn_slug}+{self._search_backend_slug}_{self._model_name}_{ds}_n{self._n}"

    @property
    def label(self) -> str:
        return (
            f"Embed: {self._embed_fn_slug}+{self._search_backend_slug} / "
            f"{self._model_name} / {self._dataset} / N={self._n}"
        )

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (self._n, self._dim, self._model_name, self._embed_fn_slug, self._search_backend_slug)

    def params_dict(self) -> dict[str, Any]:
        return {
            "embed_fn": self._embed_fn_slug,
            "search_backend": self._search_backend_slug,
            "model": self._model_name,
            "dim": self._dim,
            "dataset": self._dataset,
            "n": self._actual_n,
            "k": K,
            "n_queries": self._n_queries,
            "seed": self._seed,
        }

    def setup(self, conn: sqlite3.Connection, db_path: Path) -> dict[str, Any]:
        """Load text datasets and sample N docs + M queries."""
        self._seed = random.randint(0, 2**32 - 1)
        rng = np.random.default_rng(seed=self._seed)

        # Load all doc texts
        all_doc_texts = _load_doc_texts(self._dataset, max_n=max(self._n * 2, 10_000))

        # Sample N docs
        self._actual_n = min(self._n, len(all_doc_texts))
        if self._actual_n < self._n:
            log.warning("  Doc texts has %d items, need %d — using available", len(all_doc_texts), self._n)

        if self._actual_n < len(all_doc_texts):
            indices = rng.choice(len(all_doc_texts), size=self._actual_n, replace=False)
            self._doc_indices = sorted(indices.tolist())
            self._doc_texts = [all_doc_texts[i] for i in self._doc_indices]
        else:
            self._doc_indices = list(range(self._actual_n))
            self._doc_texts = all_doc_texts[: self._actual_n]

        # Load and sample query texts
        all_query_texts = _load_query_texts(self._dataset, doc_texts=all_doc_texts)
        self._n_queries = min(N_QUERIES, len(all_query_texts))
        if self._n_queries < N_QUERIES:
            log.warning("  Query texts has %d items, need %d — using available", len(all_query_texts), N_QUERIES)

        query_indices = rng.choice(len(all_query_texts), size=self._n_queries, replace=False)
        self._query_indices = sorted(query_indices.tolist())
        self._query_texts = [all_query_texts[i] for i in self._query_indices]

        return {
            "n_doc_texts": self._actual_n,
            "n_query_texts": self._n_queries,
            "seed": self._seed,
            "total_doc_pool": len(all_doc_texts),
            "total_query_pool": len(all_query_texts),
        }

    def run(self, conn: sqlite3.Connection) -> dict[str, Any]:
        """Execute the full embed benchmark pipeline."""
        model_info = EMBEDDING_MODELS[self._model_name]
        doc_prefix = model_info["doc_prefix"]
        query_prefix = model_info["query_prefix"]
        gguf_path = str(GGUF_MODELS_DIR / model_info["gguf_filename"])

        # Load the appropriate embedding extension
        if self._embed_fn_slug == "muninn_embed":
            metrics = self._run_with_muninn_embed(conn, gguf_path, doc_prefix, query_prefix)
        elif self._embed_fn_slug == "lembed":
            metrics = self._run_with_lembed(conn, gguf_path, doc_prefix, query_prefix)
        else:
            raise ValueError(f"Unknown embed_fn: {self._embed_fn_slug}")

        return metrics

    def teardown(self, conn: sqlite3.Connection) -> None:
        self._doc_texts = None
        self._query_texts = None
        self._doc_indices = None
        self._query_indices = None

    # ── Embedding function dispatchers ────────────────────────────

    def _run_with_muninn_embed(
        self, conn: sqlite3.Connection, gguf_path: str, doc_prefix: str, query_prefix: str
    ) -> dict[str, Any]:
        """Run the full pipeline using muninn_embed as the embedding function."""
        # Task 1: Model load time
        with self.step("Model load (muninn_embed)"):
            t0 = time.perf_counter()
            conn.execute(
                "INSERT INTO temp.muninn_models(name, model) SELECT ?, muninn_embed_model(?)",
                (self._model_name, gguf_path),
            )
            model_load_time_ms = (time.perf_counter() - t0) * 1000

        embed_sql_doc = f"muninn_embed('{self._model_name}', '{doc_prefix}' || ?)"
        embed_sql_query = f"muninn_embed('{self._model_name}', '{query_prefix}' || ?)"

        args = (conn, embed_sql_doc, embed_sql_query, model_load_time_ms, doc_prefix, query_prefix)
        return self._run_search_backend(*args)

    def _run_with_lembed(
        self, conn: sqlite3.Connection, gguf_path: str, doc_prefix: str, query_prefix: str
    ) -> dict[str, Any]:
        """Run the full pipeline using lembed as the embedding function."""
        _load_lembed(conn)

        # Task 1: Model load time
        with self.step("Model load (lembed)"):
            t0 = time.perf_counter()
            conn.execute(
                "INSERT INTO temp.lembed_models(name, model) SELECT ?, lembed_model_from_file(?)",
                (self._model_name, gguf_path),
            )
            model_load_time_ms = (time.perf_counter() - t0) * 1000

        embed_sql_doc = f"lembed('{self._model_name}', '{doc_prefix}' || ?)"
        embed_sql_query = f"lembed('{self._model_name}', '{query_prefix}' || ?)"

        args = (conn, embed_sql_doc, embed_sql_query, model_load_time_ms, doc_prefix, query_prefix)
        return self._run_search_backend(*args)

    # ── Search backend dispatcher ─────────────────────────────────

    def _run_search_backend(
        self,
        conn: sqlite3.Connection,
        embed_sql_doc: str,
        embed_sql_query: str,
        model_load_time_ms: float,
        doc_prefix: str,
        query_prefix: str,
    ) -> dict[str, Any]:
        """Dispatch to the correct search backend implementation."""
        assert self._doc_texts is not None
        assert self._query_texts is not None

        backend = self._search_backend_slug
        args = (conn, embed_sql_doc, embed_sql_query, model_load_time_ms, doc_prefix, query_prefix)

        if backend == "muninn-hnsw":
            return self._backend_muninn_hnsw(*args)
        elif backend == "sqlite-vector-pq":
            return self._backend_sqlite_vector_pq(*args)
        elif backend == "sqlite-vec-brute":
            return self._backend_sqlite_vec_brute(*args)
        else:
            raise ValueError(f"Unknown search_backend: {backend}")

    # ── Backend: muninn HNSW ──────────────────────────────────────

    def _backend_muninn_hnsw(
        self,
        conn: sqlite3.Connection,
        embed_sql_doc: str,
        embed_sql_query: str,
        model_load_time_ms: float,
        doc_prefix: str,
        query_prefix: str,
    ) -> dict[str, Any]:
        assert self._doc_texts is not None
        assert self._query_texts is not None
        dim = self._dim

        # Create documents table
        conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT NOT NULL)")

        # Create HNSW index
        conn.execute(
            f"CREATE VIRTUAL TABLE bench_vec USING hnsw_index("
            f"dimensions={dim}, metric='l2', m={HNSW_M}, "
            f"ef_construction={HNSW_EF_CONSTRUCTION})"
        )

        # Task 2: Bulk embed + insert
        n = self._actual_n
        with self.step(f"Bulk embed + insert ({n} items)"):
            tracker = self.progress(n)
            t0 = time.perf_counter()
            for pos, text in enumerate(self._doc_texts):
                conn.execute("INSERT INTO documents(id, content) VALUES (?, ?)", (pos + 1, text))
                embed_blob = conn.execute(f"SELECT {embed_sql_doc}", (text,)).fetchone()[0]
                conn.execute("INSERT INTO bench_vec(rowid, vector) VALUES (?, ?)", (pos + 1, embed_blob))
                tracker.update(pos + 1)
            bulk_time = time.perf_counter() - t0

        # Task 3: Trigger-based incremental embedding
        with self.step(f"Trigger embed ({TRIGGER_SAMPLE_SIZE} inserts)"):
            trigger_prefix_expr = f"'{doc_prefix}' || " if doc_prefix else ""
            embed_fn_name = embed_sql_doc.split("(")[0]  # muninn_embed or lembed
            conn.execute(
                f"""
                CREATE TEMP TRIGGER auto_embed AFTER INSERT ON documents
                BEGIN
                  INSERT INTO bench_vec(rowid, vector)
                    VALUES (NEW.id, {embed_fn_name}('{self._model_name}', {trigger_prefix_expr}NEW.content));
                END
                """
            )

            trigger_latencies: list[float] = []
            base_id = self._actual_n + 1
            sample_texts = self._doc_texts[:TRIGGER_SAMPLE_SIZE]
            for i, text in enumerate(sample_texts):
                row_id = base_id + i
                t0 = time.perf_counter()
                conn.execute("INSERT INTO documents(id, content) VALUES (?, ?)", (row_id, text))
                trigger_latencies.append((time.perf_counter() - t0) * 1000)

            conn.execute("DROP TRIGGER auto_embed")
            trigger_embed_latency_ms = sum(trigger_latencies) / len(trigger_latencies) if trigger_latencies else 0

        # Task 4: Live query embed + search
        n_queries = len(self._query_texts)
        with self.step(f"Query embed + search ({n_queries} queries)"):
            query_latencies: list[float] = []
            embed_latencies: list[float] = []
            search_results: list[set[int]] = []

            for query_text in self._query_texts:
                # Time embedding only
                t0 = time.perf_counter()
                query_blob = conn.execute(f"SELECT {embed_sql_query}", (query_text,)).fetchone()[0]
                embed_ms = (time.perf_counter() - t0) * 1000
                embed_latencies.append(embed_ms)

                # Time search only
                t0 = time.perf_counter()
                rows = conn.execute(
                    "SELECT rowid, distance FROM bench_vec WHERE vector MATCH ? AND k = ? AND ef_search = ?",
                    (query_blob, K, HNSW_EF_SEARCH),
                ).fetchall()
                search_ms = (time.perf_counter() - t0) * 1000

                query_latencies.append(embed_ms + search_ms)
                search_results.append({r[0] for r in rows})

            conn.commit()

        # Recall computation via pre-cached pools
        recall = self._compute_recall_if_possible(search_results)

        return self._build_metrics(
            model_load_time_ms=model_load_time_ms,
            bulk_time=bulk_time,
            trigger_embed_latency_ms=trigger_embed_latency_ms,
            query_latencies=query_latencies,
            embed_latencies=embed_latencies,
            recall=recall,
        )

    # ── Backend: sqlite-vector PQ ─────────────────────────────────

    def _backend_sqlite_vector_pq(
        self,
        conn: sqlite3.Connection,
        embed_sql_doc: str,
        embed_sql_query: str,
        model_load_time_ms: float,
        doc_prefix: str,
        query_prefix: str,
    ) -> dict[str, Any]:
        assert self._doc_texts is not None
        assert self._query_texts is not None
        dim = self._dim

        # Load sqlite-vector extension
        ext_path = _sqlite_vector_ext_path()
        conn.enable_load_extension(True)
        conn.load_extension(ext_path)

        # Create documents and vector tables
        conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT NOT NULL)")
        conn.execute("CREATE TABLE bench(id INTEGER PRIMARY KEY, embedding BLOB)")

        # Task 2: Bulk embed + insert
        n = self._actual_n
        with self.step(f"Bulk embed + insert ({n} items)"):
            tracker = self.progress(n)
            t0 = time.perf_counter()
            for pos, text in enumerate(self._doc_texts):
                conn.execute("INSERT INTO documents(id, content) VALUES (?, ?)", (pos + 1, text))
                embed_blob = conn.execute(f"SELECT {embed_sql_doc}", (text,)).fetchone()[0]
                conn.execute("INSERT INTO bench(id, embedding) VALUES (?, ?)", (pos + 1, embed_blob))
                tracker.update(pos + 1)
            bulk_time = time.perf_counter() - t0

        # Initialize and quantize
        conn.execute(f"SELECT vector_init('bench', 'embedding', 'dimension={dim},type=FLOAT32,distance=L2')")
        conn.execute("SELECT vector_quantize('bench', 'embedding')")

        # Task 3: Trigger-based incremental embedding
        with self.step(f"Trigger embed ({TRIGGER_SAMPLE_SIZE} inserts)"):
            trigger_prefix_expr = f"'{doc_prefix}' || " if doc_prefix else ""
            embed_fn_name = embed_sql_doc.split("(")[0]
            conn.execute(
                f"""
                CREATE TEMP TRIGGER auto_embed AFTER INSERT ON documents
                BEGIN
                  INSERT INTO bench(id, embedding)
                    VALUES (NEW.id, {embed_fn_name}('{self._model_name}', {trigger_prefix_expr}NEW.content));
                END
                """
            )

            trigger_latencies: list[float] = []
            base_id = self._actual_n + 1
            sample_texts = self._doc_texts[:TRIGGER_SAMPLE_SIZE]
            for i, text in enumerate(sample_texts):
                row_id = base_id + i
                t0 = time.perf_counter()
                conn.execute("INSERT INTO documents(id, content) VALUES (?, ?)", (row_id, text))
                trigger_latencies.append((time.perf_counter() - t0) * 1000)

            conn.execute("DROP TRIGGER auto_embed")
            # Re-quantize after trigger inserts
            conn.execute("SELECT vector_quantize('bench', 'embedding')")
            trigger_embed_latency_ms = sum(trigger_latencies) / len(trigger_latencies) if trigger_latencies else 0

        # Task 4: Live query embed + search
        n_queries = len(self._query_texts)
        with self.step(f"Query embed + search ({n_queries} queries)"):
            query_latencies: list[float] = []
            embed_latencies: list[float] = []
            search_results: list[set[int]] = []

            for query_text in self._query_texts:
                t0 = time.perf_counter()
                query_blob = conn.execute(f"SELECT {embed_sql_query}", (query_text,)).fetchone()[0]
                embed_ms = (time.perf_counter() - t0) * 1000
                embed_latencies.append(embed_ms)

                t0 = time.perf_counter()
                rows = conn.execute(
                    "SELECT rowid, distance FROM vector_quantize_scan('bench', 'embedding', ?, ?)",
                    (query_blob, K),
                ).fetchall()
                search_ms = (time.perf_counter() - t0) * 1000

                query_latencies.append(embed_ms + search_ms)
                search_results.append({r[0] for r in rows})

            conn.commit()

        recall = self._compute_recall_if_possible(search_results)

        return self._build_metrics(
            model_load_time_ms=model_load_time_ms,
            bulk_time=bulk_time,
            trigger_embed_latency_ms=trigger_embed_latency_ms,
            query_latencies=query_latencies,
            embed_latencies=embed_latencies,
            recall=recall,
        )

    # ── Backend: sqlite-vec brute ─────────────────────────────────

    def _backend_sqlite_vec_brute(
        self,
        conn: sqlite3.Connection,
        embed_sql_doc: str,
        embed_sql_query: str,
        model_load_time_ms: float,
        doc_prefix: str,
        query_prefix: str,
    ) -> dict[str, Any]:
        import sqlite_vec

        assert self._doc_texts is not None
        assert self._query_texts is not None
        dim = self._dim

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)

        # Create documents and vec0 tables
        conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT NOT NULL)")
        conn.execute(f"CREATE VIRTUAL TABLE bench_sv USING vec0(embedding float[{dim}])")

        # Task 2: Bulk embed + insert
        n = self._actual_n
        with self.step(f"Bulk embed + insert ({n} items)"):
            tracker = self.progress(n)
            t0 = time.perf_counter()
            for pos, text in enumerate(self._doc_texts):
                conn.execute("INSERT INTO documents(id, content) VALUES (?, ?)", (pos + 1, text))
                embed_blob = conn.execute(f"SELECT {embed_sql_doc}", (text,)).fetchone()[0]
                conn.execute("INSERT INTO bench_sv(rowid, embedding) VALUES (?, ?)", (pos + 1, embed_blob))
                tracker.update(pos + 1)
            bulk_time = time.perf_counter() - t0

        # Task 3: Trigger-based incremental embedding
        with self.step(f"Trigger embed ({TRIGGER_SAMPLE_SIZE} inserts)"):
            trigger_prefix_expr = f"'{doc_prefix}' || " if doc_prefix else ""
            embed_fn_name = embed_sql_doc.split("(")[0]
            conn.execute(
                f"""
                CREATE TEMP TRIGGER auto_embed AFTER INSERT ON documents
                BEGIN
                  INSERT INTO bench_sv(rowid, embedding)
                    VALUES (NEW.id, {embed_fn_name}('{self._model_name}', {trigger_prefix_expr}NEW.content));
                END
                """
            )

            trigger_latencies: list[float] = []
            base_id = self._actual_n + 1
            sample_texts = self._doc_texts[:TRIGGER_SAMPLE_SIZE]
            for i, text in enumerate(sample_texts):
                row_id = base_id + i
                t0 = time.perf_counter()
                conn.execute("INSERT INTO documents(id, content) VALUES (?, ?)", (row_id, text))
                trigger_latencies.append((time.perf_counter() - t0) * 1000)

            conn.execute("DROP TRIGGER auto_embed")
            trigger_embed_latency_ms = sum(trigger_latencies) / len(trigger_latencies) if trigger_latencies else 0

        # Task 4: Live query embed + search
        n_queries = len(self._query_texts)
        with self.step(f"Query embed + search ({n_queries} queries)"):
            query_latencies: list[float] = []
            embed_latencies: list[float] = []
            search_results: list[set[int]] = []

            for query_text in self._query_texts:
                t0 = time.perf_counter()
                query_blob = conn.execute(f"SELECT {embed_sql_query}", (query_text,)).fetchone()[0]
                embed_ms = (time.perf_counter() - t0) * 1000
                embed_latencies.append(embed_ms)

                t0 = time.perf_counter()
                rows = conn.execute(
                    "SELECT rowid, distance FROM bench_sv WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                    (query_blob, K),
                ).fetchall()
                search_ms = (time.perf_counter() - t0) * 1000

                query_latencies.append(embed_ms + search_ms)
                search_results.append({r[0] for r in rows})

            conn.commit()

        recall = self._compute_recall_if_possible(search_results)

        return self._build_metrics(
            model_load_time_ms=model_load_time_ms,
            bulk_time=bulk_time,
            trigger_embed_latency_ms=trigger_embed_latency_ms,
            query_latencies=query_latencies,
            embed_latencies=embed_latencies,
            recall=recall,
        )

    # ── Shared metric builders ────────────────────────────────────

    def _compute_recall_if_possible(self, search_results: list[set[int]]) -> float | None:
        """Try to compute recall using pre-cached vector pools from VSS prep."""
        if self._doc_indices is None or self._query_indices is None:
            return None
        ground_truth = _compute_ground_truth_from_pool(
            self._model_name, self._dataset, self._doc_indices, self._query_indices, K
        )
        if ground_truth is None:
            return None
        return _compute_recall(search_results, ground_truth)

    def _build_metrics(
        self,
        model_load_time_ms: float,
        bulk_time: float,
        trigger_embed_latency_ms: float,
        query_latencies: list[float],
        embed_latencies: list[float],
        recall: float | None,
    ) -> dict[str, Any]:
        """Build the standard metrics dict from timing measurements."""
        n = self._actual_n
        n_queries = len(query_latencies)

        avg_query_latency = sum(query_latencies) / n_queries if n_queries else 0
        avg_embed_latency = sum(embed_latencies) / n_queries if n_queries else 0
        avg_search_latency = avg_query_latency - avg_embed_latency

        metrics: dict[str, Any] = {
            "model_load_time_ms": round(model_load_time_ms, 3),
            "bulk_embed_insert_rate_vps": round(n / bulk_time, 1) if bulk_time > 0 else 0,
            "bulk_embed_insert_total_s": round(bulk_time, 3),
            "trigger_embed_latency_ms": round(trigger_embed_latency_ms, 3),
            "query_embed_search_latency_ms": round(avg_query_latency, 3),
            "query_embed_only_ms": round(avg_embed_latency, 3),
            "query_search_only_ms": round(avg_search_latency, 3),
        }

        if recall is not None:
            metrics["recall_at_k"] = round(recall, 4)

        return metrics
