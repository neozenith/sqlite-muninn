"""Prep module: generate .npy embedding caches using GGUF models via llama-cpp-python.

Produces separate document and query embedding pools per (model, dataset):
    {model}_{dataset}_docs.npy      — document embeddings (float32, L2-normalized)
    {model}_{dataset}_queries.npy   — query embeddings (float32, L2-normalized)
    {dataset}_docs.json             — raw doc texts + metadata (model-independent)
    {dataset}_queries.json          — query texts + metadata (model-independent)

The prep step creates maximum-size pools. Each benchmark trial randomly samples
from these pools, so embedding (expensive) happens once and benchmarks can run
many trials cheaply with different random subsets.
"""

import ctypes
import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset as hf_load_dataset
from llama_cpp import Llama, llama_log_callback, llama_log_set

from benchmarks.harness.common import DATASETS, EMBEDDING_MODELS, GGUF_MODELS_DIR, TEXTS_DIR, VECTORS_DIR, VSS_SIZES
from benchmarks.harness.prep.base import PrepTask
from benchmarks.harness.prep.common import fmt_size
from benchmarks.harness.s3_mirror import get_s3_mirror

log = logging.getLogger(__name__)


# ── PrepTask ─────────────────────────────────────────────────────


class VectorPrepTask(PrepTask):
    """PrepTask for generating doc + query .npy embedding caches."""

    def __init__(self, model_label: str, model_info: dict[str, Any], dataset_key: str):
        self._model_label = model_label
        self._model_info = model_info
        self._dataset_key = dataset_key

    @property
    def task_id(self) -> str:
        return f"vector:{self._model_label}:{self._dataset_key}"

    @property
    def label(self) -> str:
        return f"{self._model_label} / {self._dataset_key}"

    def outputs(self) -> list[Path]:
        return [
            VECTORS_DIR / f"{self._model_label}_{self._dataset_key}_docs.npy",
            VECTORS_DIR / f"{self._model_label}_{self._dataset_key}_queries.npy",
        ]

    def fetch(self, force: bool = False) -> None:
        # fetch + transform combined — handled by prep_vectors batch runner
        pass


VECTOR_PREP_TASKS: list[PrepTask] = [
    VectorPrepTask(model_label, model_info, dataset_key)
    for model_label, model_info in EMBEDDING_MODELS.items()
    for dataset_key in DATASETS
]


# ── Utility functions ────────────────────────────────────────────


def _download_gutenberg(gutenberg_id):
    """Fetch plain text from Project Gutenberg, strip boilerplate, cache locally."""
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TEXTS_DIR / f"gutenberg_{gutenberg_id}.txt"

    if cache_path.exists():
        log.info("  Gutenberg #%d: cached at %s", gutenberg_id, cache_path)
        return cache_path

    url = f"https://www.gutenberg.org/cache/epub/{gutenberg_id}/pg{gutenberg_id}.txt"
    log.info("  Downloading Gutenberg #%d from %s...", gutenberg_id, url)

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
    log.info("  Gutenberg #%d: saved %d chars to %s", gutenberg_id, len(clean_text), cache_path)
    return cache_path


def _chunk_fixed_tokens(text, window=256, overlap=50):
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


def _load_doc_texts(dataset_key, max_n):
    """Load document texts from a dataset. Returns list of strings."""
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


def _load_query_texts(dataset_key, doc_texts=None):
    """Load or generate query texts for a dataset.

    For ag_news: loads the FULL HuggingFace test split (disjoint from train docs).
    For wealth_of_nations: extracts first sentence of each document chunk.

    Returns list of query text strings.
    """
    ds_config = DATASETS[dataset_key]

    if ds_config["source_type"] == "huggingface":
        # Load the test split — disjoint from the training docs
        hf_dataset = hf_load_dataset(ds_config["hf_name"], split="test")
        field = ds_config["text_field"]
        return [row[field] for row in hf_dataset]

    if ds_config["source_type"] == "gutenberg":
        # Extract first sentence of each chunk as query
        if doc_texts is None:
            raise ValueError("doc_texts required for gutenberg query generation")
        queries = []
        for chunk in doc_texts:
            # Extract first sentence (split on . ! ?)
            match = re.match(r"(.+?[.!?])\s", chunk)
            if match:
                queries.append(match.group(1).strip())
            else:
                # No sentence boundary found — use first 50 words
                words = chunk.split()[:50]
                queries.append(" ".join(words))
        return queries

    raise ValueError(f"Unknown dataset source_type: {ds_config['source_type']}")


def _get_doc_texts_path(dataset_key):
    """Return the path to the doc texts JSON file for a dataset."""
    return VECTORS_DIR / f"{dataset_key}_docs.json"


def _save_doc_texts(dataset_key, doc_texts, source_desc):
    """Save doc texts to a JSON file for reuse by downstream treatments."""
    data = {
        "dataset": dataset_key,
        "source": source_desc,
        "n_docs": len(doc_texts),
        "docs": doc_texts,
    }
    path = _get_doc_texts_path(dataset_key)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    log.info("  %s: saved %d doc texts to %s", dataset_key, len(doc_texts), path)
    return path


def _load_cached_doc_texts(dataset_key):
    """Load cached doc texts from JSON. Returns list of strings or None if not cached."""
    path = _get_doc_texts_path(dataset_key)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["docs"]


def _get_query_texts_path(dataset_key):
    """Return the path to the query texts JSON file for a dataset."""
    return VECTORS_DIR / f"{dataset_key}_queries.json"


def _save_query_texts(dataset_key, queries, source_desc):
    """Save query texts to a JSON file."""
    data = {
        "dataset": dataset_key,
        "source": source_desc,
        "n_queries": len(queries),
        "queries": queries,
    }
    path = _get_query_texts_path(dataset_key)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    log.info("  %s: saved %d query texts to %s", dataset_key, len(queries), path)
    return path


def _load_cached_query_texts(dataset_key):
    """Load cached query texts from JSON. Returns list of strings or None if not cached."""
    path = _get_query_texts_path(dataset_key)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["queries"]


def _suppress_llama_log():
    """Install a no-op C callback to suppress llama.cpp stderr spam.

    llama-cpp-python's verbose=False only controls its Python-level logger,
    but ggml_metal_init "skipping kernel" messages and the per-decode
    "embeddings required but some input tokens were not marked as outputs"
    warning bypass that and go directly to stderr via the C log callback.

    Returns a reference to the callback that MUST be kept alive (prevent GC).
    """

    @llama_log_callback
    def _noop(_level, _text, _user_data):
        pass

    llama_log_set(_noop, ctypes.c_void_p(0))
    return _noop  # prevent garbage collection of ctypes callback


def _embed_texts(model_path, texts, prefix="", log_every=512):
    """Embed texts using a GGUF model via llama-cpp-python.

    Embeds one text at a time — embedding models have no cross-text attention,
    and batching multiple texts into a single llm.embed() call overflows the
    model's context window (llama_decode returns -1).

    Returns L2-normalized float32 numpy array of shape [N, dim].
    """
    _log_cb_ref = _suppress_llama_log()

    llm = Llama(model_path=str(model_path), embedding=True, n_gpu_layers=-1, verbose=False)

    prefixed = [prefix + t for t in texts] if prefix else list(texts)

    all_embeddings = []
    for i, text in enumerate(prefixed):
        result = llm.embed(text)
        all_embeddings.append(result)
        if len(prefixed) > log_every and (i + 1) % log_every == 0:
            log.info("    Embedded %d / %d texts", i + 1, len(prefixed))

    del llm
    del _log_cb_ref

    arr = np.array(all_embeddings, dtype=np.float32)
    # L2-normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norms, 1e-12)
    return arr


# ── Status display ───────────────────────────────────────────────


def _print_vector_status(datasets_to_prep, models_to_prep):
    """Print status table of .npy embedding cache files."""
    print("=== Vector Cache Status ===\n")
    print(
        f"  {'MODEL':<12s}   {'DATASET':<20s}   {'DIM':>5s}   {'KIND':<8s}   "
        f"{'VECTORS':>8s}   {'SIZE':>10s}   {'STATUS'}"
    )
    print(f"  {'-' * 12}   {'-' * 20}   {'-' * 5}   {'-' * 8}   {'-' * 8}   {'-' * 10}   {'-' * 8}")

    for dataset_key in datasets_to_prep:
        for model_label, model_info in models_to_prep.items():
            dim = model_info["dim"]
            for kind in ("docs", "queries"):
                cache_path = VECTORS_DIR / f"{model_label}_{dataset_key}_{kind}.npy"

                if cache_path.exists():
                    arr = np.load(cache_path)
                    size = cache_path.stat().st_size
                    print(
                        f"  {model_label:<12s}   {dataset_key:<20s}   {dim:>5d}   {kind:<8s}   "
                        f"{len(arr):>8,d}   {fmt_size(size):>10s}   CACHED"
                    )
                else:
                    print(
                        f"  {model_label:<12s}   {dataset_key:<20s}   {dim:>5d}   {kind:<8s}   "
                        f"{'':>8s}   {'':>10s}   MISSING"
                    )

    # Also show query text files
    print()
    print("  Query text files:")
    for dataset_key in datasets_to_prep:
        qt_path = _get_query_texts_path(dataset_key)
        if qt_path.exists():
            data = json.loads(qt_path.read_text(encoding="utf-8"))
            print(f"    {dataset_key}: {data['n_queries']} queries — {data['source']}")
        else:
            print(f"    {dataset_key}: MISSING")

    print(f"\n  Directory: {VECTORS_DIR}")
    print()


# ── Main entry point ─────────────────────────────────────────────


def prep_vectors(only_model=None, only_dataset=None, status_only=False, force=False):
    """Pre-download models, datasets, and generate all .npy cache files.

    Each (model, dataset) pair gets two .npy files: _docs.npy and _queries.npy.
    Query texts are generated once per dataset (model-independent) as JSON.

    Args:
        only_model: Only prep this embedding model (e.g., "MiniLM").
        only_dataset: Only prep this dataset (e.g., "ag_news").
        status_only: If True, show cache status and return.
        force: If True, re-create caches even if they exist.
    """
    datasets_to_prep = list(DATASETS.keys())
    models_to_prep = dict(EMBEDDING_MODELS)

    if only_dataset:
        datasets_to_prep = [only_dataset]
    if only_model:
        models_to_prep = {only_model: EMBEDDING_MODELS[only_model]}

    if status_only:
        _print_vector_status(datasets_to_prep, models_to_prep)
        return

    max_n = max(VSS_SIZES)

    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Generate query texts for each dataset (model-independent, done once)
    dataset_doc_texts: dict[str, list[str]] = {}
    dataset_query_texts: dict[str, list[str]] = {}

    for dataset_key in datasets_to_prep:
        qt_path = _get_query_texts_path(dataset_key)
        dt_path = _get_doc_texts_path(dataset_key)

        # Load doc texts — use cache if available, otherwise fetch and save
        cached_docs = _load_cached_doc_texts(dataset_key) if (dt_path.exists() and not force) else None
        if cached_docs is not None and len(cached_docs) >= max_n:
            doc_texts = cached_docs[:max_n]
            log.info("  %s: doc texts cached (%d docs)", dataset_key, len(doc_texts))
        else:
            doc_texts = _load_doc_texts(dataset_key, max_n=max_n)
            ds_config = DATASETS[dataset_key]
            if ds_config["source_type"] == "huggingface":
                source_desc = f"{ds_config['hf_name']} HuggingFace {ds_config['hf_split']} split"
            else:
                source_desc = f"Gutenberg #{ds_config['gutenberg_id']} chunked"
            _save_doc_texts(dataset_key, doc_texts, source_desc)
        dataset_doc_texts[dataset_key] = doc_texts

        # Generate or load query texts
        if qt_path.exists() and not force:
            query_texts = _load_cached_query_texts(dataset_key)
            log.info("  %s: query texts cached (%d queries)", dataset_key, len(query_texts))
        else:
            ds_config = DATASETS[dataset_key]
            if ds_config["source_type"] == "huggingface":
                source_desc = f"{ds_config['hf_name']} HuggingFace test split"
            else:
                source_desc = f"Gutenberg #{ds_config['gutenberg_id']} first sentence per chunk"
            query_texts = _load_query_texts(dataset_key, doc_texts=doc_texts)
            _save_query_texts(dataset_key, query_texts, source_desc)

        dataset_query_texts[dataset_key] = query_texts

    # Step 2: Model-outer loop — load each GGUF model once, embed all datasets
    for model_label, model_info in models_to_prep.items():
        gguf_filename = model_info["gguf_filename"]
        model_path = GGUF_MODELS_DIR / gguf_filename
        doc_prefix = model_info["doc_prefix"]
        query_prefix = model_info["query_prefix"]
        dim = model_info["dim"]

        mirror = get_s3_mirror()
        if not mirror.ensure_local(model_path):
            log.error(
                "  %s: GGUF model not found locally or in S3 at %s. Run 'prep gguf' first.",
                model_label,
                model_path,
            )
            continue

        # Check which datasets need encoding
        datasets_needing_docs = []
        datasets_needing_queries = []

        for dataset_key in datasets_to_prep:
            docs_path = VECTORS_DIR / f"{model_label}_{dataset_key}_docs.npy"
            queries_path = VECTORS_DIR / f"{model_label}_{dataset_key}_queries.npy"

            if force or not mirror.ensure_local(docs_path):
                datasets_needing_docs.append(dataset_key)
            else:
                log.info("  %s/%s docs: cached (%s)", model_label, dataset_key, fmt_size(docs_path.stat().st_size))

            if force or not mirror.ensure_local(queries_path):
                datasets_needing_queries.append(dataset_key)
            else:
                log.info(
                    "  %s/%s queries: cached (%s)",
                    model_label,
                    dataset_key,
                    fmt_size(queries_path.stat().st_size),
                )

        if not datasets_needing_docs and not datasets_needing_queries:
            continue  # This model is fully cached, skip loading it

        log.info("Loading GGUF model: %s (%s, dim=%d)...", model_label, gguf_filename, dim)

        # Embed documents
        for dataset_key in datasets_needing_docs:
            docs_path = VECTORS_DIR / f"{model_label}_{dataset_key}_docs.npy"
            if docs_path.exists() and force:
                log.info("  %s/%s docs: --force, re-encoding", model_label, dataset_key)
                docs_path.unlink()

            doc_texts = dataset_doc_texts[dataset_key]
            n = len(doc_texts)
            log.info(
                "  %s/%s: encoding %d doc texts (dim=%d, prefix=%r)...",
                model_label,
                dataset_key,
                n,
                dim,
                doc_prefix[:30] if doc_prefix else "(none)",
            )
            doc_embeddings = _embed_texts(model_path, doc_texts, prefix=doc_prefix)
            np.save(docs_path, doc_embeddings)
            mirror.sync_to_s3(docs_path)
            log.info("  %s/%s docs: cached %d embeddings to %s", model_label, dataset_key, n, docs_path)

        # Embed queries
        for dataset_key in datasets_needing_queries:
            queries_path = VECTORS_DIR / f"{model_label}_{dataset_key}_queries.npy"
            if queries_path.exists() and force:
                log.info("  %s/%s queries: --force, re-encoding", model_label, dataset_key)
                queries_path.unlink()

            query_texts = dataset_query_texts[dataset_key]
            n = len(query_texts)
            log.info(
                "  %s/%s: encoding %d query texts (dim=%d, prefix=%r)...",
                model_label,
                dataset_key,
                n,
                dim,
                query_prefix[:30] if query_prefix else "(none)",
            )
            query_embeddings = _embed_texts(model_path, query_texts, prefix=query_prefix)
            np.save(queries_path, query_embeddings)
            mirror.sync_to_s3(queries_path)
            log.info("  %s/%s queries: cached %d embeddings to %s", model_label, dataset_key, n, queries_path)

        log.info("Done with model: %s", model_label)

    log.info("Vector prep complete. Cached in %s", VECTORS_DIR)
