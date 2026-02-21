"""Prep module: download models and build .npy embedding caches.

Ports from benchmarks/scripts/benchmark_vss.py --prep-models.
"""

import logging
import re
import urllib.request

import numpy as np

from benchmarks.harness.common import DATASETS, EMBEDDING_MODELS, TEXTS_DIR, VECTORS_DIR, VSS_SIZES
from benchmarks.harness.prep.common import fmt_size

log = logging.getLogger(__name__)

try:
    from datasets import load_dataset as hf_load_dataset
    from sentence_transformers import SentenceTransformer

    HAS_MODEL_DEPS = True
except ImportError:
    HAS_MODEL_DEPS = False


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


def _load_texts(dataset_key, max_n):
    """Load texts from a dataset. Returns list of strings."""
    ds_config = DATASETS[dataset_key]

    if ds_config["source_type"] == "huggingface":
        if not HAS_MODEL_DEPS:
            raise RuntimeError("HuggingFace datasets require: uv pip install datasets sentence-transformers")
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


def _print_vector_status(datasets_to_prep, models_to_prep):
    """Print status table of .npy embedding cache files."""
    print("=== Vector Cache Status ===\n")
    print(f"  {'MODEL':<12s}   {'DATASET':<20s}   {'DIM':>5s}   {'VECTORS':>8s}   {'SIZE':>10s}   {'STATUS'}")
    print(f"  {'-' * 12}   {'-' * 20}   {'-' * 5}   {'-' * 8}   {'-' * 10}   {'-' * 8}")

    for dataset_key in datasets_to_prep:
        for model_label, model_info in models_to_prep.items():
            cache_path = VECTORS_DIR / f"{model_label}_{dataset_key}.npy"
            dim = model_info["dim"]

            if cache_path.exists():
                arr = np.load(cache_path)
                size = cache_path.stat().st_size
                print(
                    f"  {model_label:<12s}   {dataset_key:<20s}   {dim:>5d}   {len(arr):>8,d}   "
                    f"{fmt_size(size):>10s}   CACHED"
                )
            else:
                print(f"  {model_label:<12s}   {dataset_key:<20s}   {dim:>5d}   {'':>8s}   {'':>10s}   MISSING")

    print(f"\n  Directory: {VECTORS_DIR}")
    print()


def prep_vectors(only_model=None, only_dataset=None, status_only=False, force=False):
    """Pre-download models, datasets, and generate all .npy cache files.

    Each (model, dataset) pair gets one .npy file at the maximum needed size.

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

    if not HAS_MODEL_DEPS:
        raise RuntimeError("Vector prep requires: uv pip install datasets sentence-transformers")

    max_n = max(VSS_SIZES)

    VECTORS_DIR.mkdir(parents=True, exist_ok=True)

    # Model-outer loop: load each embedding model once, encode all datasets that need it
    for model_label, model_info in models_to_prep.items():
        # Pre-check which datasets need encoding with this model.
        # If the .npy cache exists, skip it. Use --force to re-encode.
        datasets_needing_encoding = []
        for dataset_key in datasets_to_prep:
            cache_path = VECTORS_DIR / f"{model_label}_{dataset_key}.npy"
            if cache_path.exists() and not force:
                log.info("  %s/%s: cached (%s)", model_label, dataset_key, fmt_size(cache_path.stat().st_size))
                continue
            datasets_needing_encoding.append(dataset_key)

        if not datasets_needing_encoding:
            continue  # This model is fully cached, skip loading it

        log.info("Loading model: %s (%s)...", model_label, model_info["model_id"])
        model = SentenceTransformer(model_info["model_id"])

        for dataset_key in datasets_needing_encoding:
            cache_path = VECTORS_DIR / f"{model_label}_{dataset_key}.npy"

            if cache_path.exists() and force:
                log.info("  %s/%s: --force, re-encoding", model_label, dataset_key)
                cache_path.unlink()

            texts = _load_texts(dataset_key, max_n=max_n)
            n = len(texts)
            log.info("  %s/%s: encoding %d texts (dim=%d)...", model_label, dataset_key, n, model_info["dim"])
            embeddings = model.encode(texts[:n], show_progress_bar=True, batch_size=256, normalize_embeddings=True)

            np.save(cache_path, embeddings)
            log.info("  %s/%s: cached %d embeddings to %s", model_label, dataset_key, n, cache_path)

        del model
        log.info("Unloaded model: %s", model_label)

    log.info("Vector prep complete. Cached in %s", VECTORS_DIR)
