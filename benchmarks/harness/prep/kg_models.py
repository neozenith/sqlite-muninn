"""Prep module: download all ML models needed for KG benchmarks to local cache.

Run once while online. After that, benchmarks load models offline.

Covers:
  GLiNER (3 size variants)       urchade/gliner_{small,medium,large}-v2.1 + deberta-v3-base backbone
  NuNerZero                      numind/NuNerZero + backbone (read from config)
  GNER-T5 (base + large)         dyyyyyyyy/GNER-T5-{base,large}  (self-contained, no backbone)
  GLiREL large-v0                jackboyla/glirel-large-v0 + deberta-v3-large backbone
  SentenceTransformer (2 models) sentence-transformers/all-MiniLM-L6-v2, nomic-ai/nomic-embed-text-v1.5
  spaCy en_core_web_lg           installed via `spacy download` (not HuggingFace)

Usage:
  uv run -m benchmarks.harness prep kg-models            # download all models
  uv run -m benchmarks.harness prep kg-models --model gliner_medium-v2.1
  uv run -m benchmarks.harness prep kg-models --status   # check cache status
"""

from __future__ import annotations

import logging
import subprocess
import sys

import spacy
from huggingface_hub import snapshot_download

from benchmarks.demo_builder.common import _read_backbone

log = logging.getLogger(__name__)


# ── Model registry ────────────────────────────────────────────────
#
# Each entry covers one HuggingFace repo. backbone=True means the snapshot
# ships only custom weights and the tokenizer/encoder backbone is a separate
# repo that must be downloaded via _read_backbone() + snapshot_download().

KG_MODEL_REGISTRY: list[dict] = [
    {
        "slug": "gliner_small-v2.1",
        "repo_id": "urchade/gliner_small-v2.1",
        "backbone": True,
        "description": "GLiNER small-v2.1 zero-shot NER (DeBERTa-v3-base backbone)",
    },
    {
        "slug": "gliner_medium-v2.1",
        "repo_id": "urchade/gliner_medium-v2.1",
        "backbone": True,
        "description": "GLiNER medium-v2.1 zero-shot NER (DeBERTa-v3-base backbone)",
    },
    {
        "slug": "gliner_large-v2.1",
        "repo_id": "urchade/gliner_large-v2.1",
        "backbone": True,
        "description": "GLiNER large-v2.1 zero-shot NER (DeBERTa-v3-large backbone)",
    },
    {
        "slug": "numind_NuNerZero",
        "repo_id": "numind/NuNerZero",
        "backbone": True,
        "description": "NuNerZero zero-shot NER (backbone read from config)",
    },
    {
        "slug": "gner-t5-base",
        "repo_id": "dyyyyyyyy/GNER-T5-base",
        "backbone": False,
        "description": "GNER-T5 base generative NER (self-contained T5)",
    },
    {
        "slug": "gner-t5-large",
        "repo_id": "dyyyyyyyy/GNER-T5-large",
        "backbone": False,
        "description": "GNER-T5 large generative NER (self-contained T5)",
    },
    {
        "slug": "glirel",
        "repo_id": "jackboyla/glirel-large-v0",
        "backbone": True,
        "description": "GLiREL large-v0 zero-shot RE (DeBERTa-v3-large backbone)",
    },
    {
        "slug": "all-MiniLM-L6-v2",
        "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
        "backbone": False,
        "description": "SentenceTransformer MiniLM (self-contained, dim=384)",
    },
    {
        "slug": "nomic-embed-text-v1.5",
        "repo_id": "nomic-ai/nomic-embed-text-v1.5",
        "backbone": False,
        "description": "SentenceTransformer nomic-embed (self-contained, dim=768)",
    },
]

KG_MODEL_SLUGS: list[str] = [m["slug"] for m in KG_MODEL_REGISTRY]


# ── Status display ────────────────────────────────────────────────


def _check_cached(repo_id: str) -> bool:
    """Return True if the HuggingFace snapshot is present in local cache."""
    try:
        snapshot_download(repo_id, local_files_only=True)
        return True
    except EnvironmentError:
        return False


def _check_spacy() -> bool:
    """Return True if spaCy en_core_web_lg is installed."""
    try:
        spacy.load("en_core_web_lg")
        return True
    except OSError:
        return False


def print_status() -> None:
    """Print cache status for all KG models."""
    print("=== KG Model Cache Status ===\n")
    print(f"  {'SLUG':<24s}  {'STATUS':<8s}  DESCRIPTION")
    print(f"  {'-' * 24}  {'-' * 8}  {'-' * 50}")

    for entry in KG_MODEL_REGISTRY:
        cached = _check_cached(entry["repo_id"])
        status = "READY" if cached else "MISSING"
        print(f"  {entry['slug']:<24s}  {status:<8s}  {entry['description']}")

        if cached and entry["backbone"]:
            try:
                path = snapshot_download(entry["repo_id"], local_files_only=True)
                backbone = _read_backbone(path)
                backbone_cached = _check_cached(backbone)
                backbone_status = "READY" if backbone_cached else "MISSING"
                print(f"  {'  backbone: ' + backbone:<24s}  {backbone_status}")
            except Exception as exc:
                print(f"  {'  backbone: (error)':<24s}  ERROR: {exc}")

    spacy_status = "READY" if _check_spacy() else "MISSING"
    print(f"  {'spacy:en_core_web_lg':<24s}  {spacy_status:<8s}  spaCy large English pipeline")
    print()


# ── Download logic ────────────────────────────────────────────────


def _download_entry(entry: dict) -> None:
    """Download a single model entry (snapshot + backbone if applicable)."""
    slug = entry["slug"]
    repo_id = entry["repo_id"]

    log.info("Downloading %s (%s)...", slug, repo_id)
    path = snapshot_download(repo_id)
    log.info("  snapshot → %s", path)

    if entry["backbone"]:
        backbone = _read_backbone(path)
        log.info("  Downloading backbone: %s", backbone)
        backbone_path = snapshot_download(backbone)
        log.info("  backbone → %s", backbone_path)


def _download_spacy() -> None:
    """Install spaCy en_core_web_lg into the active virtualenv."""
    if _check_spacy():
        log.info("spaCy en_core_web_lg already installed")
        return
    log.info("Downloading spaCy en_core_web_lg...")
    subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_lg"],
        check=True,
    )
    log.info("  spaCy en_core_web_lg installed")


# ── Main entry point ──────────────────────────────────────────────


def prep_kg_models(model_name: str | None = None, status_only: bool = False) -> None:
    """Download all KG benchmark ML models to the local HuggingFace cache.

    Args:
        model_name: Specific model slug to download (e.g., 'gliner_medium-v2.1').
                    If None, downloads all models.
        status_only: If True, print cache status and return without downloading.
    """
    if status_only:
        print_status()
        return

    if model_name:
        matched = [e for e in KG_MODEL_REGISTRY if e["slug"] == model_name]
        if not matched:
            log.error("Unknown model slug: %s. Available: %s", model_name, ", ".join(KG_MODEL_SLUGS))
            return
        entries = matched
        include_spacy = False
    else:
        entries = list(KG_MODEL_REGISTRY)
        include_spacy = True

    log.info("Downloading %d KG model(s)...", len(entries))
    for entry in entries:
        _download_entry(entry)

    if include_spacy:
        _download_spacy()

    log.info("KG model prep complete.")
