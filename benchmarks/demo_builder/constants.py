"""Pure data constants — no behavior, no heavy imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ── Path constants ────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARKS_ROOT = PROJECT_ROOT / "benchmarks"
VECTORS_DIR = BENCHMARKS_ROOT / "vectors"
KG_DIR = BENCHMARKS_ROOT / "kg"
TEXTS_DIR = BENCHMARKS_ROOT / "texts"
MUNINN_PATH = str(PROJECT_ROOT / "build" / "muninn")

# ── Embedding model registry ─────────────────────────────────────

EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    "MiniLM": {
        "st_name": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
    },
    "NomicEmbed": {
        "st_name": "nomic-ai/nomic-embed-text-v1.5",
        "dim": 768,
        "trust_remote_code": True,
    },
}

# Known Gutenberg book ID -> dataset slug (for finding cached .npy vectors
# produced by `benchmarks harness CLI prep vectors`)
BOOK_ID_TO_DATASET: dict[int, str] = {
    3300: "wealth_of_nations",
}

# ── NER labels for GLiNER ────────────────────────────────────────

GLINER_LABELS = [
    "person",
    "organization",
    "location",
    "economic concept",
    "commodity",
    "institution",
    "legal concept",
    "occupation",
]

# ── Relation labels for GLiREL ───────────────────────────────────

# GLiREL uses fixed_relation_types=True, so labels must be a flat list of strings.
GLIREL_LABELS = [
    "produces",
    "trades_with",
    "regulates",
    "employs",
    "located_in",
    "influences",
    "part_of",
    "opposes",
]

# ── Phase names (for _build_progress table and manifest display) ─

PHASE_NAMES = [
    "chunks+fts+embeddings",
    "ner",
    "relations",
    "entity_embeddings",
    "umap",
    "entity_resolution",
    "node2vec",
    "metadata+validation",
]
