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
        "max_tokens": 256,
        "chunk_chars": 768,
    },
    "NomicEmbed": {
        "st_name": "nomic-ai/nomic-embed-text-v1.5",
        "dim": 768,
        "max_tokens": 8192,
        "chunk_chars": 4096,
        "trust_remote_code": True,
    },
}

# ── NER/RE model limits ─────────────────────────────────────────
# GLiNER and GLiREL both truncate at 384 word-level tokens.
# Chunks must fit within this limit to avoid silent entity/relation loss.

NER_MAX_TOKENS = 384  # GLiNER medium-v2.1 config.max_len
RE_MAX_TOKENS = 384  # GLiREL large-v0 base_config.max_len

# Conservative chars-per-word-token ratio for English text.
# English averages ~5.5 chars/word, but we use 5.0 for safety margin
# (short words, punctuation tokens, etc.)
CHARS_PER_WORD_TOKEN = 5.0

# Maximum chunk size in chars that fits within NER/RE model limits
NER_RE_CHUNK_CHARS_MAX = int(min(NER_MAX_TOKENS, RE_MAX_TOKENS) * CHARS_PER_WORD_TOKEN)  # = 1920

# ── Default output ──────────────────────────────────────────────

DEFAULT_OUTPUT_FOLDER = "viz/frontend/public/demos"

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
