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
# GLiNER truncates at 384 word-level tokens (\w+(?:-\w+)*|\S regex —
# every punctuation character is its own token).
# GLiREL (DeBERTa-v3-large) additionally truncates at 512 DeBERTa subword
# tokens — a separate constraint from the 384 word-token GLiNER limit.
# Chunks must fit within BOTH limits to avoid silent entity/relation loss.

NER_MAX_TOKENS = 384  # GLiNER medium-v2.1 config.max_len
RE_MAX_TOKENS = 384  # GLiREL large-v0 base_config.max_len (span encoder)

# chars-per-word-token ratio for English literary prose (Gutenberg text).
#
# EMPIRICALLY MEASURED from full Wealth of Nations build (book 3300):
#   NER phase produced chunks with 393–475 word tokens at 1920 chars.
#   Worst observed: 1920 chars → 475 word tokens = 4.04 chars/token.
#   Safe limit at 384 tokens: 384/475 × 1920 = 1552 chars.
#   We use 3.9 (→ 1497 chars) for ~4% additional buffer below 1552.
#
# IMPORTANT: scripts/kg_chunk_size_fix.py tested lines 34486-34544 of WoN
# (the book's rhetorical conclusion — atypically sparse, ~5+ chars/token).
# That excerpt did NOT represent the book's dense enumeration chapters
# (wages, rent, manufactures). The correct ratio was found via the full build.
#
# GLiREL DeBERTa-512 cross-check at 1497 chars:
#   Observed 562 subword tokens at 1920 → 1497/1920 × 562 = 438 subword tokens
#   438 < 512 ✓ — the GLiREL subword limit is satisfied at 1497 chars.
#
# NOTE: Code-dense content (Python, JSON, file paths) tokenises at ~3.1
# chars/word-token — sessions_demo uses CHUNK_MAX_CHARS = 1200.
CHARS_PER_WORD_TOKEN = 3.9

# Maximum chunk size in chars fitting within NER/RE model limits (dense prose).
# 1497 chars → worst-case WoN passage produces ~370 word tokens (< 384 ✓).
NER_RE_CHUNK_CHARS_MAX = int(min(NER_MAX_TOKENS, RE_MAX_TOKENS) * CHARS_PER_WORD_TOKEN)  # = 1497

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

# ── NER labels for GLiNER2 (zero-shot, Gutenberg literary prose domain) ──────

GLINER2_NER_LABELS = [
    "person",
    "organization",
    "location",
    "economic concept",
    "commodity",
    "institution",
    "legal concept",
    "occupation",
]

# ── Relation labels for GLiNER2 (zero-shot RE, Gutenberg literary prose) ─────

GLINER2_RE_LABELS = [
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
    "chunks",  # text chunking + FTS only
    "chunks_embeddings",  # SentenceTransformer → chunks_vec HNSW
    "chunks_umap",  # UMAP on chunks_vec_nodes (independent of ner path)
    "ner",  # GLiNER entity extraction (parallel with chunks_embeddings)
    "relations",  # GLiREL relation extraction
    "entity_embeddings",  # SentenceTransformer → entities_vec HNSW
    "entities_umap",  # UMAP on entities_vec_nodes
    "entity_resolution",  # HNSW blocking + Jaro-Winkler + Leiden
    "node2vec",  # Node2Vec structural embeddings
    "metadata",  # meta table + validation (terminal)
]
