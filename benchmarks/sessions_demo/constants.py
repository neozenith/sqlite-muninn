"""Pure data constants — no behavior, no heavy imports."""

from __future__ import annotations

from pathlib import Path

# ── Path constants ────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARKS_ROOT = PROJECT_ROOT / "benchmarks"
MUNINN_PATH = str(PROJECT_ROOT / "build" / "muninn")

# ── Claude Code paths ────────────────────────────────────────────

CLAUDE_HOME = Path.home() / ".claude"
PROJECTS_PATH = CLAUDE_HOME / "projects"

# ── Output ────────────────────────────────────────────────────────
# Default to the same folder as demo_builder so both tools write
# to the same viz/frontend/public/demos/ directory and share manifest.json.

DEFAULT_OUTPUT_FOLDER = PROJECT_ROOT / "viz" / "frontend" / "public" / "demos"
DEFAULT_DB_NAME = "sessions_demo.db"

# Unique ID used in the meta table and manifest.json
SESSION_DB_ID = "sessions_demo"

# ── GGUF embedding model ─────────────────────────────────────────

GGUF_MODEL_PATH = str(PROJECT_ROOT / "models" / "nomic-embed-text-v1.5.Q8_0.gguf")
GGUF_MODEL_NAME = "nomic"
GGUF_EMBEDDING_DIM = 768

# ── Schema version ───────────────────────────────────────────────
# Bump this when the schema changes to trigger a rebuild.

SCHEMA_VERSION = "2"

# ── Chunking parameters ──────────────────────────────────────────
# Chunk size is constrained by the smallest model window in the pipeline.
# GLiNER/GLiREL have 384 word-token windows (~5.0 chars/word-token).
# All chunks must fit within this limit to avoid silent truncation.

MODEL_MIN_TOKENS = 384
CHARS_PER_WORD_TOKEN = 5.0
CHUNK_MAX_CHARS = int(MODEL_MIN_TOKENS * CHARS_PER_WORD_TOKEN)  # 1920
CHUNK_MIN_CHARS = 100

# Hard character limit fed to muninn_embed(). Chunks are stored at CHUNK_MAX_CHARS
# chars (good for NER/RE/FTS), but nomic-embed's subword tokenizer encodes
# code-heavy content at up to ~1.3 tokens/char — a 1920-char chunk can reach
# 2416 tokens, exceeding the 2048-token model context. 1500 chars is a safe
# ceiling: worst-case observed ratio yields ~1950 tokens, well under the limit.
EMBED_MAX_CHARS = 1500

# ── NER labels for Claude Code session logs ───────────────────────
# GLiNER is zero-shot, so labels are domain-adapted for session content.

SESSION_NER_LABELS = [
    "tool",  # Tool invocations: Bash, Read, Write, Edit, Task, etc.
    "file path",  # Source file paths and directories
    "model",  # AI model identifiers: claude-sonnet-4-6, etc.
    "concept",  # Technical/programming concepts: HNSW, BFS, Leiden
    "error",  # Error types and exception messages
    "person",  # People referenced in session messages
    "organization",  # Companies, projects, products
]

# ── RE labels for Claude Code session logs ────────────────────────
# GLiREL uses fixed_relation_types=True, so labels must be a flat list.

SESSION_GLIREL_LABELS = [
    "uses",  # Tool usage relationships
    "modifies",  # File modification
    "creates",  # Resource creation
    "references",  # Cross-references between concepts
    "causes",  # Error causation chains
    "implements",  # Concept implementation
    "part_of",  # Component/module relationships
    "depends_on",  # Dependency relationships
]

# ── Phase names ───────────────────────────────────────────────────

PHASE_NAMES = [
    "ingest",
    "chunks",
    "embeddings",
    "chunks_vec_umap",
    "ner",
    "relations",
    "entity_embeddings",
    "entities_vec_umap",
    "entity_resolution",
    "node2vec",
    "metadata",
]
