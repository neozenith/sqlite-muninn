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

DEFAULT_OUTPUT_DIR = BENCHMARKS_ROOT / "sessions_demo" / "output"
DEFAULT_DB_NAME = "sessions_demo.db"

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
# All chunks must fit within this limit to avoid silent truncation
# when KG extraction phases are added later.

MODEL_MIN_TOKENS = 384
CHARS_PER_WORD_TOKEN = 5.0
CHUNK_MAX_CHARS = int(MODEL_MIN_TOKENS * CHARS_PER_WORD_TOKEN)  # 1920
CHUNK_MIN_CHARS = 100

# ── Phase names ───────────────────────────────────────────────────

PHASE_NAMES = [
    "ingest",
    "chunks",
    "embeddings",
]
