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

# ── GGUF chat model (for muninn NER/RE backend) ─────────────────

MUNINN_CHAT_MODEL_NAME = "Qwen3.5-4B"
MUNINN_CHAT_MODEL_FILE = "Qwen3.5-4B-Q4_K_M.gguf"
MUNINN_CHAT_MODELS_DIR = PROJECT_ROOT / "models"

# ── GGUF embedding model ─────────────────────────────────────────

GGUF_MODEL_PATH = str(PROJECT_ROOT / "models" / "nomic-embed-text-v1.5.Q8_0.gguf")
GGUF_MODEL_NAME = "nomic"
GGUF_EMBEDDING_DIM = 768

# ── Schema version ───────────────────────────────────────────────
# Bump this when the schema changes to trigger a rebuild.

SCHEMA_VERSION = "6"

# ── Chunking parameters ──────────────────────────────────────────
# Chunk size is constrained by the smallest model window in the pipeline.
# GLiNER/GLiREL have 384 word-token windows.
#
# IMPORTANT: sessions_demo processes Claude Code session logs — Python code,
# JSON structures, file paths, UUIDs. This content tokenises at ~3.1 chars
# per GLiNER word-token (each {, }, ", :, ,, /, - and path segment is a
# separate word token), vs English prose at ~5.5 chars/word-token.
#
# Empirically confirmed by scripts/kg_chunk_size_fix.py diagnostic:
#   1920 chars → "Sentence of length 590 has been truncated to 384"  ← BUG
#   1400 chars → "Sentence of length 428 has been truncated to 384"  ← BUG
#   1200 chars → OK (all GLiNER small/medium/large and NuNerZero)    ← SAFE
#
# DO NOT restore to 1920. That formula assumed English prose density.
MODEL_MIN_TOKENS = 384
CHARS_PER_WORD_TOKEN = 3.1  # code-dense content: Python/JSON/paths — empirically measured
CHUNK_MAX_CHARS = 1200  # safe boundary confirmed by scripts/kg_chunk_size_fix.py
CHUNK_MIN_CHARS = 100

# Hard character limit fed to muninn_embed(). Chunks are stored at CHUNK_MAX_CHARS
# chars (good for NER/RE/FTS), but nomic-embed's subword tokenizer encodes
# code-heavy content at up to ~1.3 tokens/char — a 1920-char chunk can reach
# 2416 tokens, exceeding the 2048-token model context. 1500 chars is a safe
# ceiling: worst-case observed ratio yields ~1950 tokens, well under the limit.
EMBED_MAX_CHARS = 1500

# ── Message type filtering ───────────────────────────────────────
# Controls which JSONL event types are chunked and fed into the KG pipeline.
# DEFAULT_MESSAGE_TYPES focuses the KG on human intent signals only.
# Change at build time via --message-types; switching filters mid-run requires
# --run-from chunks to rebuild the chunks table from scratch.

DEFAULT_MESSAGE_TYPES: list[str] = ["human"]
ALL_MESSAGE_TYPES: list[str] = [
    # ── Named filter aliases ─────────────────────────────────────
    "human",  # Maps to msg_kind='human'. Matches only genuine
    # human-typed prompts (~9% of all events, ~6% of user events).
    # Excludes tool_result blocks, isMeta wrappers, and injections.
    # ── Raw event_type values ────────────────────────────────────
    "user",  # All user-role events: human prompts + tool results + injections.
    "assistant",  # Claude responses: text, thinking, and tool_use stub blocks.
    "system",  # System-injected context (CLAUDE.md, permissions, env info).
]

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

# ── NER labels for GLiNER2 (zero-shot, Claude Code session log domain) ───────

SESSION_GLINER2_NER_LABELS = [
    "tool",  # Tool invocations: Bash, Read, Write, Edit, Task, etc.
    "file path",  # Source file paths and directories
    "model",  # AI model identifiers: claude-sonnet-4-6, etc.
    "concept",  # Technical/programming concepts: HNSW, BFS, Leiden
    "error",  # Error types and exception messages
    "person",  # People referenced in session messages
    "organization",  # Companies, projects, products
]

# ── RE labels for GLiNER2 (zero-shot RE, Claude Code session log domain) ──────

SESSION_GLINER2_RE_LABELS = [
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
    "chunks_vec",
    "chunks_vec_umap",
    "ner",
    "relations",
    "entity_embeddings",
    "entities_vec_umap",
    "entity_resolution",
    "node2vec",
    "communities",
    "community_naming",
    "metadata",
]
