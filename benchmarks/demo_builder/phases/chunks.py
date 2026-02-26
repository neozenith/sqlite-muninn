"""Phase 1: Chunks + FTS + Chunk Embeddings.

Reads raw Gutenberg text, splits into model-aware overlapping chunks
(snapped to sentence boundaries), builds FTS5, loads/computes embeddings,
and inserts into HNSW.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

from benchmarks.demo_builder.common import load_chunk_vectors, pack_vector
from benchmarks.demo_builder.constants import EMBEDDING_MODELS, NER_RE_CHUNK_CHARS_MAX, TEXTS_DIR
from benchmarks.demo_builder.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext

log = logging.getLogger(__name__)


def _chunk_text(text: str, chunk_chars: int, overlap_frac: float = 0.1) -> list[str]:
    """Split text into overlapping chunks, snapping to sentence boundaries.

    Args:
        text: Raw text to split.
        chunk_chars: Maximum characters per chunk (sized to model's context window).
        overlap_frac: Fraction of chunk_chars to overlap between consecutive chunks.

    Returns:
        List of non-empty stripped chunks covering the full text.
    """
    overlap = int(chunk_chars * overlap_frac)
    chunks: list[str] = []
    pos = 0
    while pos < len(text):
        end = min(pos + chunk_chars, len(text))
        chunk = text[pos:end]
        # Try to snap end to sentence boundary (look back from end)
        if end < len(text):
            for sep in (". ", ".\n", "? ", "!\n", "! ", "?\n"):
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_chars // 2:  # Don't snap if it would lose >50%
                    chunk = chunk[: last_sep + 1]  # Include the period/punctuation
                    end = pos + len(chunk)
                    break
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        pos = end - overlap  # Overlap by re-reading some chars
    return [c for c in chunks if c]


class PhaseChunks(Phase):
    """Read raw text, chunk, build FTS5 index, load/compute embeddings, insert into HNSW."""

    def __init__(self, book_id: int, model_name: str) -> None:
        self._book_id = book_id
        self._model_name = model_name
        self._st_model: SentenceTransformer | None = None

    @property
    def name(self) -> str:
        return "chunks+fts+embeddings"

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        model_info = EMBEDDING_MODELS[self._model_name]
        dim = model_info["dim"]

        conn.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
        conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(  text, content=chunks, content_rowid=chunk_id)")
        conn.execute(
            f"CREATE VIRTUAL TABLE chunks_vec USING hnsw_index("
            f"  dimensions={dim}, metric='cosine', m=16, ef_construction=200"
            f")"
        )

        log.info("  Loading SentenceTransformer %s...", model_info["st_name"])
        st_kwargs: dict[str, bool] = {}
        if model_info.get("trust_remote_code"):
            st_kwargs["trust_remote_code"] = True
        self._st_model = SentenceTransformer(model_info["st_name"], **st_kwargs)

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        assert self._st_model is not None, "setup() must be called before run()"

        # ── Read raw text and chunk ────────────────────────────────
        text_path = TEXTS_DIR / f"gutenberg_{self._book_id}.txt"
        assert text_path.exists(), f"Text file not found: {text_path}"

        raw_text = text_path.read_text(encoding="utf-8")
        model_info = EMBEDDING_MODELS[self._model_name]
        model_chunk_chars = model_info["chunk_chars"]
        chunk_chars = min(model_chunk_chars, NER_RE_CHUNK_CHARS_MAX)

        chunk_texts = _chunk_text(raw_text, chunk_chars)
        if chunk_chars < model_chunk_chars:
            log.info(
                "  Chunked %d chars into %d chunks (chunk_chars=%d, capped from %d by NER/RE limit, model=%s)",
                len(raw_text),
                len(chunk_texts),
                chunk_chars,
                model_chunk_chars,
                self._model_name,
            )
        else:
            log.info(
                "  Chunked %d chars into %d chunks (chunk_chars=%d, model=%s)",
                len(raw_text),
                len(chunk_texts),
                chunk_chars,
                self._model_name,
            )

        # ── Insert chunks ──────────────────────────────────────────
        rows = [(i, text) for i, text in enumerate(chunk_texts)]
        conn.executemany("INSERT INTO chunks (chunk_id, text) VALUES (?, ?)", rows)
        num_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        log.info("  Inserted %d chunks", num_chunks)

        # ── Rebuild FTS5 index ─────────────────────────────────────
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        log.info("  Built FTS5 index (chunks_fts)")

        # ── Load or compute chunk embeddings ───────────────────────
        chunk_vectors = load_chunk_vectors(self._book_id, num_chunks, self._model_name, self._st_model, chunk_texts)

        # ── Insert vectors into HNSW ───────────────────────────────
        for i in range(num_chunks):
            conn.execute(
                "INSERT INTO chunks_vec (rowid, vector) VALUES (?, ?)",
                (i, pack_vector(chunk_vectors[i])),
            )

        log.info("  Inserted %d vectors into chunks_vec HNSW index", num_chunks)

        ctx.num_chunks = num_chunks
        ctx.chunk_vectors = chunk_vectors
