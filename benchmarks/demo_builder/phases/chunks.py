"""Phase 1: Chunks + FTS.

Reads raw Gutenberg text, splits into model-aware overlapping chunks
(snapped to sentence boundaries), and builds FTS5.

Chunk embeddings (SentenceTransformer + HNSW) are handled separately by
PhaseChunksEmbeddings (phase 2), which enables ner (phase 3) to run in
parallel with chunks_embeddings once chunking is done.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

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
    """Read raw text, chunk it, and build FTS5 index.

    Does NOT load SentenceTransformer or create chunks_vec — that is
    PhaseChunksEmbeddings (phase 2). Splitting these lets ner (phase 3)
    start in parallel with chunks_embeddings as soon as text is chunked.
    """

    def __init__(self, book_id: int, model_name: str) -> None:
        self._book_id = book_id
        self._model_name = model_name

    @property
    def name(self) -> str:
        return "chunks"

    def restore_ctx(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        try:
            ctx.num_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        except sqlite3.OperationalError:
            pass

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        conn.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
        conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(  text, content=chunks, content_rowid=chunk_id)")

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        model_info = EMBEDDING_MODELS[self._model_name]
        model_chunk_chars = model_info["chunk_chars"]
        chunk_chars = min(model_chunk_chars, NER_RE_CHUNK_CHARS_MAX)

        text_path = TEXTS_DIR / f"gutenberg_{self._book_id}.txt"
        assert text_path.exists(), f"Text file not found: {text_path}"
        raw_text = text_path.read_text(encoding="utf-8")

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

        rows = [(i, text) for i, text in enumerate(chunk_texts)]
        conn.executemany("INSERT INTO chunks (chunk_id, text) VALUES (?, ?)", rows)
        num_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        log.info("  Inserted %d chunks", num_chunks)

        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        log.info("  Built FTS5 index (chunks_fts)")

        ctx.num_chunks = num_chunks
