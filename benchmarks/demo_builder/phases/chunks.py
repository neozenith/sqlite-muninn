"""Phase 1: Chunks + FTS + Chunk Embeddings."""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

from benchmarks.demo_builder.common import load_chunk_vectors, pack_vector
from benchmarks.demo_builder.constants import EMBEDDING_MODELS, KG_DIR
from benchmarks.demo_builder.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext

log = logging.getLogger(__name__)


class PhaseChunks(Phase):
    """Import chunks, build FTS5 index, load/compute embeddings, insert into HNSW."""

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

        # ── Import chunks from pre-built chunks DB ────────────────────
        chunks_db_path = KG_DIR / f"{self._book_id}_chunks.db"
        assert chunks_db_path.exists(), f"Chunks DB not found: {chunks_db_path}"

        src = sqlite3.connect(str(chunks_db_path))
        rows = src.execute("SELECT id, text FROM text_chunks ORDER BY id").fetchall()
        src.close()

        conn.executemany("INSERT INTO chunks (chunk_id, text) VALUES (?, ?)", rows)
        num_chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        log.info("  Imported %d chunks from %s", num_chunks, chunks_db_path.name)

        # ── Rebuild FTS5 index ────────────────────────────────────────
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        log.info("  Built FTS5 index (chunks_fts)")

        # ── Load or compute chunk embeddings ──────────────────────────
        chunk_texts = [text for _, text in rows]
        chunk_vectors = load_chunk_vectors(self._book_id, num_chunks, self._model_name, self._st_model, chunk_texts)

        # ── Insert vectors into HNSW ──────────────────────────────────
        for i in range(num_chunks):
            conn.execute(
                "INSERT INTO chunks_vec (rowid, vector) VALUES (?, ?)",
                (i, pack_vector(chunk_vectors[i])),
            )

        log.info("  Inserted %d vectors into chunks_vec HNSW index", num_chunks)

        ctx.num_chunks = num_chunks
        ctx.chunk_vectors = chunk_vectors
