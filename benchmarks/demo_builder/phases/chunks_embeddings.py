"""Phase 2: Chunk Embeddings (SentenceTransformer + HNSW).

Loads or computes embedding vectors for all chunks and inserts them into
the chunks_vec HNSW index.

Separated from PhaseChunks so that ner (phase 3) can run in parallel with
chunks_embeddings once chunking is done: NER only needs the chunks table
(raw text), not the HNSW vectors.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from benchmarks.demo_builder.common import load_chunk_vectors, pack_vector
from benchmarks.demo_builder.constants import EMBEDDING_MODELS
from benchmarks.demo_builder.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext

log = logging.getLogger(__name__)


class PhaseChunksEmbeddings(Phase):
    """Load or compute chunk embeddings and insert into chunks_vec HNSW index."""

    def __init__(self, book_id: int, model_name: str) -> None:
        self._book_id = book_id
        self._model_name = model_name
        self._st_model: SentenceTransformer | None = None

    @property
    def name(self) -> str:
        return "chunks_embeddings"

    def restore_ctx(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        try:
            ctx.num_chunks = conn.execute("SELECT count(*) FROM chunks_vec_nodes").fetchone()[0]
        except sqlite3.OperationalError:
            pass

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        model_info = EMBEDDING_MODELS[self._model_name]
        dim = model_info["dim"]

        conn.execute(
            f"CREATE VIRTUAL TABLE chunks_vec USING hnsw_index("
            f"  dimensions={dim}, metric='cosine', m=16, ef_construction=200"
            f")"
        )

        log.info("  Loading SentenceTransformer %s...", model_info["st_name"])
        st_kwargs: dict[str, bool] = {}
        if model_info.get("trust_remote_code"):
            st_kwargs["trust_remote_code"] = True
        path = snapshot_download(model_info["st_name"], local_files_only=True)
        self._st_model = SentenceTransformer(path, **st_kwargs)

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        assert self._st_model is not None, "setup() must be called before run()"

        chunk_rows = conn.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
        assert chunk_rows, "chunks table is empty — run PhaseChunks first"

        chunk_texts = [r[1] for r in chunk_rows]
        num_chunks = len(chunk_texts)

        chunk_vectors = load_chunk_vectors(self._book_id, num_chunks, self._model_name, self._st_model, chunk_texts)

        for i in range(num_chunks):
            conn.execute(
                "INSERT INTO chunks_vec (rowid, vector) VALUES (?, ?)",
                (i, pack_vector(chunk_vectors[i])),
            )

        log.info("  Inserted %d vectors into chunks_vec HNSW index", num_chunks)
        ctx.num_chunks = num_chunks
