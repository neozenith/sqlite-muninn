"""Phase 3: Embeddings — HNSW vector index on event_message_chunks via GGUF.

Embeds chunks (not raw events) so that every vector fits within the
model's context window. Chunks are already sized to 1920 chars (384
word tokens) by the chunks phase.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

from benchmarks.demo_builder.common import load_muninn
from benchmarks.sessions_demo.constants import GGUF_EMBEDDING_DIM, GGUF_MODEL_NAME, GGUF_MODEL_PATH

if TYPE_CHECKING:
    from benchmarks.sessions_demo.build import PhaseContext

log = logging.getLogger(__name__)


class PhaseEmbeddings:
    """Create HNSW vector index on message chunks.

    Loads the muninn extension, registers the NomicEmbed GGUF model,
    creates a chunks_vec hnsw_index virtual table, and embeds each
    chunk's text using muninn_embed(). Chunks are pre-sized to fit
    within all model context windows.
    """

    @property
    def name(self) -> str:
        return "embeddings"

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        load_muninn(conn)

        # Register the GGUF embedding model
        conn.execute(
            """INSERT INTO temp.muninn_models(name, model)
               SELECT ?, muninn_embed_model(?)""",
            (GGUF_MODEL_NAME, GGUF_MODEL_PATH),
        )
        log.info("Registered GGUF model: %s", GGUF_MODEL_PATH)

        # Create the HNSW virtual table keyed by chunk_id
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec
            USING hnsw_index(dimensions={GGUF_EMBEDDING_DIM}, metric='cosine')
        """)
        conn.commit()
        log.info("Created chunks_vec (dim=%d, metric=cosine)", GGUF_EMBEDDING_DIM)

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        # Count chunks needing embedding for progress estimate
        total_to_embed = conn.execute("""
            SELECT COUNT(*) FROM event_message_chunks c
            WHERE c.chunk_id NOT IN (SELECT rowid FROM chunks_vec)
        """).fetchone()[0]
        log.info("Embedding %d chunks (dim=%d)", total_to_embed, GGUF_EMBEDDING_DIM)

        if total_to_embed == 0:
            ctx.chunks_embedded = 0
            return

        cursor = conn.execute("""
            SELECT c.chunk_id, c.text
            FROM event_message_chunks c
            WHERE c.chunk_id NOT IN (SELECT rowid FROM chunks_vec)
        """)

        total_embedded = 0
        for row in cursor:
            chunk_id, text = row[0], row[1]
            try:
                result = conn.execute(
                    "SELECT muninn_embed(?, ?)",
                    (GGUF_MODEL_NAME, text),
                ).fetchone()
                if result and result[0]:
                    conn.execute(
                        "INSERT INTO chunks_vec(rowid, vector) VALUES (?, ?)",
                        (chunk_id, result[0]),
                    )
                    total_embedded += 1
            except sqlite3.OperationalError as e:
                log.warning("Failed to embed chunk %d: %s", chunk_id, e)

            if total_embedded % 200 == 0 and total_embedded > 0:
                log.info("  Embedded %d/%d chunks", total_embedded, total_to_embed)

        conn.commit()
        ctx.chunks_embedded = total_embedded
        log.info("Embedded %d/%d chunks into chunks_vec", total_embedded, total_to_embed)

    def teardown(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        pass

    def __call__(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        self.setup(conn, ctx)
        self.run(conn, ctx)
        self.teardown(conn, ctx)
