"""Phase 3: chunks_vec — HNSW vector index on event_message_chunks via GGUF.

Embeds chunks (not raw events) so that every vector fits within the
model's context window. Chunks are already sized to CHUNK_MAX_CHARS by
the chunks phase.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

from benchmarks.sessions_demo.common import ProgressTracker, load_muninn
from benchmarks.sessions_demo.constants import EMBED_MAX_CHARS, GGUF_EMBEDDING_DIM, GGUF_MODEL_NAME, GGUF_MODEL_PATH
from benchmarks.sessions_demo.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.sessions_demo.build import PhaseContext

log = logging.getLogger(__name__)


class PhaseChunksVec(Phase):
    """Create HNSW vector index on message chunks.

    Loads the muninn extension, registers the NomicEmbed GGUF model,
    creates a chunks_vec hnsw_index virtual table, and embeds each
    chunk's text using muninn_embed(). Chunks are pre-sized to fit
    within all model context windows.
    """

    @property
    def name(self) -> str:
        return "chunks_vec"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        """Return True if any chunks lack a vector in chunks_vec_nodes."""
        try:
            pending: int = conn.execute("""
                SELECT COUNT(*) FROM event_message_chunks c
                WHERE c.chunk_id NOT IN (SELECT id FROM chunks_vec_nodes)
            """).fetchone()[0]
            return pending > 0
        except sqlite3.OperationalError:
            return True

    def restore_ctx(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        """Restore embedding count from DB when phase is skipped."""
        try:
            ctx.chunks_embedded = 0  # nothing embedded this run
        except Exception:
            pass

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
        # Count chunks needing embedding for progress estimate.
        # Use chunks_vec_nodes (shadow table, regular SQLite) not chunks_vec
        # (HNSW virtual table): HNSW VTs don't support full table scans —
        # an unconstrained SELECT returns an empty set, causing every chunk
        # to appear "not yet embedded" and triggering duplicate-rowid errors.
        total_to_embed = conn.execute("""
            SELECT COUNT(*) FROM event_message_chunks c
            WHERE c.chunk_id NOT IN (SELECT id FROM chunks_vec_nodes)
        """).fetchone()[0]
        log.info("Embedding %d chunks (dim=%d)", total_to_embed, GGUF_EMBEDDING_DIM)

        if total_to_embed == 0:
            ctx.chunks_embedded = 0
            return

        cursor = conn.execute("""
            SELECT c.chunk_id, c.text
            FROM event_message_chunks c
            WHERE c.chunk_id NOT IN (SELECT id FROM chunks_vec_nodes)
        """)

        total_embedded = 0
        tracker = ProgressTracker(total_to_embed, min_interval_s=30.0)
        for row in cursor:
            chunk_id, text = row[0], row[1]
            embedded_before = total_embedded
            try:
                # Truncate to EMBED_MAX_CHARS before embedding. Stored chunk text
                # is unchanged (NER/RE/FTS use the full text); only the vector is
                # generated from a prefix. nomic-embed's subword tokenizer encodes
                # code at ~1.3 tokens/char — full chunks can exceed the 2048-token
                # context window. Beginning-of-chunk truncation preserves the
                # highest-signal content.
                embed_text = text[:EMBED_MAX_CHARS]
                result = conn.execute(
                    "SELECT muninn_embed(?, ?)",
                    (GGUF_MODEL_NAME, embed_text),
                ).fetchone()
                if result and result[0]:
                    conn.execute(
                        "INSERT INTO chunks_vec(rowid, vector) VALUES (?, ?)",
                        (chunk_id, result[0]),
                    )
                    total_embedded += 1
            except sqlite3.OperationalError as e:
                log.warning("Failed to embed chunk %d: %s", chunk_id, e)

            tracker.update()
            tracker.record_output(total_embedded - embedded_before)
            if tracker.should_log():
                log.info("  Embedded %s", tracker.report())

        conn.commit()
        ctx.chunks_embedded = total_embedded
        log.info("Embedded %d/%d chunks into chunks_vec", total_embedded, total_to_embed)

    def teardown(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        pass

    def __call__(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        self.setup(conn, ctx)
        self.run(conn, ctx)
        self.teardown(conn, ctx)
