"""Phase: Entity Embeddings via muninn GGUF.

Embeds unique entity names using muninn_embed() (llama.cpp GGUF inference)
and inserts into the entities_vec HNSW index.

Incremental: only embeds entity names not yet present in entity_vec_map.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

from benchmarks.sessions_demo.constants import GGUF_EMBEDDING_DIM, GGUF_MODEL_NAME, GGUF_MODEL_PATH
from benchmarks.sessions_demo.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.sessions_demo.build import PhaseContext

log = logging.getLogger(__name__)


class PhaseEntityEmbeddings(Phase):
    """Embed unique entity names via muninn_embed() GGUF and insert into HNSW index."""

    @property
    def name(self) -> str:
        return "entity_embeddings"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        """Return True if any entity names lack a vector in entity_vec_map."""
        try:
            new_names: int = conn.execute("""
                SELECT count(*) FROM (
                    SELECT DISTINCT name FROM entities
                    EXCEPT
                    SELECT name FROM entity_vec_map
                )
            """).fetchone()[0]
            return new_names > 0
        except sqlite3.OperationalError:
            return True

    def restore_ctx(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        try:
            ctx.num_unique_entities = conn.execute("SELECT count(*) FROM entity_vec_map").fetchone()[0]
        except sqlite3.OperationalError:
            pass

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        # Register the GGUF embedding model if not already registered by chunks_vec.
        existing = conn.execute("SELECT 1 FROM temp.muninn_models WHERE name = ?", (GGUF_MODEL_NAME,)).fetchone()
        if not existing:
            conn.execute(
                """INSERT INTO temp.muninn_models(name, model)
                   SELECT ?, muninn_embed_model(?)""",
                (GGUF_MODEL_NAME, GGUF_MODEL_PATH),
            )
            log.info("  Registered GGUF model: %s → %s", GGUF_MODEL_NAME, GGUF_MODEL_PATH)

        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS entities_vec USING hnsw_index("
            f"  dimensions={GGUF_EMBEDDING_DIM}, metric='cosine', m=16, ef_construction=200"
            f")"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS entity_vec_map (rowid INTEGER PRIMARY KEY, name TEXT NOT NULL)")

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        # ── Find entity names not yet embedded ───────────────────────
        new_names = conn.execute("""
            SELECT DISTINCT name FROM entities
            WHERE name NOT IN (SELECT name FROM entity_vec_map)
            ORDER BY name
        """).fetchall()
        new_names_list = [r[0] for r in new_names]

        if not new_names_list:
            log.info("  All entity names already embedded")
        else:
            log.info("  Embedding %d new entity names with %s (GGUF)...", len(new_names_list), GGUF_MODEL_NAME)

            max_rowid = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM entity_vec_map").fetchone()[0]
            embedded = 0
            for i, ent_name in enumerate(new_names_list):
                rowid = max_rowid + i + 1
                result = conn.execute("SELECT muninn_embed(?, ?)", (GGUF_MODEL_NAME, ent_name)).fetchone()
                if result and result[0]:
                    conn.execute(
                        "INSERT INTO entities_vec (rowid, vector) VALUES (?, ?)",
                        (rowid, result[0]),
                    )
                    conn.execute(
                        "INSERT INTO entity_vec_map (rowid, name) VALUES (?, ?)",
                        (rowid, ent_name),
                    )
                    embedded += 1
            log.info("  Inserted %d new entity embeddings", embedded)

        ctx.num_unique_entities = conn.execute("SELECT count(*) FROM entity_vec_map").fetchone()[0]
        log.info("  Total entity embeddings: %d", ctx.num_unique_entities)
