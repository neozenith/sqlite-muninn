"""Phase 4: Entity Embeddings.

Incremental: only embeds entity names not yet present in entity_vec_map.
After embedding new names, reloads ALL entity vectors from the DB into ctx
so downstream phases (UMAP, entity_resolution) have the full set.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from benchmarks.demo_builder.common import pack_vector
from benchmarks.demo_builder.constants import EMBEDDING_MODELS
from benchmarks.demo_builder.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext

log = logging.getLogger(__name__)


class PhaseEntityEmbeddings(Phase):
    """Embed unique entity names and insert into HNSW index."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._st_model: SentenceTransformer | None = None

    @property
    def name(self) -> str:
        return "entity_embeddings"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        """Return True if any entity names lack a vector in entity_vec_map."""
        try:
            new_names = conn.execute("""
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
        """Load all entity vectors from the DB into ctx for UMAP/entity_resolution."""
        try:
            rows = conn.execute("SELECT id, vector FROM entities_vec_nodes ORDER BY id").fetchall()
            ctx.num_unique_entities = len(rows)
            if rows:
                ctx.entity_vectors = np.stack([np.frombuffer(bytes(r[1]), dtype=np.float32) for r in rows])
        except sqlite3.OperationalError:
            pass

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        model_info = EMBEDDING_MODELS[self._model_name]
        dim = model_info["dim"]

        # IF NOT EXISTS: preserve existing embeddings on incremental re-runs.
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS entities_vec USING hnsw_index("
            f"  dimensions={dim}, metric='cosine', m=16, ef_construction=200"
            f")"
        )
        conn.execute("CREATE TABLE IF NOT EXISTS entity_vec_map (rowid INTEGER PRIMARY KEY, name TEXT NOT NULL)")

        log.info("  Loading SentenceTransformer %s...", model_info["st_name"])
        st_kwargs: dict[str, bool] = {}
        if model_info.get("trust_remote_code"):
            st_kwargs["trust_remote_code"] = True
        path = snapshot_download(model_info["st_name"], local_files_only=True)
        self._st_model = SentenceTransformer(path, **st_kwargs)

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        assert self._st_model is not None, "setup() must be called before run()"

        model_info = EMBEDDING_MODELS[self._model_name]

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
            log.info("  Encoding %d new entity names with %s...", len(new_names_list), model_info["st_name"])
            new_vectors = self._st_model.encode(new_names_list, show_progress_bar=True, normalize_embeddings=True)
            new_vectors = new_vectors.astype(np.float32)

            # Assign rowids sequentially from max existing + 1.
            max_rowid = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM entity_vec_map").fetchone()[0]
            for i, (ent_name, vec) in enumerate(zip(new_names_list, new_vectors, strict=True)):
                rowid = max_rowid + i + 1
                conn.execute(
                    "INSERT INTO entities_vec (rowid, vector) VALUES (?, ?)",
                    (rowid, pack_vector(vec)),
                )
                conn.execute(
                    "INSERT INTO entity_vec_map (rowid, name) VALUES (?, ?)",
                    (rowid, ent_name),
                )
            log.info("  Inserted %d new entity embeddings", len(new_names_list))

        # ── Reload ALL entity vectors (old + new) into ctx ───────────
        # UMAP and entity_resolution need the full set, not just incremental.
        all_rows = conn.execute("SELECT id, vector FROM entities_vec_nodes ORDER BY id").fetchall()
        ctx.num_unique_entities = len(all_rows)
        if all_rows:
            ctx.entity_vectors = np.stack([np.frombuffer(bytes(r[1]), dtype=np.float32) for r in all_rows])
        log.info("  Total entity embeddings: %d", ctx.num_unique_entities)
