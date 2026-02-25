"""Phase 4: Entity Embeddings."""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

import numpy as np
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

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        model_info = EMBEDDING_MODELS[self._model_name]
        dim = model_info["dim"]

        conn.execute(
            f"CREATE VIRTUAL TABLE entities_vec USING hnsw_index("
            f"  dimensions={dim}, metric='cosine', m=16, ef_construction=200"
            f")"
        )
        conn.execute("CREATE TABLE entity_vec_map (rowid INTEGER PRIMARY KEY, name TEXT NOT NULL)")

        log.info("  Loading SentenceTransformer %s...", model_info["st_name"])
        st_kwargs: dict[str, bool] = {}
        if model_info.get("trust_remote_code"):
            st_kwargs["trust_remote_code"] = True
        self._st_model = SentenceTransformer(model_info["st_name"], **st_kwargs)

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        assert self._st_model is not None, "setup() must be called before run()"

        model_info = EMBEDDING_MODELS[self._model_name]

        # ── Get unique entity names ───────────────────────────────────
        rows = conn.execute("SELECT DISTINCT name FROM entities ORDER BY name").fetchall()
        entity_names = [r[0] for r in rows]
        log.info("  Found %d unique entity names", len(entity_names))

        # ── Embed with sentence-transformers ──────────────────────────
        log.info("  Encoding %d entity names with %s...", len(entity_names), model_info["st_name"])
        entity_vectors = self._st_model.encode(entity_names, show_progress_bar=True, normalize_embeddings=True)
        entity_vectors = entity_vectors.astype(np.float32)
        log.info("  Encoded entity embeddings: shape=%s", entity_vectors.shape)

        # ── Insert embeddings + build mapping table ───────────────────
        for i, (ent_name, vec) in enumerate(zip(entity_names, entity_vectors, strict=True)):
            rowid = i + 1
            conn.execute(
                "INSERT INTO entities_vec (rowid, vector) VALUES (?, ?)",
                (rowid, pack_vector(vec)),
            )
            conn.execute(
                "INSERT INTO entity_vec_map (rowid, name) VALUES (?, ?)",
                (rowid, ent_name),
            )

        log.info("  Inserted %d entity embeddings into entities_vec", len(entity_names))

        ctx.num_unique_entities = len(entity_names)
        ctx.entity_vectors = entity_vectors
