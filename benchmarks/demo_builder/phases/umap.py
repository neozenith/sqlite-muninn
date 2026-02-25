"""Phase 5: UMAP Dimensionality Reduction."""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

import numpy as np
import umap

from benchmarks.demo_builder.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext

log = logging.getLogger(__name__)


class PhaseUMAP(Phase):
    """Compute UMAP 2D + 3D projections for chunks and entities."""

    @property
    def name(self) -> str:
        return "umap"

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        assert ctx.chunk_vectors is not None, "UMAP requires chunk_vectors from Phase 1"
        assert ctx.entity_vectors is not None, "UMAP requires entity_vectors from Phase 4"

        conn.execute(
            "CREATE TABLE chunks_vec_umap (  id INTEGER PRIMARY KEY, x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL)"
        )
        conn.execute(
            "CREATE TABLE entities_vec_umap ("
            "  id INTEGER PRIMARY KEY, x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL"
            ")"
        )

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        assert ctx.chunk_vectors is not None
        assert ctx.entity_vectors is not None

        chunk_vectors = ctx.chunk_vectors
        entity_vectors = ctx.entity_vectors
        num_chunks = ctx.num_chunks

        # ── UMAP 2D ──────────────────────────────────────────────────
        log.info("  Computing 2D UMAP on %d chunk vectors...", len(chunk_vectors))
        reducer_2d = umap.UMAP(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
        all_vectors = np.vstack([chunk_vectors, entity_vectors])
        proj_2d = reducer_2d.fit_transform(all_vectors)
        chunk_2d = proj_2d[: len(chunk_vectors)]
        entity_2d = proj_2d[len(chunk_vectors) :]

        # ── UMAP 3D ──────────────────────────────────────────────────
        log.info("  Computing 3D UMAP on %d vectors...", len(all_vectors))
        reducer_3d = umap.UMAP(n_components=3, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
        proj_3d = reducer_3d.fit_transform(all_vectors)
        chunk_3d = proj_3d[: len(chunk_vectors)]
        entity_3d = proj_3d[len(chunk_vectors) :]

        # ── Store chunk projections ───────────────────────────────────
        for i in range(num_chunks):
            conn.execute(
                "INSERT INTO chunks_vec_umap (id, x2d, y2d, x3d, y3d, z3d) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    i,
                    float(chunk_2d[i, 0]),
                    float(chunk_2d[i, 1]),
                    float(chunk_3d[i, 0]),
                    float(chunk_3d[i, 1]),
                    float(chunk_3d[i, 2]),
                ),
            )

        # ── Store entity projections ──────────────────────────────────
        for i in range(len(entity_vectors)):
            rowid = i + 1  # Match entity_vec_map rowids (1-based)
            conn.execute(
                "INSERT INTO entities_vec_umap (id, x2d, y2d, x3d, y3d, z3d) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    rowid,
                    float(entity_2d[i, 0]),
                    float(entity_2d[i, 1]),
                    float(entity_3d[i, 0]),
                    float(entity_3d[i, 1]),
                    float(entity_3d[i, 2]),
                ),
            )

        log.info("  Stored UMAP projections: %d chunks + %d entities", num_chunks, len(entity_vectors))
