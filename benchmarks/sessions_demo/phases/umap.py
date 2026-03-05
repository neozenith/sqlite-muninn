"""Phases 4 & 8: UMAP Dimensionality Reduction for sessions_demo.

Split into two independent phases, each managing its own fitted models:

  PhaseChunksUMAP    (phase 4)  depends on: embeddings (chunks_vec_nodes)
  PhaseEntitiesUMAP  (phase 8)  depends on: entity_embeddings (entities_vec_nodes)

Model files stored next to the DB:
  {db_stem}_chunks_umap2d.joblib     {db_stem}_chunks_umap3d.joblib
  {db_stem}_entities_umap2d.joblib   {db_stem}_entities_umap3d.joblib

Two modes per phase:

FULL FIT (first run or missing model files):
  - fit_transform on ALL current vectors for that type.
  - Saves the fitted 2D and 3D reducers as joblib files.
  - Inserts all projections into the umap table.

INCREMENTAL (model files exist):
  - Loads the saved reducers.
  - Calls reducer.transform() on ONLY new vectors (not yet in umap table).
  - Inserts only the new rows — existing rows are untouched.
  - O(n_new) rather than O(n_total): fast and stable for incremental builds.

Note: chunks and entities each have their own independent UMAP coordinate space.
They are shown in separate visualizations in the viz frontend and do not need
to share a joint embedding.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import umap

from benchmarks.sessions_demo.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.sessions_demo.build import PhaseContext

log = logging.getLogger(__name__)


def _chunks_model_paths(db_path: Path) -> tuple[Path, Path]:
    """Return (2d_path, 3d_path) for chunk UMAP joblib model files."""
    stem = db_path.stem
    parent = db_path.parent
    return parent / f"{stem}_chunks_umap2d.joblib", parent / f"{stem}_chunks_umap3d.joblib"


def _entities_model_paths(db_path: Path) -> tuple[Path, Path]:
    """Return (2d_path, 3d_path) for entity UMAP joblib model files."""
    stem = db_path.stem
    parent = db_path.parent
    return parent / f"{stem}_entities_umap2d.joblib", parent / f"{stem}_entities_umap3d.joblib"


class PhaseChunksUMAP(Phase):
    """Compute UMAP 2D + 3D projections for chunk vectors.

    Depends on: embeddings (chunks_vec_nodes)
    Produces:   chunks_vec_umap
    """

    @property
    def name(self) -> str:
        return "chunks_vec_umap"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        """Return True if any chunk vectors lack UMAP coordinates."""
        try:
            chunk_umap: int = conn.execute("SELECT count(*) FROM chunks_vec_umap").fetchone()[0]
            chunk_vecs: int = conn.execute("SELECT count(*) FROM chunks_vec_nodes").fetchone()[0]
            return chunk_umap != chunk_vecs
        except sqlite3.OperationalError:
            return True

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        umap2d_path, _ = _chunks_model_paths(ctx.db_path)
        if umap2d_path.exists():
            conn.execute(
                "CREATE TABLE IF NOT EXISTS chunks_vec_umap"
                " (id INTEGER PRIMARY KEY, x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL)"
            )
            log.info("  Chunk UMAP model found — will use transform() for new vectors")
        else:
            conn.execute("DROP TABLE IF EXISTS chunks_vec_umap")
            conn.execute(
                "CREATE TABLE chunks_vec_umap"
                " (id INTEGER PRIMARY KEY, x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL)"
            )
            log.info("  No chunk UMAP model — will run full fit_transform")

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        umap2d_path, umap3d_path = _chunks_model_paths(ctx.db_path)

        all_rows = conn.execute("SELECT id, vector FROM chunks_vec_nodes ORDER BY id").fetchall()
        if not all_rows:
            log.warning("  No chunk vectors found — skipping chunk UMAP")
            return

        if umap2d_path.exists() and umap3d_path.exists():
            self._run_incremental(conn, umap2d_path, umap3d_path, all_rows)
        else:
            self._run_full_fit(conn, umap2d_path, umap3d_path, all_rows)

    def _run_full_fit(
        self,
        conn: sqlite3.Connection,
        umap2d_path: Path,
        umap3d_path: Path,
        all_rows: list[Any],
    ) -> None:
        ids = [r[0] for r in all_rows]
        vecs = np.stack([np.frombuffer(bytes(r[1]), dtype=np.float32) for r in all_rows])
        log.info("  Chunk UMAP full fit: %d vectors (dim=%d)", len(ids), vecs.shape[1])

        log.info("  Computing 2D UMAP...")
        reducer_2d = umap.UMAP(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
        proj_2d = reducer_2d.fit_transform(vecs)

        log.info("  Computing 3D UMAP...")
        reducer_3d = umap.UMAP(n_components=3, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
        proj_3d = reducer_3d.fit_transform(vecs)

        joblib.dump(reducer_2d, umap2d_path)
        joblib.dump(reducer_3d, umap3d_path)
        log.info("  Saved chunk UMAP models: %s, %s", umap2d_path.name, umap3d_path.name)

        conn.executemany(
            "INSERT INTO chunks_vec_umap (id, x2d, y2d, x3d, y3d, z3d) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    ids[i],
                    float(proj_2d[i, 0]),
                    float(proj_2d[i, 1]),
                    float(proj_3d[i, 0]),
                    float(proj_3d[i, 1]),
                    float(proj_3d[i, 2]),
                )
                for i in range(len(ids))
            ],
        )
        log.info("  Stored UMAP projections for %d chunks", len(ids))

    def _run_incremental(
        self,
        conn: sqlite3.Connection,
        umap2d_path: Path,
        umap3d_path: Path,
        all_rows: list[Any],
    ) -> None:
        reducer_2d = joblib.load(umap2d_path)
        reducer_3d = joblib.load(umap3d_path)

        existing_ids = {r[0] for r in conn.execute("SELECT id FROM chunks_vec_umap").fetchall()}
        new_rows = [(r[0], r[1]) for r in all_rows if r[0] not in existing_ids]

        if not new_rows:
            log.info("  Chunk UMAP: all vectors already projected")
            return

        new_vecs = np.stack([np.frombuffer(bytes(r[1]), dtype=np.float32) for r in new_rows])
        log.info("  Chunk UMAP transform: %d new vectors", len(new_rows))
        new_2d = reducer_2d.transform(new_vecs)
        new_3d = reducer_3d.transform(new_vecs)
        conn.executemany(
            "INSERT INTO chunks_vec_umap (id, x2d, y2d, x3d, y3d, z3d) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    new_rows[i][0],
                    float(new_2d[i, 0]),
                    float(new_2d[i, 1]),
                    float(new_3d[i, 0]),
                    float(new_3d[i, 1]),
                    float(new_3d[i, 2]),
                )
                for i in range(len(new_rows))
            ],
        )
        log.info("  Chunk UMAP incremental: +%d projections", len(new_rows))


class PhaseEntitiesUMAP(Phase):
    """Compute UMAP 2D + 3D projections for entity vectors.

    Depends on: entity_embeddings (entities_vec_nodes)
    Produces:   entities_vec_umap
    """

    @property
    def name(self) -> str:
        return "entities_vec_umap"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        """Return True if any entity vectors lack UMAP coordinates."""
        try:
            entity_umap: int = conn.execute("SELECT count(*) FROM entities_vec_umap").fetchone()[0]
            entity_vecs: int = conn.execute("SELECT count(*) FROM entity_vec_map").fetchone()[0]
            return entity_umap != entity_vecs
        except sqlite3.OperationalError:
            return True

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        umap2d_path, _ = _entities_model_paths(ctx.db_path)
        if umap2d_path.exists():
            conn.execute(
                "CREATE TABLE IF NOT EXISTS entities_vec_umap"
                " (id INTEGER PRIMARY KEY, x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL)"
            )
            log.info("  Entity UMAP model found — will use transform() for new vectors")
        else:
            conn.execute("DROP TABLE IF EXISTS entities_vec_umap")
            conn.execute(
                "CREATE TABLE entities_vec_umap"
                " (id INTEGER PRIMARY KEY, x2d REAL, y2d REAL, x3d REAL, y3d REAL, z3d REAL)"
            )
            log.info("  No entity UMAP model — will run full fit_transform")

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        umap2d_path, umap3d_path = _entities_model_paths(ctx.db_path)

        all_rows = conn.execute("SELECT id, vector FROM entities_vec_nodes ORDER BY id").fetchall()
        if not all_rows:
            log.warning("  No entity vectors found — skipping entity UMAP")
            return

        if umap2d_path.exists() and umap3d_path.exists():
            self._run_incremental(conn, umap2d_path, umap3d_path, all_rows)
        else:
            self._run_full_fit(conn, umap2d_path, umap3d_path, all_rows)

    def _run_full_fit(
        self,
        conn: sqlite3.Connection,
        umap2d_path: Path,
        umap3d_path: Path,
        all_rows: list[Any],
    ) -> None:
        ids = [r[0] for r in all_rows]
        vecs = np.stack([np.frombuffer(bytes(r[1]), dtype=np.float32) for r in all_rows])
        log.info("  Entity UMAP full fit: %d vectors (dim=%d)", len(ids), vecs.shape[1])

        log.info("  Computing 2D UMAP...")
        reducer_2d = umap.UMAP(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
        proj_2d = reducer_2d.fit_transform(vecs)

        log.info("  Computing 3D UMAP...")
        reducer_3d = umap.UMAP(n_components=3, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42)
        proj_3d = reducer_3d.fit_transform(vecs)

        joblib.dump(reducer_2d, umap2d_path)
        joblib.dump(reducer_3d, umap3d_path)
        log.info("  Saved entity UMAP models: %s, %s", umap2d_path.name, umap3d_path.name)

        conn.executemany(
            "INSERT INTO entities_vec_umap (id, x2d, y2d, x3d, y3d, z3d) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    ids[i],
                    float(proj_2d[i, 0]),
                    float(proj_2d[i, 1]),
                    float(proj_3d[i, 0]),
                    float(proj_3d[i, 1]),
                    float(proj_3d[i, 2]),
                )
                for i in range(len(ids))
            ],
        )
        log.info("  Stored UMAP projections for %d entities", len(ids))

    def _run_incremental(
        self,
        conn: sqlite3.Connection,
        umap2d_path: Path,
        umap3d_path: Path,
        all_rows: list[Any],
    ) -> None:
        reducer_2d = joblib.load(umap2d_path)
        reducer_3d = joblib.load(umap3d_path)

        existing_ids = {r[0] for r in conn.execute("SELECT id FROM entities_vec_umap").fetchall()}
        new_rows = [(r[0], r[1]) for r in all_rows if r[0] not in existing_ids]

        if not new_rows:
            log.info("  Entity UMAP: all vectors already projected")
            return

        new_vecs = np.stack([np.frombuffer(bytes(r[1]), dtype=np.float32) for r in new_rows])
        log.info("  Entity UMAP transform: %d new vectors", len(new_rows))
        new_2d = reducer_2d.transform(new_vecs)
        new_3d = reducer_3d.transform(new_vecs)
        conn.executemany(
            "INSERT INTO entities_vec_umap (id, x2d, y2d, x3d, y3d, z3d) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    new_rows[i][0],
                    float(new_2d[i, 0]),
                    float(new_2d[i, 1]),
                    float(new_3d[i, 0]),
                    float(new_3d[i, 1]),
                    float(new_3d[i, 2]),
                )
                for i in range(len(new_rows))
            ],
        )
        log.info("  Entity UMAP incremental: +%d projections", len(new_rows))
