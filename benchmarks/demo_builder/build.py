"""DemoBuild: lifecycle manager for building a single demo database.

Handles staging directories, hierarchical logging (debug/info/error log files),
phase orchestration, atomic move on success, and staging preservation on failure.
"""

from __future__ import annotations

import datetime
import logging
import shutil
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from benchmarks.demo_builder.common import load_muninn
from benchmarks.demo_builder.constants import PHASE_NAMES
from benchmarks.demo_builder.phases import (
    phase_1_chunks,
    phase_2_ner,
    phase_3_re,
    phase_4_entity_embeddings,
    phase_5_umap,
    phase_6_entity_resolution,
    phase_7_node2vec,
    phase_8_metadata,
)

if TYPE_CHECKING:
    from benchmarks.demo_builder.models import ModelPool


@dataclass
class PhaseContext:
    """Mutable bag carrying inter-phase data.

    Replaces the chain of return values from the old monolithic build function.
    Each phase writes its outputs here; later phases read what they need.
    """

    num_chunks: int = 0
    chunk_vectors: np.ndarray | None = field(default=None, repr=False)
    num_entity_mentions: int = 0
    num_relations: int = 0
    num_unique_entities: int = 0
    entity_vectors: np.ndarray | None = field(default=None, repr=False)
    num_nodes: int = 0
    num_edges: int = 0
    num_n2v: int = 0


class DemoBuild:
    """Manages the lifecycle of building a single demo database.

    Builds in a staging area at {output_folder}/_build/{perm_id}/ with
    hierarchical log files (debug/info/error). On success, atomically moves
    the DB to the final output path and cleans up staging. On failure,
    preserves staging for inspection.
    """

    def __init__(
        self,
        book_id: int,
        model_name: str,
        output_folder: Path,
        models: ModelPool,
    ) -> None:
        self._book_id = book_id
        self._model_name = model_name
        self._output_folder = output_folder
        self._models = models
        self._ctx = PhaseContext()
        self._conn: sqlite3.Connection | None = None
        # Attach file handlers to the package-level logger so that all
        # module loggers (phases, common, etc.) propagate to our files.
        self._pkg_logger = logging.getLogger("benchmarks.demo_builder")
        self._log = logging.getLogger(f"benchmarks.demo_builder.build.{self.perm_id}")
        self._file_handlers: list[logging.FileHandler] = []

    @property
    def _db(self) -> sqlite3.Connection:
        """Return the open database connection; raises if not yet opened."""
        assert self._conn is not None, "Database not opened — call _open_db() first"
        return self._conn

    @property
    def perm_id(self) -> str:
        return f"{self._book_id}_{self._model_name}"

    @property
    def staging_dir(self) -> Path:
        return self._output_folder / "_build" / self.perm_id

    @property
    def staging_db_path(self) -> Path:
        return self.staging_dir / f"{self.perm_id}.db"

    @property
    def final_path(self) -> Path:
        return self._output_folder / f"{self.perm_id}.db"

    # ── Public API ────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full build lifecycle."""
        self._setup_staging()
        self._setup_logging()
        try:
            self._open_db()
            self._run_phases()
            self._drop_build_progress()
            self._vacuum()
            self._atomic_move()
            self._cleanup_staging()
        except Exception:
            self._log.error("Build FAILED -- staging preserved: %s", self.staging_dir)
            if self._conn is not None:
                self._conn.close()
            raise
        finally:
            self._teardown_logging()

    # ── Staging lifecycle ─────────────────────────────────────────

    def _setup_staging(self) -> None:
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup_staging(self) -> None:
        """Remove staging dir (including logs) on success."""
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
        # Clean up _build/ dir if now empty
        build_dir = self._output_folder / "_build"
        if build_dir.exists() and not any(build_dir.iterdir()):
            build_dir.rmdir()

    # ── Hierarchical logging ──────────────────────────────────────

    def _setup_logging(self) -> None:
        """Create three log file handlers on the package-level logger.

        By attaching to 'benchmarks.demo_builder' (the package logger), we
        capture output from all module loggers (phases, common, models, etc.)
        via Python's logger hierarchy propagation.
        """
        self._pkg_logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

        levels = [
            (f"{self.perm_id}.debug.log", logging.DEBUG),
            (f"{self.perm_id}.info.log", logging.INFO),
            (f"{self.perm_id}.error.log", logging.ERROR),
        ]

        for filename, level in levels:
            handler = logging.FileHandler(self.staging_dir / filename)
            handler.setLevel(level)
            handler.setFormatter(fmt)
            self._pkg_logger.addHandler(handler)
            self._file_handlers.append(handler)

    def _teardown_logging(self) -> None:
        """Close and remove all file handlers."""
        for handler in self._file_handlers:
            handler.close()
            self._pkg_logger.removeHandler(handler)
        self._file_handlers.clear()

    # ── Database lifecycle ────────────────────────────────────────

    def _open_db(self) -> None:
        self._conn = sqlite3.connect(str(self.staging_db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        load_muninn(self._conn)
        self._init_build_progress()

    def _init_build_progress(self) -> None:
        """Create the _build_progress tracking table."""
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS _build_progress (
                phase INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                completed_at TEXT NOT NULL
            )
        """)
        self._db.commit()

    def _record_phase(self, phase_num: int) -> None:
        """Record completion of a build phase (1-indexed)."""
        self._db.execute(
            "INSERT INTO _build_progress (phase, name, completed_at) VALUES (?, ?, ?)",
            (phase_num, PHASE_NAMES[phase_num - 1], datetime.datetime.now(datetime.UTC).isoformat()),
        )

    def _drop_build_progress(self) -> None:
        """Drop internal bookkeeping table (not a deliverable)."""
        self._db.execute("DROP TABLE _build_progress")
        self._db.commit()

    def _vacuum(self) -> None:
        """VACUUM the database to reclaim space."""
        self._log.info("VACUUMing database...")
        self._db.execute("PRAGMA journal_mode=DELETE")
        self._db.execute("VACUUM")
        self._db.close()
        self._conn = None

    def _atomic_move(self) -> None:
        """Move the built DB from staging to final output path."""
        self._output_folder.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.staging_db_path), str(self.final_path))

        db_size = self.final_path.stat().st_size
        self._log.info("Done! %s (%.1f MB)", self.final_path.name, db_size / 1e6)

    # ── Phase orchestration ───────────────────────────────────────

    def _run_phases(self) -> None:
        """Execute all 8 build phases sequentially."""
        t_total = time.monotonic()

        self._log.info("=" * 60)
        self._log.info("Building %s", self.perm_id)
        self._log.info("  Book: %d | Model: %s", self._book_id, self._model_name)
        self._log.info("  Staging: %s", self.staging_db_path)
        self._log.info("=" * 60)

        phase_fns: list[Callable[[], None]] = [
            lambda: phase_1_chunks(self._db, self._ctx, self._book_id, self._model_name, self._models),
            lambda: phase_2_ner(self._db, self._ctx, self._models),
            lambda: phase_3_re(self._db, self._ctx, self._models),
            lambda: phase_4_entity_embeddings(self._db, self._ctx, self._model_name, self._models),
            lambda: phase_5_umap(self._db, self._ctx),
            lambda: phase_6_entity_resolution(self._db, self._ctx),
            lambda: phase_7_node2vec(self._db, self._ctx),
            lambda: phase_8_metadata(self._db, self._ctx, self._book_id, self._model_name),
        ]

        for i, fn in enumerate(phase_fns, 1):
            self._log.info("Phase %d/%d: %s", i, len(PHASE_NAMES), PHASE_NAMES[i - 1])
            t0 = time.monotonic()
            fn()
            self._db.commit()
            self._record_phase(i)
            self._db.commit()
            self._log.info("Phase %d complete (%.1fs)", i, time.monotonic() - t0)

        elapsed = time.monotonic() - t_total
        self._log.info("All phases complete (%.1fs total)", elapsed)
