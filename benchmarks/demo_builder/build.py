"""DemoBuild: lifecycle manager for building a single demo database.

Handles staging directories, hierarchical logging (debug/info/error log files),
phase orchestration, atomic move on success, and staging preservation on failure.

Public API follows a setup/run/teardown lifecycle:
    build = DemoBuild(book_id, model_name, output_folder)
    build.setup()
    try:
        build.run()
    finally:
        build.teardown()
"""

from __future__ import annotations

import datetime
import logging
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.demo_builder.common import _fmt_elapsed, load_muninn
from benchmarks.demo_builder.phases import Phase, default_phases


@dataclass
class PhaseContext:
    """Mutable bag carrying inter-phase data.

    Replaces the chain of return values from the old monolithic build function.
    Each phase writes its outputs here; later phases read what they need.
    """

    db_path: Path = field(default_factory=Path)
    num_chunks: int = 0
    num_entity_mentions: int = 0
    num_relations: int = 0
    num_unique_entities: int = 0
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
        legacy_models: bool = False,
        muninn_model: str | None = None,
    ) -> None:
        self._book_id = book_id
        self._model_name = model_name
        self._output_folder = output_folder
        self._legacy_models = legacy_models
        self._muninn_model = muninn_model
        self._ctx = PhaseContext()
        self._conn: sqlite3.Connection | None = None
        # Attach file handlers to the package-level logger so that all
        # module loggers (phases, common, etc.) propagate to our files.
        self._pkg_logger = logging.getLogger("benchmarks.demo_builder")
        self._log = logging.getLogger(f"benchmarks.demo_builder.build.{self.perm_id}")
        self._file_handlers: list[logging.FileHandler] = []
        self._build_succeeded = False

    @property
    def _db(self) -> sqlite3.Connection:
        """Return the open database connection; raises if not yet opened."""
        assert self._conn is not None, "Database not opened — call setup() first"
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

    def setup(self) -> None:
        """Create staging dir, set up hierarchical log files, open DB."""
        self._setup_staging()
        self._setup_logging()
        self._open_db()

    def run(self, phases: list[Phase] | None = None) -> None:
        """Execute phases sequentially. Default: default_phases()."""
        if phases is None:
            phases = default_phases(
                self._book_id,
                self._model_name,
                legacy_models=self._legacy_models,
                muninn_model=self._muninn_model,
            )

        t_total = time.monotonic()

        self._log.info("=" * 60)
        self._log.info("Building %s", self.perm_id)
        self._log.info("  Book: %d | Model: %s", self._book_id, self._model_name)
        self._log.info("  Staging: %s", self.staging_db_path)
        self._log.info("=" * 60)

        for i, phase in enumerate(phases, 1):
            self._log.info("Phase %d/%d: %s", i, len(phases), phase.name)
            t0 = time.monotonic()
            phase(self._db, self._ctx)
            self._db.commit()
            self._record_phase(i, phase.name)
            self._db.commit()
            self._log.info("Phase %d complete (%s)", i, _fmt_elapsed(time.monotonic() - t0))

        elapsed = time.monotonic() - t_total
        self._log.info("All phases complete (%s total)", _fmt_elapsed(elapsed))
        self._build_succeeded = True

    def teardown(self) -> None:
        """Finalize on success; preserve staging on failure; always close logs."""
        try:
            if self._build_succeeded:
                self._drop_build_progress()
                self._vacuum()
                self._atomic_move()
                self._cleanup_staging()
            else:
                self._log.error("Build FAILED -- staging preserved: %s", self.staging_dir)
                if self._conn is not None:
                    self._conn.close()
                    self._conn = None
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
        capture output from all module loggers (phases, common, etc.)
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
        self._conn.execute("PRAGMA busy_timeout = 120000")
        load_muninn(self._conn)
        self._ctx.db_path = self.staging_db_path
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

    def _record_phase(self, phase_num: int, phase_name: str) -> None:
        """Record completion of a build phase (1-indexed)."""
        self._db.execute(
            "INSERT INTO _build_progress (phase, name, completed_at) VALUES (?, ?, ?)",
            (phase_num, phase_name, datetime.datetime.now(datetime.UTC).isoformat()),
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
        """Move the built DB and any joblib model files from staging to final output."""
        self._output_folder.mkdir(parents=True, exist_ok=True)
        shutil.move(str(self.staging_db_path), str(self.final_path))

        # Move any UMAP joblib model files produced alongside the DB.
        for joblib_file in self.staging_dir.glob("*.joblib"):
            dest = self._output_folder / joblib_file.name
            shutil.move(str(joblib_file), str(dest))

        db_size = self.final_path.stat().st_size
        self._log.info("Done! %s (%.1f MB)", self.final_path.name, db_size / 1e6)


# ── Standalone status query ──────────────────────────────────────


def get_build_status(db_path: Path) -> dict[str, Any]:
    """Return a status dict for the given demo database.

    Works on both staging (in-progress) and final (complete) databases.
    Does NOT require the muninn extension or ML dependencies — only standard
    SQLite queries against regular tables and HNSW shadow tables.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        status: dict[str, Any] = {}
        status["db_path"] = str(db_path)
        status["db_size_bytes"] = db_path.stat().st_size

        # ── Phase progress from _build_progress ──────────────────
        # This table exists during build and is dropped after successful finalization.
        # Missing table → build completed (all phases done).
        try:
            rows = conn.execute("SELECT phase, name, completed_at FROM _build_progress ORDER BY phase").fetchall()
            status["completed_phases"] = {name: (phase, ts) for phase, name, ts in rows}
            status["build_finalized"] = False
        except sqlite3.OperationalError:
            status["completed_phases"] = {}
            status["build_finalized"] = True

        # ── Table row counts ─────────────────────────────────────
        def _count(table: str) -> int:
            try:
                return int(conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
            except sqlite3.OperationalError:
                return 0

        status["chunks"] = _count("chunks")
        status["chunks_embedded"] = _count("chunks_vec_nodes")
        status["chunks_umap"] = _count("chunks_vec_umap")
        status["entities"] = _count("entities")
        status["relations"] = _count("relations")
        status["entities_embedded"] = _count("entities_vec_nodes")
        status["entities_umap"] = _count("entities_vec_umap")
        status["entity_clusters"] = _count("entity_clusters")
        status["nodes"] = _count("nodes")
        status["edges"] = _count("edges")
        status["n2v_embeddings"] = _count("node2vec_emb_nodes")
        status["meta"] = _count("meta")

        # NER/RE processing logs (incremental tracking tables)
        status["ner_logged"] = _count("ner_chunks_log")
        status["re_logged"] = _count("re_chunks_log")

        # ── Per-phase done/pending counts ────────────────────────
        ch = status["chunks"]
        emb = status["chunks_embedded"]
        ent = status["entities"]
        ent_emb = status["entities_embedded"]
        nodes = status["nodes"]

        status["phase_counts"] = {
            "chunks": (ch, 0),
            "chunks_embeddings": (emb, max(0, ch - emb)),
            "chunks_umap": (status["chunks_umap"], max(0, emb - status["chunks_umap"])),
            "ner": (status["ner_logged"] or ent, max(0, ch - (status["ner_logged"] or ch))),
            "relations": (status["re_logged"] or status["relations"], max(0, ch - (status["re_logged"] or ch))),
            "entity_embeddings": (ent_emb, max(0, ent - ent_emb)),
            "entities_umap": (status["entities_umap"], max(0, ent_emb - status["entities_umap"])),
            "entity_resolution": (status["entity_clusters"], max(0, ent - status["entity_clusters"])),
            "node2vec": (status["n2v_embeddings"], max(0, nodes - status["n2v_embeddings"])),
            "metadata": (status["meta"], 0 if status["meta"] > 0 else 1),
        }

        return status
    finally:
        conn.close()
