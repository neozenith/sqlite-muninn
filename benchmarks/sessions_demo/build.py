"""SessionsBuild: lifecycle manager for building the sessions demo database.

Handles schema initialization, phase orchestration, and cleanup.

Public API follows a setup/run/teardown lifecycle:
    build = SessionsBuild(db_path)
    build.setup()
    try:
        build.run()
    finally:
        build.teardown()

## Incremental KG pipeline

Pre-KG phases (ingest, chunks, chunks_vec) are always executed — they are
already individually incremental: ingest picks up new JSONL files, chunks
splits new events only, chunks_vec skips already-embedded chunks.

KG phases (ner → metadata) are NOT individually incremental. They are skipped
when all of the following are true:
  - All KG phase names are recorded in _build_progress (previous run succeeded)
  - No new chunks were embedded in this run (ctx.chunks_embedded == 0)

When KG phases need to re-run (new chunks OR incomplete previous run), ALL KG
output tables are dropped first to ensure a clean rebuild from the full chunk
set. This avoids partial-state corruption from crashed mid-run phases.

_build_progress is kept permanently (not dropped at the end of a successful
build) so the skip decision survives across restarts.
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.demo_builder.common import _fmt_elapsed
from benchmarks.sessions_demo.phases import Phase, default_phases

log = logging.getLogger(__name__)

# Names of KG pipeline phases (everything after chunks_vec).
# Used to split the phase list into "always run" vs "conditional" groups.
_KG_PHASE_NAMES: frozenset[str] = frozenset(
    [
        "ner",
        "relations",
        "entity_embeddings",
        "chunks_vec_umap",
        "entities_vec_umap",
        "entity_resolution",
        "node2vec",
        "metadata",
    ]
)

# All tables created by KG phases. Dropped before a KG rebuild so each run
# starts from a clean slate. HNSW virtual tables (entities_vec, node2vec_emb)
# are included — DROP TABLE IF EXISTS works on virtual tables too and cascades
# to their shadow tables via xDestroy.
_KG_TABLES: list[str] = [
    "meta",
    "entities",
    "relations",
    "entity_vec_map",
    "entities_vec",
    "entity_clusters",
    "nodes",
    "edges",
    "chunks_vec_umap",
    "entities_vec_umap",
    "node2vec_emb",
    "_match_edges",
]


@dataclass
class PhaseContext:
    """Mutable bag carrying inter-phase data.

    Each phase writes its outputs here; later phases read what they need.
    Fields prefixed with num_ mirror demo_builder.PhaseContext so that
    demo_builder phases can be imported and run against this context via
    Python's structural (duck) typing.
    """

    db_path: Path = Path("")

    # Phase 1: ingest
    events_ingested: int = 0
    files_updated: int = 0

    # Phase 2: chunks
    chunks_created: int = 0

    # Phase 3: chunks_vec
    chunks_embedded: int = 0

    # KG pipeline fields (match demo_builder.PhaseContext names so imported
    # phases write into the correct attributes without modification)
    num_chunks: int = 0  # total chunks in DB, updated before KG run
    num_entity_mentions: int = 0  # set by PhaseNER
    num_relations: int = 0  # set by PhaseRE
    num_unique_entities: int = 0  # set by PhaseEntityEmbeddings
    entity_vectors: np.ndarray | None = field(default=None, repr=False)  # set by PhaseEntityEmbeddings
    num_nodes: int = 0  # set by PhaseEntityResolution
    num_edges: int = 0  # set by PhaseEntityResolution
    num_n2v: int = 0  # set by PhaseNode2Vec


class SessionsBuild:
    """Manages the lifecycle of building the sessions demo database.

    Unlike DemoBuild (which uses staging dirs and atomic moves), this builder
    operates in-place on the target DB since the cache is designed for
    incremental updates.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._ctx = PhaseContext(db_path=db_path)
        self._conn: sqlite3.Connection | None = None
        self._log = logging.getLogger("benchmarks.sessions_demo.build")

    @property
    def _db(self) -> sqlite3.Connection:
        """Return the open database connection; raises if not yet opened."""
        assert self._conn is not None, "Database not opened — call setup() first"
        return self._conn

    # ── Public API ────────────────────────────────────────────────

    def setup(self) -> None:
        """Create output directory and open DB."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_build_progress()
        self._log.info("Database opened: %s", self._db_path)

    def run(
        self,
        phases: list[Phase] | None = None,
        run_from: str | None = None,
        message_types: list[str] | None = None,
        legacy_models: bool = False,
    ) -> None:
        """Execute phases using per-phase staleness checks.

        Each phase decides independently whether it has pending work via
        is_stale(conn). Stale phases run; up-to-date phases restore ctx fields
        from the DB and are skipped.

        run_from: force-skip all phases before PHASE, restoring their ctx
        from the DB. Normal staleness logic applies from PHASE onward.

        muninn is loaded once upfront so HNSW phases can create/query virtual
        tables even when chunks_vec is skipped (all chunks already embedded).
        """
        if phases is None:
            phases = default_phases(message_types=message_types, legacy_models=legacy_models)

        # Force-skip phases before run_from: restore ctx, do not execute.
        start_idx = 0
        if run_from is not None:
            names = [p.name for p in phases]
            if run_from not in names:
                raise ValueError(f"Unknown phase {run_from!r}. Valid phase names: {', '.join(names)}")
            start_idx = names.index(run_from)
            self._log.info("--run-from %s: force-skipping phases 1-%d", run_from, start_idx)
            skipped = phases[:start_idx]
            phases = phases[start_idx:]
            for phase in skipped:
                phase.restore_ctx(self._db, self._ctx)

        self._run_sequential(phases)

        # Ensure num_chunks reflects total in DB — metadata reads it from ctx.
        try:
            self._ctx.num_chunks = self._db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        except sqlite3.OperationalError:
            pass

    def _run_sequential(self, phases: list[Phase]) -> None:
        """Sequential phase execution on the shared DB connection."""
        total = len(phases)
        t_total = time.monotonic()

        self._log.info("=" * 60)
        self._log.info("Sequential build: %s", self._db_path.name)
        self._log.info("=" * 60)

        # Load muninn once for the shared connection.
        from benchmarks.demo_builder.common import load_muninn

        load_muninn(self._db)

        for i, phase in enumerate(phases, 1):
            t0 = time.monotonic()
            stale = phase.is_stale(self._db)

            if not stale:
                self._log.info("Phase %d/%d: %s  (up to date, skipping)", i, total, phase.name)
                phase.restore_ctx(self._db, self._ctx)
                continue

            self._log.info("Phase %d/%d: %s", i, total, phase.name)
            phase(self._db, self._ctx)
            self._db.commit()
            self._record_phase(i, phase.name)
            self._db.commit()
            self._log.info("Phase %d complete (%s)", i, _fmt_elapsed(time.monotonic() - t0))

        self._log.info("Sequential build complete (%s total)", _fmt_elapsed(time.monotonic() - t_total))

    def run_single_phase(
        self,
        phase_name: str,
        message_types: list[str] | None = None,
        legacy_models: bool = False,
    ) -> None:
        """Run a single named phase, restoring ctx from DB for all preceding phases.

        Equivalent to --run-from PHASE followed by stopping after that one phase.
        Useful for targeted re-runs, debugging, and standalone phase testing:

            uv run -m benchmarks.sessions_demo run-phase ner
            uv run -m benchmarks.sessions_demo run-phase entity_resolution

        All phases before PHASE have their restore_ctx() called so that ctx fields
        (num_chunks, num_entity_mentions, etc.) are correct for the target phase.

        muninn is loaded upfront since any HNSW phase needs it registered.
        """
        phases = default_phases(message_types=message_types, legacy_models=legacy_models)
        names = [p.name for p in phases]
        if phase_name not in names:
            raise ValueError(f"Unknown phase {phase_name!r}. Valid phase names: {', '.join(names)}")

        target_idx = names.index(phase_name)
        target = phases[target_idx]

        # muninn must be registered before any HNSW phase creates or queries a VT.
        from benchmarks.demo_builder.common import load_muninn

        load_muninn(self._db)

        # Restore ctx from DB for every phase that precedes the target.
        for phase in phases[:target_idx]:
            phase.restore_ctx(self._db, self._ctx)
            self._log.debug("  Restored ctx from phase: %s", phase.name)

        # Ensure num_chunks reflects the total in `chunks` — metadata reads it
        # directly from ctx and this fixup brings it in line with the DB count.
        try:
            self._ctx.num_chunks = self._db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        except Exception:
            pass

        t0 = time.monotonic()
        self._log.info("run-phase: %s", phase_name)
        target(self._db, self._ctx)
        self._db.commit()
        self._record_phase(target_idx + 1, phase_name)
        self._db.commit()
        self._log.info("Phase %s complete (%s)", phase_name, _fmt_elapsed(time.monotonic() - t0))

    def teardown(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._log.info("Database closed")

    # ── Status reporting ─────────────────────────────────────────

    def get_build_status(self) -> dict[str, Any]:
        """Return a status dict describing current build state and pending work.

        Safe to call on a partially-built or empty DB. Requires setup() to have
        been called (DB connection open).
        """
        from benchmarks.sessions_demo.cache import CacheManager

        db = self._db
        status: dict[str, Any] = {}

        # ── DB file info ──────────────────────────────────────────
        status["db_path"] = str(self._db_path)
        status["db_size_bytes"] = self._db_path.stat().st_size if self._db_path.exists() else 0

        # ── Phase progress ────────────────────────────────────────
        try:
            progress_rows = db.execute(
                "SELECT phase, name, completed_at FROM _build_progress ORDER BY phase"
            ).fetchall()
            status["completed_phases"] = {name: (phase, completed_at) for phase, name, completed_at in progress_rows}
        except sqlite3.OperationalError:
            status["completed_phases"] = {}

        # ── Current data counts ───────────────────────────────────
        def _count(table: str, fallback: int = 0) -> int:
            try:
                return int(db.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
            except sqlite3.OperationalError:
                return fallback

        status["events"] = _count("events")
        status["chunks"] = _count("chunks")
        status["chunks_embedded"] = _count("chunks_vec_nodes")
        status["entities"] = _count("entities")
        status["relations"] = _count("relations")
        status["nodes"] = _count("nodes")
        status["edges"] = _count("edges")
        status["n2v_embeddings"] = _count("node2vec_emb_nodes")

        # ── Pending work ──────────────────────────────────────────
        # JSONL files changed since last ingest.
        # CacheManager.get_files_needing_update() accesses results as cached["mtime"]
        # (dict-style), which requires row_factory=sqlite3.Row.
        try:
            db.row_factory = sqlite3.Row
            cache = CacheManager(self._db_path)
            cache._conn = db
            all_files = cache.discover_files()
            status["jsonl_files_changed"] = len(cache.get_files_needing_update(all_files))
            status["jsonl_files_total"] = len(all_files)
        except Exception:
            status["jsonl_files_changed"] = -1
            status["jsonl_files_total"] = -1

        # Chunks not yet embedded
        status["chunks_to_embed"] = max(0, status["chunks"] - status["chunks_embedded"])

        # Per-phase stale status via is_stale() (read-only queries, no model loading).
        from benchmarks.sessions_demo.phases import default_phases as _default_phases

        phase_stale: dict[str, bool] = {}
        for _phase in _default_phases():
            try:
                phase_stale[_phase.name] = _phase.is_stale(db)
            except Exception:
                phase_stale[_phase.name] = True
        status["phase_stale"] = phase_stale

        # KG rebuild needed: any KG phase reports stale
        status["kg_rebuild_needed"] = any(phase_stale.get(n, True) for n in _KG_PHASE_NAMES)
        status["kg_incomplete_phases"] = [n for n in _KG_PHASE_NAMES if phase_stale.get(n, True)]

        # ── Per-phase done/pending counts ─────────────────────────
        # done  = items this phase has already produced / processed
        # pending = upstream items not yet consumed by this phase
        def _count_q(sql: str) -> int:
            try:
                return int(db.execute(sql).fetchone()[0])
            except sqlite3.OperationalError:
                return 0

        ev = status["events"]
        ch = status["chunks"]
        emb = status["chunks_embedded"]
        ent = status["entities"]
        n2v = status["n2v_embeddings"]
        nodes = status["nodes"]

        ner_log = _count_q("SELECT COUNT(*) FROM ner_chunks_log")
        re_log = _count_q("SELECT COUNT(*) FROM re_chunks_log")
        umap_ch = _count_q("SELECT COUNT(*) FROM chunks_vec_umap")
        ev_vec = _count_q("SELECT COUNT(*) FROM entities_vec_nodes")
        umap_ent = _count_q("SELECT COUNT(*) FROM entities_vec_umap")
        ec = _count_q("SELECT COUNT(*) FROM entity_clusters")
        ev_chunked = _count_q("SELECT COUNT(DISTINCT event_id) FROM event_message_chunks")
        ev_pending_chunks = _count_q("""
            SELECT COUNT(*) FROM events e
            WHERE e.message_content IS NOT NULL AND e.message_content != ''
              AND e.id NOT IN (SELECT DISTINCT event_id FROM event_message_chunks)
        """)

        status["phase_counts"] = {
            "ingest": (ev, max(0, status.get("jsonl_files_changed", 0))),
            "chunks": (ev_chunked, ev_pending_chunks),
            "chunks_vec": (emb, max(0, ch - emb)),
            "chunks_vec_umap": (umap_ch, max(0, emb - umap_ch)),
            "ner": (ner_log, max(0, ch - ner_log)),
            "relations": (re_log, max(0, ch - re_log)),
            "entity_embeddings": (ev_vec, max(0, ent - ev_vec)),
            "entities_vec_umap": (umap_ent, max(0, ev_vec - umap_ent)),
            "entity_resolution": (ec, max(0, ent - ec)),
            "node2vec": (n2v, max(0, nodes - n2v)),
            "metadata": (_count_q("SELECT COUNT(*) FROM meta"), 0 if not phase_stale.get("metadata", True) else 1),
        }

        return status

    # ── Build progress tracking ──────────────────────────────────

    def _init_build_progress(self) -> None:
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS _build_progress (
                phase INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                completed_at TEXT NOT NULL
            )
        """)
        self._db.commit()

    def _record_phase(self, phase_num: int, phase_name: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO _build_progress (phase, name, completed_at) VALUES (?, ?, ?)",
            (phase_num, phase_name, datetime.datetime.now(datetime.UTC).isoformat()),
        )

    # ── KG phase management ──────────────────────────────────────

    def _kg_phases_complete(self, kg_phases: list[Phase]) -> bool:
        """Return True iff every KG phase name is recorded in _build_progress."""
        completed = {row[0] for row in self._db.execute("SELECT name FROM _build_progress").fetchall()}
        return all(p.name in completed for p in kg_phases)

    def _restore_kg_ctx(self) -> None:
        """Restore KG-related PhaseContext fields from existing DB tables.

        Called when KG phases are skipped so that downstream callers (e.g.
        write_manifest_json()) still have accurate counts.
        """
        db = self._db
        ctx = self._ctx
        ctx.num_chunks = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        ctx.num_entity_mentions = db.execute("SELECT count(*) FROM entities").fetchone()[0]
        ctx.num_relations = db.execute("SELECT count(*) FROM relations").fetchone()[0]
        ctx.num_unique_entities = db.execute("SELECT count(*) FROM entity_vec_map").fetchone()[0]
        ctx.num_nodes = db.execute("SELECT count(*) FROM nodes").fetchone()[0]
        ctx.num_edges = db.execute("SELECT count(*) FROM edges").fetchone()[0]
        ctx.num_n2v = db.execute("SELECT count(*) FROM node2vec_emb_nodes").fetchone()[0]

    def _drop_kg_tables(self) -> None:
        """Drop all KG output tables and UMAP model files for a clean rebuild.

        HNSW virtual tables (entities_vec, node2vec_emb) are included —
        DROP TABLE IF EXISTS cascades to their shadow tables via xDestroy.
        muninn must already be loaded (phase 3 always runs before this).
        UMAP joblib models are deleted so the next run does a full refit.
        """
        for table in _KG_TABLES:
            self._db.execute(f'DROP TABLE IF EXISTS "{table}"')
        self._db.commit()

        # Delete saved UMAP models so the next UMAP run does a full refit.
        from benchmarks.sessions_demo.phases.umap import _chunks_model_paths, _entities_model_paths

        all_model_paths = _chunks_model_paths(self._db_path) + _entities_model_paths(self._db_path)
        for model_path in all_model_paths:
            if model_path.exists():
                model_path.unlink()
                self._log.info("Deleted UMAP model: %s", model_path.name)

        self._log.info("Dropped KG output tables (clean rebuild)")

    def _clear_kg_progress(self, kg_phases: list[Phase]) -> None:
        """Remove KG phase records from _build_progress before a rebuild."""
        names = [p.name for p in kg_phases]
        placeholders = ",".join("?" * len(names))
        self._db.execute(f"DELETE FROM _build_progress WHERE name IN ({placeholders})", names)
        self._db.commit()
