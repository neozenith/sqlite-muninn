"""DAG dependency declarations and concurrent execution engine for sessions_demo.

Phase dependencies verified from SQL queries in each phase source file:

  ner reads `chunks`   (ner.py:114-117)    — NOT chunks_vec (no HNSW dependency)
  relations (gliner2)  (re.py:158-162)     — reads `chunks` only, NOT entities
  entity_resolution    (entity_resolution.py:50,66,80,237) — reads entities + entity_vec_map
                                                               + entities_vec + relations

True parallel fan-out from `chunks`:
  chunks_vec  ∥  ner  ∥  relations     (3-way parallel after chunks)
  entity_embeddings starts after ner
  entities_vec_umap starts after entity_embeddings
  entity_resolution starts after entity_embeddings + relations (join)
  node2vec starts after entity_resolution
  metadata joins: chunks_vec_umap + entities_vec_umap + node2vec
"""

from __future__ import annotations

import graphlib
import logging
import queue
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.sessions_demo.common import load_muninn

if TYPE_CHECKING:
    from benchmarks.sessions_demo.build import PhaseContext
    from benchmarks.sessions_demo.phases.base import Phase

log = logging.getLogger(__name__)


# Dependency graph: phase name → set of direct predecessor phase names.
# Each entry lists only DIRECT dependencies (transitive ones are implied).
PHASE_DEPS: dict[str, set[str]] = {
    "ingest": set(),
    "chunks": {"ingest"},
    "chunks_vec": {"chunks"},
    "chunks_vec_umap": {"chunks_vec"},
    "ner": {"chunks"},  # reads `chunks`, NOT chunks_vec
    "relations": {"chunks"},  # gliner2: reads `chunks` only, NOT entities
    "entity_embeddings": {"ner"},
    "entities_vec_umap": {"entity_embeddings"},
    "entity_resolution": {"entity_embeddings", "relations"},  # reads entities+entity_vec_map+entities_vec+relations
    "node2vec": {"entity_resolution"},
    "metadata": {"chunks_vec_umap", "entities_vec_umap", "node2vec"},
}


def _open_conn(db_path: Path) -> sqlite3.Connection:
    """Open a fresh WAL-mode connection with busy-retry enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _worker(
    db_path: Path,
    phase: Phase,
    ctx: PhaseContext,
    done_q: queue.Queue[tuple[str, BaseException | None]],
) -> None:
    """Execute a single phase in a worker thread with its own SQLite connection.

    muninn is loaded on every worker connection. It is required by any phase
    that creates or queries an HNSW virtual table (chunks_vec, entity_embeddings,
    entity_resolution, node2vec). Loading it on a connection that doesn't need it
    is harmless — the extension registration is idempotent.
    """
    conn = _open_conn(db_path)
    load_muninn(conn)
    try:
        phase(conn, ctx)
        conn.commit()
        done_q.put((phase.name, None))
    except BaseException as exc:
        done_q.put((phase.name, exc))
    finally:
        conn.close()


def run_concurrent(
    db_path: Path,
    phases: list[Phase],
    ctx: PhaseContext,
    workers: int = 4,
    record_phase_fn: Callable[[int, str], None] | None = None,
) -> None:
    """Execute phases respecting PHASE_DEPS, running independent phases in parallel.

    Uses graphlib.TopologicalSorter as a DAG state machine driven by a single
    coordinator thread (TopologicalSorter is not thread-safe). Worker threads
    communicate completion back via a Queue.

    Each worker opens its own SQLite connection. WAL mode serializes concurrent
    writes; phases that overlap in parallel pairs never write to the same tables
    so write contention is rare.

    record_phase_fn: optional callback(phase_num, phase_name) called by the
    coordinator after each phase completes — used to record _build_progress rows.

    Parallel pairs (from PHASE_DEPS):
      After chunks:           chunks_vec ∥ ner ∥ relations        (3-way)
      After ner:              entity_embeddings (serial — ner is sole dep)
      After entity_embeddings: entities_vec_umap ∥ entity_resolution (2-way, but
                               entity_resolution also waits for relations)
      After entity_resolution: node2vec
      After all leaf phases:  metadata
    """
    phase_map: dict[str, Phase] = {p.name: p for p in phases}
    phase_names: list[str] = [p.name for p in phases]

    # Restrict graph to phases present in this run (supports run_from slicing).
    graph: dict[str, set[str]] = {name: deps & set(phase_map) for name, deps in PHASE_DEPS.items() if name in phase_map}

    ts = graphlib.TopologicalSorter(graph)
    ts.prepare()

    done_q: queue.Queue[tuple[str, BaseException | None]] = queue.Queue()
    in_flight: set[str] = set()
    t_start = time.monotonic()
    t_phase_start: dict[str, float] = {}

    log.info("=" * 60)
    log.info("Concurrent build (workers=%d): %s", workers, db_path.name)
    log.info("=" * 60)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="phase") as pool:
        while ts.is_active():
            # Drain all phases whose predecessors are now done.
            for name in ts.get_ready():
                phase = phase_map[name]

                # is_stale() uses read-only queries — safe on a fresh connection.
                conn = _open_conn(db_path)
                try:
                    stale = phase.is_stale(conn)
                finally:
                    conn.close()

                if stale:
                    log.info(
                        "  [dag] submit   %-22s  (in-flight after: %d)",
                        name,
                        len(in_flight) + 1,
                    )
                    t_phase_start[name] = time.monotonic()
                    in_flight.add(name)
                    pool.submit(_worker, db_path, phase, ctx, done_q)
                else:
                    log.info("  [dag] skip     %-22s  (up to date)", name)
                    conn2 = _open_conn(db_path)
                    try:
                        phase.restore_ctx(conn2, ctx)
                    finally:
                        conn2.close()
                    ts.done(name)

            # If nothing is in-flight all ready phases were up-to-date and have
            # been ts.done()-ed; their dependents will appear in the next
            # get_ready() call on the next loop iteration.
            if not in_flight:
                continue

            # Block until one worker completes.
            name, exc = done_q.get()
            in_flight.discard(name)

            if exc is not None:
                raise RuntimeError(f"Phase '{name}' failed") from exc

            phase_elapsed = time.monotonic() - t_phase_start.get(name, t_start)
            total_elapsed = time.monotonic() - t_start
            log.info(
                "  [dag] done     %-22s  (phase=%.1fs  total=%.1fs)",
                name,
                phase_elapsed,
                total_elapsed,
            )

            if record_phase_fn is not None:
                phase_num = phase_names.index(name) + 1
                record_phase_fn(phase_num, name)

            ts.done(name)

    elapsed = time.monotonic() - t_start
    log.info("Concurrent build complete (%.1fs total)", elapsed)
