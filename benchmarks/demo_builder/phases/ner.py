"""Phase 2: Entity Extraction (GLiNER2 or GLiNER zero-shot NER).

Incremental: tracks processed chunks in `ner_chunks_log`. Each run only
processes chunks that don't yet have a ner_chunks_log entry, so re-runs
after new ingest only process the new chunks.

Backends:
  "gliner2" (default) — fastino/gliner2-base-v1, 205M DeBERTa-v3-base,
      loaded via the shared _gliner2_loader cache (one load per process).
  "gliner"  (legacy)  — urchade/gliner_medium-v2.1, requires gliner package.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from gliner import GLiNER
from huggingface_hub import snapshot_download

from benchmarks.demo_builder.common import ProgressTracker, offline_mode
from benchmarks.demo_builder.constants import GLINER2_NER_LABELS, GLINER_LABELS
from benchmarks.demo_builder.phases._gliner2_loader import get_gliner2
from benchmarks.demo_builder.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext

log = logging.getLogger(__name__)

# Outer loop batch size for write-lock release between batches.
# NER holds the SQLite write lock while inserting entities; batching at 32
# gives chunks_embeddings windows to acquire the lock (~4-5s per batch).
_NER_OUTER_BATCH = 32

# Internal batch size passed to GLiNER2.batch_extract_entities().
# DeBERTa-v3-base: 8 chunks at ~1400 chars each ≈ manageable peak memory.
_GLINER2_BATCH_SIZE = 8


class PhaseNER(Phase):
    """Extract entities from all unprocessed chunks using NER model."""

    def __init__(self, labels: list[str] | None = None, backend: str = "gliner2") -> None:
        self._model: Any = None
        self._backend = backend
        if labels is not None:
            self._labels = labels
        elif backend == "gliner2":
            self._labels = GLINER2_NER_LABELS
        else:
            self._labels = GLINER_LABELS

    @property
    def name(self) -> str:
        return "ner"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        """Return True if any chunks have not yet been NER-processed."""
        try:
            unprocessed: int = conn.execute("""
                SELECT count(*) FROM chunks c
                WHERE c.chunk_id NOT IN (SELECT chunk_id FROM ner_chunks_log)
            """).fetchone()[0]
            return unprocessed > 0
        except sqlite3.OperationalError:
            return True  # ner_chunks_log doesn't exist yet → never run

    def restore_ctx(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        try:
            ctx.num_entity_mentions = conn.execute("SELECT count(*) FROM entities").fetchone()[0]
        except sqlite3.OperationalError:
            pass

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        # IF NOT EXISTS so re-runs after crash don't wipe partial progress.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                entity_type TEXT,
                source TEXT NOT NULL,
                chunk_id INTEGER REFERENCES chunks(chunk_id),
                confidence REAL DEFAULT 1.0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_chunk ON entities(chunk_id)")

        # ner_chunks_log: one row per processed chunk (even if no entities found).
        # Enables incremental re-runs: only unlogged chunks are processed.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ner_chunks_log (
                chunk_id INTEGER PRIMARY KEY,
                processed_at TEXT NOT NULL
            )
        """)

        if self._backend == "gliner2":
            log.info("  Loading GLiNER2 (fastino/gliner2-base-v1)...")
            self._model = get_gliner2()
        else:
            log.info("  Loading GLiNER medium-v2.1...")
            path = snapshot_download("urchade/gliner_medium-v2.1", local_files_only=True)
            with offline_mode():
                self._model = GLiNER.from_pretrained(path, local_files_only=True)

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        assert self._model is not None, "setup() must be called before run()"

        # Only process chunks that haven't been logged yet.
        chunks = conn.execute("""
            SELECT chunk_id, text FROM chunks
            WHERE chunk_id NOT IN (SELECT chunk_id FROM ner_chunks_log)
            ORDER BY chunk_id
        """).fetchall()

        if not chunks:
            log.info("  All chunks already NER-processed")
            ctx.num_entity_mentions = conn.execute("SELECT count(*) FROM entities").fetchone()[0]
            return

        log.info("  NER processing %d unprocessed chunks [%s] (labels: %s)", len(chunks), self._backend, self._labels)

        total_entities = 0
        ts = datetime.now(UTC).isoformat()
        tracker = ProgressTracker(len(chunks))

        for batch_start in range(0, len(chunks), _NER_OUTER_BATCH):
            batch = chunks[batch_start : batch_start + _NER_OUTER_BATCH]
            texts = [text for _, text in batch]
            chunk_ids = [cid for cid, _ in batch]

            if self._backend == "gliner2":
                results = self._model.batch_extract_entities(
                    texts, self._labels, batch_size=_GLINER2_BATCH_SIZE, include_confidence=True
                )
                insert_rows = []
                for chunk_id, result in zip(chunk_ids, results, strict=True):
                    for label, ents in result.get("entities", {}).items():
                        for ent in ents:
                            insert_rows.append((ent["text"], label, "gliner2", chunk_id, ent.get("confidence", 1.0)))
            else:
                results = self._model.batch_predict_entities(texts, self._labels, threshold=0.3)
                insert_rows = []
                for chunk_id, entities in zip(chunk_ids, results, strict=True):
                    for ent in entities:
                        insert_rows.append((ent["text"], ent["label"], "gliner", chunk_id, ent["score"]))

            if insert_rows:
                conn.executemany(
                    "INSERT INTO entities (name, entity_type, source, chunk_id, confidence) VALUES (?, ?, ?, ?, ?)",
                    insert_rows,
                )
            total_entities += len(insert_rows)

            # Mark all chunks in this batch as processed (even if no entities found).
            conn.executemany(
                "INSERT OR IGNORE INTO ner_chunks_log (chunk_id, processed_at) VALUES (?, ?)",
                [(cid, ts) for cid in chunk_ids],
            )

            # Commit after each batch to release the write lock between batches.
            # Required for parallel builds: ner and chunks_embeddings both write
            # to the same staging DB; per-batch commits give chunks_embeddings
            # windows to acquire the lock (~4-5s per NER batch at 32 chunks).
            conn.commit()

            tracker.update(len(batch))
            tracker.record_output(len(insert_rows))
            if tracker.should_log():
                log.info("  NER %s", tracker.report())

        log.info("  Extracted %d entity mentions from %d chunks", total_entities, len(chunks))

        ctx.num_entity_mentions = conn.execute("SELECT count(*) FROM entities").fetchone()[0]
