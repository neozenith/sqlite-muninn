"""Phase 2: Entity Extraction (GLiNER zero-shot NER).

Incremental: tracks processed chunks in `ner_chunks_log`. Each run only
processes chunks that don't yet have a ner_chunks_log entry, so re-runs
after new ingest only process the new chunks.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from gliner import GLiNER
from huggingface_hub import snapshot_download

from benchmarks.demo_builder.common import ProgressTracker, offline_mode
from benchmarks.demo_builder.constants import GLINER_LABELS
from benchmarks.demo_builder.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext

log = logging.getLogger(__name__)


class PhaseNER(Phase):
    """Extract entities from all unprocessed chunks using GLiNER zero-shot NER."""

    def __init__(self, labels: list[str] | None = None) -> None:
        self._ner_model: GLiNER | None = None
        self._labels = labels if labels is not None else GLINER_LABELS

    @property
    def name(self) -> str:
        return "ner"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        """Return True if any chunks have not yet been NER-processed."""
        try:
            unprocessed = conn.execute("""
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

        log.info("  Loading GLiNER medium-v2.1...")
        path = snapshot_download("urchade/gliner_medium-v2.1", local_files_only=True)
        with offline_mode():
            self._ner_model = GLiNER.from_pretrained(path, local_files_only=True)

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        assert self._ner_model is not None, "setup() must be called before run()"

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

        log.info("  NER processing %d unprocessed chunks (labels: %s)", len(chunks), self._labels)

        batch_size = 32
        total_entities = 0
        ts = datetime.now(UTC).isoformat()
        tracker = ProgressTracker(len(chunks))

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            texts = [text for _, text in batch]
            chunk_ids = [cid for cid, _ in batch]

            results = self._ner_model.batch_predict_entities(texts, self._labels, threshold=0.3)

            insert_rows = []
            for chunk_id, entities in zip(chunk_ids, results, strict=True):
                for ent in entities:
                    insert_rows.append(
                        (
                            ent["text"],
                            ent["label"],
                            "gliner",
                            chunk_id,
                            ent["score"],
                        )
                    )

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

            tracker.update(len(batch))
            if tracker.should_log():
                log.info("  NER %s  (%d entities so far)", tracker.report(), total_entities)

        log.info("  Extracted %d entity mentions from %d chunks", total_entities, len(chunks))

        ctx.num_entity_mentions = conn.execute("SELECT count(*) FROM entities").fetchone()[0]
