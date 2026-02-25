"""Phase 2: Entity Extraction (GLiNER zero-shot NER)."""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

from gliner import GLiNER

from benchmarks.demo_builder.constants import GLINER_LABELS
from benchmarks.demo_builder.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext

log = logging.getLogger(__name__)


class PhaseNER(Phase):
    """Extract entities from all chunks using GLiNER zero-shot NER."""

    def __init__(self) -> None:
        self._ner_model: GLiNER | None = None

    @property
    def name(self) -> str:
        return "ner"

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        conn.execute("""
            CREATE TABLE entities (
                entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                entity_type TEXT,
                source TEXT NOT NULL,
                chunk_id INTEGER REFERENCES chunks(chunk_id),
                confidence REAL DEFAULT 1.0
            )
        """)
        conn.execute("CREATE INDEX idx_entities_name ON entities(name)")
        conn.execute("CREATE INDEX idx_entities_chunk ON entities(chunk_id)")

        log.info("  Loading GLiNER medium-v2.1...")
        self._ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        assert self._ner_model is not None, "setup() must be called before run()"

        chunks = conn.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
        batch_size = 32
        total_entities = 0

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            texts = [text for _, text in batch]
            chunk_ids = [cid for cid, _ in batch]

            results = self._ner_model.batch_predict_entities(texts, GLINER_LABELS, threshold=0.3)

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

            conn.executemany(
                "INSERT INTO entities (name, entity_type, source, chunk_id, confidence) VALUES (?, ?, ?, ?, ?)",
                insert_rows,
            )
            total_entities += len(insert_rows)

            if (batch_start // batch_size) % 10 == 0:
                log.info(
                    "  Processed %d/%d chunks (%d entities so far)",
                    min(batch_start + batch_size, len(chunks)),
                    len(chunks),
                    total_entities,
                )

        log.info("  Extracted %d entity mentions across %d chunks", total_entities, len(chunks))

        ctx.num_entity_mentions = total_entities
