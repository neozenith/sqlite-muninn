"""Phase: Entity Extraction (GLiNER2 or GLiNER zero-shot NER, or muninn LLM).

Incremental: tracks processed chunks in `ner_chunks_log`. Each run only
processes chunks that don't yet have a ner_chunks_log entry, so re-runs
after new ingest only process the new chunks.

Backends:
  "gliner2" (default) — fastino/gliner2-base-v1, 205M DeBERTa-v3-base,
      loaded via the shared _gliner2_loader cache (one load per process).
  "gliner"  (legacy)  — urchade/gliner_medium-v2.1, requires gliner package.
  "muninn"            — muninn_extract_ner_re() via llama.cpp GGUF chat model.
      Combined NER+RE in one LLM pass; also populates relations table.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from gliner import GLiNER
from huggingface_hub import snapshot_download

from benchmarks.sessions_demo.common import ProgressTracker, offline_mode
from benchmarks.sessions_demo.constants import (
    MUNINN_CHAT_MODEL_FILE,
    MUNINN_CHAT_MODEL_NAME,
    MUNINN_CHAT_MODELS_DIR,
    SESSION_GLINER2_NER_LABELS,
    SESSION_GLINER2_RE_LABELS,
    SESSION_NER_LABELS,
)
from benchmarks.sessions_demo.phases._gliner2_loader import get_gliner2
from benchmarks.sessions_demo.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.sessions_demo.build import PhaseContext

log = logging.getLogger(__name__)

# Outer loop batch size for write-lock release between batches.
_NER_OUTER_BATCH = 32

# Internal batch size passed to GLiNER2.batch_extract_entities().
_GLINER2_BATCH_SIZE = 8


class PhaseNER(Phase):
    """Extract entities from all unprocessed chunks using NER model."""

    def __init__(
        self,
        labels: list[str] | None = None,
        backend: str = "gliner2",
        relation_labels: list[str] | None = None,
        gguf_model: str | None = None,
    ) -> None:
        self._model: Any = None
        self._backend = backend
        self._gguf_model = gguf_model or MUNINN_CHAT_MODEL_FILE
        self._muninn_model_name = MUNINN_CHAT_MODEL_NAME
        if labels is not None:
            self._labels = labels
        elif backend == "gliner2":
            self._labels = SESSION_GLINER2_NER_LABELS
        elif backend == "muninn":
            self._labels = SESSION_GLINER2_NER_LABELS
        else:
            self._labels = SESSION_NER_LABELS
        # Relation labels used by muninn combined NER+RE
        self._relation_labels = relation_labels or SESSION_GLINER2_RE_LABELS

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
            return True

    def restore_ctx(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        try:
            ctx.num_entity_mentions = conn.execute("SELECT count(*) FROM entities").fetchone()[0]
        except sqlite3.OperationalError:
            pass

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ner_chunks_log (
                chunk_id INTEGER PRIMARY KEY,
                processed_at TEXT NOT NULL
            )
        """)

        if self._backend == "muninn":
            log.info("  Loading muninn chat model: %s", self._gguf_model)
            model_path = str(MUNINN_CHAT_MODELS_DIR / self._gguf_model)
            try:
                conn.execute(
                    "INSERT INTO temp.muninn_chat_models(name, model) SELECT ?, muninn_chat_model(?)",
                    (self._muninn_model_name, model_path),
                )
            except sqlite3.OperationalError as e:
                if "already loaded" not in str(e):
                    raise
                log.info("  Model %s already loaded", self._muninn_model_name)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    src TEXT NOT NULL,
                    dst TEXT NOT NULL,
                    rel_type TEXT,
                    weight REAL DEFAULT 1.0,
                    chunk_id INTEGER,
                    source TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_src ON relations(src)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_dst ON relations(dst)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS re_chunks_log (
                    chunk_id INTEGER PRIMARY KEY,
                    processed_at TEXT NOT NULL
                )
            """)
        elif self._backend == "gliner2":
            log.info("  Loading GLiNER2 (fastino/gliner2-base-v1)...")
            self._model = get_gliner2()
        else:
            log.info("  Loading GLiNER medium-v2.1...")
            path = snapshot_download("urchade/gliner_medium-v2.1", local_files_only=True)
            with offline_mode():
                self._model = GLiNER.from_pretrained(path, local_files_only=True)

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        if self._backend == "muninn":
            self._run_muninn(conn, ctx)
            return
        assert self._model is not None, "setup() must be called before run()"

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

            conn.executemany(
                "INSERT OR IGNORE INTO ner_chunks_log (chunk_id, processed_at) VALUES (?, ?)",
                [(cid, ts) for cid in chunk_ids],
            )
            conn.commit()

            tracker.update(len(batch))
            tracker.record_output(len(insert_rows))
            if tracker.should_log():
                log.info("  NER %s", tracker.report())

        log.info("  Extracted %d entity mentions from %d chunks", total_entities, len(chunks))
        ctx.num_entity_mentions = conn.execute("SELECT count(*) FROM entities").fetchone()[0]

    # ── muninn backend (combined NER+RE via llama.cpp) ────────────────────────

    def _run_muninn(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        """Extract entities AND relations in a single LLM call per chunk."""
        chunks = conn.execute("""
            SELECT chunk_id, text FROM chunks
            WHERE chunk_id NOT IN (SELECT chunk_id FROM ner_chunks_log)
            ORDER BY chunk_id
        """).fetchall()

        if not chunks:
            log.info("  All chunks already NER-processed")
            ctx.num_entity_mentions = conn.execute("SELECT count(*) FROM entities").fetchone()[0]
            return

        entity_labels_csv = ",".join(self._labels)
        relation_labels_csv = ",".join(self._relation_labels)
        model_name = self._muninn_model_name

        log.info(
            "  NER+RE processing %d chunks [muninn] (entity labels: %s, relation labels: %s)",
            len(chunks),
            self._labels,
            self._relation_labels,
        )

        total_entities = 0
        total_relations = 0
        ts = datetime.now(UTC).isoformat()
        tracker = ProgressTracker(len(chunks), window=10, min_interval_s=30.0)

        for chunk_id, text in chunks:
            result_json = conn.execute(
                "SELECT muninn_extract_ner_re(?, ?, ?, ?)",
                (model_name, text, entity_labels_csv, relation_labels_csv),
            ).fetchone()[0]

            parsed = json.loads(result_json)

            entity_rows = []
            for ent in parsed.get("entities", []):
                entity_rows.append(
                    (
                        ent.get("text", ""),
                        ent.get("type", ""),
                        "muninn",
                        chunk_id,
                        ent.get("score", 1.0),
                    )
                )
            if entity_rows:
                conn.executemany(
                    "INSERT INTO entities (name, entity_type, source, chunk_id, confidence) VALUES (?, ?, ?, ?, ?)",
                    entity_rows,
                )
            total_entities += len(entity_rows)

            relation_rows = []
            for rel in parsed.get("relations", []):
                head = rel.get("head", "")
                tail = rel.get("tail", "")
                if head and tail and head != tail:
                    relation_rows.append(
                        (
                            head,
                            tail,
                            rel.get("rel", ""),
                            rel.get("score", 1.0),
                            chunk_id,
                            "muninn",
                        )
                    )
            if relation_rows:
                conn.executemany(
                    "INSERT INTO relations (src, dst, rel_type, weight, chunk_id, source) VALUES (?, ?, ?, ?, ?, ?)",
                    relation_rows,
                )
            total_relations += len(relation_rows)

            conn.execute(
                "INSERT OR IGNORE INTO ner_chunks_log (chunk_id, processed_at) VALUES (?, ?)",
                (chunk_id, ts),
            )
            conn.execute(
                "INSERT OR IGNORE INTO re_chunks_log (chunk_id, processed_at) VALUES (?, ?)",
                (chunk_id, ts),
            )
            conn.commit()

            tracker.update(1)
            tracker.record_output(len(entity_rows))
            if tracker.should_log():
                log.info("  NER+RE %s (rels: %d)", tracker.report(), total_relations)

        log.info(
            "  Extracted %d entities + %d relations from %d chunks",
            total_entities,
            total_relations,
            len(chunks),
        )
        ctx.num_entity_mentions = conn.execute("SELECT count(*) FROM entities").fetchone()[0]
