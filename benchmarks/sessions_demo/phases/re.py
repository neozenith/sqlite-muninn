"""Phase: Relation Extraction (GLiNER2 or GLiREL zero-shot RE, or muninn LLM).

Incremental: tracks processed chunks in `re_chunks_log`. Each run only
processes chunks that haven't been RE-logged yet.

Backends:
  "gliner2" (default) — fastino/gliner2-base-v1, 205M DeBERTa-v3-base.
      Processes ALL chunks (discovers entity spans from raw text autonomously).
  "glirel"  (legacy)  — jackboyla/glirel-large-v0 + spaCy en_core_web_lg.
      Only processes chunks that already have >=2 NER entities.
  "muninn"            — No-op when following PhaseNER(backend="muninn"), which
      already populates relations via the combined muninn_extract_ner_re() call.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import spacy
from glirel import GLiREL
from huggingface_hub import snapshot_download

from benchmarks.sessions_demo.common import ProgressTracker, char_span_to_token_span, offline_mode
from benchmarks.sessions_demo.constants import SESSION_GLINER2_RE_LABELS, SESSION_GLIREL_LABELS
from benchmarks.sessions_demo.phases._gliner2_loader import get_gliner2
from benchmarks.sessions_demo.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.sessions_demo.build import PhaseContext

log = logging.getLogger(__name__)

RE_BATCH_SIZE = 8


def _find_entity_overlap(pos: list[int], span_to_name: dict[tuple[int, int], str]) -> str | None:
    """Return the entity name whose token span best overlaps with pos (legacy path)."""
    r_start, r_end = pos[0], pos[1]
    best_name = None
    best_overlap = 0
    for (s, e), found_name in span_to_name.items():
        overlap = max(0, min(e, r_end) - max(s, r_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = found_name
    return best_name


class PhaseRE(Phase):
    """Extract relations per chunk using GLiNER2 (default) or GLiREL + spaCy (legacy)."""

    def __init__(self, labels: list[str] | None = None, backend: str = "gliner2") -> None:
        self._model: Any = None
        self._nlp: Any = None
        self._backend = backend
        if labels is not None:
            self._labels = labels
        elif backend == "gliner2":
            self._labels = SESSION_GLINER2_RE_LABELS
        else:
            self._labels = SESSION_GLIREL_LABELS

    @property
    def name(self) -> str:
        return "relations"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        try:
            unprocessed: int
            if self._backend == "gliner2":
                unprocessed = conn.execute("""
                    SELECT count(*) FROM chunks c
                    WHERE c.chunk_id NOT IN (SELECT chunk_id FROM re_chunks_log)
                """).fetchone()[0]
            else:
                unprocessed = conn.execute("""
                    SELECT count(DISTINCT c.chunk_id)
                    FROM chunks c
                    JOIN entities e ON e.chunk_id = c.chunk_id
                    WHERE c.chunk_id NOT IN (SELECT chunk_id FROM re_chunks_log)
                """).fetchone()[0]
            return unprocessed > 0
        except sqlite3.OperationalError:
            return True

    def restore_ctx(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        try:
            ctx.num_relations = conn.execute("SELECT count(*) FROM relations").fetchone()[0]
        except sqlite3.OperationalError:
            pass

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
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

        if self._backend == "muninn":
            log.info("  RE [muninn]: tables created (relations populated by NER phase)")
            return
        elif self._backend == "gliner2":
            log.info("  Loading GLiNER2 (fastino/gliner2-base-v1)...")
            self._model = get_gliner2()
        else:
            log.info("  Loading GLiREL large-v0...")
            glirel_dir = snapshot_download("jackboyla/glirel-large-v0", local_files_only=True)
            with offline_mode():
                self._model = GLiREL._from_pretrained(
                    model_id=glirel_dir,
                    revision=None,
                    cache_dir=None,
                    force_download=False,
                    proxies=None,
                    resume_download=False,
                    local_files_only=True,
                    token=None,
                )
            log.info("  Loading spaCy en_core_web_lg...")
            self._nlp = spacy.load("en_core_web_lg")

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        if self._backend == "muninn":
            try:
                ctx.num_relations = conn.execute("SELECT count(*) FROM relations").fetchone()[0]
            except sqlite3.OperationalError:
                ctx.num_relations = 0
            log.info("  RE [muninn]: %d relations already extracted by NER phase", ctx.num_relations)
            return
        assert self._model is not None, "setup() must be called before run()"
        if self._backend == "gliner2":
            self._run_gliner2(conn, ctx)
        else:
            assert self._nlp is not None, "setup() must be called before run() (legacy backend requires spaCy)"
            self._run_legacy(conn, ctx)

    # ── GLiNER2 backend ───────────────────────────────────────────────────────

    def _run_gliner2(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        chunks = conn.execute("""
            SELECT chunk_id, text FROM chunks
            WHERE chunk_id NOT IN (SELECT chunk_id FROM re_chunks_log)
            ORDER BY chunk_id
        """).fetchall()

        if not chunks:
            log.info("  All chunks already RE-processed")
            ctx.num_relations = conn.execute("SELECT count(*) FROM relations").fetchone()[0]
            return

        log.info("  RE processing %d unprocessed chunks in batches of %d [gliner2]", len(chunks), RE_BATCH_SIZE)

        total_relations = 0
        ts = datetime.now(UTC).isoformat()
        tracker = ProgressTracker(len(chunks))

        re_commit_batches = 6

        for batch_num, batch_start in enumerate(range(0, len(chunks), RE_BATCH_SIZE)):
            batch = chunks[batch_start : batch_start + RE_BATCH_SIZE]
            chunk_ids = [cid for cid, _ in batch]
            texts = [text for _, text in batch]

            conn.executemany(
                "INSERT OR IGNORE INTO re_chunks_log (chunk_id, processed_at) VALUES (?, ?)",
                [(cid, ts) for cid in chunk_ids],
            )

            if batch_num > 0 and batch_num % re_commit_batches == 0:
                conn.commit()

            tracker.update(len(batch))
            if tracker.should_log():
                log.info("  RE %s", tracker.report())

            results = self._model.batch_extract_relations(
                texts, self._labels, batch_size=RE_BATCH_SIZE, include_confidence=True
            )

            insert_rows = []
            for chunk_id, result in zip(chunk_ids, results, strict=True):
                for rel_type, rels in result.get("relation_extraction", {}).items():
                    for rel in rels:
                        head = rel["head"]["text"]
                        tail = rel["tail"]["text"]
                        if head == tail:
                            continue
                        weight = (rel["head"].get("confidence", 1.0) + rel["tail"].get("confidence", 1.0)) / 2
                        insert_rows.append((head, tail, rel_type, weight, chunk_id, "gliner2"))

            if insert_rows:
                conn.executemany(
                    "INSERT INTO relations (src, dst, rel_type, weight, chunk_id, source) VALUES (?, ?, ?, ?, ?, ?)",
                    insert_rows,
                )
                total_relations += len(insert_rows)
                tracker.record_output(len(insert_rows))

        conn.commit()
        log.info("  Extracted %d new relations from %d chunks", total_relations, len(chunks))
        ctx.num_relations = conn.execute("SELECT count(*) FROM relations").fetchone()[0]

    # ── Legacy backend (GLiREL + spaCy) ──────────────────────────────────────

    def _run_legacy(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        nlp = self._nlp

        chunks = conn.execute("""
            SELECT DISTINCT c.chunk_id, c.text
            FROM chunks c
            JOIN entities e ON e.chunk_id = c.chunk_id
            WHERE c.chunk_id NOT IN (SELECT chunk_id FROM re_chunks_log)
            ORDER BY c.chunk_id
        """).fetchall()

        if not chunks:
            log.info("  All chunks already RE-processed")
            ctx.num_relations = conn.execute("SELECT count(*) FROM relations").fetchone()[0]
            return

        log.info("  RE processing %d unprocessed chunks in batches of %d [glirel]", len(chunks), RE_BATCH_SIZE)

        entity_rows = conn.execute("""
            SELECT chunk_id, name, entity_type, confidence FROM entities
            WHERE chunk_id NOT IN (SELECT chunk_id FROM re_chunks_log)
            ORDER BY chunk_id
        """).fetchall()
        entities_by_chunk: dict[int, list[tuple[str, str, float]]] = {}
        for chunk_id, ent_name, etype, conf in entity_rows:
            entities_by_chunk.setdefault(chunk_id, []).append((ent_name, etype, conf))

        total_relations = 0
        ts = datetime.now(UTC).isoformat()
        tracker = ProgressTracker(len(chunks))

        re_commit_batches = 6

        for batch_num, batch_start in enumerate(range(0, len(chunks), RE_BATCH_SIZE)):
            batch = chunks[batch_start : batch_start + RE_BATCH_SIZE]
            chunk_ids = [cid for cid, _ in batch]
            texts = [text for _, text in batch]

            conn.executemany(
                "INSERT OR IGNORE INTO re_chunks_log (chunk_id, processed_at) VALUES (?, ?)",
                [(cid, ts) for cid in chunk_ids],
            )

            if batch_num > 0 and batch_num % re_commit_batches == 0:
                conn.commit()

            tracker.update(len(batch))
            if tracker.should_log():
                log.info("  RE %s", tracker.report())

            eligible_mask = [len(entities_by_chunk.get(cid, [])) >= 2 for cid in chunk_ids]
            if not any(eligible_mask):
                continue

            eligible_texts = [t for t, ok in zip(texts, eligible_mask, strict=True) if ok]
            eligible_ids = [cid for cid, ok in zip(chunk_ids, eligible_mask, strict=True) if ok]
            docs = list(nlp.pipe(eligible_texts))

            re_batch: list[tuple[int, list[str], list[Any], dict[tuple[int, int], str]]] = []
            for chunk_id, text, doc in zip(eligible_ids, eligible_texts, docs, strict=True):
                chunk_entities = entities_by_chunk[chunk_id]
                tokens = [token.text for token in doc]

                ner_spans: list[Any] = []
                span_to_name: dict[tuple[int, int], str] = {}
                for ent_name, ent_type, _conf in chunk_entities:
                    start_char = text.find(ent_name)
                    if start_char == -1:
                        start_char = text.lower().find(ent_name.lower())
                    if start_char == -1:
                        continue
                    end_char = start_char + len(ent_name)
                    token_span = char_span_to_token_span(doc, start_char, end_char)
                    if token_span is None:
                        continue
                    ner_spans.append([token_span[0], token_span[1], ent_type, ent_name])
                    span_to_name[(token_span[0], token_span[1])] = ent_name

                if len(ner_spans) >= 2:
                    re_batch.append((chunk_id, tokens, ner_spans, span_to_name))

            if not re_batch:
                continue

            batch_results = self._model.batch_predict_relations(
                [e[1] for e in re_batch],
                self._labels,
                threshold=0.5,
                ner=[e[2] for e in re_batch],
                top_k=10,
            )

            insert_rows = []
            for (chunk_id, _, _, span_to_name), relations in zip(re_batch, batch_results, strict=True):
                for rel in relations:
                    head = _find_entity_overlap(rel["head_pos"], span_to_name)
                    tail = _find_entity_overlap(rel["tail_pos"], span_to_name)
                    if head is None or tail is None or head == tail:
                        continue
                    insert_rows.append((head, tail, rel["label"], rel.get("score", 1.0), chunk_id, "glirel"))

            if insert_rows:
                conn.executemany(
                    "INSERT INTO relations (src, dst, rel_type, weight, chunk_id, source) VALUES (?, ?, ?, ?, ?, ?)",
                    insert_rows,
                )
                total_relations += len(insert_rows)
                tracker.record_output(len(insert_rows))

        log.info("  Extracted %d new relations from %d chunks", total_relations, len(chunks))
        ctx.num_relations = conn.execute("SELECT count(*) FROM relations").fetchone()[0]
