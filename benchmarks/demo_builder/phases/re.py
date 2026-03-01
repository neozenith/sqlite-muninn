"""Phase 3: Relation Extraction (GLiREL zero-shot RE).

Incremental: tracks processed chunks in `re_chunks_log`. Each run only
processes chunks that have entities but haven't been RE-processed yet.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import spacy
from glirel import GLiREL
from huggingface_hub import snapshot_download

from benchmarks.demo_builder.common import ProgressTracker, char_span_to_token_span, offline_mode
from benchmarks.demo_builder.constants import GLIREL_LABELS
from benchmarks.demo_builder.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext

log = logging.getLogger(__name__)


class PhaseRE(Phase):
    """Extract relations per chunk using GLiREL and spaCy."""

    def __init__(self, labels: list[str] | None = None) -> None:
        self._re_model: GLiREL | None = None
        self._nlp: spacy.language.Language | None = None
        self._labels = labels if labels is not None else GLIREL_LABELS

    @property
    def name(self) -> str:
        return "relations"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        """Return True if any chunks with entities have not yet been RE-processed."""
        try:
            unprocessed = conn.execute("""
                SELECT count(DISTINCT c.chunk_id)
                FROM chunks c
                JOIN entities e ON e.chunk_id = c.chunk_id
                WHERE c.chunk_id NOT IN (SELECT chunk_id FROM re_chunks_log)
            """).fetchone()[0]
            return unprocessed > 0
        except sqlite3.OperationalError:
            return True  # re_chunks_log doesn't exist yet → never run

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

        # re_chunks_log: tracks chunks that have been RE-processed.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS re_chunks_log (
                chunk_id INTEGER PRIMARY KEY,
                processed_at TEXT NOT NULL
            )
        """)

        log.info("  Loading GLiREL large-v0...")
        glirel_dir = snapshot_download("jackboyla/glirel-large-v0", local_files_only=True)
        with offline_mode():
            self._re_model = GLiREL._from_pretrained(
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
        assert self._re_model is not None, "setup() must be called before run()"
        assert self._nlp is not None, "setup() must be called before run()"

        re_model = self._re_model
        nlp = self._nlp

        # Only process chunks with entities that haven't been RE-logged yet.
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

        log.info("  RE processing %d unprocessed chunks", len(chunks))

        # Pre-load entities grouped by chunk_id (only for chunks not yet RE-processed).
        # Use a subquery instead of IN (?, ?, ...) to avoid SQLite's variable limit.
        entity_rows = conn.execute("""
            SELECT chunk_id, name, entity_type, confidence FROM entities
            WHERE chunk_id NOT IN (SELECT chunk_id FROM re_chunks_log)
            ORDER BY chunk_id
        """).fetchall()
        entities_by_chunk: dict[int, list[tuple[str, str, float]]] = {}
        for chunk_id, ent_name, etype, conf in entity_rows:
            entities_by_chunk.setdefault(chunk_id, []).append((ent_name, etype, conf))

        total_relations = 0
        chunks_processed = 0
        ts = datetime.now(UTC).isoformat()
        tracker = ProgressTracker(len(chunks))

        for chunk_id, text in chunks:
            chunk_entities = entities_by_chunk.get(chunk_id, [])

            # Always log this chunk as processed (even if skipped for <2 entities).
            conn.execute(
                "INSERT OR IGNORE INTO re_chunks_log (chunk_id, processed_at) VALUES (?, ?)",
                (chunk_id, ts),
            )

            if len(chunk_entities) < 2:
                continue  # Need at least 2 entities for relations

            doc = nlp(text)
            tokens = [token.text for token in doc]

            ner_spans = []
            for ent_name, ent_type, _conf in chunk_entities:
                start_char = text.find(ent_name)
                if start_char == -1:
                    lower_text = text.lower()
                    start_char = lower_text.find(ent_name.lower())
                if start_char == -1:
                    continue
                end_char = start_char + len(ent_name)
                token_span = char_span_to_token_span(doc, start_char, end_char)
                if token_span is None:
                    continue
                ner_spans.append([token_span[0], token_span[1], ent_type, ent_name])

            if len(ner_spans) < 2:
                continue

            span_to_name: dict[tuple[int, int], str] = {}
            for span in ner_spans:
                s_start: int = span[0]  # type: ignore[assignment]
                s_end: int = span[1]  # type: ignore[assignment]
                s_name: str = span[3]  # type: ignore[assignment]
                span_to_name[(s_start, s_end)] = s_name

            def _find_entity(pos: list[int], _lookup: dict[tuple[int, int], str] = span_to_name) -> str | None:
                r_start, r_end = pos[0], pos[1]
                best_name = None
                best_overlap = 0
                for (s, e), found_name in _lookup.items():
                    overlap = max(0, min(e, r_end) - max(s, r_start))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_name = found_name
                return best_name

            relations = re_model.predict_relations(tokens, self._labels, threshold=0.5, ner=ner_spans, top_k=10)

            insert_rows = []
            for rel in relations:
                head = _find_entity(rel["head_pos"])
                tail = _find_entity(rel["tail_pos"])
                if head is None or tail is None or head == tail:
                    continue
                insert_rows.append(
                    (
                        head,
                        tail,
                        rel["label"],
                        rel.get("score", 1.0),
                        chunk_id,
                        "glirel",
                    )
                )

            if insert_rows:
                conn.executemany(
                    "INSERT INTO relations (src, dst, rel_type, weight, chunk_id, source) VALUES (?, ?, ?, ?, ?, ?)",
                    insert_rows,
                )
                total_relations += len(insert_rows)

            chunks_processed += 1
            tracker.update()
            if tracker.should_log():
                log.info("  RE %s  (%d relations so far)", tracker.report(), total_relations)

        log.info("  Extracted %d new relations from %d chunks", total_relations, len(chunks))

        ctx.num_relations = conn.execute("SELECT count(*) FROM relations").fetchone()[0]
