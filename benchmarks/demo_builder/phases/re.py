"""Phase 3: Relation Extraction (GLiREL zero-shot RE)."""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

import spacy
from glirel import GLiREL
from huggingface_hub import snapshot_download

from benchmarks.demo_builder.common import char_span_to_token_span
from benchmarks.demo_builder.constants import GLIREL_LABELS
from benchmarks.demo_builder.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext

log = logging.getLogger(__name__)


class PhaseRE(Phase):
    """Extract relations per chunk using GLiREL and spaCy."""

    def __init__(self) -> None:
        self._re_model: GLiREL | None = None
        self._nlp: spacy.language.Language | None = None

    @property
    def name(self) -> str:
        return "relations"

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        conn.execute("""
            CREATE TABLE relations (
                relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                src TEXT NOT NULL,
                dst TEXT NOT NULL,
                rel_type TEXT,
                weight REAL DEFAULT 1.0,
                chunk_id INTEGER,
                source TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX idx_relations_src ON relations(src)")
        conn.execute("CREATE INDEX idx_relations_dst ON relations(dst)")

        log.info("  Loading GLiREL large-v0...")
        glirel_dir = snapshot_download("jackboyla/glirel-large-v0")
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

        # ── Process each chunk that has entities ──────────────────────
        chunks = conn.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()

        # Pre-load entities grouped by chunk_id
        entity_rows = conn.execute(
            "SELECT chunk_id, name, entity_type, confidence FROM entities ORDER BY chunk_id"
        ).fetchall()
        entities_by_chunk: dict[int, list[tuple[str, str, float]]] = {}
        for chunk_id, ent_name, etype, conf in entity_rows:
            entities_by_chunk.setdefault(chunk_id, []).append((ent_name, etype, conf))

        total_relations = 0

        for chunk_id, text in chunks:
            chunk_entities = entities_by_chunk.get(chunk_id, [])
            if len(chunk_entities) < 2:
                continue  # Need at least 2 entities for relations

            # Tokenize with spaCy
            doc = nlp(text)
            tokens = [token.text for token in doc]

            # Convert entity mentions to token-level spans for GLiREL
            ner_spans = []
            for ent_name, ent_type, _conf in chunk_entities:
                # Find entity text in the chunk
                start_char = text.find(ent_name)
                if start_char == -1:
                    # Try case-insensitive search
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

            # Build position -> entity name lookup for mapping GLiREL output back to NER entities.
            # GLiREL may extend span boundaries, so we match by overlap with NER spans.
            span_to_name: dict[tuple[int, int], str] = {}
            for span in ner_spans:
                s_start: int = span[0]  # type: ignore[assignment]
                s_end: int = span[1]  # type: ignore[assignment]
                s_name: str = span[3]  # type: ignore[assignment]
                span_to_name[(s_start, s_end)] = s_name

            def _find_entity(pos: list[int], _lookup: dict[tuple[int, int], str] = span_to_name) -> str | None:
                """Map GLiREL head_pos/tail_pos to NER entity name via span overlap."""
                r_start, r_end = pos[0], pos[1]
                best_name = None
                best_overlap = 0
                for (s, e), found_name in _lookup.items():
                    overlap = max(0, min(e, r_end) - max(s, r_start))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_name = found_name
                return best_name

            # Extract relations
            relations = re_model.predict_relations(tokens, GLIREL_LABELS, threshold=0.5, ner=ner_spans, top_k=10)

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

            if chunk_id % 200 == 0:
                log.info("  Processed chunk %d (%d relations so far)", chunk_id, total_relations)

        log.info("  Extracted %d relations", total_relations)

        ctx.num_relations = total_relations
