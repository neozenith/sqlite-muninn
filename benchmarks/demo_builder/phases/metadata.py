"""Phase 8: Metadata + Validation."""

from __future__ import annotations

import datetime
import logging
import sqlite3
from typing import TYPE_CHECKING

from benchmarks.demo_builder.constants import EMBEDDING_MODELS
from benchmarks.demo_builder.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext

log = logging.getLogger(__name__)


class PhaseMetadata(Phase):
    """Write metadata table and validate all tables."""

    def __init__(self, book_id: int, model_name: str) -> None:
        self._book_id = book_id
        self._model_name = model_name

    @property
    def name(self) -> str:
        return "metadata+validation"

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        # ── Write meta table ──────────────────────────────────────────
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")

        meta_rows = [
            ("book_id", str(self._book_id)),
            ("text_file", f"gutenberg_{self._book_id}.txt"),
            ("embedding_model", self._model_name),
            ("embedding_dim", str(EMBEDDING_MODELS[self._model_name]["dim"])),
            ("ner_model", "urchade/gliner_medium-v2.1"),
            ("re_model", "jackboyla/glirel-large-v0"),
            ("strategies", "gliner+glirel"),
            ("num_chunks", str(ctx.num_chunks)),
            ("total_entities", str(ctx.num_entity_mentions)),
            ("total_relations", str(ctx.num_relations)),
            ("num_nodes", str(ctx.num_nodes)),
            ("num_edges", str(ctx.num_edges)),
            ("num_n2v_embeddings", str(ctx.num_n2v)),
            ("build_timestamp", datetime.datetime.now(datetime.UTC).isoformat()),
            ("builder", "benchmarks.demo_builder"),
        ]
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_rows)

        # ── Validation pass ───────────────────────────────────────────
        log.info("  Validating tables...")
        expected_tables = [
            ("chunks", ctx.num_chunks),
            ("entities", None),
            ("relations", None),
            ("entity_clusters", None),
            ("entity_vec_map", None),
            ("nodes", ctx.num_nodes),
            ("edges", ctx.num_edges),
            ("chunks_vec_umap", ctx.num_chunks),
            ("entities_vec_umap", None),
            ("meta", len(meta_rows)),
        ]

        for table_name, expected_count in expected_tables:
            actual = conn.execute(f'SELECT count(*) FROM "{table_name}"').fetchone()[0]
            if expected_count is not None:
                assert actual == expected_count, f"Table {table_name}: expected {expected_count} rows, got {actual}"
            assert actual > 0, f"Table {table_name} is empty!"
            log.info("    %s: %d rows", table_name, actual)

        # Validate virtual tables (HNSW)
        for vt_name in ["chunks_vec", "entities_vec", "node2vec_emb"]:
            count = conn.execute(f'SELECT count(*) FROM "{vt_name}_nodes"').fetchone()[0]
            log.info("    %s: %d vectors", vt_name, count)

        # Validate FTS5
        fts_count = conn.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'labour'").fetchone()[0]
        log.info("    chunks_fts: FTS5 working (%d matches for 'labour')", fts_count)
