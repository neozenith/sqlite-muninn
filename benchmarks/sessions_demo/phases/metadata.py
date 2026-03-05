"""Phase 10: Metadata + Validation for sessions_demo.

Writes a `meta` key/value table in the same schema as demo_builder so that:
  - The viz frontend can discover this DB via manifest.json
  - write_manifest_json() in demo_builder/manifest.py picks up db_id + embedding_model
  - The KG Pipeline Explorer shows correct counts for all stages

Also validates that all expected tables are non-empty after a full build.
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.sessions_demo.constants import (
    GGUF_EMBEDDING_DIM,
    GGUF_MODEL_PATH,
    SESSION_DB_ID,
)
from benchmarks.sessions_demo.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.sessions_demo.build import PhaseContext

log = logging.getLogger(__name__)


class PhaseSessionMetadata(Phase):
    """Write the meta table and validate all expected output tables."""

    @property
    def name(self) -> str:
        return "metadata"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        """Always re-run metadata — it's fast and counts must reflect latest state."""
        return True

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        # ── Write meta table (drop first so re-runs get fresh counts) ──
        conn.execute("DROP TABLE IF EXISTS meta")
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")

        # Read num_chunks directly from the DB rather than ctx.num_chunks.
        # ctx.num_chunks tracks newly-created chunks per run (delta), not the
        # total. build.py applies a post-run fixup, but run_single_phase()
        # can call metadata in isolation — reading from DB is always correct.
        try:
            num_chunks_db = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        except Exception:
            num_chunks_db = ctx.num_chunks

        # Detect which NER/RE backend was actually used from the source column.
        # entities.source is "gliner2" (default) or "gliner" (legacy).
        # relations.source is "gliner2" (default) or "glirel" (legacy).
        try:
            ner_sources = {r[0] for r in conn.execute("SELECT DISTINCT source FROM entities").fetchall()}
        except Exception:
            ner_sources = set()
        try:
            re_sources = {r[0] for r in conn.execute("SELECT DISTINCT source FROM relations").fetchall()}
        except Exception:
            re_sources = set()

        if "gliner2" in ner_sources:
            ner_model_name = "fastino/gliner2-base-v1"
        else:
            ner_model_name = "urchade/gliner_medium-v2.1"

        if "gliner2" in re_sources:
            re_model_name = "fastino/gliner2-base-v1"
        elif "glirel" in re_sources:
            re_model_name = "jackboyla/glirel-large-v0"
        else:
            re_model_name = ner_model_name  # fallback: same as NER

        strategies = "+".join(sorted(ner_sources | re_sources)) or "unknown"

        meta_rows = [
            ("db_id", SESSION_DB_ID),
            ("source", "claude_code_sessions"),
            # embedding_model and embedding_dim are read by write_manifest_json()
            ("embedding_model", Path(GGUF_MODEL_PATH).name),
            ("embedding_dim", str(GGUF_EMBEDDING_DIM)),
            ("ner_model", ner_model_name),
            ("re_model", re_model_name),
            ("strategies", strategies),
            ("num_chunks", str(num_chunks_db)),
            ("total_entities", str(ctx.num_entity_mentions)),
            ("total_relations", str(ctx.num_relations)),
            ("num_nodes", str(ctx.num_nodes)),
            ("num_edges", str(ctx.num_edges)),
            ("num_n2v_embeddings", str(ctx.num_n2v)),
            ("build_timestamp", datetime.datetime.now(datetime.UTC).isoformat()),
            ("builder", "benchmarks.sessions_demo"),
        ]
        conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", meta_rows)

        # ── Validation pass ───────────────────────────────────────────
        log.info("  Validating tables...")

        # Tables that must exist and be non-empty
        plain_tables = [
            "chunks",  # real table (chunk_id, text) for FK compat
            "entities",
            "relations",
            "entity_clusters",
            "nodes",
            "edges",
            "chunks_vec_umap",
            "entities_vec_umap",
            "meta",
        ]
        for table_name in plain_tables:
            actual = conn.execute(f'SELECT count(*) FROM "{table_name}"').fetchone()[0]
            assert actual > 0, f"Table {table_name} is empty!"
            log.info("    %s: %d rows", table_name, actual)

        # HNSW virtual tables — check via shadow nodes table
        for vt_name in ["chunks_vec", "entities_vec", "node2vec_emb"]:
            count = conn.execute(f'SELECT count(*) FROM "{vt_name}_nodes"').fetchone()[0]
            log.info("    %s: %d vectors", vt_name, count)

        # FTS sanity check
        fts_count = conn.execute("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'error'").fetchone()[0]
        log.info("    chunks_fts: FTS5 working (%d matches for 'error')", fts_count)

        log.info("  Metadata written and validation complete")
