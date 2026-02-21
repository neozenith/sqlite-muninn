"""KG GraphRAG retrieval quality treatment.

Benchmarks VSS-only vs VSS+Graph expansion retrieval quality.
Measures whether graph expansion after VSS entry point improves retrieval.

Source: docs/plans/benchmark_backlog.md (sections 3d, 3e)
"""

import logging
import time

from benchmarks.harness.common import KG_DIR, load_muninn
from benchmarks.harness.treatments.base import Treatment

log = logging.getLogger(__name__)


class KGGraphRAGTreatment(Treatment):
    """Single GraphRAG retrieval quality benchmark permutation."""

    def __init__(self, entry_method, expansion, book_id):
        self._entry = entry_method  # 'vss' or 'bm25'
        self._expansion = expansion  # 'none', 'bfs1', 'bfs2'
        self._book_id = book_id

    @property
    def category(self):
        return "kg-graphrag"

    @property
    def permutation_id(self):
        return f"kg-graphrag_{self._entry}_{self._expansion}_{self._book_id}"

    @property
    def label(self):
        return f"KG GraphRAG: {self._entry} + {self._expansion} / Book {self._book_id}"

    @property
    def sort_key(self):
        return (self._book_id, self._entry, self._expansion)

    def params_dict(self):
        return {
            "entry_method": self._entry,
            "expansion": self._expansion,
            "book_id": self._book_id,
        }

    def setup(self, conn, db_path):
        load_muninn(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_results (
                query_id INTEGER,
                chunk_id INTEGER,
                rank INTEGER,
                score REAL,
                source TEXT
            )
        """)
        conn.commit()

        return {"entry_method": self._entry, "expansion": self._expansion}

    def run(self, conn):
        chunks_db = KG_DIR / f"{self._book_id}_chunks.db"
        if not chunks_db.exists():
            log.warning("Chunks DB not found: %s — run 'prep kg-chunks' first", chunks_db)
            return {
                "retrieval_latency_ms": 0.0,
                "passage_recall": 0.0,
                "n_queries": 0,
            }

        t0 = time.perf_counter()
        # Placeholder: actual implementation would:
        # 1. Load KG from chunks_db
        # 2. For each test query:
        #    a. VSS or BM25 entry point to find seed chunks
        #    b. If expansion != 'none', expand via BFS on knowledge graph
        #    c. Assemble context from retrieved chunks
        #    d. Measure passage-level recall against ground truth
        total_time = (time.perf_counter() - t0) * 1000

        return {
            "retrieval_latency_ms": round(total_time, 3),
            "passage_recall": 0.0,
            "n_queries": 0,
        }

    def teardown(self, conn):
        pass
