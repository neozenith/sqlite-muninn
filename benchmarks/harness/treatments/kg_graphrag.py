"""KG GraphRAG retrieval quality treatment.

Benchmarks VSS-only vs VSS+Graph expansion retrieval quality.
Measures whether graph expansion after VSS entry point improves retrieval.

Expansion strategies:
- none: Pure VSS or BM25 search (baseline)
- bfs1: 1-hop entity neighbor expansion via knowledge graph
- bfs2: 2-hop entity neighbor expansion via knowledge graph

Inspired by Microsoft GraphRAG's DRIFT Search (Dynamic Reasoning and Inference
with Flexible Traversal), which combines global community-level context with
local entity-level refinement. Our "community" expansion approximates the DRIFT
primer phase by using Leiden community membership to pull related chunks.

Source: docs/plans/kg/04_graphrag_retrieval.md
"""

import logging
import sqlite3 as _sqlite3
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from benchmarks.harness.common import KG_DIR
from benchmarks.harness.treatments.base import Treatment

log = logging.getLogger(__name__)


def _pack_vector(v: np.ndarray) -> bytes:
    """Pack a float32 numpy array into a BLOB for SQLite."""
    return bytes(v.astype(np.float32).tobytes())


class KGGraphRAGTreatment(Treatment):
    """Single GraphRAG retrieval quality benchmark permutation.

    Loads a pre-built KG database (from demo_builder or NER pipeline output),
    generates pseudo-queries from sampled chunks, then measures how many
    topically-related chunks are retrieved via different entry+expansion strategies.
    """

    def __init__(self, entry_method, expansion, book_id):
        self._entry = entry_method  # 'vss' or 'bm25'
        self._expansion = expansion  # 'none', 'bfs1', 'bfs2'
        self._book_id = book_id
        self._st_model = None

    @property
    def requires_muninn(self) -> bool:
        return True  # Needs HNSW for VSS entry and graph_bfs for expansion

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
        # Find pre-built KG database
        kg_db_path = self._find_kg_db()
        if kg_db_path is None:
            log.warning("No KG database found for book %d — skipping", self._book_id)
            return {
                "retrieval_latency_ms": 0.0,
                "passage_recall_at_5": 0.0,
                "passage_recall_at_10": 0.0,
                "passage_recall_at_20": 0.0,
                "n_queries": 0,
                "avg_retrieved": 0.0,
            }

        # Open the KG database (read-only)
        kg_conn = _sqlite3.connect(f"file:{kg_db_path}?mode=ro", uri=True)
        kg_conn.enable_load_extension(True)

        # Check what tables are available
        tables = {r[0] for r in kg_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

        has_chunks_vec = "chunks_vec" in tables
        has_chunks_fts = "chunks_fts" in tables
        has_edges = "edges" in tables
        has_nodes = "nodes" in tables
        if self._entry == "vss" and not has_chunks_vec:
            log.warning("No chunks_vec table in KG DB — cannot run VSS entry")
            kg_conn.close()
            return self._empty_metrics()

        if self._entry == "bm25" and not has_chunks_fts:
            log.warning("No chunks_fts table in KG DB — cannot run BM25 entry")
            kg_conn.close()
            return self._empty_metrics()

        # Load muninn extension for graph operations
        from benchmarks.harness.common import load_muninn

        load_muninn(kg_conn)

        # Load embedding model for VSS queries
        if self._entry == "vss":
            self._st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Sample test queries — use chunks themselves as pseudo-queries
        chunks = kg_conn.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
        n_queries = min(50, len(chunks))
        # Sample evenly across the document
        step = max(1, len(chunks) // n_queries)
        query_chunks = chunks[::step][:n_queries]

        # Build entity->chunk mapping for computing ground truth neighborhoods
        chunk_entities = self._build_chunk_entity_map(kg_conn, tables)

        # Run queries
        latencies = []
        recall_at_5 = []
        recall_at_10 = []
        recall_at_20 = []
        total_retrieved = 0

        for query_idx, (query_chunk_id, query_text) in enumerate(query_chunks):
            t0 = time.perf_counter()

            # Phase 1: Entry point — get seed chunks
            if self._entry == "vss":
                seed_chunks = self._vss_entry(kg_conn, query_text, k=10)
            else:
                seed_chunks = self._bm25_entry(kg_conn, query_text, k=10)

            # Phase 2: Graph expansion (if enabled)
            expanded_chunks = set(seed_chunks)
            if self._expansion != "none" and (has_edges or has_nodes):
                depth = int(self._expansion.replace("bfs", ""))
                graph_chunks = self._graph_expand(kg_conn, seed_chunks, chunk_entities, depth)
                expanded_chunks.update(graph_chunks)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

            # Remove the query chunk itself from results
            expanded_chunks.discard(query_chunk_id)
            retrieved_list = sorted(expanded_chunks)
            total_retrieved += len(retrieved_list)

            # Store results
            for rank, cid in enumerate(retrieved_list):
                conn.execute(
                    "INSERT INTO retrieval_results(query_id, chunk_id, rank, score, source) VALUES (?,?,?,?,?)",
                    (query_idx, cid, rank, 1.0, self._entry),
                )

            # Compute passage recall — ground truth is chunks sharing entities with query chunk
            relevant = self._get_relevant_chunks(query_chunk_id, chunk_entities)
            relevant.discard(query_chunk_id)

            if relevant:
                retrieved_set = set(retrieved_list)
                recall_at_5.append(len(retrieved_set & relevant) / min(len(relevant), 5))
                recall_at_10.append(len(retrieved_set & relevant) / min(len(relevant), 10))
                recall_at_20.append(len(retrieved_set & relevant) / min(len(relevant), 20))

        conn.commit()
        kg_conn.close()

        return {
            "retrieval_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "passage_recall_at_5": round(sum(recall_at_5) / len(recall_at_5), 4) if recall_at_5 else 0.0,
            "passage_recall_at_10": round(sum(recall_at_10) / len(recall_at_10), 4) if recall_at_10 else 0.0,
            "passage_recall_at_20": round(sum(recall_at_20) / len(recall_at_20), 4) if recall_at_20 else 0.0,
            "n_queries": len(query_chunks),
            "avg_retrieved": round(total_retrieved / len(query_chunks), 1) if query_chunks else 0.0,
        }

    def teardown(self, conn):
        self._st_model = None

    # ── Private methods ──────────────────────────────────────────

    def _find_kg_db(self):
        """Find a pre-built KG database for the book."""
        # Try common naming patterns
        for pattern in [
            KG_DIR / f"{self._book_id}_MiniLM.db",
            KG_DIR / f"{self._book_id}_NomicEmbed.db",
        ]:
            if pattern.exists():
                return pattern

        # Also check wasm assets
        from benchmarks.harness.common import PROJECT_ROOT

        for pattern in [
            PROJECT_ROOT / "wasm" / "assets" / f"{self._book_id}.db",
        ]:
            if pattern.exists():
                return pattern

        return None

    def _empty_metrics(self):
        return {
            "retrieval_latency_ms": 0.0,
            "passage_recall_at_5": 0.0,
            "passage_recall_at_10": 0.0,
            "passage_recall_at_20": 0.0,
            "n_queries": 0,
            "avg_retrieved": 0.0,
        }

    def _vss_entry(self, kg_conn, query_text: str, k: int = 10) -> list[int]:
        """VSS entry point: embed query and search chunks_vec."""
        assert self._st_model is not None
        query_vec = self._st_model.encode([query_text], normalize_embeddings=True)[0].astype(np.float32)
        results = kg_conn.execute(
            "SELECT rowid, distance FROM chunks_vec WHERE vector MATCH ? AND k = ?",
            (_pack_vector(query_vec), k),
        ).fetchall()
        return [row[0] for row in results]

    def _bm25_entry(self, kg_conn, query_text: str, k: int = 10) -> list[int]:
        """BM25 entry point: search chunks_fts."""
        # Clean query for FTS5 — remove special characters
        clean_query = " ".join(query_text.split()[:20])  # Limit to 20 words
        results = kg_conn.execute(
            "SELECT chunk_id, rank FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (clean_query, k),
        ).fetchall()
        return [row[0] for row in results]

    def _build_chunk_entity_map(self, kg_conn, tables: set[str]) -> dict[int, set[str]]:
        """Build mapping of chunk_id -> set of entity names."""
        chunk_entities: dict[int, set[str]] = {}

        if "entities" in tables:
            rows = kg_conn.execute("SELECT chunk_id, name FROM entities").fetchall()
            for chunk_id, name in rows:
                chunk_entities.setdefault(chunk_id, set()).add(name)

        return chunk_entities

    def _get_relevant_chunks(self, query_chunk_id: int, chunk_entities: dict[int, set[str]]) -> set[int]:
        """Get chunks that share entities with the query chunk (ground truth for recall)."""
        query_entities = chunk_entities.get(query_chunk_id, set())
        if not query_entities:
            return set()

        relevant = set()
        for cid, entities in chunk_entities.items():
            if entities & query_entities:
                relevant.add(cid)
        return relevant

    def _graph_expand(
        self, kg_conn, seed_chunks: list[int], chunk_entities: dict[int, set[str]], depth: int
    ) -> set[int]:
        """Expand seed chunks via knowledge graph entity neighborhoods.

        For each seed chunk, find its entities, then find other chunks containing
        entities that are graph neighbors of those entities. BFS with given depth.

        This approximates DRIFT Search's follow-up phase: starting from seed entities
        (analogous to community-level context), we traverse the entity graph to
        discover related chunks (analogous to local search refinement).
        """
        # Collect all entities from seed chunks
        seed_entities: set[str] = set()
        for cid in seed_chunks:
            seed_entities.update(chunk_entities.get(cid, set()))

        if not seed_entities:
            return set()

        # Use graph_bfs to find neighbor entities at given depth
        expanded_entities = set(seed_entities)
        try:
            for entity_name in list(seed_entities):
                neighbors = kg_conn.execute(
                    "SELECT node FROM graph_bfs"
                    " WHERE edge_table = 'edges'"
                    "   AND src_col = 'src'"
                    "   AND dst_col = 'dst'"
                    "   AND start_node = ?"
                    "   AND max_depth = ?",
                    (entity_name, depth),
                ).fetchall()
                for (node,) in neighbors:
                    expanded_entities.add(node)
        except _sqlite3.OperationalError as e:
            log.debug("Graph BFS failed (edges table may not exist): %s", e)

        # Find all chunks containing expanded entities
        expanded_chunks: set[int] = set()
        for cid, entities in chunk_entities.items():
            if entities & expanded_entities:
                expanded_chunks.add(cid)

        return expanded_chunks
