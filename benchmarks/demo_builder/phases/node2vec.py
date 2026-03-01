"""Phase 7: Node2Vec Structural Embeddings."""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

from benchmarks.demo_builder.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext

log = logging.getLogger(__name__)

N2V_DIM = 64


class PhaseNode2Vec(Phase):
    """Train Node2Vec on the coalesced graph and store structural embeddings."""

    def __init__(self) -> None:
        self._n2v_edge_count = 0

    @property
    def name(self) -> str:
        return "node2vec"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        """Return True if node2vec output is missing or out-of-sync with nodes table."""
        try:
            n2v_count = conn.execute("SELECT count(*) FROM node2vec_emb_nodes").fetchone()[0]
            node_count = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
            return n2v_count != node_count
        except sqlite3.OperationalError:
            return True

    def restore_ctx(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        try:
            ctx.num_n2v = conn.execute("SELECT count(*) FROM node2vec_emb_nodes").fetchone()[0]
        except sqlite3.OperationalError:
            pass

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        # Drop + recreate: node2vec must be retrained from scratch when the graph changes.
        conn.execute("DROP TABLE IF EXISTS node2vec_emb")  # VT + all shadow tables via xDestroy
        conn.execute("DROP TABLE IF EXISTS n2v_edges")
        conn.execute(
            f"CREATE VIRTUAL TABLE node2vec_emb USING hnsw_index("
            f"  dimensions={N2V_DIM}, metric='cosine', m=16, ef_construction=200"
            f")"
        )

        conn.execute("CREATE TABLE n2v_edges (src INTEGER NOT NULL, dst INTEGER NOT NULL)")

        conn.execute("""
            INSERT INTO n2v_edges (src, dst)
            SELECT n1.node_id, n2.node_id
            FROM edges e
            JOIN nodes n1 ON n1.name = e.src
            JOIN nodes n2 ON n2.name = e.dst
            ORDER BY n1.node_id, n2.node_id
        """)

        self._n2v_edge_count = conn.execute("SELECT count(*) FROM n2v_edges").fetchone()[0]
        log.info("  Prepared %d integer edges for Node2Vec", self._n2v_edge_count)

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        if self._n2v_edge_count == 0:
            log.info("  No edges -- skipping Node2Vec training")
            ctx.num_n2v = 0
            return

        result = conn.execute(
            "SELECT node2vec_train(  'n2v_edges', 'src', 'dst', 'node2vec_emb',  64, 0.5, 0.5, 10, 40, 5, 5, 0.025, 5)"
        ).fetchone()

        num_embedded = result[0]
        log.info("  Node2Vec embedded %d nodes (dim=%d)", num_embedded, N2V_DIM)

        ctx.num_n2v = num_embedded

    def teardown(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        conn.execute("DROP TABLE IF EXISTS n2v_edges")
