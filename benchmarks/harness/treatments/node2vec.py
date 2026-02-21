"""Node2Vec training and embedding quality treatment.

Benchmarks Node2Vec training time and embedding quality via silhouette score.

Source: docs/plans/benchmark_backlog.md (section 2a)
"""

import logging
import time

from benchmarks.harness.common import generate_barabasi_albert, generate_erdos_renyi, load_muninn
from benchmarks.harness.treatments.base import Treatment

log = logging.getLogger(__name__)


class Node2VecTreatment(Treatment):
    """Single Node2Vec training benchmark permutation."""

    def __init__(self, graph_model, n_nodes, avg_degree, p, q, dim):
        self._graph_model = graph_model
        self._n_nodes = n_nodes
        self._avg_degree = avg_degree
        self._p = p
        self._q = q
        self._dim = dim
        self._edges = None

    @property
    def category(self):
        return "node2vec"

    @property
    def permutation_id(self):
        # Format p and q without dots for filesystem safety
        p_str = str(self._p).replace(".", "p")
        q_str = str(self._q).replace(".", "p")
        return f"node2vec_{self._graph_model}_n{self._n_nodes}_p{p_str}_q{q_str}_dim{self._dim}"

    @property
    def label(self):
        return f"Node2Vec: {self._graph_model} / N={self._n_nodes} / p={self._p} / q={self._q} / dim={self._dim}"

    @property
    def sort_key(self):
        return (self._n_nodes, self._dim, self._p, self._q)

    def params_dict(self):
        return {
            "graph_model": self._graph_model,
            "n_nodes": self._n_nodes,
            "avg_degree": self._avg_degree,
            "p": self._p,
            "q": self._q,
            "dim": self._dim,
        }

    def setup(self, conn, db_path):
        if self._graph_model == "erdos_renyi":
            self._edges, _ = generate_erdos_renyi(self._n_nodes, self._avg_degree, seed=42)
        else:
            self._edges, _ = generate_barabasi_albert(self._n_nodes, int(self._avg_degree), seed=42)

        load_muninn(conn)

        # Create edge table for node2vec
        conn.execute("CREATE TABLE bench_edges(src INTEGER, dst INTEGER, weight REAL)")
        conn.execute("CREATE INDEX idx_src ON bench_edges(src)")
        conn.execute("CREATE INDEX idx_dst ON bench_edges(dst)")
        conn.executemany("INSERT INTO bench_edges(src, dst, weight) VALUES (?, ?, ?)", self._edges)

        # Create HNSW index for embeddings output
        conn.execute(f"CREATE VIRTUAL TABLE n2v_embeddings USING hnsw_index(dimensions={self._dim}, metric='cosine')")
        conn.commit()

        return {"n_edges": len(self._edges)}

    def run(self, conn):
        t0 = time.perf_counter()
        conn.execute(
            "SELECT node2vec_train("
            "  'bench_edges', 'src', 'dst', 'weight',"
            "  'n2v_embeddings',"
            f"  {self._dim},"  # dimensions
            f"  {self._p},"  # p
            f"  {self._q},"  # q
            "  10,"  # walk_length
            "  80,"  # num_walks
            "  5,"  # window
            "  1,"  # min_count
            "  4,"  # workers (unused in C impl)
            "  100"  # epochs
            ")"
        )
        training_time = time.perf_counter() - t0

        # Count embeddings generated
        n_embeddings = conn.execute("SELECT COUNT(*) FROM n2v_embeddings").fetchone()[0]

        return {
            "training_time_s": round(training_time, 3),
            "n_embeddings": n_embeddings,
            "training_rate_nodes_per_s": round(self._n_nodes / training_time, 1) if training_time > 0 else 0,
        }

    def teardown(self, conn):
        self._edges = None
