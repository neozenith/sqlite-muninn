"""
Performance benchmarks for sqlite-vec-graph.

Measures insert rate, search latency, and recall at various dataset sizes.
Run: python python/benchmark.py
"""
import os
import sqlite3
import struct
import time
import random
import math
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTENSION_PATH = os.path.join(PROJECT_ROOT, "vec_graph")


def random_vector(dim):
    """Generate a random unit vector."""
    v = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-10:
        v[0] = 1.0
        norm = 1.0
    return [x / norm for x in v]


def pack_vector(v):
    """Pack a float list into bytes."""
    return struct.pack(f"{len(v)}f", *v)


def unpack_vector(blob):
    """Unpack bytes into a float list."""
    dim = len(blob) // 4
    return struct.unpack(f"{dim}f", blob)


def brute_force_knn(query, vectors, k):
    """Brute force KNN by L2 distance."""
    dists = []
    for rowid, v in vectors.items():
        d = sum((a - b) ** 2 for a, b in zip(query, v))
        dists.append((d, rowid))
    dists.sort()
    return [rowid for _, rowid in dists[:k]]


def benchmark_hnsw(n_values, dim=64, k=10, m=16, ef_construction=100, ef_search=64):
    """Benchmark HNSW insert and search at various N."""
    print(f"\n{'='*60}")
    print(f"HNSW Benchmark: dim={dim}, k={k}, M={m}, ef_c={ef_construction}, ef_s={ef_search}")
    print(f"{'='*60}")
    print(f"{'N':>8} | {'Insert (vec/s)':>14} | {'Search (ms)':>11} | {'Recall@{k}':>10}")
    print(f"{'-'*8}-+-{'-'*14}-+-{'-'*11}-+-{'-'*10}")

    for n in n_values:
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        conn.load_extension(EXTENSION_PATH)

        conn.execute(
            f"CREATE VIRTUAL TABLE bench_vec USING hnsw_index("
            f"dimensions={dim}, metric='l2', m={m}, ef_construction={ef_construction})"
        )

        # Generate vectors
        vectors = {}
        for i in range(1, n + 1):
            v = random_vector(dim)
            vectors[i] = v

        # Benchmark insert
        t0 = time.perf_counter()
        for rowid, v in vectors.items():
            conn.execute(
                "INSERT INTO bench_vec (rowid, vector) VALUES (?, ?)",
                (rowid, pack_vector(v)),
            )
        t_insert = time.perf_counter() - t0
        insert_rate = n / t_insert if t_insert > 0 else float("inf")

        # Benchmark search (100 queries, average latency)
        n_queries = min(100, n)
        query_ids = random.sample(range(1, n + 1), n_queries)

        t0 = time.perf_counter()
        hnsw_results = []
        for qid in query_ids:
            q = vectors[qid]
            rows = conn.execute(
                "SELECT rowid, distance FROM bench_vec"
                " WHERE vector MATCH ? AND k = ? AND ef_search = ?",
                (pack_vector(q), k, ef_search),
            ).fetchall()
            hnsw_results.append(set(r[0] for r in rows))
        t_search = time.perf_counter() - t0
        avg_search_ms = (t_search / n_queries) * 1000

        # Compute recall vs brute force
        recalls = []
        for i, qid in enumerate(query_ids):
            bf = set(brute_force_knn(vectors[qid], vectors, k))
            if len(bf) > 0:
                recall = len(hnsw_results[i] & bf) / len(bf)
                recalls.append(recall)
        avg_recall = sum(recalls) / len(recalls) if recalls else 0

        print(f"{n:>8} | {insert_rate:>14.0f} | {avg_search_ms:>11.3f} | {avg_recall:>10.1%}")

        conn.close()


def benchmark_graph_tvfs():
    """Benchmark graph TVFs on synthetic graphs."""
    print(f"\n{'='*60}")
    print("Graph TVF Benchmark")
    print(f"{'='*60}")

    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION_PATH)

    # Generate a random graph with N nodes and E edges
    for n_nodes, n_edges in [(100, 300), (1000, 3000), (5000, 15000)]:
        conn.execute("DROP TABLE IF EXISTS bench_edges")
        conn.execute("CREATE TABLE bench_edges (src TEXT, dst TEXT)")

        edges = set()
        while len(edges) < n_edges:
            a = random.randint(1, n_nodes)
            b = random.randint(1, n_nodes)
            if a != b:
                edges.add((str(a), str(b)))
        conn.executemany("INSERT INTO bench_edges VALUES (?, ?)", list(edges))

        start = str(random.randint(1, n_nodes))

        print(f"\n  Graph: {n_nodes} nodes, {n_edges} edges")

        # BFS
        t0 = time.perf_counter()
        rows = conn.execute(
            "SELECT node FROM graph_bfs"
            " WHERE edge_table = 'bench_edges'"
            "   AND src_col = 'src' AND dst_col = 'dst'"
            f"   AND start_node = '{start}'"
            "   AND max_depth = 100 AND direction = 'forward'"
        ).fetchall()
        t_bfs = (time.perf_counter() - t0) * 1000
        print(f"    BFS:        {t_bfs:8.2f} ms  ({len(rows)} nodes reached)")

        # Components
        t0 = time.perf_counter()
        rows = conn.execute(
            "SELECT node, component_id FROM graph_components"
            " WHERE edge_table = 'bench_edges'"
            "   AND src_col = 'src' AND dst_col = 'dst'"
        ).fetchall()
        t_comp = (time.perf_counter() - t0) * 1000
        n_components = len(set(r[1] for r in rows))
        print(f"    Components: {t_comp:8.2f} ms  ({n_components} components)")

        # PageRank
        t0 = time.perf_counter()
        rows = conn.execute(
            "SELECT node, rank FROM graph_pagerank"
            " WHERE edge_table = 'bench_edges'"
            "   AND src_col = 'src' AND dst_col = 'dst'"
        ).fetchall()
        t_pr = (time.perf_counter() - t0) * 1000
        print(f"    PageRank:   {t_pr:8.2f} ms  (20 iterations)")

    conn.close()


def main():
    random.seed(42)

    print("sqlite-vec-graph Performance Benchmark")
    print(f"Extension: {EXTENSION_PATH}")

    # HNSW benchmarks
    benchmark_hnsw([100, 500, 1000, 5000, 10000])

    # Graph TVF benchmarks
    benchmark_graph_tvfs()

    print(f"\n{'='*60}")
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
