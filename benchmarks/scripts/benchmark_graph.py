"""
Graph traversal benchmark: vec_graph TVFs vs recursive CTEs vs GraphQLite.

Compares graph traversal operations (BFS, DFS, shortest path, connected components,
PageRank) across different engines on synthetic graphs with controlled topology.

Graph models:
    erdos_renyi     — Random graph with uniform edge probability
    barabasi_albert — Scale-free graph via preferential attachment

Engines:
    vec_graph   — This project's graph TVFs (graph_bfs, graph_dfs, etc.)
    cte         — Recursive CTEs in plain SQLite (baseline)
    graphqlite  — GraphQLite library (Python API)

Profiles:
    small       — N<=1K, avg_degree 5,20           (~5 min)
    medium      — N<=10K, avg_degree 5,20,50        (~20 min)
    large       — N<=100K, avg_degree 5,20          (~1 hr)
    scale_free  — Barabasi-Albert, N<=50K, m=3,5,10 (~30 min)

Run:
    python python/benchmark_graph.py --nodes 100 --avg-degree 5
    python python/benchmark_graph.py --nodes 1000 --avg-degree 20 --engine vec_graph
"""
import argparse
import collections
import datetime
import heapq
import json
import logging
import platform
import random
import sqlite3
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

try:
    from graphqlite import Graph

    HAS_GRAPHQLITE = True
except ImportError:
    HAS_GRAPHQLITE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VEC_GRAPH_PATH = str(PROJECT_ROOT / "vec_graph")
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"

N_PER_QUERY_OPS = 50  # random start nodes for per-query operations

ALL_OPERATIONS = ["bfs", "dfs", "shortest_path", "components", "pagerank"]

# Profile definitions (mirrors Makefile targets for manifest tracking)
GRAPH_PROFILES = {
    "small": {
        "graph_model": "erdos_renyi",
        "configs": [(100, 5), (100, 20), (500, 5), (500, 20), (1000, 5), (1000, 20)],
    },
    "medium": {
        "graph_model": "erdos_renyi",
        "configs": [
            (1000, 5), (1000, 20), (1000, 50),
            (5000, 5), (5000, 20), (5000, 50),
            (10000, 5), (10000, 20), (10000, 50),
        ],
    },
    "large": {
        "graph_model": "erdos_renyi",
        "configs": [
            (10000, 5), (10000, 20),
            (50000, 5), (50000, 20),
            (100000, 5), (100000, 20),
        ],
    },
    "scale_free": {
        "graph_model": "barabasi_albert",
        "configs": [
            (1000, 3), (1000, 5), (1000, 10),
            (5000, 3), (5000, 5), (5000, 10),
            (10000, 3), (10000, 5), (10000, 10),
            (50000, 3), (50000, 5), (50000, 10),
        ],
    },
}


# ── Graph generators ──────────────────────────────────────────────


def generate_erdos_renyi(n_nodes, avg_degree, weighted=False, seed=42):
    """Generate Erdos-Renyi random graph.

    Returns (edges, adjacency_dict) where edges is a list of (src, dst, weight)
    tuples and adjacency_dict maps node -> [(neighbor, weight)].
    """
    rng = random.Random(seed)
    p = avg_degree / max(1, n_nodes - 1)

    edges = []
    adj = collections.defaultdict(list)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < p:
                w = round(rng.uniform(0.1, 10.0), 2) if weighted else 1.0
                edges.append((i, j, w))
                edges.append((j, i, w))
                adj[i].append((j, w))
                adj[j].append((i, w))

    # Ensure all nodes exist in adjacency (even isolates)
    for i in range(n_nodes):
        if i not in adj:
            adj[i] = []

    log.info("  Erdos-Renyi: %d nodes, %d directed edges (p=%.4f)", n_nodes, len(edges), p)
    return edges, dict(adj)


def generate_barabasi_albert(n_nodes, m, weighted=False, seed=42):
    """Generate Barabasi-Albert scale-free graph via preferential attachment.

    Each new node connects to m existing nodes. Returns (edges, adjacency_dict).
    """
    rng = random.Random(seed)
    adj = collections.defaultdict(list)
    edges = []
    degree = [0] * n_nodes
    targets = list(range(min(m, n_nodes)))

    # Start with a complete graph on the first m+1 nodes
    for i in range(min(m + 1, n_nodes)):
        for j in range(i + 1, min(m + 1, n_nodes)):
            w = round(rng.uniform(0.1, 10.0), 2) if weighted else 1.0
            edges.append((i, j, w))
            edges.append((j, i, w))
            adj[i].append((j, w))
            adj[j].append((i, w))
            degree[i] += 1
            degree[j] += 1

    # Preferential attachment for remaining nodes
    for new_node in range(m + 1, n_nodes):
        # Build weighted probability distribution
        total_degree = sum(degree[:new_node])
        if total_degree == 0:
            targets = rng.sample(range(new_node), min(m, new_node))
        else:
            targets = set()
            while len(targets) < min(m, new_node):
                r = rng.random() * total_degree
                cumulative = 0
                for node in range(new_node):
                    cumulative += degree[node]
                    if cumulative >= r:
                        targets.add(node)
                        break

        for target in targets:
            w = round(rng.uniform(0.1, 10.0), 2) if weighted else 1.0
            edges.append((new_node, target, w))
            edges.append((target, new_node, w))
            adj[new_node].append((target, w))
            adj[target].append((new_node, w))
            degree[new_node] += 1
            degree[target] += 1

    for i in range(n_nodes):
        if i not in adj:
            adj[i] = []

    log.info("  Barabasi-Albert: %d nodes, %d directed edges (m=%d)", n_nodes, len(edges), m)
    return edges, dict(adj)


def largest_component_nodes(adj):
    """Find nodes in the largest connected component."""
    visited = set()
    components = []

    for start in adj:
        if start in visited:
            continue
        component = set()
        queue = collections.deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor, _ in adj.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)

    return max(components, key=len) if components else set()


# ── Python ground truth implementations ───────────────────────────


def python_bfs(adj, start):
    """BFS from start node. Returns {node: depth}."""
    visited = {start: 0}
    queue = collections.deque([start])
    while queue:
        node = queue.popleft()
        for neighbor, _ in adj.get(node, []):
            if neighbor not in visited:
                visited[neighbor] = visited[node] + 1
                queue.append(neighbor)
    return visited


def python_dfs(adj, start):
    """DFS from start node. Returns [(node, depth)] in visit order."""
    visited = []
    seen = set()
    stack = [(start, 0)]
    while stack:
        node, depth = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        visited.append((node, depth))
        # Push neighbors in reverse order for consistent ordering
        for neighbor, _ in reversed(adj.get(node, [])):
            if neighbor not in seen:
                stack.append((neighbor, depth + 1))
    return visited


def python_dijkstra(adj, start, end):
    """Dijkstra's shortest path. Returns (distance, path) or (None, []) if unreachable."""
    dist = {start: 0.0}
    prev = {}
    heap = [(0.0, start)]

    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, float("inf")):
            continue
        if node == end:
            break
        for neighbor, weight in adj.get(node, []):
            new_dist = d + weight
            if new_dist < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_dist
                prev[neighbor] = node
                heapq.heappush(heap, (new_dist, neighbor))

    if end not in dist:
        return None, []

    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return dist[end], path


def python_components(adj):
    """Connected components via Union-Find. Returns {node: component_id}."""
    parent = {n: n for n in adj}
    rank = {n: 0 for n in adj}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    for node in adj:
        for neighbor, _ in adj[node]:
            union(node, neighbor)

    # Normalize component IDs
    component_map = {}
    result = {}
    for node in sorted(adj):
        root = find(node)
        if root not in component_map:
            component_map[root] = len(component_map)
        result[node] = component_map[root]

    return result


def python_pagerank(adj, damping=0.85, iterations=100):
    """PageRank via power iteration. Returns {node: rank}."""
    n = len(adj)
    if n == 0:
        return {}

    rank = {node: 1.0 / n for node in adj}
    out_degree = {node: len(neighbors) for node, neighbors in adj.items()}

    for _ in range(iterations):
        new_rank = {}
        for node in adj:
            incoming = 0.0
            for other in adj:
                for neighbor, _ in adj[other]:
                    if neighbor == node and out_degree[other] > 0:
                        incoming += rank[other] / out_degree[other]
            new_rank[node] = (1 - damping) / n + damping * incoming
        rank = new_rank

    return rank


# ── vec_graph runner ──────────────────────────────────────────────


def setup_vec_graph_edges(conn, edges):
    """Create edge table and load edges for vec_graph TVFs."""
    conn.execute("CREATE TABLE IF NOT EXISTS bench_edges(src INTEGER, dst INTEGER, weight REAL)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bench_src ON bench_edges(src)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bench_dst ON bench_edges(dst)")
    conn.executemany("INSERT INTO bench_edges(src, dst, weight) VALUES (?, ?, ?)", edges)
    conn.commit()


def run_graph_vec_graph(conn, operation, adj, start_nodes, end_nodes=None):
    """Run a graph operation using vec_graph TVFs. Returns (results, timing)."""
    if operation == "bfs":
        times = []
        results = []
        for start in start_nodes:
            t0 = time.perf_counter()
            rows = conn.execute(
                "SELECT node, depth FROM graph_bfs"
                " WHERE edge_table = 'bench_edges' AND src_col = 'src' AND dst_col = 'dst'"
                " AND start_node = ? AND max_depth = 1000 AND direction = 'forward'",
                (start,),
            ).fetchall()
            times.append(time.perf_counter() - t0)
            results.append({int(r[0]): r[1] for r in rows})
        return results, times

    if operation == "dfs":
        times = []
        results = []
        for start in start_nodes:
            t0 = time.perf_counter()
            rows = conn.execute(
                "SELECT node, depth FROM graph_dfs"
                " WHERE edge_table = 'bench_edges' AND src_col = 'src' AND dst_col = 'dst'"
                " AND start_node = ? AND max_depth = 1000 AND direction = 'forward'",
                (start,),
            ).fetchall()
            times.append(time.perf_counter() - t0)
            results.append([(int(r[0]), r[1]) for r in rows])
        return results, times

    if operation == "shortest_path":
        times = []
        results = []
        for start, end in zip(start_nodes, end_nodes):
            t0 = time.perf_counter()
            rows = conn.execute(
                "SELECT node, distance, path_order FROM graph_shortest_path"
                " WHERE edge_table = 'bench_edges' AND src_col = 'src' AND dst_col = 'dst'"
                " AND start_node = ? AND end_node = ?",
                (start, end),
            ).fetchall()
            times.append(time.perf_counter() - t0)
            path = [int(r[0]) for r in rows]
            results.append(path)
        return results, times

    if operation == "components":
        t0 = time.perf_counter()
        rows = conn.execute(
            "SELECT node, component_id FROM graph_components"
            " WHERE edge_table = 'bench_edges' AND src_col = 'src' AND dst_col = 'dst'",
        ).fetchall()
        elapsed = time.perf_counter() - t0
        result = {int(r[0]): r[1] for r in rows}
        return result, [elapsed]

    if operation == "pagerank":
        t0 = time.perf_counter()
        rows = conn.execute(
            "SELECT node, rank FROM graph_pagerank"
            " WHERE edge_table = 'bench_edges' AND src_col = 'src' AND dst_col = 'dst'"
            " AND damping = 0.85 AND iterations = 100",
        ).fetchall()
        elapsed = time.perf_counter() - t0
        result = {int(r[0]): r[1] for r in rows}
        return result, [elapsed]

    log.error("Unknown operation: %s", operation)
    return None, []


# ── CTE baseline runner ──────────────────────────────────────────


def setup_cte_edges(conn, edges):
    """Create edge table for CTE-based traversal."""
    conn.execute("CREATE TABLE IF NOT EXISTS bench_edges(src INTEGER, dst INTEGER, weight REAL)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bench_src ON bench_edges(src)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bench_dst ON bench_edges(dst)")
    conn.executemany("INSERT INTO bench_edges(src, dst, weight) VALUES (?, ?, ?)", edges)
    conn.commit()


def run_graph_cte(conn, operation, adj, start_nodes, end_nodes=None):
    """Run a graph operation using recursive CTEs. Returns (results, timing)."""
    if operation == "bfs":
        times = []
        results = []
        for start in start_nodes:
            t0 = time.perf_counter()
            rows = conn.execute(
                """
                WITH RECURSIVE reachable(node, depth) AS (
                    SELECT ?, 0
                    UNION
                    SELECT e.dst, r.depth + 1
                    FROM bench_edges e JOIN reachable r ON e.src = r.node
                    WHERE r.depth < 1000
                )
                SELECT node, MIN(depth) as depth FROM reachable GROUP BY node
                """,
                (start,),
            ).fetchall()
            times.append(time.perf_counter() - t0)
            results.append({r[0]: r[1] for r in rows})
        return results, times

    if operation == "shortest_path":
        times = []
        results = []
        for start, end in zip(start_nodes, end_nodes):
            t0 = time.perf_counter()
            rows = conn.execute(
                """
                WITH RECURSIVE path(node, depth) AS (
                    SELECT ?, 0
                    UNION
                    SELECT e.dst, p.depth + 1
                    FROM bench_edges e JOIN path p ON e.src = p.node
                    WHERE p.depth < 1000
                )
                SELECT node, MIN(depth) as depth FROM path GROUP BY node
                """,
                (start,),
            ).fetchall()
            times.append(time.perf_counter() - t0)
            # Extract distances for the end node
            dist_map = {r[0]: r[1] for r in rows}
            results.append(dist_map.get(end))
        return results, times

    if operation == "components":
        # CTE-based component detection: iterate BFS from unvisited nodes
        t0 = time.perf_counter()
        all_nodes = set()
        for row in conn.execute("SELECT DISTINCT src FROM bench_edges UNION SELECT DISTINCT dst FROM bench_edges"):
            all_nodes.add(row[0])

        visited = set()
        comp_id = 0
        result = {}
        for node in sorted(all_nodes):
            if node in visited:
                continue
            rows = conn.execute(
                """
                WITH RECURSIVE reachable(node) AS (
                    SELECT ?
                    UNION
                    SELECT e.dst FROM bench_edges e JOIN reachable r ON e.src = r.node
                )
                SELECT node FROM reachable
                """,
                (node,),
            ).fetchall()
            for r in rows:
                result[r[0]] = comp_id
                visited.add(r[0])
            comp_id += 1
        elapsed = time.perf_counter() - t0
        return result, [elapsed]

    # DFS, PageRank: skip for CTE (not expressible efficiently)
    if operation in ("dfs", "pagerank"):
        log.info("    CTE: skipping %s (not efficiently expressible)", operation)
        return None, []

    log.error("Unknown operation: %s", operation)
    return None, []


# ── GraphQLite runner ─────────────────────────────────────────────


def run_graph_graphqlite(adj, edges, operation, n_nodes, start_nodes, end_nodes=None):
    """Run a graph operation using GraphQLite. Returns (results, timing)."""
    if not HAS_GRAPHQLITE:
        log.warning("  GraphQLite not available: pip install graphqlite")
        return None, []

    # Build graph
    g = Graph(":memory:")
    nodes_batch = [(str(i), {}, "Node") for i in range(n_nodes)]
    edges_batch = [(str(src), str(dst), {"weight": w}, "EDGE") for src, dst, w in edges if src < dst]
    g.upsert_nodes_batch(nodes_batch)
    g.upsert_edges_batch(edges_batch)

    if operation == "bfs":
        times = []
        results = []
        for start in start_nodes:
            t0 = time.perf_counter()
            rows = g.bfs(start_id=str(start), max_depth=-1)
            times.append(time.perf_counter() - t0)
            result = {int(r["user_id"]): r["depth"] for r in rows}
            results.append(result)
        return results, times

    if operation == "dfs":
        times = []
        results = []
        for start in start_nodes:
            t0 = time.perf_counter()
            rows = g.dfs(start_id=str(start), max_depth=-1)
            times.append(time.perf_counter() - t0)
            result = [(int(r["user_id"]), r["depth"]) for r in rows]
            results.append(result)
        return results, times

    if operation == "shortest_path":
        times = []
        results = []
        for start, end in zip(start_nodes, end_nodes):
            t0 = time.perf_counter()
            result = g.shortest_path(source_id=str(start), target_id=str(end))
            times.append(time.perf_counter() - t0)
            path = [int(n) for n in result.get("path", [])] if result.get("found") else []
            results.append(path)
        return results, times

    if operation == "components":
        t0 = time.perf_counter()
        rows = g.weakly_connected_components()
        elapsed = time.perf_counter() - t0
        result = {int(r["node_id"]): r["component"] for r in rows}
        return result, [elapsed]

    if operation == "pagerank":
        t0 = time.perf_counter()
        rows = g.pagerank(damping=0.85, iterations=100)
        elapsed = time.perf_counter() - t0
        result = {int(r["node_id"]): r["score"] for r in rows}
        return result, [elapsed]

    log.error("Unknown operation: %s", operation)
    return None, []


# ── Correctness verification ──────────────────────────────────────


def verify_bfs(engine_result, truth, start):
    """Verify BFS results match ground truth."""
    if engine_result is None:
        return False
    truth_result = python_bfs(truth, start)
    return set(engine_result.keys()) == set(truth_result.keys())


def verify_components(engine_result, truth):
    """Verify component detection: same grouping, not necessarily same IDs."""
    if engine_result is None:
        return False
    truth_result = python_components(truth)

    # Build equivalence classes
    def to_groups(comp_map):
        groups = collections.defaultdict(set)
        for node, comp in comp_map.items():
            groups[comp].add(node)
        return set(frozenset(g) for g in groups.values())

    return to_groups(engine_result) == to_groups(truth_result)


# ── JSONL output ──────────────────────────────────────────────────


def write_jsonl_record(filepath, record):
    """Append a single JSON record to the JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def make_graph_record(
    engine, operation, graph_model, n_nodes, n_edges, avg_degree,
    weighted, setup_time_s, query_times, correct, nodes_visited_mean,
    storage="memory", engine_params=None,
):
    """Build a JSONL record for graph benchmark results."""
    n_queries = len(query_times)
    mean_time_ms = (sum(query_times) / n_queries * 1000) if n_queries > 0 else 0

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "benchmark_type": "graph",
        "engine": engine,
        "operation": operation,
        "graph_model": graph_model,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": round(avg_degree, 2),
        "weighted": weighted,
        "setup_time_s": round(setup_time_s, 4),
        "query_time_ms": round(mean_time_ms, 3),
        "n_queries": n_queries,
        "correct": correct,
        "nodes_visited_mean": round(nodes_visited_mean, 1) if nodes_visited_mean else None,
        "storage": storage,
        "db_size_bytes": None,
        "platform": f"{sys.platform}-{platform.machine()}",
        "python_version": platform.python_version(),
        "engine_params": engine_params or {},
    }


# ── Main benchmark ────────────────────────────────────────────────


def run_graph_benchmark(
    graph_model, n_nodes, avg_degree_or_m, engines, output_path,
    weighted=False, storage="memory", operations=None,
):
    """Run graph traversal benchmarks for a single graph configuration."""
    operations = operations or ALL_OPERATIONS

    # Generate graph
    log.info("\n  Generating %s graph (n=%d)...", graph_model, n_nodes)
    if graph_model == "erdos_renyi":
        edges, adj = generate_erdos_renyi(n_nodes, avg_degree_or_m, weighted=weighted)
    elif graph_model == "barabasi_albert":
        edges, adj = generate_barabasi_albert(n_nodes, int(avg_degree_or_m), weighted=weighted)
    else:
        log.error("Unknown graph model: %s", graph_model)
        return

    n_edges = len(edges)
    actual_avg_degree = n_edges / max(1, n_nodes)

    # Pick random start/end nodes from largest component
    lc_nodes = largest_component_nodes(adj)
    lc_list = sorted(lc_nodes)
    rng = random.Random(42)
    n_queries = min(N_PER_QUERY_OPS, len(lc_list))
    start_nodes = rng.sample(lc_list, n_queries)
    end_nodes = rng.sample(lc_list, n_queries)

    for engine in engines:
        log.info("\n  Engine: %s", engine)

        if engine == "vec_graph":
            conn = sqlite3.connect(":memory:")
            conn.enable_load_extension(True)
            conn.load_extension(VEC_GRAPH_PATH)

            t0 = time.perf_counter()
            setup_vec_graph_edges(conn, edges)
            setup_time = time.perf_counter() - t0

            for op in operations:
                log.info("    Operation: %s", op)
                results, times = run_graph_vec_graph(conn, op, adj, start_nodes, end_nodes)
                if results is None:
                    continue

                # Verify correctness
                correct = True
                nodes_visited = None
                if op == "bfs" and isinstance(results, list):
                    correct = all(verify_bfs(r, adj, s) for r, s in zip(results, start_nodes))
                    nodes_visited = sum(len(r) for r in results) / len(results) if results else 0
                elif op == "components":
                    correct = verify_components(results, adj)
                    nodes_visited = len(results)
                elif op == "dfs" and isinstance(results, list):
                    nodes_visited = sum(len(r) for r in results) / len(results) if results else 0

                record = make_graph_record(
                    engine="vec_graph", operation=op, graph_model=graph_model,
                    n_nodes=n_nodes, n_edges=n_edges, avg_degree=actual_avg_degree,
                    weighted=weighted, setup_time_s=setup_time,
                    query_times=times, correct=correct,
                    nodes_visited_mean=nodes_visited, storage=storage,
                )
                write_jsonl_record(output_path, record)
                log.info("      %s: %.3fms (correct=%s)", op, record["query_time_ms"], correct)

            conn.close()

        elif engine == "cte":
            conn = sqlite3.connect(":memory:")

            t0 = time.perf_counter()
            setup_cte_edges(conn, edges)
            setup_time = time.perf_counter() - t0

            for op in operations:
                log.info("    Operation: %s", op)
                results, times = run_graph_cte(conn, op, adj, start_nodes, end_nodes)
                if results is None:
                    continue

                correct = True
                nodes_visited = None
                if op == "bfs" and isinstance(results, list):
                    correct = all(verify_bfs(r, adj, s) for r, s in zip(results, start_nodes))
                    nodes_visited = sum(len(r) for r in results) / len(results) if results else 0
                elif op == "components":
                    correct = verify_components(results, adj)
                    nodes_visited = len(results)

                record = make_graph_record(
                    engine="cte", operation=op, graph_model=graph_model,
                    n_nodes=n_nodes, n_edges=n_edges, avg_degree=actual_avg_degree,
                    weighted=weighted, setup_time_s=setup_time,
                    query_times=times, correct=correct,
                    nodes_visited_mean=nodes_visited, storage=storage,
                )
                write_jsonl_record(output_path, record)
                log.info("      %s: %.3fms (correct=%s)", op, record["query_time_ms"], correct)

            conn.close()

        elif engine == "graphqlite":
            if not HAS_GRAPHQLITE:
                log.warning("  GraphQLite not installed, skipping")
                continue

            t0 = time.perf_counter()
            # Setup time measured inside the runner (graph construction)
            setup_time = 0  # will be measured per-run

            for op in operations:
                log.info("    Operation: %s", op)
                t_setup = time.perf_counter()
                results, times = run_graph_graphqlite(adj, edges, op, n_nodes, start_nodes, end_nodes)
                if results is None:
                    continue
                setup_time = time.perf_counter() - t_setup - sum(times)

                correct = True
                nodes_visited = None
                if op == "bfs" and isinstance(results, list):
                    correct = all(verify_bfs(r, adj, s) for r, s in zip(results, start_nodes))
                    nodes_visited = sum(len(r) for r in results) / len(results) if results else 0
                elif op == "components":
                    correct = verify_components(results, adj)
                    nodes_visited = len(results)
                elif op == "dfs" and isinstance(results, list):
                    nodes_visited = sum(len(r) for r in results) / len(results) if results else 0

                record = make_graph_record(
                    engine="graphqlite", operation=op, graph_model=graph_model,
                    n_nodes=n_nodes, n_edges=n_edges, avg_degree=actual_avg_degree,
                    weighted=weighted, setup_time_s=setup_time,
                    query_times=times, correct=correct,
                    nodes_visited_mean=nodes_visited, storage=storage,
                )
                write_jsonl_record(output_path, record)
                log.info("      %s: %.3fms (correct=%s)", op, record["query_time_ms"], correct)


ALL_GRAPH_ENGINES = ["vec_graph", "graphqlite"]


def verify_graph_extensions():
    """Check which graph engines are available. Returns dict of {engine: bool}."""
    status = {}

    try:
        c = sqlite3.connect(":memory:")
        c.enable_load_extension(True)
        c.load_extension(VEC_GRAPH_PATH)
        c.close()
        log.info("  vec_graph:     OK")
        status["vec_graph"] = True
    except Exception as e:
        log.error("  vec_graph:     FAILED — %s", e)
        status["vec_graph"] = False

    # CTE is always available (built-in SQLite)
    status["cte"] = True
    log.info("  cte:           OK (built-in)")

    if HAS_GRAPHQLITE:
        try:
            g = Graph(":memory:")
            g.stats()
            log.info("  graphqlite:    OK")
            status["graphqlite"] = True
        except Exception as e:
            log.error("  graphqlite:    FAILED — %s", e)
            status["graphqlite"] = False
    else:
        log.warning("  graphqlite:    not installed (pip install graphqlite)")
        status["graphqlite"] = False

    return status


def parse_args():
    parser = argparse.ArgumentParser(
        description="Graph traversal benchmark: vec_graph TVFs vs CTEs vs GraphQLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python python/benchmark_graph.py --nodes 100 --avg-degree 5
  python python/benchmark_graph.py --graph-model barabasi_albert --nodes 1000 --avg-degree 5
  python python/benchmark_graph.py --nodes 1000 --avg-degree 20 --engine vec_graph
        """,
    )
    parser.add_argument("--graph-model", choices=["erdos_renyi", "barabasi_albert"], default="erdos_renyi")
    parser.add_argument("--nodes", type=int, required=True, help="Number of nodes")
    parser.add_argument("--avg-degree", type=float, required=True, help="Average degree (or m for Barabasi-Albert)")
    parser.add_argument("--weighted", action="store_true", help="Use weighted edges")
    parser.add_argument(
        "--engine",
        choices=["all"] + ALL_GRAPH_ENGINES,
        default="all",
        help="Which engine(s) to benchmark",
    )
    parser.add_argument(
        "--operations",
        help="Comma-separated operations (default: all). Options: bfs,dfs,shortest_path,components,pagerank",
    )
    parser.add_argument(
        "--storage",
        choices=["memory", "disk"],
        default="memory",
        help="Storage backend (default: memory)",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    random.seed(42)

    args = parse_args()

    log.info("Checking graph engines...")
    ext_status = verify_graph_extensions()

    if args.engine == "all":
        engines = [e for e in ALL_GRAPH_ENGINES if ext_status.get(e)]
    else:
        engines = [args.engine] if ext_status.get(args.engine) else []

    if not engines:
        log.error("No graph engines available. Exiting.")
        sys.exit(1)

    log.info("Engines: %s", ", ".join(engines))

    operations = args.operations.split(",") if args.operations else None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"graph_{timestamp}.jsonl"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Results: %s", output_path)

    run_graph_benchmark(
        graph_model=args.graph_model,
        n_nodes=args.nodes,
        avg_degree_or_m=args.avg_degree,
        engines=engines,
        output_path=output_path,
        weighted=args.weighted,
        storage=args.storage,
        operations=operations,
    )

    log.info("\nGraph benchmark complete. Results: %s", output_path)
    log.info("Run 'make benchmark-graph-analyze' to generate charts.")


if __name__ == "__main__":
    main()
