"""
Social Network — Friends of Friends with BFS and DFS

Demonstrates: graph_bfs (direction='both', max_depth), graph_dfs.

8 people in two clusters connected by a bridge node. BFS with
direction='both' traverses undirected edges stored in one direction.
"""
import sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
EXTENSION_PATH = str(PROJECT_ROOT / "muninn")

# ── Data: Two friend clusters with a bridge ──────────────────────────
# Cluster 1: Alice, Bob, Carol, Dave (tightly connected)
# Bridge:    Dave -- Eve
# Cluster 2: Eve, Frank, Grace, Heidi (tightly connected)
FRIENDSHIPS = [
    # Cluster 1
    ("Alice", "Bob"),
    ("Alice", "Carol"),
    ("Bob", "Carol"),
    ("Bob", "Dave"),
    ("Carol", "Dave"),
    # Bridge
    ("Dave", "Eve"),
    # Cluster 2
    ("Eve", "Frank"),
    ("Eve", "Grace"),
    ("Frank", "Grace"),
    ("Frank", "Heidi"),
    ("Grace", "Heidi"),
]


def main() -> None:
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXTENSION_PATH)

    print("=== Social Network Example ===\n")

    db.execute("CREATE TABLE friendships (person_a TEXT, person_b TEXT)")
    db.executemany("INSERT INTO friendships VALUES (?, ?)", FRIENDSHIPS)
    print(f"Created social network with {len(FRIENDSHIPS)} friendships.\n")

    # ── BFS depth=1: Direct friends of Alice ─────────────────────────
    print("--- BFS from Alice (depth=1) — Direct friends ---")
    results = db.execute(
        """
        SELECT node, depth, parent FROM graph_bfs
        WHERE edge_table = 'friendships'
          AND src_col = 'person_a'
          AND dst_col = 'person_b'
          AND start_node = 'Alice'
          AND max_depth = 1
          AND direction = 'both'
        """
    ).fetchall()

    friends = {r[0] for r in results if r[1] == 1}
    for node, depth, parent in sorted(results, key=lambda r: (r[1], r[0])):
        print(f"  depth={depth}: {node}" + (f" (via {parent})" if parent else ""))

    assert friends == {"Bob", "Carol"}, f"Expected {{Bob, Carol}}, got {friends}"
    print(f"  Alice's direct friends: {sorted(friends)}\n")

    # ── BFS depth=2: Friends of friends ──────────────────────────────
    print("--- BFS from Alice (depth=2) — Friends of friends ---")
    results = db.execute(
        """
        SELECT node, depth, parent FROM graph_bfs
        WHERE edge_table = 'friendships'
          AND src_col = 'person_a'
          AND dst_col = 'person_b'
          AND start_node = 'Alice'
          AND max_depth = 2
          AND direction = 'both'
        """
    ).fetchall()

    by_depth: dict[int, list[str]] = {}
    for node, depth, _parent in results:
        by_depth.setdefault(depth, []).append(node)

    for depth in sorted(by_depth):
        names = sorted(by_depth[depth])
        print(f"  depth={depth}: {', '.join(names)}")

    assert "Dave" in by_depth.get(2, []), "Dave should be at depth 2 (friend-of-friend)"
    print()

    # ── BFS depth=10: Full network traversal ─────────────────────────
    print("--- BFS from Alice (depth=10) — Full network ---")
    results = db.execute(
        """
        SELECT node, depth, parent FROM graph_bfs
        WHERE edge_table = 'friendships'
          AND src_col = 'person_a'
          AND dst_col = 'person_b'
          AND start_node = 'Alice'
          AND max_depth = 10
          AND direction = 'both'
        """
    ).fetchall()

    all_nodes = {r[0] for r in results}
    by_depth = {}
    for node, depth, _parent in results:
        by_depth.setdefault(depth, []).append(node)

    for depth in sorted(by_depth):
        names = sorted(by_depth[depth])
        print(f"  depth={depth}: {', '.join(names)}")

    expected_all = {"Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"}
    assert all_nodes == expected_all, f"Expected all 8 people, got {all_nodes}"
    print(f"\n  Reached all {len(all_nodes)} people in the network.\n")

    # ── DFS depth=10: Same nodes, different order ────────────────────
    print("--- DFS from Alice (depth=10) — Depth-first exploration ---")
    dfs_results = db.execute(
        """
        SELECT node, depth, parent FROM graph_dfs
        WHERE edge_table = 'friendships'
          AND src_col = 'person_a'
          AND dst_col = 'person_b'
          AND start_node = 'Alice'
          AND max_depth = 10
          AND direction = 'both'
        """
    ).fetchall()

    dfs_nodes = {r[0] for r in dfs_results}
    dfs_order = [r[0] for r in dfs_results]
    bfs_order = [r[0] for r in results]

    print(f"  DFS visit order: {' → '.join(dfs_order)}")
    print(f"  BFS visit order: {' → '.join(bfs_order)}")

    assert dfs_nodes == all_nodes, "DFS should reach the same nodes as BFS"
    assert dfs_order != bfs_order, "DFS and BFS should visit nodes in different order"
    print(f"\n  Same {len(dfs_nodes)} nodes reached, different traversal order.\n")

    db.close()
    print("All assertions passed.")


if __name__ == "__main__":
    main()
