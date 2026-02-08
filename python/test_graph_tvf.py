"""
Integration tests for graph table-valued functions.

Tests BFS, DFS, and shortest path on synthetic edge tables.
"""
import pytest


def create_tree_graph(conn):
    """
    Create a simple tree:
        A
       / \\
      B   C
     / \\   \\
    D   E   F
    """
    conn.execute("CREATE TABLE edges (src TEXT, dst TEXT)")
    conn.executemany(
        "INSERT INTO edges VALUES (?, ?)",
        [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F")],
    )


def create_weighted_graph(conn):
    """
    Create a weighted graph:
        A --1-- B --2-- C
        |               |
        4               1
        |               |
        D ------3------ E
    """
    conn.execute("CREATE TABLE wedges (src TEXT, dst TEXT, weight REAL)")
    conn.executemany(
        "INSERT INTO wedges VALUES (?, ?, ?)",
        [
            ("A", "B", 1.0),
            ("B", "C", 2.0),
            ("C", "E", 1.0),
            ("A", "D", 4.0),
            ("D", "E", 3.0),
        ],
    )


def create_cycle_graph(conn):
    """
    Create a cycle: A -> B -> C -> D -> A
    """
    conn.execute("CREATE TABLE cycle_edges (src TEXT, dst TEXT)")
    conn.executemany(
        "INSERT INTO cycle_edges VALUES (?, ?)",
        [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")],
    )


class TestGraphBFS:
    def test_bfs_tree_forward(self, conn):
        create_tree_graph(conn)
        results = conn.execute(
            "SELECT node, depth, parent FROM graph_bfs"
            " WHERE edge_table = 'edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND start_node = 'A'"
            "   AND max_depth = 10"
            "   AND direction = 'forward'"
        ).fetchall()

        nodes = {r[0] for r in results}
        assert nodes == {"A", "B", "C", "D", "E", "F"}

        # Check depths
        depth_map = {r[0]: r[1] for r in results}
        assert depth_map["A"] == 0
        assert depth_map["B"] == 1
        assert depth_map["C"] == 1
        assert depth_map["D"] == 2
        assert depth_map["E"] == 2
        assert depth_map["F"] == 2

    def test_bfs_max_depth(self, conn):
        create_tree_graph(conn)
        results = conn.execute(
            "SELECT node FROM graph_bfs"
            " WHERE edge_table = 'edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND start_node = 'A'"
            "   AND max_depth = 1"
            "   AND direction = 'forward'"
        ).fetchall()

        nodes = {r[0] for r in results}
        assert nodes == {"A", "B", "C"}  # depth 0 and 1 only

    def test_bfs_reverse(self, conn):
        create_tree_graph(conn)
        results = conn.execute(
            "SELECT node, depth FROM graph_bfs"
            " WHERE edge_table = 'edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND start_node = 'D'"
            "   AND max_depth = 10"
            "   AND direction = 'reverse'"
        ).fetchall()

        nodes = {r[0] for r in results}
        # D -> B -> A (reverse traversal up the tree)
        assert "D" in nodes
        assert "B" in nodes
        assert "A" in nodes

    def test_bfs_cycle(self, conn):
        create_cycle_graph(conn)
        results = conn.execute(
            "SELECT node FROM graph_bfs"
            " WHERE edge_table = 'cycle_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND start_node = 'A'"
            "   AND max_depth = 10"
            "   AND direction = 'forward'"
        ).fetchall()

        # BFS should visit all nodes exactly once despite cycle
        nodes = [r[0] for r in results]
        assert len(nodes) == len(set(nodes))  # no duplicates
        assert set(nodes) == {"A", "B", "C", "D"}

    def test_bfs_parent_tracking(self, conn):
        create_tree_graph(conn)
        results = conn.execute(
            "SELECT node, parent FROM graph_bfs"
            " WHERE edge_table = 'edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND start_node = 'A'"
            "   AND max_depth = 10"
            "   AND direction = 'forward'"
        ).fetchall()

        parent_map = {r[0]: r[1] for r in results}
        assert parent_map["A"] is None  # start node has no parent
        assert parent_map["B"] == "A"
        assert parent_map["C"] == "A"
        assert parent_map["D"] == "B"


class TestGraphDFS:
    def test_dfs_tree(self, conn):
        create_tree_graph(conn)
        results = conn.execute(
            "SELECT node FROM graph_dfs"
            " WHERE edge_table = 'edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND start_node = 'A'"
            "   AND max_depth = 10"
            "   AND direction = 'forward'"
        ).fetchall()

        nodes = {r[0] for r in results}
        assert nodes == {"A", "B", "C", "D", "E", "F"}

    def test_dfs_visits_each_once(self, conn):
        create_cycle_graph(conn)
        results = conn.execute(
            "SELECT node FROM graph_dfs"
            " WHERE edge_table = 'cycle_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND start_node = 'A'"
            "   AND max_depth = 10"
            "   AND direction = 'forward'"
        ).fetchall()

        nodes = [r[0] for r in results]
        assert len(nodes) == len(set(nodes))
        assert set(nodes) == {"A", "B", "C", "D"}


class TestGraphShortestPath:
    def test_unweighted_path(self, conn):
        create_tree_graph(conn)
        results = conn.execute(
            "SELECT node, distance, path_order FROM graph_shortest_path"
            " WHERE edge_table = 'edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND start_node = 'A'"
            "   AND end_node = 'D'"
            "   AND weight_col IS NULL"
        ).fetchall()

        path = [r[0] for r in results]
        assert path == ["A", "B", "D"]

    def test_weighted_dijkstra(self, conn):
        create_weighted_graph(conn)
        results = conn.execute(
            "SELECT node, distance, path_order FROM graph_shortest_path"
            " WHERE edge_table = 'wedges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND start_node = 'A'"
            "   AND end_node = 'E'"
            "   AND weight_col = 'weight'"
        ).fetchall()

        path = [r[0] for r in results]
        # A -> B (1) -> C (3) -> E (4) is shorter than A -> D (4) -> E (7)
        assert path == ["A", "B", "C", "E"]

        # Check cumulative distances
        dists = {r[0]: r[1] for r in results}
        assert abs(dists["A"] - 0.0) < 1e-6
        assert abs(dists["B"] - 1.0) < 1e-6
        assert abs(dists["C"] - 3.0) < 1e-6
        assert abs(dists["E"] - 4.0) < 1e-6

    def test_no_path(self, conn):
        conn.execute("CREATE TABLE edges2 (src TEXT, dst TEXT)")
        conn.execute("INSERT INTO edges2 VALUES ('A', 'B')")
        conn.execute("INSERT INTO edges2 VALUES ('C', 'D')")

        results = conn.execute(
            "SELECT node FROM graph_shortest_path"
            " WHERE edge_table = 'edges2'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND start_node = 'A'"
            "   AND end_node = 'D'"
            "   AND weight_col IS NULL"
        ).fetchall()

        assert len(results) == 0  # no path exists

    def test_same_start_end(self, conn):
        create_tree_graph(conn)
        results = conn.execute(
            "SELECT node FROM graph_shortest_path"
            " WHERE edge_table = 'edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND start_node = 'A'"
            "   AND end_node = 'A'"
            "   AND weight_col IS NULL"
        ).fetchall()

        assert len(results) == 1
        assert results[0][0] == "A"


def create_disconnected_graph(conn):
    """
    Create a graph with 3 disconnected components:
        Component 1: A - B - C
        Component 2: D - E
        Component 3: F - G
    (All nodes must appear in at least one edge to be discoverable.)
    """
    conn.execute("CREATE TABLE disc_edges (src TEXT, dst TEXT)")
    conn.executemany(
        "INSERT INTO disc_edges VALUES (?, ?)",
        [("A", "B"), ("B", "C"), ("D", "E"), ("F", "G")],
    )


def create_star_graph(conn):
    """
    Create a star graph: center H connected to I, J, K, L.
    H has highest PageRank (all edges flow to/from it).
    """
    conn.execute("CREATE TABLE star_edges (src TEXT, dst TEXT)")
    conn.executemany(
        "INSERT INTO star_edges VALUES (?, ?)",
        [("H", "I"), ("H", "J"), ("H", "K"), ("H", "L"),
         ("I", "H"), ("J", "H"), ("K", "H"), ("L", "H")],
    )


class TestGraphComponents:
    def test_single_component(self, conn):
        """A fully connected graph should have one component."""
        create_tree_graph(conn)
        results = conn.execute(
            "SELECT node, component_id, component_size FROM graph_components"
            " WHERE edge_table = 'edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
        ).fetchall()

        nodes = {r[0] for r in results}
        assert nodes == {"A", "B", "C", "D", "E", "F"}

        # All nodes should have the same component_id
        comp_ids = {r[1] for r in results}
        assert len(comp_ids) == 1

        # Component size should be 6
        sizes = {r[2] for r in results}
        assert sizes == {6}

    def test_multiple_components(self, conn):
        """A disconnected graph should have multiple components."""
        create_disconnected_graph(conn)
        results = conn.execute(
            "SELECT node, component_id, component_size FROM graph_components"
            " WHERE edge_table = 'disc_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
        ).fetchall()

        nodes = {r[0] for r in results}
        assert nodes == {"A", "B", "C", "D", "E", "F", "G"}

        # Group nodes by component_id
        components = {}
        for node, comp_id, comp_size in results:
            components.setdefault(comp_id, set()).add(node)

        assert len(components) == 3

        # Find which component has which nodes
        comp_sets = [frozenset(v) for v in components.values()]
        assert frozenset({"A", "B", "C"}) in comp_sets
        assert frozenset({"D", "E"}) in comp_sets
        assert frozenset({"F", "G"}) in comp_sets

    def test_component_sizes(self, conn):
        """Component sizes should match actual component membership."""
        create_disconnected_graph(conn)
        results = conn.execute(
            "SELECT node, component_id, component_size FROM graph_components"
            " WHERE edge_table = 'disc_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
        ).fetchall()

        size_map = {r[0]: r[2] for r in results}
        assert size_map["A"] == 3
        assert size_map["B"] == 3
        assert size_map["C"] == 3
        assert size_map["D"] == 2
        assert size_map["E"] == 2
        assert size_map["F"] == 2
        assert size_map["G"] == 2

    def test_cycle_single_component(self, conn):
        """A cycle graph should be one component."""
        create_cycle_graph(conn)
        results = conn.execute(
            "SELECT node, component_id, component_size FROM graph_components"
            " WHERE edge_table = 'cycle_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
        ).fetchall()

        comp_ids = {r[1] for r in results}
        assert len(comp_ids) == 1
        assert all(r[2] == 4 for r in results)


class TestGraphPageRank:
    def test_star_center_highest_rank(self, conn):
        """In a star graph, the center node should have the highest PageRank."""
        create_star_graph(conn)
        results = conn.execute(
            "SELECT node, rank FROM graph_pagerank"
            " WHERE edge_table = 'star_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
        ).fetchall()

        rank_map = {r[0]: r[1] for r in results}
        assert len(rank_map) == 5

        # Center H should have the highest rank
        assert rank_map["H"] > rank_map["I"]
        assert rank_map["H"] > rank_map["J"]
        assert rank_map["H"] > rank_map["K"]
        assert rank_map["H"] > rank_map["L"]

    def test_pagerank_sums_to_one(self, conn):
        """PageRank values should sum to approximately 1.0."""
        create_star_graph(conn)
        results = conn.execute(
            "SELECT node, rank FROM graph_pagerank"
            " WHERE edge_table = 'star_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
        ).fetchall()

        total = sum(r[1] for r in results)
        assert abs(total - 1.0) < 0.01

    def test_pagerank_custom_params(self, conn):
        """PageRank should accept custom damping and iterations."""
        create_star_graph(conn)
        results = conn.execute(
            "SELECT node, rank FROM graph_pagerank"
            " WHERE edge_table = 'star_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND damping = 0.5"
            "   AND iterations = 50"
        ).fetchall()

        rank_map = {r[0]: r[1] for r in results}
        # With lower damping, ranks are more uniform but center still highest
        assert rank_map["H"] > rank_map["I"]

        # Should still sum to ~1.0
        total = sum(r[1] for r in results)
        assert abs(total - 1.0) < 0.01

    def test_pagerank_symmetric_graph(self, conn):
        """In a symmetric cycle, all nodes should have equal PageRank."""
        create_cycle_graph(conn)
        # Add reverse edges to make it symmetric
        conn.executemany(
            "INSERT INTO cycle_edges VALUES (?, ?)",
            [("B", "A"), ("C", "B"), ("D", "C"), ("A", "D")],
        )

        results = conn.execute(
            "SELECT node, rank FROM graph_pagerank"
            " WHERE edge_table = 'cycle_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
        ).fetchall()

        ranks = [r[1] for r in results]
        # All ranks should be approximately equal (1/4 = 0.25)
        for r in ranks:
            assert abs(r - 0.25) < 0.05
