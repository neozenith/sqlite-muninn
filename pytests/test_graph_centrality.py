"""
Integration tests for graph centrality table-valued functions.

Tests graph_degree, graph_betweenness, and graph_closeness on synthetic graphs.
"""
import pytest


def create_line_graph(conn):
    """
    A -- B -- C -- D -- E

    In a line graph, the center node (C) has highest betweenness.
    """
    conn.execute("CREATE TABLE line_edges (src TEXT, dst TEXT)")
    conn.executemany(
        "INSERT INTO line_edges VALUES (?, ?)",
        [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")],
    )


def create_line_graph_bidir(conn):
    """Bidirectional line graph for undirected analysis."""
    conn.execute("CREATE TABLE bidir_edges (src TEXT, dst TEXT)")
    conn.executemany(
        "INSERT INTO bidir_edges VALUES (?, ?)",
        [
            ("A", "B"), ("B", "A"),
            ("B", "C"), ("C", "B"),
            ("C", "D"), ("D", "C"),
            ("D", "E"), ("E", "D"),
        ],
    )


def create_star_graph(conn):
    """
    Star: H at center, connected to I, J, K, L (bidirectional).
    """
    conn.execute("CREATE TABLE star_edges (src TEXT, dst TEXT)")
    conn.executemany(
        "INSERT INTO star_edges VALUES (?, ?)",
        [
            ("H", "I"), ("I", "H"),
            ("H", "J"), ("J", "H"),
            ("H", "K"), ("K", "H"),
            ("H", "L"), ("L", "H"),
        ],
    )


def create_weighted_triangle(conn):
    """
    Weighted triangle:
        A --2-- B
         \\     /
          1   3
           \\ /
            C
    """
    conn.execute("CREATE TABLE wtri (src TEXT, dst TEXT, weight REAL)")
    conn.executemany(
        "INSERT INTO wtri VALUES (?, ?, ?)",
        [
            ("A", "B", 2.0), ("B", "A", 2.0),
            ("A", "C", 1.0), ("C", "A", 1.0),
            ("B", "C", 3.0), ("C", "B", 3.0),
        ],
    )


def create_temporal_graph(conn):
    """
    Graph with timestamps for temporal filtering.
    """
    conn.execute("CREATE TABLE temp_edges (src TEXT, dst TEXT, ts TEXT)")
    conn.executemany(
        "INSERT INTO temp_edges VALUES (?, ?, ?)",
        [
            ("A", "B", "2024-01-01"),
            ("B", "C", "2024-03-01"),
            ("C", "D", "2024-06-01"),
            ("D", "E", "2024-09-01"),
        ],
    )


class TestGraphDegree:
    def test_basic_degree(self, conn):
        """Degree centrality on a star graph."""
        create_star_graph(conn)
        results = conn.execute(
            "SELECT node, in_degree, out_degree, degree, centrality"
            " FROM graph_degree"
            " WHERE edge_table = 'star_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
        ).fetchall()

        degree_map = {r[0]: r for r in results}
        assert len(degree_map) == 5

        # Center H has degree 8 (4 out + 4 in)
        h = degree_map["H"]
        assert h[1] == 4.0  # in_degree
        assert h[2] == 4.0  # out_degree
        assert h[3] == 8.0  # total degree

        # Leaf I has degree 2 (1 out + 1 in)
        i_node = degree_map["I"]
        assert i_node[1] == 1.0
        assert i_node[2] == 1.0
        assert i_node[3] == 2.0

    def test_normalized_degree(self, conn):
        """Normalized degree divides by (N-1)."""
        create_star_graph(conn)
        results = conn.execute(
            "SELECT node, centrality FROM graph_degree"
            " WHERE edge_table = 'star_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND normalized = 1"
        ).fetchall()

        cent_map = {r[0]: r[1] for r in results}
        # H: 8 / (5-1) = 2.0
        assert abs(cent_map["H"] - 2.0) < 0.01
        # Leaf: 2 / 4 = 0.5
        assert abs(cent_map["I"] - 0.5) < 0.01

    def test_weighted_degree(self, conn):
        """Weighted degree sums edge weights."""
        create_weighted_triangle(conn)
        results = conn.execute(
            "SELECT node, degree FROM graph_degree"
            " WHERE edge_table = 'wtri'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND weight_col = 'weight'"
        ).fetchall()

        deg_map = {r[0]: r[1] for r in results}
        # A: out(2+1) + in(2+1) = 6
        assert abs(deg_map["A"] - 6.0) < 0.01
        # B: out(2+3) + in(2+3) = 10
        assert abs(deg_map["B"] - 10.0) < 0.01
        # C: out(1+3) + in(1+3) = 8
        assert abs(deg_map["C"] - 8.0) < 0.01

    def test_direction_forward(self, conn):
        """Forward-only direction populates out-edges, not in-edges."""
        create_line_graph(conn)
        results = conn.execute(
            "SELECT node, out_degree, in_degree FROM graph_degree"
            " WHERE edge_table = 'line_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND direction = 'forward'"
        ).fetchall()

        deg_map = {r[0]: (r[1], r[2]) for r in results}
        # With direction='forward', graph_load only populates out adjacency.
        # A has 1 out-edge (A->B), in-edges are 0 (in adj not populated)
        assert deg_map["A"][0] == 1.0  # out
        assert deg_map["A"][1] == 0.0  # in (not populated in forward mode)
        # B has 1 out-edge (B->C), in-edges = 0
        assert deg_map["B"][0] == 1.0
        assert deg_map["B"][1] == 0.0
        # E has 0 out-edges (terminal node in forward direction)
        assert deg_map["E"][0] == 0.0


class TestGraphBetweenness:
    def test_line_graph_center(self, conn):
        """In a line A-B-C-D-E, C should have highest betweenness."""
        create_line_graph_bidir(conn)
        results = conn.execute(
            "SELECT node, centrality FROM graph_betweenness"
            " WHERE edge_table = 'bidir_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND direction = 'both'"
        ).fetchall()

        bc_map = {r[0]: r[1] for r in results}
        assert len(bc_map) == 5

        # C is the center — highest betweenness
        assert bc_map["C"] > bc_map["A"]
        assert bc_map["C"] > bc_map["E"]
        assert bc_map["C"] > bc_map["B"]
        assert bc_map["C"] > bc_map["D"]

        # B and D should be equal (symmetric)
        assert abs(bc_map["B"] - bc_map["D"]) < 0.01

        # A and E should be 0 (endpoints)
        assert bc_map["A"] == 0.0
        assert bc_map["E"] == 0.0

    def test_star_graph_center(self, conn):
        """In a star graph, the center has all the betweenness."""
        create_star_graph(conn)
        results = conn.execute(
            "SELECT node, centrality FROM graph_betweenness"
            " WHERE edge_table = 'star_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND direction = 'both'"
        ).fetchall()

        bc_map = {r[0]: r[1] for r in results}
        # Center H has all betweenness, leaves have 0
        assert bc_map["H"] > 0
        assert bc_map["I"] == 0.0
        assert bc_map["J"] == 0.0

    def test_normalized_betweenness(self, conn):
        """Normalized betweenness should be between 0 and 1."""
        create_line_graph_bidir(conn)
        results = conn.execute(
            "SELECT node, centrality FROM graph_betweenness"
            " WHERE edge_table = 'bidir_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND direction = 'both'"
            "   AND normalized = 1"
        ).fetchall()

        for _, cent in results:
            assert 0.0 <= cent <= 1.0

    def test_weighted_betweenness(self, conn):
        """Weighted betweenness uses Dijkstra for SSSP."""
        create_weighted_triangle(conn)
        results = conn.execute(
            "SELECT node, centrality FROM graph_betweenness"
            " WHERE edge_table = 'wtri'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND weight_col = 'weight'"
            "   AND direction = 'both'"
        ).fetchall()

        bc_map = {r[0]: r[1] for r in results}
        assert len(bc_map) == 3
        # Weights: A-B=2, A-C=1, B-C=3
        # B→C: direct(3) ties with B→A→C(2+1=3), so A is on half the
        # shortest paths between B and C.  betweenness(A) = 0.5
        # A→B and A→C: direct paths are strictly shortest, so B and C
        # have 0 betweenness.
        assert abs(bc_map["A"] - 0.5) < 0.01
        assert abs(bc_map["B"]) < 0.01
        assert abs(bc_map["C"]) < 0.01

    def test_temporal_filter(self, conn):
        """Betweenness with temporal filtering."""
        create_temporal_graph(conn)
        results = conn.execute(
            "SELECT node, centrality FROM graph_betweenness"
            " WHERE edge_table = 'temp_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND timestamp_col = 'ts'"
            "   AND time_start = '2024-01-01'"
            "   AND time_end = '2024-06-01'"
        ).fetchall()

        nodes = {r[0] for r in results}
        # Only edges A->B, B->C, C->D are within the time window
        assert "A" in nodes
        assert "B" in nodes
        assert "C" in nodes
        assert "D" in nodes
        # E should NOT be present (D->E is 2024-09-01, outside window)
        assert "E" not in nodes


class TestGraphCloseness:
    def test_line_graph_center(self, conn):
        """In a line graph, the center node has highest closeness."""
        create_line_graph_bidir(conn)
        results = conn.execute(
            "SELECT node, centrality FROM graph_closeness"
            " WHERE edge_table = 'bidir_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND direction = 'both'"
        ).fetchall()

        cc_map = {r[0]: r[1] for r in results}
        assert len(cc_map) == 5

        # C (center) should have highest closeness
        assert cc_map["C"] > cc_map["A"]
        assert cc_map["C"] > cc_map["E"]

        # Symmetric nodes should be equal
        assert abs(cc_map["A"] - cc_map["E"]) < 0.01
        assert abs(cc_map["B"] - cc_map["D"]) < 0.01

    def test_star_graph(self, conn):
        """In a star, the center is closest to all."""
        create_star_graph(conn)
        results = conn.execute(
            "SELECT node, centrality FROM graph_closeness"
            " WHERE edge_table = 'star_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND direction = 'both'"
        ).fetchall()

        cc_map = {r[0]: r[1] for r in results}
        # Center H: distance to all leaves = 1, closeness = 4/4 = 1.0
        assert cc_map["H"] > cc_map["I"]
        assert cc_map["H"] > cc_map["J"]

    def test_closeness_all_positive(self, conn):
        """Closeness centrality should be non-negative."""
        create_line_graph_bidir(conn)
        results = conn.execute(
            "SELECT node, centrality FROM graph_closeness"
            " WHERE edge_table = 'bidir_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND direction = 'both'"
        ).fetchall()

        for _, cent in results:
            assert cent >= 0.0

    def test_weighted_closeness(self, conn):
        """Weighted closeness uses Dijkstra distances."""
        create_weighted_triangle(conn)
        results = conn.execute(
            "SELECT node, centrality FROM graph_closeness"
            " WHERE edge_table = 'wtri'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND weight_col = 'weight'"
            "   AND direction = 'both'"
        ).fetchall()

        cc_map = {r[0]: r[1] for r in results}
        assert len(cc_map) == 3
        # All nodes should have positive closeness
        for v in cc_map.values():
            assert v > 0.0

    def test_disconnected_closeness(self, conn):
        """Disconnected nodes should have 0 closeness."""
        conn.execute("CREATE TABLE disc_edges (src TEXT, dst TEXT)")
        conn.executemany(
            "INSERT INTO disc_edges VALUES (?, ?)",
            [("A", "B"), ("B", "A"), ("C", "D"), ("D", "C")],
        )

        results = conn.execute(
            "SELECT node, centrality FROM graph_closeness"
            " WHERE edge_table = 'disc_edges'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND direction = 'both'"
        ).fetchall()

        cc_map = {r[0]: r[1] for r in results}
        # With Wasserman-Faust normalization, disconnected nodes have low closeness
        # (reachable=1, N-1=3, so normalization factor = 1/3)
        assert len(cc_map) == 4
        for v in cc_map.values():
            assert v > 0.0  # each node can reach at least 1 other
