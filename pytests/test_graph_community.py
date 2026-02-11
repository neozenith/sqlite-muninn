"""
Integration tests for graph community detection table-valued functions.

Tests graph_leiden on synthetic graphs with known community structure.
"""
import pytest


def create_barbell_graph(conn):
    """
    Barbell graph: two cliques (A,B,C) and (D,E,F) connected by a single bridge C-D.
    Both cliques are bidirectional (undirected).

    Expected: Leiden should find 2 communities.
    """
    conn.execute("CREATE TABLE barbell (src TEXT, dst TEXT)")
    edges = [
        # Clique 1: A, B, C
        ("A", "B"), ("B", "A"),
        ("A", "C"), ("C", "A"),
        ("B", "C"), ("C", "B"),
        # Clique 2: D, E, F
        ("D", "E"), ("E", "D"),
        ("D", "F"), ("F", "D"),
        ("E", "F"), ("F", "E"),
        # Bridge
        ("C", "D"), ("D", "C"),
    ]
    conn.executemany("INSERT INTO barbell VALUES (?, ?)", edges)


def create_triangle_graph(conn):
    """A single triangle: should be one community."""
    conn.execute("CREATE TABLE tri (src TEXT, dst TEXT)")
    edges = [
        ("A", "B"), ("B", "A"),
        ("A", "C"), ("C", "A"),
        ("B", "C"), ("C", "B"),
    ]
    conn.executemany("INSERT INTO tri VALUES (?, ?)", edges)


def create_disconnected_communities(conn):
    """
    Two completely disconnected cliques.
    Must find exactly 2 communities.
    """
    conn.execute("CREATE TABLE disc_comm (src TEXT, dst TEXT)")
    edges = [
        # Community 1
        ("A", "B"), ("B", "A"),
        ("A", "C"), ("C", "A"),
        ("B", "C"), ("C", "B"),
        # Community 2
        ("X", "Y"), ("Y", "X"),
        ("X", "Z"), ("Z", "X"),
        ("Y", "Z"), ("Z", "Y"),
    ]
    conn.executemany("INSERT INTO disc_comm VALUES (?, ?)", edges)


def create_weighted_communities(conn):
    """
    Two groups with weak inter-community edges and strong intra-community edges.
    """
    conn.execute("CREATE TABLE wcomm (src TEXT, dst TEXT, weight REAL)")
    edges = [
        # Strong clique 1
        ("A", "B", 10.0), ("B", "A", 10.0),
        ("A", "C", 10.0), ("C", "A", 10.0),
        ("B", "C", 10.0), ("C", "B", 10.0),
        # Strong clique 2
        ("D", "E", 10.0), ("E", "D", 10.0),
        ("D", "F", 10.0), ("F", "D", 10.0),
        ("E", "F", 10.0), ("F", "E", 10.0),
        # Weak bridge
        ("C", "D", 0.1), ("D", "C", 0.1),
    ]
    conn.executemany("INSERT INTO wcomm VALUES (?, ?, ?)", edges)


def create_temporal_communities(conn):
    """Graph with timestamps for temporal filtering tests."""
    conn.execute("CREATE TABLE tcomm (src TEXT, dst TEXT, ts TEXT)")
    edges = [
        # Early edges form one group
        ("A", "B", "2024-01-01"), ("B", "A", "2024-01-01"),
        ("A", "C", "2024-02-01"), ("C", "A", "2024-02-01"),
        ("B", "C", "2024-03-01"), ("C", "B", "2024-03-01"),
        # Late edges form another group
        ("D", "E", "2024-07-01"), ("E", "D", "2024-07-01"),
        ("D", "F", "2024-08-01"), ("F", "D", "2024-08-01"),
        ("E", "F", "2024-09-01"), ("F", "E", "2024-09-01"),
        # Bridge connecting the groups (mid-year)
        ("C", "D", "2024-05-01"), ("D", "C", "2024-05-01"),
    ]
    conn.executemany("INSERT INTO tcomm VALUES (?, ?, ?)", edges)


class TestGraphLeiden:
    def test_barbell_finds_two_communities(self, conn):
        """Barbell graph should split into 2 communities."""
        create_barbell_graph(conn)
        results = conn.execute(
            "SELECT node, community_id, modularity FROM graph_leiden"
            " WHERE edge_table = 'barbell'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
        ).fetchall()

        assert len(results) == 6

        # Group by community
        communities = {}
        for node, comm_id, _ in results:
            communities.setdefault(comm_id, set()).add(node)

        assert len(communities) == 2

        # The two cliques should be in different communities
        comm_sets = [frozenset(v) for v in communities.values()]
        # A, B, C should be together; D, E, F should be together
        assert frozenset({"A", "B", "C"}) in comm_sets
        assert frozenset({"D", "E", "F"}) in comm_sets

    def test_single_community(self, conn):
        """A single triangle should be one community."""
        create_triangle_graph(conn)
        results = conn.execute(
            "SELECT node, community_id FROM graph_leiden"
            " WHERE edge_table = 'tri'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
        ).fetchall()

        comm_ids = {r[1] for r in results}
        assert len(comm_ids) == 1

    def test_disconnected_communities(self, conn):
        """Completely disconnected cliques must be in separate communities."""
        create_disconnected_communities(conn)
        results = conn.execute(
            "SELECT node, community_id FROM graph_leiden"
            " WHERE edge_table = 'disc_comm'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
        ).fetchall()

        communities = {}
        for node, comm_id in results:
            communities.setdefault(comm_id, set()).add(node)

        assert len(communities) == 2
        comm_sets = [frozenset(v) for v in communities.values()]
        assert frozenset({"A", "B", "C"}) in comm_sets
        assert frozenset({"X", "Y", "Z"}) in comm_sets

    def test_modularity_positive(self, conn):
        """Modularity should be positive for well-separated communities."""
        create_barbell_graph(conn)
        results = conn.execute(
            "SELECT modularity FROM graph_leiden"
            " WHERE edge_table = 'barbell'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
        ).fetchall()

        # All rows report the same global modularity
        modularities = [r[0] for r in results]
        assert all(m > 0 for m in modularities)
        # All should be equal (global modularity)
        assert all(abs(m - modularities[0]) < 0.001 for m in modularities)

    def test_weighted_communities(self, conn):
        """Strong intra-community weights should overcome weak bridge."""
        create_weighted_communities(conn)
        results = conn.execute(
            "SELECT node, community_id FROM graph_leiden"
            " WHERE edge_table = 'wcomm'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND weight_col = 'weight'"
        ).fetchall()

        communities = {}
        for node, comm_id in results:
            communities.setdefault(comm_id, set()).add(node)

        assert len(communities) == 2
        comm_sets = [frozenset(v) for v in communities.values()]
        assert frozenset({"A", "B", "C"}) in comm_sets
        assert frozenset({"D", "E", "F"}) in comm_sets

    def test_resolution_parameter(self, conn):
        """Higher resolution should produce more communities."""
        create_barbell_graph(conn)

        # Default resolution (1.0) — 2 communities
        results_default = conn.execute(
            "SELECT DISTINCT community_id FROM graph_leiden"
            " WHERE edge_table = 'barbell'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
        ).fetchall()

        # Very high resolution — should find more (or equal) communities
        results_high = conn.execute(
            "SELECT DISTINCT community_id FROM graph_leiden"
            " WHERE edge_table = 'barbell'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND resolution = 5.0"
        ).fetchall()

        assert len(results_high) >= len(results_default)

    def test_temporal_filter(self, conn):
        """Temporal filtering should only include edges within the time window."""
        create_temporal_communities(conn)

        # Only early edges (2024-01-01 to 2024-04-01)
        results = conn.execute(
            "SELECT node, community_id FROM graph_leiden"
            " WHERE edge_table = 'tcomm'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            "   AND timestamp_col = 'ts'"
            "   AND time_start = '2024-01-01'"
            "   AND time_end = '2024-04-01'"
        ).fetchall()

        nodes = {r[0] for r in results}
        # Only A, B, C edges are within the time window
        assert nodes == {"A", "B", "C"}

    def test_all_nodes_assigned(self, conn):
        """Every node in the graph should get a community assignment."""
        create_barbell_graph(conn)
        results = conn.execute(
            "SELECT node FROM graph_leiden"
            " WHERE edge_table = 'barbell'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
        ).fetchall()

        nodes = {r[0] for r in results}
        assert nodes == {"A", "B", "C", "D", "E", "F"}

    def test_community_ids_contiguous(self, conn):
        """Community IDs should be contiguous starting from 0."""
        create_barbell_graph(conn)
        results = conn.execute(
            "SELECT DISTINCT community_id FROM graph_leiden"
            " WHERE edge_table = 'barbell'"
            "   AND src_col = 'src'"
            "   AND dst_col = 'dst'"
            " ORDER BY community_id"
        ).fetchall()

        comm_ids = [r[0] for r in results]
        assert comm_ids == list(range(len(comm_ids)))
