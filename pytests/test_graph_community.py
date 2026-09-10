"""
Integration tests for graph community detection table-valued functions.

Tests graph_leiden on synthetic graphs with known community structure, and
graph_conductance against the two-triangle graph from docs/centrality-community.md.
"""

import pysqlite3 as sqlite3
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
        ("A", "B"),
        ("B", "A"),
        ("A", "C"),
        ("C", "A"),
        ("B", "C"),
        ("C", "B"),
        # Clique 2: D, E, F
        ("D", "E"),
        ("E", "D"),
        ("D", "F"),
        ("F", "D"),
        ("E", "F"),
        ("F", "E"),
        # Bridge
        ("C", "D"),
        ("D", "C"),
    ]
    conn.executemany("INSERT INTO barbell VALUES (?, ?)", edges)


def create_triangle_graph(conn):
    """A single triangle: should be one community."""
    conn.execute("CREATE TABLE tri (src TEXT, dst TEXT)")
    edges = [
        ("A", "B"),
        ("B", "A"),
        ("A", "C"),
        ("C", "A"),
        ("B", "C"),
        ("C", "B"),
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
        ("A", "B"),
        ("B", "A"),
        ("A", "C"),
        ("C", "A"),
        ("B", "C"),
        ("C", "B"),
        # Community 2
        ("X", "Y"),
        ("Y", "X"),
        ("X", "Z"),
        ("Z", "X"),
        ("Y", "Z"),
        ("Z", "Y"),
    ]
    conn.executemany("INSERT INTO disc_comm VALUES (?, ?)", edges)


def create_weighted_communities(conn):
    """
    Two groups with weak inter-community edges and strong intra-community edges.
    """
    conn.execute("CREATE TABLE wcomm (src TEXT, dst TEXT, weight REAL)")
    edges = [
        # Strong clique 1
        ("A", "B", 10.0),
        ("B", "A", 10.0),
        ("A", "C", 10.0),
        ("C", "A", 10.0),
        ("B", "C", 10.0),
        ("C", "B", 10.0),
        # Strong clique 2
        ("D", "E", 10.0),
        ("E", "D", 10.0),
        ("D", "F", 10.0),
        ("F", "D", 10.0),
        ("E", "F", 10.0),
        ("F", "E", 10.0),
        # Weak bridge
        ("C", "D", 0.1),
        ("D", "C", 0.1),
    ]
    conn.executemany("INSERT INTO wcomm VALUES (?, ?, ?)", edges)


def create_temporal_communities(conn):
    """Graph with timestamps for temporal filtering tests."""
    conn.execute("CREATE TABLE tcomm (src TEXT, dst TEXT, ts TEXT)")
    edges = [
        # Early edges form one group
        ("A", "B", "2024-01-01"),
        ("B", "A", "2024-01-01"),
        ("A", "C", "2024-02-01"),
        ("C", "A", "2024-02-01"),
        ("B", "C", "2024-03-01"),
        ("C", "B", "2024-03-01"),
        # Late edges form another group
        ("D", "E", "2024-07-01"),
        ("E", "D", "2024-07-01"),
        ("D", "F", "2024-08-01"),
        ("F", "D", "2024-08-01"),
        ("E", "F", "2024-09-01"),
        ("F", "E", "2024-09-01"),
        # Bridge connecting the groups (mid-year)
        ("C", "D", "2024-05-01"),
        ("D", "C", "2024-05-01"),
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
            "SELECT node FROM graph_leiden WHERE edge_table = 'barbell'   AND src_col = 'src'   AND dst_col = 'dst'"
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


# ═══════════════════════════════════════════════════════════════
# graph_conductance
# ═══════════════════════════════════════════════════════════════


def create_two_triangles(conn):
    """
    Two triangles (alice, bob, carol, dave) and (eve, frank, grace) joined by
    the single bridge dave -> eve. Same graph as docs/centrality-community.md.

    total_vol = 18. left: internal 5, cut 1, vol 11. right: internal 3, cut 1, vol 7.
    """
    conn.execute("CREATE TABLE edges (src TEXT, dst TEXT, weight REAL DEFAULT 1.0, ts TEXT)")
    edges = [
        ("alice", "bob", 1.0, "2026-01-10"),
        ("alice", "carol", 1.0, "2026-01-10"),
        ("bob", "carol", 1.0, "2026-01-10"),
        ("bob", "dave", 1.0, "2026-02-10"),
        ("carol", "dave", 1.0, "2026-02-10"),
        ("dave", "eve", 1.0, "2026-03-10"),  # bridge
        ("eve", "frank", 1.0, "2026-04-10"),
        ("eve", "grace", 1.0, "2026-04-10"),
        ("frank", "grace", 1.0, "2026-04-10"),
    ]
    conn.executemany("INSERT INTO edges VALUES (?, ?, ?, ?)", edges)


def create_assignment(conn):
    conn.execute("CREATE TABLE assignment (node TEXT, team TEXT)")
    conn.executemany(
        "INSERT INTO assignment VALUES (?, ?)",
        [
            ("alice", "left"),
            ("bob", "left"),
            ("carol", "left"),
            ("dave", "left"),
            ("eve", "right"),
            ("frank", "right"),
            ("grace", "right"),
        ],
    )


CONDUCTANCE_SQL = (
    "SELECT group_id, size, internal, cut, vol, phi FROM graph_conductance"
    " WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'"
    "   AND direction = 'both'"
    "   AND membership_table = 'assignment' AND group_col = 'team' AND member_col = 'node'"
    " ORDER BY group_id"
)


class TestGraphConductance:
    def test_declared_membership_matches_issue_table(self, conn):
        """Values from the issue #31 worked example."""
        create_two_triangles(conn)
        create_assignment(conn)
        rows = conn.execute(CONDUCTANCE_SQL).fetchall()

        assert [r[0] for r in rows] == ["left", "right"]
        left, right = rows
        assert left[1:5] == (4, 5.0, 1.0, 11.0)
        assert right[1:5] == (3, 3.0, 1.0, 7.0)
        assert abs(left[5] - 1 / 7) < 1e-9
        assert abs(right[5] - 1 / 7) < 1e-9

    def test_output_columns_and_types(self, conn):
        create_two_triangles(conn)
        create_assignment(conn)
        cur = conn.execute(CONDUCTANCE_SQL)
        assert [d[0] for d in cur.description] == ["group_id", "size", "internal", "cut", "vol", "phi"]
        row = cur.fetchone()
        assert isinstance(row[0], str)
        assert isinstance(row[1], int)
        assert all(isinstance(v, float) for v in row[2:])

    def test_scores_leiden_partition(self, conn):
        """Composition with graph_leiden: INTEGER community IDs are accepted."""
        create_two_triangles(conn)
        conn.execute(
            "CREATE TEMP TABLE discovered AS"
            " SELECT node, community_id FROM graph_leiden"
            "  WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'"
            "    AND direction = 'both' AND resolution = 1.0"
        )
        rows = conn.execute(
            "SELECT group_id, size, phi FROM graph_conductance"
            " WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'"
            "   AND direction = 'both'"
            "   AND membership_table = 'discovered'"
            "   AND group_col = 'community_id' AND member_col = 'node'"
            " ORDER BY group_id"
        ).fetchall()

        # Leiden finds the two triangles; both sides have phi = 1/7
        assert len(rows) == 2
        assert {r[0] for r in rows} == {"0", "1"}
        assert sum(r[1] for r in rows) == 7
        assert all(abs(r[2] - 1 / 7) < 1e-9 for r in rows)

    def test_ungrouped_nodes_count_toward_cut(self, conn):
        create_two_triangles(conn)
        create_assignment(conn)
        conn.execute("DELETE FROM assignment WHERE node = 'grace'")
        rows = {r[0]: r[1:] for r in conn.execute(CONDUCTANCE_SQL).fetchall()}

        assert rows["left"] == (4, 5.0, 1.0, 11.0, rows["left"][4])
        assert abs(rows["left"][4] - 1 / 7) < 1e-9
        # right = {eve, frank}: internal 1, cut 3 (dave-eve, eve-grace, frank-grace), vol 5
        assert rows["right"][:4] == (2, 1.0, 3.0, 5.0)
        assert abs(rows["right"][4] - 3 / 5) < 1e-9

    def test_members_not_in_graph_are_ignored(self, conn):
        create_two_triangles(conn)
        create_assignment(conn)
        conn.execute("INSERT INTO assignment VALUES ('zed', 'left'), ('nobody', 'ghost')")
        rows = conn.execute(CONDUCTANCE_SQL).fetchall()

        assert [r[0] for r in rows] == ["left", "right"]
        assert rows[0][1] == 4  # zed did not inflate size

    def test_singleton_with_no_internal_edges_is_one(self, conn):
        create_two_triangles(conn)
        conn.execute("CREATE TABLE assignment (node TEXT, team TEXT)")
        conn.execute("INSERT INTO assignment VALUES ('dave', 'solo')")
        (row,) = conn.execute(CONDUCTANCE_SQL).fetchall()
        assert row == ("solo", 1, 0.0, 3.0, 3.0, 1.0)

    def test_whole_graph_is_zero_not_nan(self, conn):
        create_two_triangles(conn)
        conn.execute("CREATE TABLE assignment (node TEXT, team TEXT)")
        conn.execute("INSERT INTO assignment SELECT DISTINCT src, 'all' FROM edges")
        conn.execute("INSERT INTO assignment SELECT DISTINCT dst, 'all' FROM edges")
        (row,) = conn.execute(CONDUCTANCE_SQL).fetchall()
        assert row == ("all", 7, 9.0, 0.0, 18.0, 0.0)

    def test_weighted(self, conn):
        create_two_triangles(conn)
        create_assignment(conn)
        conn.execute("UPDATE edges SET weight = 0.25 WHERE src = 'dave' AND dst = 'eve'")
        rows = conn.execute(
            "SELECT group_id, internal, cut, vol, phi FROM graph_conductance"
            " WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'"
            "   AND weight_col = 'weight' AND direction = 'both'"
            "   AND membership_table = 'assignment' AND group_col = 'team' AND member_col = 'node'"
            " ORDER BY group_id"
        ).fetchall()
        left, right = rows
        assert left[1:4] == (5.0, 0.25, 10.25)
        assert right[1:4] == (3.0, 0.25, 6.25)
        assert abs(left[4] - 0.25 / 6.25) < 1e-9
        assert abs(right[4] - 0.25 / 6.25) < 1e-9

    def test_temporal_filter(self, conn):
        """Only January edges: the left triangle exists, no bridge, so phi = 0."""
        create_two_triangles(conn)
        create_assignment(conn)
        rows = conn.execute(
            "SELECT group_id, size, internal, cut, phi FROM graph_conductance"
            " WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'"
            "   AND direction = 'both' AND timestamp_col = 'ts'"
            "   AND time_start = '2026-01-01' AND time_end = '2026-01-31'"
            "   AND membership_table = 'assignment' AND group_col = 'team' AND member_col = 'node'"
        ).fetchall()
        assert rows == [("left", 3, 3.0, 0.0, 0.0)]

    def test_volume_floor_filter(self, conn):
        create_two_triangles(conn)
        create_assignment(conn)
        rows = conn.execute(CONDUCTANCE_SQL.replace(" ORDER BY", " AND vol >= 10 ORDER BY")).fetchall()
        assert [r[0] for r in rows] == ["left"]

    def test_direction_forward_and_reverse_match_both(self, conn):
        create_two_triangles(conn)
        create_assignment(conn)
        both = conn.execute(CONDUCTANCE_SQL).fetchall()
        fwd = conn.execute(CONDUCTANCE_SQL.replace("'both'", "'forward'")).fetchall()
        rev = conn.execute(CONDUCTANCE_SQL.replace("'both'", "'reverse'")).fetchall()
        assert fwd == both
        assert rev == both

    def test_graph_adjacency_as_edge_table(self, conn):
        create_two_triangles(conn)
        create_assignment(conn)
        conn.execute("CREATE VIRTUAL TABLE g USING graph_adjacency(edge_table='edges', src_col='src', dst_col='dst')")
        rows = conn.execute(CONDUCTANCE_SQL.replace("edge_table = 'edges'", "edge_table = 'g'")).fetchall()
        assert rows == conn.execute(CONDUCTANCE_SQL).fetchall()

    def test_empty_membership_yields_no_rows(self, conn):
        create_two_triangles(conn)
        conn.execute("CREATE TABLE assignment (node TEXT, team TEXT)")
        assert conn.execute(CONDUCTANCE_SQL).fetchall() == []

    def test_missing_membership_constraints_errors(self, conn):
        create_two_triangles(conn)
        with pytest.raises(sqlite3.OperationalError, match="membership_table"):
            conn.execute(
                "SELECT * FROM graph_conductance WHERE edge_table = 'edges' AND src_col = 'src' AND dst_col = 'dst'"
            ).fetchall()

    def test_invalid_identifier_rejected(self, conn):
        create_two_triangles(conn)
        create_assignment(conn)
        with pytest.raises(sqlite3.OperationalError, match="invalid"):
            conn.execute(CONDUCTANCE_SQL.replace("'assignment'", "'assignment; DROP TABLE edges'")).fetchall()
        assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 9
