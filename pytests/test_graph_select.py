"""
Integration tests for graph_select TVF — dbt-inspired selector syntax.

Tests the full pipeline: SQL → graph_select TVF → parser → evaluator → results.
"""

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

import pytest


def create_dag(conn):
    """
    Create a DAG:
        A → C → D
        B → C → E → F
        X → Y → E
    """
    conn.execute("CREATE TABLE edges (src TEXT, dst TEXT)")
    conn.executemany(
        "INSERT INTO edges VALUES (?, ?)",
        [
            ("A", "C"),
            ("B", "C"),
            ("C", "D"),
            ("C", "E"),
            ("E", "F"),
            ("X", "Y"),
            ("Y", "E"),
        ],
    )


def get_nodes(conn, selector):
    """Run graph_select and return the set of selected node names."""
    rows = conn.execute(
        "SELECT node, depth, direction FROM graph_select('edges', 'src', 'dst', ?)",
        (selector,),
    ).fetchall()
    return {row[0] for row in rows}


def get_rows(conn, selector):
    """Run graph_select and return full rows as list of tuples."""
    return conn.execute(
        "SELECT node, depth, direction FROM graph_select('edges', 'src', 'dst', ?)",
        (selector,),
    ).fetchall()


class TestBareNode:
    def test_single_node(self, conn):
        create_dag(conn)
        nodes = get_nodes(conn, "C")
        assert nodes == {"C"}

    def test_single_node_depth(self, conn):
        create_dag(conn)
        rows = get_rows(conn, "C")
        assert len(rows) == 1
        assert rows[0][1] == 0  # depth
        assert rows[0][2] == "self"  # direction


class TestDescendants:
    def test_all_descendants(self, conn):
        create_dag(conn)
        nodes = get_nodes(conn, "C+")
        assert nodes == {"C", "D", "E", "F"}

    def test_depth_limited(self, conn):
        create_dag(conn)
        nodes = get_nodes(conn, "C+1")
        assert nodes == {"C", "D", "E"}
        assert "F" not in nodes  # 2 hops away


class TestAncestors:
    def test_all_ancestors(self, conn):
        create_dag(conn)
        nodes = get_nodes(conn, "+C")
        assert nodes == {"A", "B", "C"}

    def test_depth_limited_ancestors(self, conn):
        create_dag(conn)
        nodes = get_nodes(conn, "1+E")
        # 1 hop backward from E: C and Y are direct predecessors
        assert "E" in nodes
        assert "C" in nodes
        assert "Y" in nodes
        assert "A" not in nodes  # 2 hops away


class TestBoth:
    def test_both_unlimited(self, conn):
        create_dag(conn)
        nodes = get_nodes(conn, "+C+")
        assert nodes == {"A", "B", "C", "D", "E", "F"}
        assert "X" not in nodes
        assert "Y" not in nodes

    def test_depth_limited_both(self, conn):
        create_dag(conn)
        nodes = get_nodes(conn, "1+C+1")
        # 1 hop up: A, B; 1 hop down: D, E; self: C
        assert nodes == {"A", "B", "C", "D", "E"}
        assert "F" not in nodes  # 2 hops down


class TestClosure:
    def test_build_closure(self, conn):
        create_dag(conn)
        nodes = get_nodes(conn, "@C")
        # @C = descendants of C + all ancestors of those descendants
        # Descendants of C: {C, D, E, F}
        # Ancestors of D: {A, B, C}
        # Ancestors of E: {A, B, C, X, Y}
        # Ancestors of F: {A, B, C, E, X, Y}
        # Total: all 8 nodes
        assert nodes == {"A", "B", "C", "D", "E", "F", "X", "Y"}


class TestUnion:
    def test_union_bare_nodes(self, conn):
        create_dag(conn)
        nodes = get_nodes(conn, "A B")
        assert nodes == {"A", "B"}

    def test_union_lineage(self, conn):
        create_dag(conn)
        nodes = get_nodes(conn, "+D +X")
        # +D = {A, B, C, D}; +X = {X}
        assert nodes == {"A", "B", "C", "D", "X"}


class TestIntersection:
    def test_common_ancestors(self, conn):
        create_dag(conn)
        nodes = get_nodes(conn, "+D,+E")
        # +D = {A, B, C, D}; +E = {A, B, C, E, X, Y}
        # Intersection = {A, B, C}
        assert nodes == {"A", "B", "C"}


class TestComplement:
    def test_not_single(self, conn):
        create_dag(conn)
        nodes = get_nodes(conn, "not C")
        assert "C" not in nodes
        assert len(nodes) == 7  # all except C

    def test_not_closure(self, conn):
        create_dag(conn)
        nodes = get_nodes(conn, "not @C")
        # @C selects all 8 nodes, so not @C is empty
        assert len(nodes) == 0


class TestComposite:
    def test_union_of_closures(self, conn):
        create_dag(conn)
        # @A = descendants of A + ancestors of those descendants
        # Descendants of A: {A, C, D, E, F}
        # Ancestors of those: {A, B, C, X, Y, E}
        # = all 8 nodes
        # So @A @X should also be all 8
        nodes = get_nodes(conn, "@A @X")
        assert len(nodes) == 8


class TestErrors:
    def test_empty_selector(self, conn):
        create_dag(conn)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("SELECT * FROM graph_select('edges', 'src', 'dst', '')").fetchall()

    def test_nonexistent_node(self, conn):
        create_dag(conn)
        with pytest.raises(sqlite3.OperationalError, match="not found"):
            conn.execute("SELECT * FROM graph_select('edges', 'src', 'dst', 'nonexistent')").fetchall()

    def test_invalid_table(self, conn):
        create_dag(conn)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("SELECT * FROM graph_select('no_such_table', 'src', 'dst', 'A')").fetchall()
