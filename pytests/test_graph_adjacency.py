"""Integration tests for graph_adjacency virtual table.

Tests the persistent CSR adjacency index: creation, querying, trigger-based
dirty tracking, full and incremental rebuilds, and administrative commands.
"""

try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

import pytest


@pytest.fixture
def adj_conn(conn):
    """Provide a connection with an edge table and graph_adjacency VT."""
    conn.execute("CREATE TABLE edges (src TEXT, dst TEXT, weight REAL DEFAULT 1.0)")
    conn.execute("INSERT INTO edges VALUES ('A', 'B', 1.0)")
    conn.execute("INSERT INTO edges VALUES ('B', 'C', 2.0)")
    conn.execute("INSERT INTO edges VALUES ('C', 'A', 3.0)")

    conn.execute(
        "CREATE VIRTUAL TABLE g USING graph_adjacency("
        "edge_table='edges', src_col='src', dst_col='dst', weight_col='weight')"
    )
    return conn


class TestCreation:
    """Test virtual table creation and shadow tables."""

    def test_create_basic(self, adj_conn):
        """VT creation with edge data should produce queryable results."""
        rows = adj_conn.execute("SELECT node, in_degree, out_degree FROM g ORDER BY node").fetchall()
        assert len(rows) == 3
        nodes = {r[0] for r in rows}
        assert nodes == {"A", "B", "C"}

    def test_create_unweighted(self, conn):
        """VT creation without weight_col should work."""
        conn.execute("CREATE TABLE edges (src TEXT, dst TEXT)")
        conn.execute("INSERT INTO edges VALUES ('X', 'Y')")
        conn.execute("CREATE VIRTUAL TABLE g USING graph_adjacency(edge_table='edges', src_col='src', dst_col='dst')")
        rows = conn.execute("SELECT * FROM g").fetchall()
        assert len(rows) == 2

    def test_create_empty_table(self, conn):
        """VT creation on empty edge table should produce 0 rows."""
        conn.execute("CREATE TABLE edges (src TEXT, dst TEXT)")
        conn.execute("CREATE VIRTUAL TABLE g USING graph_adjacency(edge_table='edges', src_col='src', dst_col='dst')")
        rows = conn.execute("SELECT * FROM g").fetchall()
        assert len(rows) == 0

    def test_shadow_tables_exist(self, adj_conn):
        """Shadow tables should be created."""
        tables = adj_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'g_%'").fetchall()
        names = {t[0] for t in tables}
        assert "g_config" in names
        assert "g_nodes" in names
        assert "g_degree" in names
        assert "g_csr_fwd" in names
        assert "g_csr_rev" in names
        assert "g_delta" in names

    def test_triggers_installed(self, adj_conn):
        """Triggers should be installed on the edge table."""
        triggers = adj_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'g_%'"
        ).fetchall()
        names = {t[0] for t in triggers}
        assert "g_ai" in names
        assert "g_ad" in names
        assert "g_au" in names


class TestDegreeQuery:
    """Test degree sequence queries."""

    def test_degree_values(self, adj_conn):
        """Each node in the triangle should have in_degree=1, out_degree=1."""
        rows = adj_conn.execute("SELECT node, in_degree, out_degree FROM g ORDER BY node").fetchall()
        for node, in_deg, out_deg in rows:
            assert in_deg == 1, f"Node {node} in_degree should be 1, got {in_deg}"
            assert out_deg == 1, f"Node {node} out_degree should be 1, got {out_deg}"

    def test_weighted_degree(self, adj_conn):
        """Weighted degrees should reflect edge weights."""
        rows = adj_conn.execute("SELECT node, weighted_in_degree, weighted_out_degree FROM g ORDER BY node").fetchall()
        degree_map = {r[0]: (r[1], r[2]) for r in rows}
        # A→B(1.0), C→A(3.0) → A: w_out=1.0, w_in=3.0
        assert degree_map["A"][1] == 1.0  # w_out
        assert degree_map["A"][0] == 3.0  # w_in
        # B→C(2.0), A→B(1.0) → B: w_out=2.0, w_in=1.0
        assert degree_map["B"][1] == 2.0
        assert degree_map["B"][0] == 1.0

    def test_node_idx_sequential(self, adj_conn):
        """node_idx values should be sequential starting from 0."""
        rows = adj_conn.execute("SELECT node_idx FROM g ORDER BY node_idx").fetchall()
        indices = [r[0] for r in rows]
        assert indices == [0, 1, 2]


class TestPointLookup:
    """Test node = ? constraint."""

    def test_lookup_existing(self, adj_conn):
        """Point lookup should return exactly one row."""
        rows = adj_conn.execute("SELECT * FROM g WHERE node = 'A'").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "A"

    def test_lookup_nonexistent(self, adj_conn):
        """Point lookup for missing node should return no rows."""
        rows = adj_conn.execute("SELECT * FROM g WHERE node = 'Z'").fetchall()
        assert len(rows) == 0


class TestTriggerTracking:
    """Test that edge modifications trigger delta tracking and auto-rebuild."""

    def test_insert_triggers_rebuild(self, adj_conn):
        """Inserting a new edge should trigger rebuild on next query."""
        adj_conn.execute("INSERT INTO edges VALUES ('D', 'A', 1.0)")
        rows = adj_conn.execute("SELECT node FROM g ORDER BY node").fetchall()
        nodes = [r[0] for r in rows]
        assert "D" in nodes
        assert len(nodes) == 4

    def test_delete_triggers_rebuild(self, adj_conn):
        """Deleting an edge should trigger rebuild on next query."""
        adj_conn.execute("DELETE FROM edges WHERE src = 'C' AND dst = 'A'")
        # After delete: A→B, B→C remain. C has no outgoing edges.
        rows = adj_conn.execute("SELECT node, out_degree FROM g ORDER BY node").fetchall()
        degree_map = {r[0]: r[1] for r in rows}
        assert degree_map["C"] == 0

    def test_update_triggers_rebuild(self, adj_conn):
        """Updating an edge should trigger rebuild on next query."""
        adj_conn.execute("UPDATE edges SET weight = 10.0 WHERE src = 'A' AND dst = 'B'")
        rows = adj_conn.execute("SELECT node, weighted_out_degree FROM g WHERE node = 'A'").fetchall()
        assert rows[0][1] == 10.0

    def test_multiple_inserts_batch(self, adj_conn):
        """Multiple inserts before a query should all be reflected."""
        adj_conn.execute("INSERT INTO edges VALUES ('D', 'E', 1.0)")
        adj_conn.execute("INSERT INTO edges VALUES ('E', 'F', 1.0)")
        adj_conn.execute("INSERT INTO edges VALUES ('F', 'D', 1.0)")
        rows = adj_conn.execute("SELECT node FROM g ORDER BY node").fetchall()
        nodes = [r[0] for r in rows]
        assert set(nodes) == {"A", "B", "C", "D", "E", "F"}


class TestRebuildCommand:
    """Test administrative commands via xUpdate."""

    def test_rebuild_command(self, adj_conn):
        """'rebuild' command should force a full rebuild."""
        adj_conn.execute("INSERT INTO g(g) VALUES ('rebuild')")
        rows = adj_conn.execute("SELECT COUNT(*) FROM g").fetchone()
        assert rows[0] == 3

    def test_incremental_rebuild_command(self, adj_conn):
        """'incremental_rebuild' command should work."""
        adj_conn.execute("INSERT INTO edges VALUES ('D', 'A', 1.0)")
        adj_conn.execute("INSERT INTO g(g) VALUES ('incremental_rebuild')")
        rows = adj_conn.execute("SELECT node FROM g ORDER BY node").fetchall()
        nodes = [r[0] for r in rows]
        assert "D" in nodes

    def test_unknown_command_fails(self, adj_conn):
        """Unknown command should raise an error."""
        with pytest.raises(sqlite3.OperationalError):
            adj_conn.execute("INSERT INTO g(g) VALUES ('bad_command')")


class TestDropAndRename:
    """Test cleanup and rename operations."""

    def test_drop_cleans_up(self, adj_conn):
        """DROP TABLE should remove shadow tables and triggers."""
        adj_conn.execute("DROP TABLE g")
        tables = adj_conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'g_%'").fetchall()
        assert len(tables) == 0

    def test_direct_insert_rejected(self, adj_conn):
        """Direct INSERT into VT (not a command) should be rejected."""
        with pytest.raises(sqlite3.OperationalError):
            adj_conn.execute("INSERT INTO g(node) VALUES ('Z')")


class TestIncrementalMerge:
    """Test Phase 2 incremental merge behavior."""

    def test_small_delta_uses_incremental(self, adj_conn):
        """A small delta (< 10% of edges) should use incremental merge."""
        # Add many edges to make the graph larger
        for i in range(100):
            adj_conn.execute(
                "INSERT INTO edges VALUES (?, ?, 1.0)",
                (f"N{i}", f"N{(i + 1) % 100}"),
            )
        # Force initial build
        adj_conn.execute("INSERT INTO g(g) VALUES ('rebuild')")

        # Now add a single edge — should trigger incremental merge
        adj_conn.execute("INSERT INTO edges VALUES ('N0', 'N50', 1.0)")
        rows = adj_conn.execute("SELECT COUNT(*) FROM g").fetchone()
        # 3 original + 100 ring + N0 appearing with extra edge
        assert rows[0] > 100

    def test_large_delta_uses_full_rebuild(self, adj_conn):
        """A large delta should fall back to full rebuild."""
        # Force rebuild to set baseline
        adj_conn.execute("INSERT INTO g(g) VALUES ('rebuild')")

        # Add many edges (more than 10% of current 3)
        for i in range(10):
            adj_conn.execute(
                "INSERT INTO edges VALUES (?, ?, 1.0)",
                (f"X{i}", f"X{(i + 1) % 10}"),
            )

        # Query should trigger full rebuild due to large delta
        rows = adj_conn.execute("SELECT COUNT(*) FROM g").fetchone()
        assert rows[0] == 13  # 3 original + 10 new


class TestCSRConsistency:
    """Verify CSR shadow table contents are consistent."""

    def test_csr_blobs_exist(self, adj_conn):
        """CSR BLOBs should exist in shadow tables after build."""
        fwd = adj_conn.execute("SELECT offsets, targets FROM g_csr_fwd WHERE block_id=0").fetchone()
        assert fwd is not None
        assert len(fwd[0]) > 0  # offsets BLOB not empty

        rev = adj_conn.execute("SELECT offsets, targets FROM g_csr_rev WHERE block_id=0").fetchone()
        assert rev is not None
        assert len(rev[0]) > 0

    def test_node_registry_consistent(self, adj_conn):
        """Node registry should match graph nodes."""
        nodes = adj_conn.execute("SELECT id FROM g_nodes ORDER BY idx").fetchall()
        node_set = {n[0] for n in nodes}
        assert node_set == {"A", "B", "C"}

    def test_config_has_metadata(self, adj_conn):
        """Config should have edge_table, generation, block_size, etc."""
        config = adj_conn.execute("SELECT key, value FROM g_config").fetchall()
        config_map = dict(config)
        assert config_map["edge_table"] == "edges"
        assert config_map["src_col"] == "src"
        assert config_map["dst_col"] == "dst"
        assert int(config_map["generation"]) >= 1
        assert int(config_map["node_count"]) == 3
        assert int(config_map["edge_count"]) == 3
        assert int(config_map["block_size"]) == 4096


class TestBlockedCSR:
    """Verify Phase 3 blocked CSR storage."""

    def test_small_graph_single_block(self, adj_conn):
        """A graph with 3 nodes fits in one block (block_size=4096)."""
        rows = adj_conn.execute("SELECT block_id FROM g_csr_fwd ORDER BY block_id").fetchall()
        # 3 nodes / 4096 = 1 block (block_id=0)
        assert len(rows) == 1
        assert rows[0][0] == 0

    def test_blocked_storage_preserves_degrees(self, adj_conn):
        """Degree values should be correct with blocked storage."""
        rows = adj_conn.execute("SELECT node, out_degree, in_degree FROM g ORDER BY node").fetchall()
        by_node = {r[0]: (r[1], r[2]) for r in rows}
        assert by_node["A"] == (1, 1)  # A→B, C→A
        assert by_node["B"] == (1, 1)  # B→C, A→B
        assert by_node["C"] == (1, 1)  # C→A, B→C

    def test_blocked_incremental_rebuild(self, adj_conn):
        """Incremental rebuild with blocked storage should work."""
        # Build a larger graph to have multiple nodes
        for i in range(50):
            adj_conn.execute(
                "INSERT INTO edges VALUES (?, ?, 1.0)",
                (f"R{i}", f"R{(i + 1) % 50}"),
            )
        adj_conn.execute("INSERT INTO g(g) VALUES ('rebuild')")

        # Check node count
        nc = adj_conn.execute("SELECT value FROM g_config WHERE key='node_count'").fetchone()
        assert int(nc[0]) == 53  # 3 original + 50 ring

        # Add a small delta
        adj_conn.execute("INSERT INTO edges VALUES ('R0', 'R25', 1.0)")
        adj_conn.execute("INSERT INTO g(g) VALUES ('incremental_rebuild')")

        # Verify the new edge is reflected
        rows = adj_conn.execute("SELECT out_degree FROM g WHERE node='R0'").fetchone()
        assert rows[0] >= 2  # at least R0→R1 and R0→R25

    def test_tvf_reads_blocked_csr(self, adj_conn):
        """Algorithm TVFs should work with blocked CSR storage."""
        rows = adj_conn.execute("SELECT COUNT(*) FROM graph_degree('g', 'src', 'dst', 'weight')").fetchone()
        assert rows[0] == 3  # 3 nodes
