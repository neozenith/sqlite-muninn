"""
Integration tests for the HNSW virtual table.

Tests the full stack: extension loading → CREATE VIRTUAL TABLE →
INSERT → KNN search → persistence → DELETE.
"""
import math
import random
import struct

import pytest


def make_vector(values: list[float]) -> bytes:
    """Pack a list of floats into a BLOB (little-endian float32 array)."""
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes, dim: int) -> list[float]:
    """Unpack a BLOB into a list of floats."""
    return list(struct.unpack(f"<{dim}f", blob))


def brute_force_knn(
    query: list[float],
    vectors: dict[int, list[float]],
    k: int,
) -> list[tuple[int, float]]:
    """Brute-force KNN using L2 squared distance."""
    dists = []
    for vid, vec in vectors.items():
        d = sum((a - b) ** 2 for a, b in zip(query, vec))
        dists.append((vid, d))
    dists.sort(key=lambda x: x[1])
    return dists[:k]


class TestHnswVtabCreate:
    def test_create_basic(self, conn):
        conn.execute(
            "CREATE VIRTUAL TABLE test_vec USING hnsw_index(dimensions=4, metric='l2')"
        )
        # Shadow tables should exist
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "test_vec_config" in tables
        assert "test_vec_nodes" in tables
        assert "test_vec_edges" in tables

    def test_create_missing_dimensions(self, conn):
        with pytest.raises(Exception, match="dimensions.*required"):
            conn.execute("CREATE VIRTUAL TABLE bad USING hnsw_index(metric='l2')")

    def test_create_invalid_metric(self, conn):
        with pytest.raises(Exception, match="unknown metric"):
            conn.execute(
                "CREATE VIRTUAL TABLE bad USING hnsw_index(dimensions=4, metric='hamming')"
            )

    def test_create_invalid_dimensions(self, conn):
        with pytest.raises(Exception, match="dimensions must be > 0"):
            conn.execute(
                "CREATE VIRTUAL TABLE bad USING hnsw_index(dimensions=0)"
            )

    def test_create_unknown_param(self, conn):
        with pytest.raises(Exception, match="unknown parameter"):
            conn.execute(
                "CREATE VIRTUAL TABLE bad USING hnsw_index(dimensions=4, foobar=1)"
            )


class TestHnswInsert:
    def test_insert_single(self, conn):
        conn.execute(
            "CREATE VIRTUAL TABLE vec USING hnsw_index(dimensions=3, metric='l2')"
        )
        vec = make_vector([1.0, 2.0, 3.0])
        conn.execute("INSERT INTO vec (rowid, vector) VALUES (1, ?)", (vec,))

        # Verify it's stored in shadow table
        row = conn.execute("SELECT id, level FROM vec_nodes WHERE id=1").fetchone()
        assert row is not None
        assert row[0] == 1

    def test_insert_wrong_size(self, conn):
        conn.execute(
            "CREATE VIRTUAL TABLE vec USING hnsw_index(dimensions=3, metric='l2')"
        )
        bad_vec = make_vector([1.0, 2.0])  # 2-dim instead of 3
        with pytest.raises(Exception, match="expected 3-dim"):
            conn.execute("INSERT INTO vec (rowid, vector) VALUES (1, ?)", (bad_vec,))

    def test_insert_multiple(self, conn):
        conn.execute(
            "CREATE VIRTUAL TABLE vec USING hnsw_index(dimensions=4, metric='l2', m=4)"
        )
        for i in range(20):
            vec = make_vector([float(i), float(i + 1), float(i + 2), float(i + 3)])
            conn.execute("INSERT INTO vec (rowid, vector) VALUES (?, ?)", (i, vec))

        count = conn.execute("SELECT COUNT(*) FROM vec_nodes").fetchone()[0]
        assert count == 20


class TestHnswSearch:
    def test_knn_basic(self, conn):
        """Insert 3 known points, search for nearest to one of them."""
        conn.execute(
            "CREATE VIRTUAL TABLE vec USING hnsw_index(dimensions=2, metric='l2')"
        )
        conn.execute(
            "INSERT INTO vec (rowid, vector) VALUES (1, ?)",
            (make_vector([0.0, 0.0]),),
        )
        conn.execute(
            "INSERT INTO vec (rowid, vector) VALUES (2, ?)",
            (make_vector([10.0, 0.0]),),
        )
        conn.execute(
            "INSERT INTO vec (rowid, vector) VALUES (3, ?)",
            (make_vector([0.0, 10.0]),),
        )

        # Search near origin
        query = make_vector([0.1, 0.1])
        results = conn.execute(
            "SELECT rowid, distance FROM vec WHERE vector MATCH ? AND k = 2",
            (query,),
        ).fetchall()

        assert len(results) == 2
        assert results[0][0] == 1  # origin is closest
        assert results[0][1] < 1.0  # very close

    def test_knn_recall_100_vectors(self, conn):
        """Insert 100 random 8-dim vectors, verify top-5 matches brute force."""
        dim = 8
        k = 5
        random.seed(42)

        conn.execute(
            f"CREATE VIRTUAL TABLE vec USING hnsw_index("
            f"dimensions={dim}, metric='l2', m=16, ef_construction=200"
            f")"
        )

        vectors = {}
        for i in range(100):
            v = [random.gauss(0, 1) for _ in range(dim)]
            vectors[i] = v
            conn.execute(
                "INSERT INTO vec (rowid, vector) VALUES (?, ?)",
                (i, make_vector(v)),
            )

        # Random query
        query = [random.gauss(0, 1) for _ in range(dim)]
        query_blob = make_vector(query)

        results = conn.execute(
            "SELECT rowid, distance FROM vec WHERE vector MATCH ? AND k = ? AND ef_search = 64",
            (query_blob, k),
        ).fetchall()

        assert len(results) == k

        # Compare with brute force
        bf = brute_force_knn(query, vectors, k)
        bf_ids = {r[0] for r in bf}
        hnsw_ids = {r[0] for r in results}

        recall = len(bf_ids & hnsw_ids) / k
        assert recall >= 0.8, f"Recall {recall} < 0.8"

    def test_knn_cosine(self, conn):
        """Verify cosine metric works correctly."""
        conn.execute(
            "CREATE VIRTUAL TABLE vec USING hnsw_index(dimensions=2, metric='cosine')"
        )
        # Insert vectors in different directions
        conn.execute(
            "INSERT INTO vec (rowid, vector) VALUES (1, ?)",
            (make_vector([1.0, 0.0]),),
        )  # right
        conn.execute(
            "INSERT INTO vec (rowid, vector) VALUES (2, ?)",
            (make_vector([0.0, 1.0]),),
        )  # up
        conn.execute(
            "INSERT INTO vec (rowid, vector) VALUES (3, ?)",
            (make_vector([-1.0, 0.0]),),
        )  # left

        # Query slightly right — should match node 1 first
        query = make_vector([0.95, 0.05])
        results = conn.execute(
            "SELECT rowid, distance FROM vec WHERE vector MATCH ? AND k = 3",
            (query,),
        ).fetchall()

        assert results[0][0] == 1  # right is closest by cosine

    def test_search_empty_table(self, conn):
        conn.execute(
            "CREATE VIRTUAL TABLE vec USING hnsw_index(dimensions=2, metric='l2')"
        )
        query = make_vector([0.0, 0.0])
        results = conn.execute(
            "SELECT rowid, distance FROM vec WHERE vector MATCH ? AND k = 5",
            (query,),
        ).fetchall()
        assert len(results) == 0


class TestHnswDelete:
    def test_delete_basic(self, conn):
        conn.execute(
            "CREATE VIRTUAL TABLE vec USING hnsw_index(dimensions=2, metric='l2')"
        )
        conn.execute(
            "INSERT INTO vec (rowid, vector) VALUES (1, ?)",
            (make_vector([0.0, 0.0]),),
        )
        conn.execute(
            "INSERT INTO vec (rowid, vector) VALUES (2, ?)",
            (make_vector([1.0, 0.0]),),
        )
        conn.execute(
            "INSERT INTO vec (rowid, vector) VALUES (3, ?)",
            (make_vector([2.0, 0.0]),),
        )

        conn.execute("DELETE FROM vec WHERE rowid = 2")

        # Search should not return deleted node
        query = make_vector([1.0, 0.0])  # closest to deleted node
        results = conn.execute(
            "SELECT rowid FROM vec WHERE vector MATCH ? AND k = 3",
            (query,),
        ).fetchall()

        result_ids = {r[0] for r in results}
        assert 2 not in result_ids
        assert len(results) == 2  # only 2 remain


class TestHnswPointLookup:
    def test_lookup_existing(self, conn):
        conn.execute(
            "CREATE VIRTUAL TABLE vec USING hnsw_index(dimensions=3, metric='l2')"
        )
        original = [1.5, 2.5, 3.5]
        conn.execute(
            "INSERT INTO vec (rowid, vector) VALUES (42, ?)",
            (make_vector(original),),
        )

        row = conn.execute(
            "SELECT vector FROM vec WHERE rowid = 42"
        ).fetchone()
        assert row is not None
        retrieved = unpack_vector(row[0], 3)
        for a, b in zip(original, retrieved):
            assert abs(a - b) < 1e-6


class TestHnswPersistence:
    def test_persistence_via_file(self, tmp_path):
        """Create index, close, reopen, verify search still works."""
        import sqlite3

        db_path = str(tmp_path / "test.db")
        ext_path = __import__("os").path.join(
            __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))),
            "muninn",
        )

        # Create and populate
        conn1 = sqlite3.connect(db_path)
        conn1.enable_load_extension(True)
        conn1.load_extension(ext_path)
        conn1.execute(
            "CREATE VIRTUAL TABLE vec USING hnsw_index(dimensions=3, metric='l2', m=4)"
        )
        for i in range(10):
            v = make_vector([float(i), float(i * 2), float(i * 3)])
            conn1.execute("INSERT INTO vec (rowid, vector) VALUES (?, ?)", (i, v))
        conn1.commit()
        conn1.close()

        # Reopen and search
        conn2 = sqlite3.connect(db_path)
        conn2.enable_load_extension(True)
        conn2.load_extension(ext_path)

        query = make_vector([0.0, 0.0, 0.0])
        results = conn2.execute(
            "SELECT rowid, distance FROM vec WHERE vector MATCH ? AND k = 3",
            (query,),
        ).fetchall()

        assert len(results) == 3
        assert results[0][0] == 0  # origin is closest
        conn2.close()


class TestHnswDrop:
    def test_drop_cleans_shadow_tables(self, conn):
        conn.execute(
            "CREATE VIRTUAL TABLE vec USING hnsw_index(dimensions=2, metric='l2')"
        )
        conn.execute(
            "INSERT INTO vec (rowid, vector) VALUES (1, ?)",
            (make_vector([1.0, 2.0]),),
        )

        conn.execute("DROP TABLE vec")

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "vec_config" not in tables
        assert "vec_nodes" not in tables
        assert "vec_edges" not in tables
