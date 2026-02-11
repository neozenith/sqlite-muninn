"""
Integration tests for Node2Vec graph embedding training.

Tests the full pipeline: edge table → biased walks → SGNS → HNSW embeddings.
Verifies on Zachary's Karate Club graph that structurally similar nodes
get similar embeddings.
"""
import struct
import math
import pytest


def cosine_similarity(a, b):
    """Compute cosine similarity between two float vectors (raw bytes)."""
    dim = len(a) // 4
    va = struct.unpack(f"{dim}f", a)
    vb = struct.unpack(f"{dim}f", b)
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


def create_two_cliques(conn):
    """
    Create two 4-node cliques connected by a single bridge edge:
        Clique A: 1-2, 1-3, 1-4, 2-3, 2-4, 3-4
        Clique B: 5-6, 5-7, 5-8, 6-7, 6-8, 7-8
        Bridge:   4-5
    Nodes within the same clique should get similar embeddings.
    """
    conn.execute("CREATE TABLE clique_edges (src TEXT, dst TEXT)")
    edges = [
        # Clique A
        ("1", "2"), ("1", "3"), ("1", "4"),
        ("2", "3"), ("2", "4"), ("3", "4"),
        # Clique B
        ("5", "6"), ("5", "7"), ("5", "8"),
        ("6", "7"), ("6", "8"), ("7", "8"),
        # Bridge
        ("4", "5"),
    ]
    conn.executemany("INSERT INTO clique_edges VALUES (?, ?)", edges)


KARATE_EDGES = [
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
    (1, 11), (1, 12), (1, 13), (1, 14), (1, 18), (1, 20), (1, 22), (1, 32),
    (2, 3), (2, 4), (2, 8), (2, 14), (2, 18), (2, 20), (2, 22), (2, 31),
    (3, 4), (3, 8), (3, 9), (3, 10), (3, 14), (3, 28), (3, 29), (3, 33),
    (4, 8), (4, 13), (4, 14),
    (5, 7), (5, 11),
    (6, 7), (6, 11), (6, 17),
    (7, 17),
    (9, 31), (9, 33), (9, 34),
    (10, 34),
    (14, 34),
    (15, 33), (15, 34),
    (16, 33), (16, 34),
    (19, 33), (19, 34),
    (20, 34),
    (21, 33), (21, 34),
    (23, 33),
    (24, 26), (24, 28), (24, 30), (24, 33), (24, 34),
    (25, 26), (25, 28), (25, 32),
    (26, 32),
    (27, 30), (27, 34),
    (28, 34),
    (29, 32), (29, 34),
    (30, 33), (30, 34),
    (31, 33), (31, 34),
    (32, 33), (32, 34),
    (33, 34),
]


def create_karate_club(conn):
    """
    Zachary's Karate Club: 34 nodes, 78 edges.
    Two known communities:
      - Community A (Mr Hi):  {1,2,3,4,5,6,7,8,11,12,13,14,17,18,20,22}
      - Community B (Officer): {9,10,15,16,19,21,23,24,25,26,27,28,29,30,31,32,33,34}
    """
    conn.execute("CREATE TABLE karate_edges (src TEXT, dst TEXT)")
    edges = [(str(a), str(b)) for a, b in KARATE_EDGES]
    conn.executemany("INSERT INTO karate_edges VALUES (?, ?)", edges)


class TestNode2VecBasic:
    def test_train_returns_count(self, conn):
        """node2vec_train should return the number of nodes embedded."""
        create_two_cliques(conn)

        # Create HNSW output table
        conn.execute(
            "CREATE VIRTUAL TABLE emb USING hnsw_index("
            "dimensions=16, metric='cosine', m=4, ef_construction=20)"
        )

        result = conn.execute(
            "SELECT node2vec_train("
            "'clique_edges', 'src', 'dst', 'emb', "
            "16, 1.0, 1.0, 5, 20, 3, 3, 0.025, 3)"
        ).fetchone()

        assert result[0] == 8  # 8 nodes in the two-clique graph

    def test_embeddings_stored_in_hnsw(self, conn):
        """After training, embeddings should be retrievable from the HNSW table."""
        create_two_cliques(conn)
        conn.execute(
            "CREATE VIRTUAL TABLE emb USING hnsw_index("
            "dimensions=16, metric='cosine', m=4, ef_construction=20)"
        )
        conn.execute(
            "SELECT node2vec_train("
            "'clique_edges', 'src', 'dst', 'emb', "
            "16, 1.0, 1.0, 5, 20, 3, 3, 0.025, 3)"
        )

        # Check we can look up each embedding
        for rowid in range(1, 9):
            row = conn.execute(
                "SELECT vector FROM emb WHERE rowid = ?", (rowid,)
            ).fetchone()
            assert row is not None
            assert len(row[0]) == 16 * 4  # 16 floats * 4 bytes

    def test_empty_graph(self, conn):
        """Training on an empty graph should return 0."""
        conn.execute("CREATE TABLE empty_edges (src TEXT, dst TEXT)")
        conn.execute(
            "CREATE VIRTUAL TABLE emb USING hnsw_index("
            "dimensions=8, metric='cosine', m=4, ef_construction=10)"
        )

        result = conn.execute(
            "SELECT node2vec_train("
            "'empty_edges', 'src', 'dst', 'emb', "
            "8, 1.0, 1.0, 5, 10, 3, 3, 0.025, 1)"
        ).fetchone()

        assert result[0] == 0


class TestNode2VecCommunityDetection:
    def test_two_cliques_within_vs_between(self, conn):
        """
        Nodes within the same clique should have higher cosine similarity
        than nodes in different cliques.
        """
        create_two_cliques(conn)
        conn.execute(
            "CREATE VIRTUAL TABLE emb USING hnsw_index("
            "dimensions=32, metric='cosine', m=8, ef_construction=50)"
        )
        conn.execute(
            "SELECT node2vec_train("
            "'clique_edges', 'src', 'dst', 'emb', "
            "32, 1.0, 1.0, 10, 40, 5, 5, 0.025, 5)"
        )

        # Get embeddings for nodes 1 (clique A) and 5 (clique B)
        emb = {}
        for rowid in range(1, 9):
            row = conn.execute(
                "SELECT vector FROM emb WHERE rowid = ?", (rowid,)
            ).fetchone()
            emb[rowid] = row[0]

        # Within-clique similarity (nodes 1 and 2, both in clique A)
        within_sim = cosine_similarity(emb[1], emb[2])

        # Between-clique similarity (node 1 in A, node 5 in B)
        between_sim = cosine_similarity(emb[1], emb[5])

        # Within-clique should be higher than between-clique
        assert within_sim > between_sim, (
            f"Within-clique similarity ({within_sim:.4f}) should be > "
            f"between-clique similarity ({between_sim:.4f})"
        )

    def test_karate_club_communities(self, conn):
        """
        Train on Zachary's Karate Club and verify that the two known
        communities are more similar within than between.
        """
        create_karate_club(conn)
        conn.execute(
            "CREATE VIRTUAL TABLE karate_emb USING hnsw_index("
            "dimensions=64, metric='cosine', m=16, ef_construction=100)"
        )
        conn.execute(
            "SELECT node2vec_train("
            "'karate_edges', 'src', 'dst', 'karate_emb', "
            "64, 1.0, 1.0, 10, 80, 5, 5, 0.025, 5)"
        )

        # Get all embeddings
        emb = {}
        for rowid in range(1, 35):
            row = conn.execute(
                "SELECT vector FROM karate_emb WHERE rowid = ?", (rowid,)
            ).fetchone()
            if row:
                emb[rowid] = row[0]

        # Known communities
        community_a = {1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 20, 22}
        community_b = {9, 10, 15, 16, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34}

        # Compute average within-community and between-community similarity
        within_sims = []
        between_sims = []

        nodes_a = sorted(community_a & set(emb.keys()))
        nodes_b = sorted(community_b & set(emb.keys()))

        for i in range(len(nodes_a)):
            for j in range(i + 1, len(nodes_a)):
                within_sims.append(cosine_similarity(emb[nodes_a[i]], emb[nodes_a[j]]))

        for i in range(len(nodes_b)):
            for j in range(i + 1, len(nodes_b)):
                within_sims.append(cosine_similarity(emb[nodes_b[i]], emb[nodes_b[j]]))

        for a in nodes_a:
            for b in nodes_b:
                between_sims.append(cosine_similarity(emb[a], emb[b]))

        avg_within = sum(within_sims) / len(within_sims) if within_sims else 0
        avg_between = sum(between_sims) / len(between_sims) if between_sims else 0

        # Within-community avg should be higher than between-community avg
        assert avg_within > avg_between, (
            f"Average within-community similarity ({avg_within:.4f}) should be > "
            f"average between-community similarity ({avg_between:.4f})"
        )
