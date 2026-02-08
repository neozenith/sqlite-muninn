"""
Movie Recommendations — Full Pipeline (Graph → Node2Vec → HNSW KNN)

Demonstrates: node2vec_train() with all parameters, HNSW KNN search on
learned embeddings, and the critical rowid mapping between graph nodes
and HNSW vector IDs.

15 movies in 3 genre clusters (sci-fi, action, comedy) with co-preference
edges. Node2Vec learns embeddings from graph structure, then HNSW enables
"find movies similar to X" queries.
"""
import sqlite3
import struct
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
EXTENSION_PATH = str(PROJECT_ROOT / "vec_graph")


def pack_vector(values: list[float]) -> bytes:
    """Pack floats into a little-endian float32 blob for SQLite."""
    return struct.pack(f"<{len(values)}f", *values)


# ── Data: 15 movies in 3 genre clusters ──────────────────────────────
MOVIES = {
    # Sci-fi cluster
    "The Matrix": "sci-fi",
    "Inception": "sci-fi",
    "Interstellar": "sci-fi",
    "Blade Runner": "sci-fi",
    "Arrival": "sci-fi",
    # Action cluster
    "Die Hard": "action",
    "Mad Max": "action",
    "John Wick": "action",
    "Gladiator": "action",
    "The Dark Knight": "action",
    # Comedy cluster
    "Superbad": "comedy",
    "The Hangover": "comedy",
    "Bridesmaids": "comedy",
    "Step Brothers": "comedy",
    "Anchorman": "comedy",
}

# Co-preference edges: "users who liked X also liked Y"
# Dense within clusters, 2 cross-genre bridges
CO_PREFERENCES = [
    # Sci-fi cluster (dense)
    ("The Matrix", "Inception"),
    ("The Matrix", "Blade Runner"),
    ("Inception", "Interstellar"),
    ("Inception", "Arrival"),
    ("Interstellar", "Arrival"),
    ("Blade Runner", "Arrival"),
    ("The Matrix", "Interstellar"),
    # Action cluster (dense)
    ("Die Hard", "Mad Max"),
    ("Die Hard", "John Wick"),
    ("Mad Max", "Gladiator"),
    ("John Wick", "The Dark Knight"),
    ("Gladiator", "The Dark Knight"),
    ("Die Hard", "The Dark Knight"),
    ("Mad Max", "John Wick"),
    # Comedy cluster (dense)
    ("Superbad", "The Hangover"),
    ("Superbad", "Step Brothers"),
    ("The Hangover", "Bridesmaids"),
    ("Bridesmaids", "Anchorman"),
    ("Step Brothers", "Anchorman"),
    ("The Hangover", "Step Brothers"),
    # Cross-genre bridges (weak links)
    ("The Dark Knight", "The Matrix"),  # action ↔ sci-fi
    ("Superbad", "John Wick"),  # comedy ↔ action
]


def build_rowid_mapping(db: sqlite3.Connection) -> dict[str, int]:
    """
    Replicate node2vec_train's first-seen ordering to map movie names to rowids.

    node2vec.c:graph_load_edges() runs SELECT src, dst FROM edge_table and calls
    graph_node_index() for each src then dst. graph_node_index() assigns indices
    in first-seen order (line 72-92). Embeddings are inserted with rowid = i + 1
    (line 559, 1-indexed).

    We replicate this by iterating the same SELECT and tracking first appearances.
    """
    rows = db.execute(
        'SELECT src, dst FROM co_prefs'
    ).fetchall()

    name_to_idx: dict[str, int] = {}
    for src, dst in rows:
        if src not in name_to_idx:
            name_to_idx[src] = len(name_to_idx)
        if dst not in name_to_idx:
            name_to_idx[dst] = len(name_to_idx)

    # Convert 0-based index to 1-based rowid
    return {name: idx + 1 for name, idx in name_to_idx.items()}


def main() -> None:
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXTENSION_PATH)

    print("=== Movie Recommendations Example ===\n")

    # ── Step 1: Create co-preference edges (bidirectional) ───────────
    db.execute("CREATE TABLE co_prefs (src TEXT, dst TEXT)")

    # Store both directions for undirected co-preference
    edges = []
    for a, b in CO_PREFERENCES:
        edges.append((a, b))
        edges.append((b, a))
    db.executemany("INSERT INTO co_prefs VALUES (?, ?)", edges)
    print(f"Created co-preference graph: {len(MOVIES)} movies, "
          f"{len(CO_PREFERENCES)} undirected edges ({len(edges)} directed).\n")

    # ── Step 2: Create HNSW table for embeddings ─────────────────────
    dim = 32
    db.execute(f"""
        CREATE VIRTUAL TABLE movie_emb USING hnsw_index(
            dimensions={dim}, metric='cosine', m=8, ef_construction=100
        )
    """)
    print(f"Created HNSW index (dim={dim}, cosine).\n")

    # ── Step 3: Train Node2Vec embeddings ────────────────────────────
    print("--- Training Node2Vec ---")
    print("  Parameters: p=0.5, q=0.5 (BFS-like, community detection)")
    print("  10 walks/node, length=40, window=5, 5 negative samples, 5 epochs")

    result = db.execute(
        """
        SELECT node2vec_train(
            'co_prefs', 'src', 'dst', 'movie_emb',
            32,       -- dimensions
            0.5,      -- p (low = stay close, detect communities)
            0.5,      -- q (low = explore structure)
            10,       -- num_walks per node
            40,       -- walk_length
            5,        -- window_size
            5,        -- negative_samples
            0.025,    -- learning_rate
            5         -- epochs
        )
        """
    ).fetchone()
    num_embedded = result[0]
    print(f"  Embedded {num_embedded} movie nodes.\n")

    assert num_embedded == len(MOVIES), (
        f"Expected {len(MOVIES)} embeddings, got {num_embedded}"
    )

    # ── Step 4: Build rowid ↔ name mapping ───────────────────────────
    rowid_map = build_rowid_mapping(db)
    name_map = {v: k for k, v in rowid_map.items()}

    print("--- Rowid Mapping (first-seen order from edge iteration) ---")
    for name in sorted(rowid_map, key=rowid_map.get):  # type: ignore[arg-type]
        print(f"  rowid={rowid_map[name]:2d} → {name} [{MOVIES[name]}]")
    print()

    # ── Step 5: KNN search for "The Matrix" ──────────────────────────
    matrix_rowid = rowid_map["The Matrix"]
    matrix_vec = db.execute(
        "SELECT vector FROM movie_emb WHERE rowid = ?", (matrix_rowid,)
    ).fetchone()
    assert matrix_vec is not None, "Could not retrieve The Matrix embedding"

    print('--- KNN Search: "Find movies similar to The Matrix" ---')
    knn_results = db.execute(
        """
        SELECT rowid, distance FROM movie_emb
        WHERE vector MATCH ? AND k = 8 AND ef_search = 32
        """,
        (matrix_vec[0],),
    ).fetchall()

    print(f"  {'Rank':>4s}  {'Movie':<20s}  {'Genre':<8s}  {'Distance':>8s}")
    print(f"  {'─' * 4}  {'─' * 20}  {'─' * 8}  {'─' * 8}")
    matrix_genre_count = 0
    for i, (rowid, distance) in enumerate(knn_results, 1):
        name = name_map[rowid]
        genre = MOVIES[name]
        marker = " ←" if name == "The Matrix" else ""
        print(f"  {i:4d}  {name:<20s}  {genre:<8s}  {distance:8.4f}{marker}")
        if genre == "sci-fi" and name != "The Matrix":
            matrix_genre_count += 1

    # At least 2 of top-7 neighbors should be sci-fi (excluding self)
    assert matrix_genre_count >= 2, (
        f"Expected at least 2 sci-fi movies near The Matrix, got {matrix_genre_count}"
    )
    print(f"\n  {matrix_genre_count} sci-fi movies in top-7 neighbors.\n")

    # ── Step 6: KNN search for "Die Hard" ────────────────────────────
    diehard_rowid = rowid_map["Die Hard"]
    diehard_vec = db.execute(
        "SELECT vector FROM movie_emb WHERE rowid = ?", (diehard_rowid,)
    ).fetchone()
    assert diehard_vec is not None, "Could not retrieve Die Hard embedding"

    print('--- KNN Search: "Find movies similar to Die Hard" ---')
    knn_results = db.execute(
        """
        SELECT rowid, distance FROM movie_emb
        WHERE vector MATCH ? AND k = 8 AND ef_search = 32
        """,
        (diehard_vec[0],),
    ).fetchall()

    print(f"  {'Rank':>4s}  {'Movie':<20s}  {'Genre':<8s}  {'Distance':>8s}")
    print(f"  {'─' * 4}  {'─' * 20}  {'─' * 8}  {'─' * 8}")
    action_genre_count = 0
    for i, (rowid, distance) in enumerate(knn_results, 1):
        name = name_map[rowid]
        genre = MOVIES[name]
        marker = " ←" if name == "Die Hard" else ""
        print(f"  {i:4d}  {name:<20s}  {genre:<8s}  {distance:8.4f}{marker}")
        if genre == "action" and name != "Die Hard":
            action_genre_count += 1

    assert action_genre_count >= 2, (
        f"Expected at least 2 action movies near Die Hard, got {action_genre_count}"
    )
    print(f"\n  {action_genre_count} action movies in top-7 neighbors.\n")

    db.close()
    print("All assertions passed.")


if __name__ == "__main__":
    main()
