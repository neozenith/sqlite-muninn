"""
Semantic Search — Document Similarity with HNSW

Demonstrates: CREATE VIRTUAL TABLE, INSERT, KNN MATCH, point lookup, DELETE.

12 tech articles across 3 topics (AI, Web, Database) with hand-crafted
8-dimensional topic-aligned vectors and cosine similarity search.
"""
import sqlite3
import struct
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # examples/{name}/ → project root
EXTENSION_PATH = str(PROJECT_ROOT / "muninn")  # auto-appends .dylib/.so


def pack_vector(values: list[float]) -> bytes:
    """Pack floats into a little-endian float32 blob for SQLite."""
    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes, dim: int) -> list[float]:
    """Unpack a float32 blob back into a list of floats."""
    return list(struct.unpack(f"<{dim}f", blob))


# ── Data: 12 articles with 8-dim topic vectors ──────────────────────
# Dimensions roughly encode: [AI, ML, NLP, Web, Frontend, Backend, DB, SQL]
DOCUMENTS = [
    # AI cluster — high values in dims 0-2
    (1, "Introduction to Neural Networks", "ai", [0.9, 0.8, 0.3, 0.1, 0.0, 0.1, 0.1, 0.0]),
    (2, "Deep Learning for Image Recognition", "ai", [0.8, 0.9, 0.2, 0.0, 0.1, 0.0, 0.1, 0.0]),
    (3, "Transformer Models Explained", "ai", [0.7, 0.6, 0.9, 0.1, 0.0, 0.1, 0.0, 0.1]),
    (4, "GPT and Large Language Models", "ai", [0.8, 0.7, 0.8, 0.1, 0.1, 0.0, 0.0, 0.1]),
    # Web cluster — high values in dims 3-5
    (5, "Building REST APIs with FastAPI", "web", [0.1, 0.0, 0.1, 0.8, 0.2, 0.9, 0.1, 0.1]),
    (6, "React Hooks Deep Dive", "web", [0.0, 0.1, 0.0, 0.7, 0.9, 0.3, 0.0, 0.0]),
    (7, "CSS Grid Layout Guide", "web", [0.0, 0.0, 0.1, 0.8, 0.8, 0.2, 0.0, 0.0]),
    (8, "Node.js Event Loop Internals", "web", [0.1, 0.0, 0.0, 0.7, 0.3, 0.8, 0.1, 0.1]),
    # Database cluster — high values in dims 6-7
    (9, "PostgreSQL Query Optimization", "db", [0.1, 0.0, 0.1, 0.1, 0.0, 0.2, 0.9, 0.8]),
    (10, "Introduction to DuckDB Analytics", "db", [0.2, 0.1, 0.0, 0.0, 0.0, 0.1, 0.8, 0.9]),
    (11, "SQLite Internals and Extensions", "db", [0.1, 0.0, 0.0, 0.1, 0.0, 0.2, 0.9, 0.7]),
    (12, "Database Indexing Strategies", "db", [0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.8, 0.8]),
]


def main() -> None:
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXTENSION_PATH)

    # ── Step 1: Create tables ────────────────────────────────────────
    print("=== Semantic Search Example ===\n")

    db.execute("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            topic TEXT NOT NULL
        )
    """)

    db.execute("""
        CREATE VIRTUAL TABLE doc_vectors USING hnsw_index(
            dimensions=8, metric='cosine', m=16, ef_construction=200
        )
    """)
    print("Created documents table and HNSW vector index (dim=8, cosine).\n")

    # ── Step 2: Insert documents and vectors ─────────────────────────
    for doc_id, title, topic, vec in DOCUMENTS:
        db.execute("INSERT INTO documents VALUES (?, ?, ?)", (doc_id, title, topic))
        db.execute(
            "INSERT INTO doc_vectors (rowid, vector) VALUES (?, ?)",
            (doc_id, pack_vector(vec)),
        )
    print(f"Inserted {len(DOCUMENTS)} documents with vectors.\n")

    # ── Step 3: KNN search with AI-biased query ──────────────────────
    # Query vector emphasizes AI/ML dimensions
    query_vec = pack_vector([0.85, 0.75, 0.7, 0.05, 0.0, 0.05, 0.0, 0.05])

    print("--- KNN Search (k=5, query biased toward AI) ---")
    results = db.execute(
        """
        SELECT v.rowid, v.distance, d.title, d.topic
        FROM doc_vectors v
        JOIN documents d ON d.id = v.rowid
        WHERE v.vector MATCH ? AND k = 5
        """,
        (query_vec,),
    ).fetchall()

    for rowid, distance, title, topic in results:
        print(f"  [{topic:3s}] {title:<45s}  dist={distance:.4f}")

    # Assert: top results should all be AI docs
    top_topics = [r[3] for r in results[:3]]
    assert all(t == "ai" for t in top_topics), f"Expected top-3 to be AI docs, got {top_topics}"
    print(f"\n  Top 3 results are all AI articles.\n")

    # ── Step 4: Point lookup ─────────────────────────────────────────
    print("--- Point Lookup (rowid=3: 'Transformer Models Explained') ---")
    row = db.execute("SELECT vector FROM doc_vectors WHERE rowid = 3").fetchone()
    assert row is not None, "Point lookup returned no result"
    stored_vec = unpack_vector(row[0], 8)
    expected = [0.7, 0.6, 0.9, 0.1, 0.0, 0.1, 0.0, 0.1]
    print(f"  Stored vector: [{', '.join(f'{v:.1f}' for v in stored_vec)}]")
    # float32 round-trip means we compare approximately
    assert all(
        abs(a - b) < 1e-5 for a, b in zip(stored_vec, expected)
    ), f"Vector mismatch: {stored_vec}"
    print("  Matches original vector.\n")

    # ── Step 5: Delete and re-search ─────────────────────────────────
    print("--- Delete doc #2 ('Deep Learning for Image Recognition') ---")
    db.execute("DELETE FROM doc_vectors WHERE rowid = 2")
    db.execute("DELETE FROM documents WHERE id = 2")

    results_after = db.execute(
        """
        SELECT v.rowid, d.title, d.topic
        FROM doc_vectors v
        JOIN documents d ON d.id = v.rowid
        WHERE v.vector MATCH ? AND k = 5
        """,
        (query_vec,),
    ).fetchall()

    result_ids = {r[0] for r in results_after}
    assert 2 not in result_ids, "Deleted document should not appear in results"
    print("  Re-search results (k=5):")
    for rowid, title, topic in results_after:
        print(f"    [{topic:3s}] {title}")
    print(f"\n  Doc #2 correctly absent from results.\n")

    db.close()
    print("All assertions passed.")


if __name__ == "__main__":
    main()
