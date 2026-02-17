#!/usr/bin/env python3
"""
Standalone verification: cosine similarity between a query and a chunk
that contains the query as a substring, using all-MiniLM-L6-v2.

This helps diagnose whether low similarity scores in the WASM demo
are expected model behavior or indicate a bug in vector storage/search.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - similarity) between two vectors."""
    return 1.0 - cosine_similarity(a, b)


def main() -> None:
    query = (
        "In those great manufactures on the contrary which are destined to supply "
        "the great wants of the great body of the people every different branch of "
        "the work employs so great a number of workmen that it is impossible to "
        "collect them all into the same workhouse"
    )

    chunk = (
        "It is commonly supposed to be carried furthest in some very trifling ones; "
        "not perhaps that it really is carried further in them than in others of more "
        "importance: but in those trifling manufactures which are destined to supply "
        "the small wants of but a small number of people, the whole number of workmen "
        "must necessarily be small; and those employed in every different branch of the "
        "work can often be collected into the same workhouse, and placed at once under "
        "the view of the spectator. In those great manufactures, on the contrary, which "
        "are destined to supply the great wants of the great body of the people, every "
        "different branch of the work employs so great a number of workmen, that it is "
        "impossible to collect them all into the same workhouse. We can seldom see more, "
        "at one time, than those employed in one single branch. Though in such "
        "manufactures, therefore, the work may really be divided into a much greater "
        "number of parts, than in those of a more trifling nature, the division is not "
        "near so obvious, and has accordingly been much less observed. To take an example, "
        "therefore, from a very trifling manufacture, but one in which the division of "
        "labour has been very often taken notice of, the trade of a pin-maker: a workman "
        "not educated to this business (which the division of labour has rendered a "
        "distinct trade), nor acquainted with the use of the machinery employed in it "
        "(to the invention of which the same division of labour"
    )

    print("=" * 60)
    print("Cosine Similarity Verification")
    print("Model: all-MiniLM-L6-v2")
    print("=" * 60)
    print(f"\nQuery ({len(query)} chars):\n  {query!r}\n")
    print(f"Chunk ({len(chunk)} chars):\n  {chunk!r}\n")
    print(f"Query is substring of chunk: {query in chunk}\n")

    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode both texts
    embeddings = model.encode([query, chunk], normalize_embeddings=True)
    query_vec = embeddings[0]
    chunk_vec = embeddings[1]

    print(f"Vector dimension: {query_vec.shape[0]}")
    print(f"Query  vector norm: {np.linalg.norm(query_vec):.6f}")
    print(f"Chunk  vector norm: {np.linalg.norm(chunk_vec):.6f}")

    sim = cosine_similarity(query_vec, chunk_vec)
    dist = cosine_distance(query_vec, chunk_vec)

    print(f"\n{'─' * 40}")
    print(f"Cosine similarity: {sim:.6f}  ({sim * 100:.1f}%)")
    print(f"Cosine distance:   {dist:.6f}  ({dist * 100:.1f}%)")
    print(f"{'─' * 40}")

    # Also test: what does the model give for an identical string?
    identical = model.encode([query, query], normalize_embeddings=True)
    identical_sim = cosine_similarity(identical[0], identical[1])
    print(f"\nSanity check (query vs itself): {identical_sim:.6f}  ({identical_sim * 100:.1f}%)")

    # Test with raw float32 blob round-trip (simulating SQLite storage)
    print(f"\n{'=' * 60}")
    print("Float32 blob round-trip test (simulating SQLite storage)")
    print(f"{'=' * 60}")

    # Convert to float32 blob and back (this is what muninn/sqlite-vec do)
    query_blob = query_vec.astype(np.float32).tobytes()
    chunk_blob = chunk_vec.astype(np.float32).tobytes()

    query_restored = np.frombuffer(query_blob, dtype=np.float32)
    chunk_restored = np.frombuffer(chunk_blob, dtype=np.float32)

    sim_restored = cosine_similarity(query_restored, chunk_restored)
    print(f"After float32 round-trip: {sim_restored:.6f}  ({sim_restored * 100:.1f}%)")
    print(f"Difference from original: {abs(sim - sim_restored):.10f}")

    # Diagnosis
    print(f"\n{'=' * 60}")
    print("DIAGNOSIS")
    print(f"{'=' * 60}")
    if sim > 0.85:
        print(f"Model similarity is {sim:.1%} — this is EXPECTED for a substring match.")
        print("If the WASM demo shows ~57%, the bug is likely in vector storage or distance calc.")
    elif sim > 0.70:
        print(f"Model similarity is {sim:.1%} — somewhat lower than expected.")
        print("May be model-dependent. Check if the WASM demo uses a different model variant.")
    else:
        print(f"Model similarity is {sim:.1%} — surprisingly low even from the model.")
        print("This might be expected for this particular model and text pair.")


if __name__ == "__main__":
    main()
