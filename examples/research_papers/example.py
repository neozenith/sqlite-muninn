"""
Research Papers — Citation Analysis with PageRank and Components

Demonstrates: graph_pagerank (damping, iterations), graph_components.

12 papers forming 3 disconnected research clusters. Directed citation
edges with hub papers that accumulate high PageRank from many citations.
"""
import sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
EXTENSION_PATH = str(PROJECT_ROOT / "vec_graph")

# ── Data: 12 papers in 3 clusters ───────────────────────────────────
# ML cluster (5 papers): ML-Survey is heavily cited hub
# DB cluster (4 papers): DB-Indexing is heavily cited hub
# Isolated cluster (3 papers): small group, no hub

PAPERS = {
    "ML-Survey": "A Survey of Machine Learning Methods",
    "ML-CNN": "Convolutional Neural Networks for Vision",
    "ML-RNN": "Recurrent Networks for Sequences",
    "ML-GAN": "Generative Adversarial Networks",
    "ML-Transfer": "Transfer Learning Techniques",
    "DB-Indexing": "Modern Database Indexing",
    "DB-Query": "Query Optimization Strategies",
    "DB-MVCC": "Multi-Version Concurrency Control",
    "DB-Column": "Columnar Storage Engines",
    "NLP-Embed": "Word Embedding Methods",
    "NLP-Parse": "Dependency Parsing Advances",
    "NLP-Sent": "Sentiment Analysis Review",
}

# Citations: (citing_paper, cited_paper)
# Many papers cite the hub papers (ML-Survey, DB-Indexing)
CITATIONS = [
    # ML cluster — ML-Survey is the most cited
    ("ML-CNN", "ML-Survey"),
    ("ML-RNN", "ML-Survey"),
    ("ML-GAN", "ML-Survey"),
    ("ML-Transfer", "ML-Survey"),
    ("ML-Transfer", "ML-CNN"),
    ("ML-GAN", "ML-CNN"),
    # DB cluster — DB-Indexing is the most cited
    ("DB-Query", "DB-Indexing"),
    ("DB-MVCC", "DB-Indexing"),
    ("DB-Column", "DB-Indexing"),
    ("DB-Column", "DB-Query"),
    # NLP cluster — isolated, no dominant hub
    ("NLP-Parse", "NLP-Embed"),
    ("NLP-Sent", "NLP-Embed"),
    ("NLP-Sent", "NLP-Parse"),
]


def main() -> None:
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXTENSION_PATH)

    print("=== Research Papers Example ===\n")

    db.execute("CREATE TABLE citations (citing TEXT, cited TEXT)")
    db.executemany("INSERT INTO citations VALUES (?, ?)", CITATIONS)
    print(f"Created citation graph: {len(PAPERS)} papers, {len(CITATIONS)} citations.\n")

    # ── Step 1: Connected components ─────────────────────────────────
    print("--- Connected Components ---")
    components = db.execute(
        """
        SELECT node, component_id, component_size FROM graph_components
        WHERE edge_table = 'citations'
          AND src_col = 'citing'
          AND dst_col = 'cited'
        """
    ).fetchall()

    # Group by component
    clusters: dict[int, list[str]] = {}
    sizes: dict[int, int] = {}
    for node, comp_id, comp_size in components:
        clusters.setdefault(comp_id, []).append(node)
        sizes[comp_id] = comp_size

    print(f"  Found {len(clusters)} research clusters:\n")
    for comp_id in sorted(clusters, key=lambda c: -sizes[c]):
        members = sorted(clusters[comp_id])
        print(f"  Cluster {comp_id} ({sizes[comp_id]} papers):")
        for paper in members:
            print(f"    - {paper}: {PAPERS[paper]}")
        print()

    assert len(clusters) == 3, f"Expected 3 clusters, got {len(clusters)}"

    # Verify cluster sizes
    cluster_sizes = sorted(sizes.values(), reverse=True)
    assert cluster_sizes == [5, 4, 3], f"Expected sizes [5, 4, 3], got {cluster_sizes}"

    # ── Step 2: PageRank ─────────────────────────────────────────────
    print("--- PageRank (damping=0.85, iterations=20) ---")
    pagerank = db.execute(
        """
        SELECT node, rank FROM graph_pagerank
        WHERE edge_table = 'citations'
          AND src_col = 'citing'
          AND dst_col = 'cited'
          AND damping = 0.85
          AND iterations = 20
        """
    ).fetchall()

    rank_map = {r[0]: r[1] for r in pagerank}

    # Sort by rank descending
    ranked = sorted(pagerank, key=lambda r: -r[1])

    print(f"\n  {'Rank':>4s}  {'Paper':<15s}  {'PageRank':>10s}  Title")
    print(f"  {'─' * 4}  {'─' * 15}  {'─' * 10}  {'─' * 35}")
    for i, (paper, rank) in enumerate(ranked, 1):
        print(f"  {i:4d}  {paper:<15s}  {rank:10.4f}  {PAPERS[paper]}")

    total_rank = sum(r[1] for r in pagerank)
    print(f"\n  Total PageRank: {total_rank:.4f} (should be ~1.0)")

    # ── Assertions ───────────────────────────────────────────────────
    # Hub papers should have the highest PageRank
    assert rank_map["ML-Survey"] > rank_map["ML-GAN"], "ML-Survey should outrank ML-GAN"
    assert rank_map["ML-Survey"] > rank_map["ML-Transfer"], "ML-Survey should outrank ML-Transfer"
    assert rank_map["DB-Indexing"] > rank_map["DB-MVCC"], "DB-Indexing should outrank DB-MVCC"

    # ML-Survey should be #1 (4 incoming citations)
    assert ranked[0][0] == "ML-Survey", f"Expected ML-Survey as #1, got {ranked[0][0]}"

    # Ranks should sum to ~1.0
    assert abs(total_rank - 1.0) < 0.05, f"PageRank sum should be ~1.0, got {total_rank}"

    print()
    db.close()
    print("All assertions passed.")


if __name__ == "__main__":
    main()
