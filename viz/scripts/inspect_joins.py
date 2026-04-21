"""Exploration-only: verify the joins I'll need for the embed/kg endpoints."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB = PROJECT_ROOT / "viz" / "frontend" / "public" / "demos" / "3300_MiniLM.db"
EXTENSION = str(PROJECT_ROOT / "build" / "muninn")


def main() -> None:
    conn = sqlite3.connect(str(DB))
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION)

    print("--- Sample chunks UMAP (3D) ---")
    for row in conn.execute(
        "SELECT u.id, u.x3d, u.y3d, u.z3d, SUBSTR(c.text, 1, 60) "
        "FROM chunks_vec_umap u JOIN chunks c ON u.id = c.chunk_id LIMIT 3"
    ):
        print(f"  {row}")

    print("\n--- Sample entities UMAP (3D) ---")
    for row in conn.execute(
        "SELECT u.id, u.x3d, u.y3d, u.z3d, m.name, n.entity_type "
        "FROM entities_vec_umap u "
        "JOIN entity_vec_map m ON u.id = m.rowid "
        "LEFT JOIN nodes n ON n.name = m.name LIMIT 3"
    ):
        print(f"  {row}")

    print("\n--- Leiden resolutions available ---")
    for row in conn.execute(
        "SELECT resolution, COUNT(DISTINCT community_id) AS n_communities, "
        "       COUNT(*) AS n_rows FROM leiden_communities GROUP BY resolution ORDER BY resolution"
    ):
        print(f"  resolution={row[0]}  communities={row[1]}  members={row[2]}")

    print("\n--- Community labels sample ---")
    for row in conn.execute(
        "SELECT resolution, community_id, label, member_count FROM community_labels "
        "ORDER BY resolution, member_count DESC LIMIT 5"
    ):
        print(f"  {row}")

    print("\n--- Base graph edge sample ---")
    for row in conn.execute("SELECT src, dst, rel_type, weight FROM edges LIMIT 3"):
        print(f"  {row}")

    print("\n--- Entity clusters sample ---")
    for row in conn.execute(
        "SELECT ec.name, ec.canonical, ecl.label "
        "FROM entity_clusters ec LEFT JOIN entity_cluster_labels ecl ON ec.canonical = ecl.canonical LIMIT 5"
    ):
        print(f"  {row}")

    print("\n--- Counts ---")
    for q in [
        "SELECT COUNT(DISTINCT name) FROM nodes",
        "SELECT COUNT(DISTINCT canonical) FROM entity_clusters",
        "SELECT COUNT(DISTINCT src) + COUNT(DISTINCT dst) FROM edges",
    ]:
        print(f"  {q} -> {conn.execute(q).fetchone()[0]}")

    conn.close()


if __name__ == "__main__":
    main()
