"""Exploration-only: check case consistency between leiden_communities.node and nodes.name."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB = PROJECT_ROOT / "viz" / "frontend" / "public" / "demos" / "3300_MiniLM.db"
EXT = str(PROJECT_ROOT / "build" / "muninn")


def main() -> None:
    conn = sqlite3.connect(str(DB))
    conn.enable_load_extension(True)
    conn.load_extension(EXT)

    leiden_nodes = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT node FROM leiden_communities WHERE resolution = 0.25"
        )
    }
    node_names = {row[0] for row in conn.execute("SELECT DISTINCT name FROM nodes")}
    edge_src = {row[0] for row in conn.execute("SELECT DISTINCT src FROM edges")}
    edge_dst = {row[0] for row in conn.execute("SELECT DISTINCT dst FROM edges")}

    print(f"leiden distinct nodes: {len(leiden_nodes)}")
    print(f"nodes distinct names:  {len(node_names)}")
    print(f"edge src distinct:     {len(edge_src)}")
    print(f"edge dst distinct:     {len(edge_dst)}")

    print(f"\nleiden ∩ nodes:     {len(leiden_nodes & node_names)}")
    print(f"leiden ∩ edge.src:  {len(leiden_nodes & edge_src)}")
    print(f"leiden ∩ edge.dst:  {len(leiden_nodes & edge_dst)}")

    print("\nSample leiden nodes:")
    for n in list(leiden_nodes)[:5]:
        print(f"  {n!r}")
    print("Sample nodes.name:")
    for n in list(node_names)[:5]:
        print(f"  {n!r}")

    print("\nSample overlaps with edge.src:")
    overlap = list(leiden_nodes & edge_src)[:5]
    for n in overlap:
        print(f"  {n!r}")

    conn.close()


if __name__ == "__main__":
    main()
