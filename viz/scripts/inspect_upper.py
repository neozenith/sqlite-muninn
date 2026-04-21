"""Check for uppercase entries in leiden / edges / nodes."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB = PROJECT_ROOT / "viz" / "frontend" / "public" / "demos" / "3300_MiniLM.db"
EXT = str(PROJECT_ROOT / "build" / "muninn")


def main() -> None:
    conn = sqlite3.connect(str(DB))
    conn.enable_load_extension(True)
    conn.load_extension(EXT)

    for table, col in [
        ("leiden_communities", "node"),
        ("nodes", "name"),
        ("edges", "src"),
        ("edges", "dst"),
    ]:
        count = conn.execute(
            f"SELECT COUNT(DISTINCT {col}) FROM {table} WHERE {col} = UPPER({col}) AND {col} != LOWER({col})"
        ).fetchone()[0]
        print(f"{table}.{col} all-uppercase entries: {count}")

    print("\nLEIDEN at resolution 0.25 where UPPER:")
    for row in conn.execute(
        "SELECT node, community_id FROM leiden_communities WHERE resolution = 0.25 "
        "AND node = UPPER(node) AND node != LOWER(node) LIMIT 5"
    ):
        print(f"  {row}")

    print("\nEDGES.src UPPER sample:")
    for row in conn.execute(
        "SELECT src, dst FROM edges WHERE src = UPPER(src) AND src != LOWER(src) LIMIT 5"
    ):
        print(f"  {row}")

    conn.close()


if __name__ == "__main__":
    main()
