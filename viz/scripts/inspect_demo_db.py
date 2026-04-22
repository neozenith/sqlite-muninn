"""Exploration-only: dump schema + sample rows from a demo database.

Run:
    uv run --directory viz scripts/inspect_demo_db.py 3300_MiniLM
"""

import argparse
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEMOS_DIR = PROJECT_ROOT / "viz" / "frontend" / "public" / "demos"
EXTENSION = str(PROJECT_ROOT / "build" / "muninn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db_id", help="e.g. 3300_MiniLM")
    args = parser.parse_args()

    db_path = DEMOS_DIR / f"{args.db_id}.db"
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    conn.load_extension(EXTENSION)

    print(f"=== {db_path} ===\n")
    print("TABLES / VIEWS:")
    rows = list(
        conn.execute(
            "SELECT name, type FROM sqlite_master "
            "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
    )
    for name, ttype in rows:
        try:
            count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        except sqlite3.OperationalError:
            count = "?"
        print(f"  {name:45s} ({ttype:5s})  {count} rows")

    print("\nCOLUMN DETAIL (non-shadow tables):")
    for name, _ttype in rows:
        if name.startswith("_") or "_config" in name or "_nodes" in name:
            continue
        try:
            cols = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
        except sqlite3.OperationalError:
            continue
        if not cols:
            continue
        print(f"\n  {name}:")
        for col in cols:
            print(f"    {col[1]:30s} {col[2]}")

    conn.close()


if __name__ == "__main__":
    main()
