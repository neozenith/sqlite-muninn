"""
Transit Routes — Shortest Paths (BFS vs Dijkstra)

Demonstrates: graph_shortest_path with unweighted (BFS) and weighted (Dijkstra).

A transit network where the fewest-stops path and the fastest-time path differ,
showing why weighted shortest paths matter for real routing problems.
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

_IN_COLAB = "google.colab" in sys.modules

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    if _IN_COLAB:
        _REPO = Path("/content/sqlite-muninn")
        if not _REPO.exists():
            subprocess.run(
                ["git", "clone", "--recursive", "https://github.com/neozenith/sqlite-muninn.git", str(_REPO)],
                check=True,
            )
        if not list((_REPO / "build").glob("muninn.*")):
            subprocess.run(["apt-get", "install", "-y", "libsqlite3-dev"], check=True)
            subprocess.run(["make", "all"], cwd=str(_REPO), check=True)
        PROJECT_ROOT = _REPO
    else:
        PROJECT_ROOT = Path.cwd().parent.parent  # local notebook CWD is examples/{name}/
EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")

# ── Data: Transit network with travel times ──────────────────────────
#
#   Central ──15min──→ North ──10min──→ Airport
#     │                  ↑
#     │ 5min          6min │
#     ↓                  │
#   South ──12min──→ East
#     │
#     │ 7min
#     ↓
#    West ──────7min──────→ Airport
#
# Fewest stops:  Central → North → Airport           (2 hops, 25 min)
# Fastest route: Central → South → West → Airport    (3 hops, 19 min)

ROUTES = [
    ("Central", "North", 15.0),
    ("North", "Airport", 10.0),
    ("Central", "South", 5.0),
    ("South", "East", 12.0),
    ("East", "North", 6.0),
    ("South", "West", 7.0),
    ("West", "Airport", 7.0),
]


def main() -> None:
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    db.load_extension(EXTENSION_PATH)

    print("=== Transit Routes Example ===\n")

    db.execute("CREATE TABLE routes (src TEXT, dst TEXT, travel_time REAL)")
    db.executemany("INSERT INTO routes VALUES (?, ?, ?)", ROUTES)

    print("Transit network:")
    print("  Central ──15min──→ North ──10min──→ Airport")
    print("    │                  ↑")
    print("    │ 5min          6min │")
    print("    ↓                  │")
    print("  South ──12min──→  East")
    print("    │")
    print("    │ 7min")
    print("    ↓")
    print("   West ──────7min──────→ Airport\n")

    # ── Unweighted shortest path (fewest stops) ──────────────────────
    print("--- Unweighted: Fewest Stops (BFS) ---")
    unweighted = db.execute(
        """
        SELECT node, distance, path_order FROM graph_shortest_path
        WHERE edge_table = 'routes'
          AND src_col = 'src'
          AND dst_col = 'dst'
          AND start_node = 'Central'
          AND end_node = 'Airport'
          AND weight_col IS NULL
        """
    ).fetchall()

    unw_path = [r[0] for r in unweighted]
    unw_hops = len(unw_path) - 1
    print(f"  Path: {' → '.join(unw_path)}")
    print(f"  Hops: {unw_hops}")

    # Calculate actual travel time for this path
    unw_time = 0.0
    for i in range(len(unw_path) - 1):
        row = db.execute(
            "SELECT travel_time FROM routes WHERE src = ? AND dst = ?",
            (unw_path[i], unw_path[i + 1]),
        ).fetchone()
        assert row is not None, f"No route from {unw_path[i]} to {unw_path[i + 1]}"
        unw_time += row[0]
    print(f"  Total travel time: {unw_time:.0f} min\n")

    # ── Weighted shortest path (fastest time) ────────────────────────
    print("--- Weighted: Fastest Time (Dijkstra) ---")
    weighted = db.execute(
        """
        SELECT node, distance, path_order FROM graph_shortest_path
        WHERE edge_table = 'routes'
          AND src_col = 'src'
          AND dst_col = 'dst'
          AND start_node = 'Central'
          AND end_node = 'Airport'
          AND weight_col = 'travel_time'
        """
    ).fetchall()

    wt_path = [r[0] for r in weighted]
    wt_hops = len(wt_path) - 1
    wt_time = weighted[-1][1] if weighted else 0.0  # cumulative distance at end
    print(f"  Path: {' → '.join(wt_path)}")
    print(f"  Hops: {wt_hops}")
    print(f"  Total travel time: {wt_time:.0f} min\n")

    # ── Side-by-side comparison ──────────────────────────────────────
    print("--- Comparison ---")
    print(f"  {'':20s} {'Fewest Stops':>15s}  {'Fastest Time':>15s}")
    print(f"  {'Path':20s} {' → '.join(unw_path):>15s}  {' → '.join(wt_path):>15s}")
    print(f"  {'Hops':20s} {unw_hops:>15d}  {wt_hops:>15d}")
    print(f"  {'Travel time':20s} {unw_time:>14.0f}m  {wt_time:>14.0f}m\n")

    # ── Assertions ───────────────────────────────────────────────────
    assert unw_path == ["Central", "North", "Airport"], f"Unexpected unweighted path: {unw_path}"
    assert wt_path == ["Central", "South", "West", "Airport"], f"Unexpected weighted path: {wt_path}"
    assert unw_path != wt_path, "Paths should differ"
    assert unw_hops < wt_hops, "Unweighted path should have fewer hops"
    assert wt_time < unw_time, "Weighted path should have less travel time"
    assert abs(unw_time - 25.0) < 0.01, f"Unweighted travel time should be 25, got {unw_time}"
    assert abs(wt_time - 19.0) < 0.01, f"Weighted travel time should be 19, got {wt_time}"

    db.close()
    print("All assertions passed.")


if __name__ == "__main__":
    main()
