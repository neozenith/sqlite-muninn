"""Phase: Communities — precompute Leiden communities at multiple resolutions."""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

from benchmarks.sessions_demo.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.sessions_demo.build import PhaseContext

log = logging.getLogger(__name__)

# Resolutions: coarse (macro topics), medium (default), fine (sub-topics).
LEIDEN_RESOLUTIONS = [0.25, 1.0, 3.0]


class PhaseCommunities(Phase):
    """Precompute Leiden community assignments on the relations graph.

    Runs graph_leiden at multiple resolutions and stores results in
    leiden_communities(node, resolution, community_id, modularity).
    """

    @property
    def name(self) -> str:
        return "communities"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        try:
            n_resolutions: int = conn.execute("SELECT count(DISTINCT resolution) FROM leiden_communities").fetchone()[0]
            return n_resolutions < len(LEIDEN_RESOLUTIONS)
        except sqlite3.OperationalError:
            return True

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        try:
            edge_count: int = conn.execute("SELECT count(*) FROM relations").fetchone()[0]
        except sqlite3.OperationalError:
            log.info("  No relations table — skipping community detection")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS leiden_communities (
                    node        TEXT NOT NULL,
                    resolution  REAL NOT NULL,
                    community_id INTEGER NOT NULL,
                    modularity  REAL,
                    PRIMARY KEY (node, resolution)
                )
            """)
            return

        if edge_count == 0:
            log.info("  Relations table empty — skipping community detection")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS leiden_communities (
                    node        TEXT NOT NULL,
                    resolution  REAL NOT NULL,
                    community_id INTEGER NOT NULL,
                    modularity  REAL,
                    PRIMARY KEY (node, resolution)
                )
            """)
            return

        conn.execute("DROP TABLE IF EXISTS leiden_communities")
        conn.execute("""
            CREATE TABLE leiden_communities (
                node        TEXT NOT NULL,
                resolution  REAL NOT NULL,
                community_id INTEGER NOT NULL,
                modularity  REAL,
                PRIMARY KEY (node, resolution)
            )
        """)

        total_rows = 0
        for resolution in LEIDEN_RESOLUTIONS:
            rows = conn.execute(
                "SELECT node, community_id, modularity "
                "FROM graph_leiden "
                "WHERE edge_table = 'relations' AND src_col = 'src' AND dst_col = 'dst' "
                "  AND direction = 'both' AND resolution = ?",
                (resolution,),
            ).fetchall()

            n_communities = len({r[1] for r in rows})
            modularity = rows[0][2] if rows else 0.0

            conn.executemany(
                "INSERT INTO leiden_communities (node, resolution, community_id, modularity) VALUES (?, ?, ?, ?)",
                [(r[0], resolution, r[1], r[2]) for r in rows],
            )

            total_rows += len(rows)
            log.info(
                "  Resolution %.2f: %d nodes → %d communities (Q=%.4f)",
                resolution,
                len(rows),
                n_communities,
                modularity,
            )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_leiden_communities_resolution "
            "ON leiden_communities (resolution, community_id)"
        )

        log.info(
            "  Leiden communities complete: %d total assignments across %d resolutions",
            total_rows,
            len(LEIDEN_RESOLUTIONS),
        )
