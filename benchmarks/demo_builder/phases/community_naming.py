"""Phase: Community Naming — LLM-generated labels via muninn_label_groups TVF.

Uses the C TVF that internally handles grouping, prompt construction,
LLM inference (with skip_think), and label cleaning. The Python phase
just creates views/temp tables with the right schema and calls the TVF.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from benchmarks.demo_builder.constants import MUNINN_CHAT_MODEL_FILE, MUNINN_CHAT_MODEL_NAME, MUNINN_CHAT_MODELS_DIR
from benchmarks.demo_builder.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext

log = logging.getLogger(__name__)

MIN_COMMUNITY_SIZE = 3
MAX_MEMBERS_IN_PROMPT = 10

_REGISTER_MODEL_SQL = "INSERT INTO temp.muninn_chat_models(name, model) SELECT ?, muninn_chat_model(?)"


class PhaseCommunityNaming(Phase):
    """Generate human-readable labels for entity clusters and Leiden communities.

    Uses muninn_label_groups TVF — all prompt construction and LLM calling
    happens in C. This phase just creates the membership views and inserts
    the TVF results into output tables.
    """

    @property
    def name(self) -> str:
        return "community_naming"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        try:
            ecl: int = conn.execute("SELECT count(*) FROM entity_cluster_labels").fetchone()[0]
            cl: int = conn.execute("SELECT count(*) FROM community_labels").fetchone()[0]
            return ecl == 0 and cl == 0
        except sqlite3.OperationalError:
            return True

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        model_path = str(MUNINN_CHAT_MODELS_DIR / MUNINN_CHAT_MODEL_FILE)
        log.info("  Registering chat model: %s from %s", MUNINN_CHAT_MODEL_NAME, model_path)
        conn.execute(_REGISTER_MODEL_SQL, (MUNINN_CHAT_MODEL_NAME, model_path))

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        model_name = MUNINN_CHAT_MODEL_NAME
        now = datetime.now(UTC).isoformat()

        self._label_entity_clusters(conn, model_name, now)
        self._label_leiden_communities(conn, model_name, now)

    def _label_entity_clusters(self, conn: sqlite3.Connection, model_name: str, now: str) -> None:
        """Name entity clusters via muninn_label_groups TVF."""
        conn.execute("DROP TABLE IF EXISTS entity_cluster_labels")
        conn.execute("""
            CREATE TABLE entity_cluster_labels (
                canonical    TEXT PRIMARY KEY,
                label        TEXT NOT NULL,
                member_count INTEGER NOT NULL,
                model        TEXT NOT NULL,
                generated_at TEXT NOT NULL
            )
        """)

        # Create a membership view: (canonical, member_name)
        # The TVF needs a flat table with group_col and member_col.
        conn.execute("DROP VIEW IF EXISTS _ecl_membership")
        try:
            conn.execute("""
                CREATE TEMP VIEW _ecl_membership AS
                SELECT ec.canonical AS group_id,
                       ec.name || ' (' || COALESCE(n.entity_type, 'unknown') || ')' AS member_name
                FROM entity_clusters ec
                LEFT JOIN nodes n ON n.name = ec.canonical
            """)
        except sqlite3.OperationalError:
            log.info("  No entity_clusters table — skipping entity cluster naming")
            return

        total = conn.execute("SELECT count(DISTINCT group_id) FROM _ecl_membership").fetchone()[0]
        log.info("  Labelling entity clusters via muninn_label_groups (%d groups)...", total)

        conn.execute(
            """
            INSERT INTO entity_cluster_labels(canonical, label, member_count, model, generated_at)
            SELECT group_id, label, member_count, ?, ?
            FROM muninn_label_groups
            WHERE model = ?
              AND membership_table = '_ecl_membership'
              AND group_col = 'group_id'
              AND member_col = 'member_name'
              AND min_group_size = ?
              AND max_members_in_prompt = ?
              AND system_prompt = 'Output ONLY a concise label (3-8 words). No explanation.'
            """,
            (model_name, now, model_name, MIN_COMMUNITY_SIZE, MAX_MEMBERS_IN_PROMPT),
        )

        count = conn.execute("SELECT count(*) FROM entity_cluster_labels").fetchone()[0]
        log.info("  Entity cluster naming complete: %d labels", count)
        conn.execute("DROP VIEW IF EXISTS _ecl_membership")

    def _label_leiden_communities(self, conn: sqlite3.Connection, model_name: str, now: str) -> None:
        """Name Leiden communities at each resolution via muninn_label_groups TVF."""
        conn.execute("DROP TABLE IF EXISTS community_labels")
        conn.execute("""
            CREATE TABLE community_labels (
                resolution    REAL NOT NULL,
                community_id  INTEGER NOT NULL,
                label         TEXT NOT NULL,
                member_count  INTEGER NOT NULL,
                model         TEXT NOT NULL,
                generated_at  TEXT NOT NULL,
                PRIMARY KEY (resolution, community_id)
            )
        """)

        try:
            resolutions = conn.execute(
                "SELECT DISTINCT resolution FROM leiden_communities ORDER BY resolution"
            ).fetchall()
        except sqlite3.OperationalError:
            log.info("  No leiden_communities table — skipping community labelling")
            return

        if not resolutions:
            log.info("  No Leiden communities found — skipping community labelling")
            return

        total_generated = 0
        for (resolution,) in resolutions:
            # Create a temp membership table for this resolution
            # The TVF needs a real table (not a parameterised view)
            conn.execute("DROP TABLE IF EXISTS temp._comm_membership")
            conn.execute(
                """
                CREATE TEMP TABLE _comm_membership AS
                SELECT CAST(community_id AS TEXT) AS group_id, node AS member_name
                FROM leiden_communities
                WHERE resolution = ?
            """,
                (resolution,),
            )

            n_communities = conn.execute("SELECT count(DISTINCT group_id) FROM _comm_membership").fetchone()[0]
            log.info("  Resolution %.2f: %d communities", resolution, n_communities)

            conn.execute(
                """
                INSERT INTO community_labels(resolution, community_id, label, member_count, model, generated_at)
                SELECT ?, CAST(group_id AS INTEGER), label, member_count, ?, ?
                FROM muninn_label_groups
                WHERE model = ?
                  AND membership_table = '_comm_membership'
                  AND group_col = 'group_id'
                  AND member_col = 'member_name'
                  AND min_group_size = ?
                  AND max_members_in_prompt = ?
                  AND system_prompt = 'Output ONLY a concise topic label (3-8 words). No explanation.'
                """,
                (resolution, model_name, now, model_name, MIN_COMMUNITY_SIZE, MAX_MEMBERS_IN_PROMPT),
            )

            generated = conn.execute(
                "SELECT count(*) FROM community_labels WHERE resolution = ?", (resolution,)
            ).fetchone()[0]
            total_generated += generated
            log.info("    Labelled %d communities at resolution %.2f", generated, resolution)

            conn.execute("DROP TABLE IF EXISTS temp._comm_membership")

        log.info("  Community labelling complete: %d labels across %d resolutions", total_generated, len(resolutions))

    def teardown(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        try:
            conn.execute("DELETE FROM temp.muninn_chat_models WHERE name = ?", (MUNINN_CHAT_MODEL_NAME,))
        except sqlite3.OperationalError:
            pass
        conn.execute("DROP VIEW IF EXISTS _ecl_membership")
        conn.execute("DROP TABLE IF EXISTS temp._comm_membership")
