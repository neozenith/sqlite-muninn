"""Phase: Community Naming — LLM-generated labels for entity clusters and Leiden communities."""

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

# Communities below this threshold are too small to warrant an LLM label.
MIN_COMMUNITY_SIZE = 3

# Above this threshold, only the top members (by mention frequency) are included
# in the prompt; the rest are summarised as "and N others".
MAX_MEMBERS_IN_PROMPT = 10

# Model registration SQL — loads the GGUF into muninn's chat model registry.
_REGISTER_MODEL_SQL = (
    "INSERT INTO temp.muninn_chat_models(name, model) "
    "SELECT ?, muninn_chat_model(?)"
)


class PhaseCommunityNaming(Phase):
    """Generate human-readable labels for entity clusters and Leiden communities.

    Two output tables:

    1. entity_cluster_labels — names for entity_clusters groups (synonym naming).
       e.g. "Adam Smith" / "Smith" / "A. Smith" → "Adam Smith (Economist)"

    2. community_labels — names for Leiden graph communities at each resolution.
       e.g. community 3 at resolution 1.0 → "Economic Trade Theory"

    Uses muninn_summarize() SQL function via the loaded GGUF chat model.
    """

    @property
    def name(self) -> str:
        return "community_naming"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        """Return True if either labels table is missing or empty."""
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

        self._label_entity_clusters(conn, model_name)
        self._label_leiden_communities(conn, model_name)

    def _label_entity_clusters(self, conn: sqlite3.Connection, model_name: str) -> None:
        """Name entity_clusters groups using muninn_summarize()."""
        conn.execute("DROP TABLE IF EXISTS entity_cluster_labels")
        conn.execute("""
            CREATE TABLE entity_cluster_labels (
                canonical   TEXT PRIMARY KEY,
                label       TEXT NOT NULL,
                member_count INTEGER NOT NULL,
                model       TEXT NOT NULL,
                generated_at TEXT NOT NULL
            )
        """)

        # Gather entity clusters: group by canonical name
        try:
            rows = conn.execute("""
                SELECT ec.canonical, n.entity_type, n.mention_count, ec.name
                FROM entity_clusters ec
                JOIN nodes n ON n.name = ec.canonical
                ORDER BY ec.canonical, ec.name
            """).fetchall()
        except sqlite3.OperationalError:
            log.info("  No entity_clusters table — skipping entity cluster naming")
            return

        # Group members by canonical
        clusters: dict[str, list[tuple[str, str, int]]] = {}
        for canonical, entity_type, mention_count, member_name in rows:
            clusters.setdefault(canonical, []).append((member_name, entity_type, mention_count))

        # Only label clusters with multiple members (singletons are self-descriptive)
        eligible = {k: v for k, v in clusters.items() if len(v) >= MIN_COMMUNITY_SIZE}

        log.info(
            "  Entity clusters: %d total, %d with >= %d members eligible for naming",
            len(clusters),
            len(eligible),
            MIN_COMMUNITY_SIZE,
        )

        generated = 0
        for canonical, members in sorted(eligible.items()):
            prompt = _build_cluster_prompt(canonical, members)
            label = conn.execute(
                "SELECT muninn_summarize(?, ?)", (model_name, prompt)
            ).fetchone()[0]

            # Strip surrounding quotes if the model wrapped the label
            label = label.strip()
            if len(label) >= 2 and label[0] == label[-1] and label[0] in ('"', "'"):
                label = label[1:-1]

            conn.execute(
                "INSERT INTO entity_cluster_labels (canonical, label, member_count, model, generated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (canonical, label, len(members), model_name, datetime.now(UTC).isoformat()),
            )
            generated += 1

            if generated % 10 == 0:
                log.info("  Named %d / %d entity clusters", generated, len(eligible))

        log.info("  Entity cluster naming complete: %d labels", generated)

    def _label_leiden_communities(self, conn: sqlite3.Connection, model_name: str) -> None:
        """Name Leiden communities at each resolution using muninn_summarize()."""
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

        # Check if leiden_communities exists
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

        # Pre-fetch relations for intra-community edge context
        try:
            all_edges = conn.execute("SELECT src, dst, rel_type FROM relations").fetchall()
        except sqlite3.OperationalError:
            all_edges = []

        total_generated = 0
        for (resolution,) in resolutions:
            # Fetch community assignments at this resolution
            rows = conn.execute(
                "SELECT node, community_id FROM leiden_communities WHERE resolution = ?",
                (resolution,),
            ).fetchall()

            # Group by community
            communities: dict[int, list[str]] = {}
            for node, comm_id in rows:
                communities.setdefault(comm_id, []).append(node)

            # Build node → community map for edge filtering
            node_to_comm = {node: comm_id for node, comm_id in rows}

            # Index intra-community edges
            comm_edges: dict[int, list[tuple[str, str, str]]] = {}
            for src, dst, rel_type in all_edges:
                src_comm = node_to_comm.get(src)
                dst_comm = node_to_comm.get(dst)
                if src_comm is not None and src_comm == dst_comm:
                    comm_edges.setdefault(src_comm, []).append((src, dst, rel_type))

            # Filter to communities meeting the size threshold
            eligible = {cid: members for cid, members in communities.items() if len(members) >= MIN_COMMUNITY_SIZE}

            log.info(
                "  Resolution %.2f: %d communities, %d with >= %d members",
                resolution,
                len(communities),
                len(eligible),
                MIN_COMMUNITY_SIZE,
            )

            generated = 0
            for comm_id, members in sorted(eligible.items()):
                edges = comm_edges.get(comm_id, [])
                prompt = _build_community_prompt(members, edges)
                label = conn.execute(
                    "SELECT muninn_summarize(?, ?)", (model_name, prompt)
                ).fetchone()[0]

                label = label.strip()
                if len(label) >= 2 and label[0] == label[-1] and label[0] in ('"', "'"):
                    label = label[1:-1]

                conn.execute(
                    "INSERT INTO community_labels "
                    "(resolution, community_id, label, member_count, model, generated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (resolution, comm_id, label, len(members), model_name, datetime.now(UTC).isoformat()),
                )
                generated += 1

                if generated % 10 == 0:
                    log.info("    Named %d / %d communities", generated, len(eligible))

            total_generated += generated

        log.info("  Community labelling complete: %d labels across %d resolutions", total_generated, len(resolutions))

    def teardown(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        # Unload the chat model to free memory
        try:
            conn.execute(
                "DELETE FROM temp.muninn_chat_models WHERE name = ?",
                (MUNINN_CHAT_MODEL_NAME,),
            )
        except sqlite3.OperationalError:
            pass


def _build_cluster_prompt(canonical: str, members: list[tuple[str, str, int]]) -> str:
    """Build a prompt describing an entity cluster for naming."""
    sorted_members = sorted(members, key=lambda m: m[2], reverse=True)

    if len(sorted_members) <= MAX_MEMBERS_IN_PROMPT:
        member_lines = [f"- {name} ({etype})" for name, etype, _ in sorted_members]
    else:
        top = sorted_members[:MAX_MEMBERS_IN_PROMPT]
        remainder = len(sorted_members) - MAX_MEMBERS_IN_PROMPT
        member_lines = [f"- {name} ({etype})" for name, etype, _ in top]
        member_lines.append(f"- ...and {remainder} others")

    parts = [
        f"Entity cluster with canonical name '{canonical}' ({len(members)} aliases):",
        *member_lines,
        "",
        "Generate a concise label (3-8 words) describing what this entity is.",
    ]
    return "\n".join(parts)


def _build_community_prompt(
    members: list[str],
    edges: list[tuple[str, str, str]],
) -> str:
    """Build a prompt describing a Leiden community for naming."""
    sorted_members = sorted(members)

    if len(sorted_members) <= MAX_MEMBERS_IN_PROMPT:
        member_lines = [f"- {name}" for name in sorted_members]
    else:
        top = sorted_members[:MAX_MEMBERS_IN_PROMPT]
        remainder = len(sorted_members) - MAX_MEMBERS_IN_PROMPT
        member_lines = [f"- {name}" for name in top]
        member_lines.append(f"- ...and {remainder} others")

    parts = [f"Knowledge graph community ({len(members)} entities):"]
    parts.extend(member_lines)

    if edges:
        parts.append("")
        parts.append("Relations between members:")
        for src, dst, rel_type in edges[:20]:
            parts.append(f"- {src} {rel_type} {dst}")
        if len(edges) > 20:
            parts.append(f"- ...and {len(edges) - 20} more relations")

    parts.append("")
    parts.append("Generate a concise label (3-8 words) describing the topic of this community.")
    return "\n".join(parts)
