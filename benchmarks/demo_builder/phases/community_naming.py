"""Phase: Community Naming — LLM-generated labels for Leiden communities."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llama_cpp import Llama

from benchmarks.demo_builder.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a concise summarization system. Generate a short descriptive label "
    "(3-8 words) for a group of related entities."
)

# Communities below this threshold are too small to warrant an LLM label.
MIN_COMMUNITY_SIZE = 3

# Above this threshold, only the top members (by mention frequency) are included
# in the prompt; the rest are summarised as "and N others".
MAX_MEMBERS_IN_PROMPT = 10


class PhaseCommunitySummarisation(Phase):
    """Generate human-readable labels for Leiden communities using an LLM.

    Runs after entity_resolution phase. For each community with >= 3 members,
    queries community membership + relations, formats a prompt, and generates
    a concise label via llama-cpp-python.

    Creates table: community_labels(community_id, label, summary, model, generated_at)
    """

    def __init__(self, model_path: str, ctx_len: int = 4096) -> None:
        self._model_path = model_path
        self._ctx_len = ctx_len

    @property
    def name(self) -> str:
        return "community_naming"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        """Return True if community_labels table is missing or empty."""
        try:
            count: int = conn.execute("SELECT count(*) FROM community_labels").fetchone()[0]
            return count == 0
        except sqlite3.OperationalError:
            return True

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        model_name = Path(self._model_path).stem
        log.info("  Loading LLM: %s (ctx_len=%d)", model_name, self._ctx_len)
        llm = Llama(model_path=self._model_path, n_ctx=self._ctx_len, n_gpu_layers=0, verbose=False)

        # ── Gather community memberships ──────────────────────────────
        # entity_clusters maps (name -> canonical), nodes has the canonical
        # entities with types and mention counts.  Leiden communities are
        # implicit: each distinct canonical name is a community of its aliases.
        # However, the *graph-level* communities come from running Leiden on
        # the coalesced edges table — those are stored as connected components
        # via the graph_leiden TVF during entity_resolution.  We reconstruct
        # community groupings from entity_clusters + nodes.
        #
        # Community id = node_id of the canonical entity (deterministic).
        rows = conn.execute("""
            SELECT ec.canonical, n.node_id, n.entity_type, n.mention_count, ec.name
            FROM entity_clusters ec
            JOIN nodes n ON n.name = ec.canonical
            ORDER BY n.node_id, ec.name
        """).fetchall()

        # Group members by community (keyed by canonical node_id).
        communities: dict[int, dict[str, Any]] = {}
        for canonical, node_id, entity_type, mention_count, member_name in rows:
            if node_id not in communities:
                communities[node_id] = {
                    "canonical": canonical,
                    "entity_type": entity_type,
                    "members": [],
                }
            communities[node_id]["members"].append((member_name, entity_type, mention_count))

        # ── Gather intra-community relations ──────────────────────────
        # Build a lookup: canonical name -> community id for edge matching.
        canonical_to_comm: dict[str, int] = {}
        for comm_id, info in communities.items():
            canonical_to_comm[info["canonical"]] = comm_id

        all_edges = conn.execute("SELECT src, dst, rel_type, weight FROM edges").fetchall()

        # Index edges by community (only intra-community edges).
        comm_edges: dict[int, list[tuple[str, str, str, float]]] = {}
        for src, dst, rel_type, weight in all_edges:
            src_comm = canonical_to_comm.get(src)
            dst_comm = canonical_to_comm.get(dst)
            if src_comm is not None and src_comm == dst_comm:
                comm_edges.setdefault(src_comm, []).append((src, dst, rel_type, weight))

        # ── Filter to communities meeting the size threshold ──────────
        eligible = {cid: info for cid, info in communities.items() if len(info["members"]) >= MIN_COMMUNITY_SIZE}

        log.info(
            "  %d communities total, %d with >= %d members eligible for naming",
            len(communities),
            len(eligible),
            MIN_COMMUNITY_SIZE,
        )

        if not eligible:
            log.info("  No communities large enough to name — creating empty table")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS community_labels (
                    community_id INTEGER PRIMARY KEY,
                    label TEXT NOT NULL,
                    summary TEXT,
                    model TEXT NOT NULL,
                    generated_at TEXT NOT NULL
                )
            """)
            return

        # ── Generate labels ───────────────────────────────────────────
        conn.execute("DROP TABLE IF EXISTS community_labels")
        conn.execute("""
            CREATE TABLE community_labels (
                community_id INTEGER PRIMARY KEY,
                label TEXT NOT NULL,
                summary TEXT,
                model TEXT NOT NULL,
                generated_at TEXT NOT NULL
            )
        """)

        generated = 0
        for comm_id, info in sorted(eligible.items()):
            prompt = _build_prompt(info["members"], comm_edges.get(comm_id, []))

            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=64,
                temperature=0.3,
            )

            label = response["choices"][0]["message"]["content"].strip()
            # Strip surrounding quotes if the model wrapped the label.
            if len(label) >= 2 and label[0] == label[-1] and label[0] in ('"', "'"):
                label = label[1:-1]

            conn.execute(
                "INSERT INTO community_labels (community_id, label, summary, model, generated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (comm_id, label, prompt, model_name, datetime.now(UTC).isoformat()),
            )
            generated += 1

            if generated % 10 == 0:
                log.info("  Named %d / %d communities", generated, len(eligible))

        log.info("  Community naming complete: %d labels generated", generated)


def _build_prompt(
    members: list[tuple[str, str, int]],
    edges: list[tuple[str, str, str, float]],
) -> str:
    """Build the user prompt describing a community's members and relations.

    members: list of (name, entity_type, mention_count) tuples.
    edges: list of (src, dst, rel_type, weight) tuples for intra-community edges.
    """
    # Sort members by mention count descending for the "top N" truncation.
    sorted_members = sorted(members, key=lambda m: m[2], reverse=True)

    if len(sorted_members) <= MAX_MEMBERS_IN_PROMPT:
        member_lines = [f"- {name} ({etype})" for name, etype, _ in sorted_members]
    else:
        top = sorted_members[:MAX_MEMBERS_IN_PROMPT]
        remainder = len(sorted_members) - MAX_MEMBERS_IN_PROMPT
        member_lines = [f"- {name} ({etype})" for name, etype, _ in top]
        member_lines.append(f"- ...and {remainder} others")

    parts = [f"Community nodes ({len(members)} members):"]
    parts.extend(member_lines)

    if edges:
        parts.append("")
        parts.append("Relations:")
        for src, dst, rel_type, _weight in edges[:20]:
            parts.append(f"- {src} {rel_type} {dst}")
        if len(edges) > 20:
            parts.append(f"- ...and {len(edges) - 20} more relations")

    return "\n".join(parts)
