"""Phase: Entity Resolution via muninn_extract_er().

Uses the C function that implements the full ER pipeline:
  HNSW blocking → JW+cosine scoring → Leiden clustering → edge betweenness cleanup.

The C function expects an `entities` table with (entity_id, name, source) and
an HNSW virtual table. We create a temp mapping table to bridge the demo_builder's
schema (entity_vec_map with entity names as IDs, entity_type as source) to the
C function's expected format.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import TYPE_CHECKING

from benchmarks.demo_builder.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext

log = logging.getLogger(__name__)


class PhaseEntityResolution(Phase):
    """Resolve entity synonyms using muninn_extract_er() C function."""

    def __init__(
        self,
        *,
        k: int = 10,
        dist_threshold: float = 0.15,
        jw_weight: float = 0.3,
        borderline_delta: float = 0.0,
        edge_betweenness_threshold: float | None = None,
    ) -> None:
        self._k = k
        self._dist = dist_threshold
        self._jw = jw_weight
        self._delta = borderline_delta
        self._eb_threshold = edge_betweenness_threshold
        self._entity_name_to_type: dict[str, str] = {}
        self._entity_name_to_count: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "entity_resolution"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        try:
            node_count: int = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
            if node_count == 0:
                return True
            distinct_names: int = conn.execute("SELECT count(DISTINCT name) FROM entities").fetchone()[0]
            cluster_count: int = conn.execute("SELECT count(*) FROM entity_clusters").fetchone()[0]
            return cluster_count < distinct_names
        except sqlite3.OperationalError:
            return True

    def restore_ctx(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        try:
            ctx.num_nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
            ctx.num_edges = conn.execute("SELECT count(*) FROM edges").fetchone()[0]
        except sqlite3.OperationalError:
            pass

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        entity_stats = conn.execute("""
            SELECT name, entity_type, count(*) as mention_count
            FROM entities
            GROUP BY name
            ORDER BY name
        """).fetchall()
        log.info("  %d unique entity names to resolve", len(entity_stats))

        self._entity_name_to_type = {name: etype for name, etype, _ in entity_stats}
        self._entity_name_to_count = {name: cnt for name, _, cnt in entity_stats}

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        entity_name_to_type = self._entity_name_to_type
        entity_name_to_count = self._entity_name_to_count

        # ── Bridge schema: create temp entities table for C function ──
        # The C function expects: entities(entity_id TEXT, name TEXT, source TEXT)
        # Demo builder has: entity_vec_map(name) with entities_vec HNSW index
        # The C function's type guard in 'diff_type' mode skips pairs where
        # entity_type differs (person ≠ location). We populate the 'source'
        # column with entity_type so the C function can filter on it.

        # ── Drop output tables before rebuilding ─────────────────────
        conn.execute("DROP TABLE IF EXISTS _match_edges")
        conn.execute("DROP TABLE IF EXISTS entity_clusters")
        conn.execute("DROP TABLE IF EXISTS nodes")
        conn.execute("DROP TABLE IF EXISTS edges")

        # ── Call muninn_extract_er() ─────────────────────────────────
        # Note: the C function reads from the "entities" table by default.
        # We need to temporarily rename our temp table.
        # Actually, the C function hardcodes "entities" as the table name.
        # Our _er_entities is a temp table. Let's create a view instead.
        conn.execute("DROP VIEW IF EXISTS temp._er_ent_view")

        # Simplest: the C function reads from "entities" table.
        # But demo_builder's entities table has different columns.
        # Solution: use a temp table named exactly "entities" with the right schema.
        # Problem: "entities" already exists with NER data.
        # Better solution: rename the entities table temporarily.

        # Actually, the cleanest approach: modify the temp table to be named
        # correctly. The C function reads "entities" — let's shadow it with a temp.
        # SQLite temp tables shadow main tables of the same name.
        conn.execute("DROP TABLE IF EXISTS temp.entities")
        conn.execute("""
            CREATE TEMP TABLE entities(entity_id TEXT, name TEXT, source TEXT)
        """)
        # Join entity_vec_map with main entities table to get entity_type.
        # entity_type goes into the 'source' column — the C function's
        # 'diff_type' guard skips pairs where source values differ.
        conn.execute("""
            INSERT INTO temp.entities(entity_id, name, source)
            SELECT m.name, m.name, COALESCE(e.entity_type, '')
            FROM entity_vec_map m
            LEFT JOIN (
                SELECT name, entity_type FROM main.entities GROUP BY name
            ) e ON e.name = m.name
        """)

        result_json = conn.execute(
            "SELECT muninn_extract_er(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "entities_vec",  # HNSW table
                "name",  # name column
                self._k,
                self._dist,
                self._jw,
                self._delta,
                None,  # chat_model (no LLM for now)
                self._eb_threshold,
                "diff_type",  # KG mode: skip pairs where entity_type differs
            ),
        ).fetchone()[0]

        # Drop the temp shadow — restore access to main entities table
        conn.execute("DROP TABLE IF EXISTS temp.entities")
        conn.execute("DROP TABLE IF EXISTS _er_entities")

        # ── Parse cluster JSON into entity_to_canonical ──────────────
        clusters = json.loads(result_json)["clusters"]

        # Group entities by cluster_id
        communities: dict[int, list[str]] = {}
        for entity_name, cluster_id in clusters.items():
            communities.setdefault(cluster_id, []).append(entity_name)

        # Pick canonical = highest mention-count member per cluster
        entity_to_canonical: dict[str, str] = {}
        for _comm_id, members in communities.items():
            canonical = max(members, key=lambda n: entity_name_to_count.get(n, 0))
            for member in members:
                entity_to_canonical[member] = canonical

        # Entities not in any cluster are their own canonical
        for ent_name in entity_name_to_type:
            if ent_name not in entity_to_canonical:
                entity_to_canonical[ent_name] = ent_name

        log.info(
            "  muninn_extract_er: %d entities → %d clusters",
            len(entity_to_canonical),
            len(communities),
        )

        # ── Populate entity_clusters table ───────────────────────────
        conn.execute("CREATE TABLE entity_clusters (name TEXT PRIMARY KEY, canonical TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO entity_clusters (name, canonical) VALUES (?, ?)",
            entity_to_canonical.items(),
        )

        # ── Build clean graph: nodes + edges ─────────────────────────
        conn.execute("""
            CREATE TABLE nodes (
                node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                entity_type TEXT,
                mention_count INTEGER DEFAULT 0
            )
        """)

        canonical_stats: dict[str, dict[str, str | int]] = {}
        for ent_name, etype, count in conn.execute(
            "SELECT name, entity_type, count(*) FROM entities GROUP BY name ORDER BY name"
        ).fetchall():
            canonical = entity_to_canonical[ent_name]
            if canonical not in canonical_stats:
                canonical_stats[canonical] = {"entity_type": etype, "mention_count": 0}
            canonical_stats[canonical]["mention_count"] += count

        for canonical in sorted(canonical_stats):
            stats = canonical_stats[canonical]
            conn.execute(
                "INSERT INTO nodes (name, entity_type, mention_count) VALUES (?, ?, ?)",
                (canonical, stats["entity_type"], stats["mention_count"]),
            )

        num_nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
        log.info("  Built nodes table: %d canonical entities", num_nodes)

        # ── Coalesce relations into edges ────────────────────────────
        conn.execute("""
            CREATE TABLE edges (
                src TEXT NOT NULL,
                dst TEXT NOT NULL,
                rel_type TEXT,
                weight REAL DEFAULT 1.0,
                PRIMARY KEY (src, dst, rel_type)
            )
        """)

        raw_relations = conn.execute("SELECT src, dst, rel_type, weight FROM relations").fetchall()

        edge_agg: dict[tuple[str, str, str], float] = {}
        for src, dst, rel_type, weight in raw_relations:
            c_src: str = entity_to_canonical.get(src, src)
            c_dst: str = entity_to_canonical.get(dst, dst)
            if c_src == c_dst:
                continue
            edge_key: tuple[str, str, str] = (c_src, c_dst, str(rel_type))
            edge_agg[edge_key] = edge_agg.get(edge_key, 0.0) + float(weight)

        conn.executemany(
            "INSERT OR IGNORE INTO edges (src, dst, rel_type, weight) VALUES (?, ?, ?, ?)",
            [(src, dst, rt, w) for (src, dst, rt), w in edge_agg.items()],
        )

        num_edges = conn.execute("SELECT count(*) FROM edges").fetchone()[0]
        log.info("  Built edges table: %d coalesced edges", num_edges)

        ctx.num_nodes = num_nodes
        ctx.num_edges = num_edges

    def teardown(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        conn.execute("DROP TABLE IF EXISTS _match_edges")
        conn.execute("DROP TABLE IF EXISTS _er_entities")
        conn.execute("DROP TABLE IF EXISTS temp.entities")
