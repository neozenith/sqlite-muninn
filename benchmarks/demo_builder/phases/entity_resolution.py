"""Phase 6: Entity Resolution (HNSW blocking + Jaro-Winkler + Leiden)."""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

from benchmarks.demo_builder.common import jaro_winkler
from benchmarks.demo_builder.phases.base import Phase

if TYPE_CHECKING:
    from benchmarks.demo_builder.build import PhaseContext

log = logging.getLogger(__name__)


class PhaseEntityResolution(Phase):
    """Resolve entity synonyms using HNSW blocking, string similarity, and Leiden clustering."""

    def __init__(self) -> None:
        self._entity_name_to_type: dict[str, str] = {}
        self._entity_name_to_count: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "entity_resolution"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        """Return True if entity_resolution output is missing or stale."""
        try:
            node_count = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
            if node_count == 0:
                return True
            # Stale if there are entity names not covered by any cluster.
            distinct_names = conn.execute("SELECT count(DISTINCT name) FROM entities").fetchone()[0]
            cluster_count = conn.execute("SELECT count(*) FROM entity_clusters").fetchone()[0]
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

        # ── HNSW blocking: find candidate match pairs ─────────────────
        entity_names_ordered = conn.execute("SELECT name FROM entity_vec_map ORDER BY rowid").fetchall()
        entity_names_ordered = [r[0] for r in entity_names_ordered]

        # Build name -> rowid mapping
        name_to_rowid = {}
        for row in conn.execute("SELECT rowid, name FROM entity_vec_map"):
            name_to_rowid[row[1]] = row[0]

        k_neighbors = 10
        candidate_pairs: list[tuple[str, str, float]] = []

        log.info("  HNSW blocking: finding %d nearest neighbors per entity...", k_neighbors)
        for ent_name in entity_names_ordered:
            rowid = name_to_rowid[ent_name]
            vec = conn.execute("SELECT vector FROM entities_vec WHERE rowid = ?", (rowid,)).fetchone()[0]

            # KNN search
            neighbors = conn.execute(
                "SELECT rowid, distance FROM entities_vec WHERE vector MATCH ? AND k = ?",
                (vec, k_neighbors + 1),
            ).fetchall()

            for neighbor_rowid, distance in neighbors:
                if neighbor_rowid == rowid:
                    continue
                if distance > 0.4:
                    continue
                neighbor_name = entity_names_ordered[neighbor_rowid - 1]
                pair = tuple(sorted([ent_name, neighbor_name]))
                candidate_pairs.append((pair[0], pair[1], distance))

        # Deduplicate candidate pairs
        seen_pairs: set[tuple[str, str]] = set()
        unique_pairs: list[tuple[str, str, float]] = []
        for n1, n2, dist in candidate_pairs:
            key = (n1, n2)
            if key not in seen_pairs:
                seen_pairs.add(key)
                unique_pairs.append((n1, n2, dist))

        log.info("  Found %d candidate pairs from HNSW blocking", len(unique_pairs))

        # ── Matching cascade: score each candidate pair ───────────────
        match_edges: list[tuple[str, str, float]] = []

        for n1, n2, cosine_dist in unique_pairs:
            cosine_sim = 1.0 - cosine_dist

            # Exact match
            if n1 == n2:
                match_edges.append((n1, n2, 1.0))
                continue

            # Case-insensitive exact
            if n1.lower() == n2.lower():
                match_edges.append((n1, n2, 0.9))
                continue

            # Jaro-Winkler on lowercased names
            jw = jaro_winkler(n1.lower(), n2.lower())

            # Combined score: 0.4 * Jaro-Winkler + 0.6 * cosine similarity
            combined = 0.4 * jw + 0.6 * cosine_sim

            if combined > 0.5:
                match_edges.append((n1, n2, combined))

        log.info("  %d match pairs above threshold 0.5", len(match_edges))

        # ── Leiden clustering on match pairs ──────────────────────────
        # Drop output tables before rebuilding — entity_resolution is a global
        # clustering pass that cannot be incrementalised. Always runs from scratch.
        conn.execute("DROP TABLE IF EXISTS _match_edges")
        conn.execute("DROP TABLE IF EXISTS entity_clusters")
        conn.execute("DROP TABLE IF EXISTS nodes")
        conn.execute("DROP TABLE IF EXISTS edges")
        conn.execute("CREATE TABLE _match_edges (src TEXT NOT NULL, dst TEXT NOT NULL, weight REAL DEFAULT 1.0)")
        conn.executemany(
            "INSERT INTO _match_edges (src, dst, weight) VALUES (?, ?, ?)",
            match_edges,
        )

        # Also insert reverse edges (Leiden expects undirected)
        conn.executemany(
            "INSERT INTO _match_edges (src, dst, weight) VALUES (?, ?, ?)",
            [(n2, n1, w) for n1, n2, w in match_edges],
        )

        # Run Leiden if we have match edges
        entity_to_canonical: dict[str, str] = {}

        if match_edges:
            leiden_results = conn.execute(
                "SELECT node, community_id FROM graph_leiden"
                " WHERE edge_table = '_match_edges'"
                "   AND src_col = 'src'"
                "   AND dst_col = 'dst'"
                "   AND weight_col = 'weight'"
            ).fetchall()

            # Group by community
            communities: dict[int, list[str]] = {}
            for node, comm_id in leiden_results:
                communities.setdefault(comm_id, []).append(node)

            # For each community, pick canonical = highest mention-count member
            for _comm_id, members in communities.items():
                canonical = max(members, key=lambda n: entity_name_to_count.get(n, 0))
                for member in members:
                    entity_to_canonical[member] = canonical

            log.info("  Leiden found %d communities from %d matched entities", len(communities), len(leiden_results))

        # Entities not in any match pair are their own canonical form
        for ent_name in entity_name_to_type:
            if ent_name not in entity_to_canonical:
                entity_to_canonical[ent_name] = ent_name

        # ── Populate entity_clusters table (fresh: was dropped above) ──
        conn.execute("CREATE TABLE entity_clusters (name TEXT PRIMARY KEY, canonical TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO entity_clusters (name, canonical) VALUES (?, ?)",
            entity_to_canonical.items(),
        )

        # ── Build clean graph: nodes + edges ──────────────────────────
        conn.execute("""
            CREATE TABLE nodes (
                node_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                entity_type TEXT,
                mention_count INTEGER DEFAULT 0
            )
        """)

        # Aggregate canonical entities
        canonical_stats: dict[str, dict[str, str | int]] = {}
        for ent_name, etype, count in conn.execute(
            "SELECT name, entity_type, count(*) FROM entities GROUP BY name ORDER BY name"
        ).fetchall():
            canonical = entity_to_canonical[ent_name]
            if canonical not in canonical_stats:
                canonical_stats[canonical] = {
                    "entity_type": etype,
                    "mention_count": 0,
                }
            canonical_stats[canonical]["mention_count"] += count

        # Insert nodes (sorted for deterministic node_ids)
        for canonical in sorted(canonical_stats):
            stats = canonical_stats[canonical]
            conn.execute(
                "INSERT INTO nodes (name, entity_type, mention_count) VALUES (?, ?, ?)",
                (canonical, stats["entity_type"], stats["mention_count"]),
            )

        num_nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
        log.info("  Built nodes table: %d canonical entities", num_nodes)

        # ── Coalesce relations into edges ─────────────────────────────
        conn.execute("""
            CREATE TABLE edges (
                src TEXT NOT NULL,
                dst TEXT NOT NULL,
                rel_type TEXT,
                weight REAL DEFAULT 1.0,
                PRIMARY KEY (src, dst, rel_type)
            )
        """)

        # Aggregate relations using canonical names
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
