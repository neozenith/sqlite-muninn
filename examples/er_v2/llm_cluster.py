"""Unified ER pipeline: blocking + type guard + scoring + LLM + Leiden + betweenness.

Pipeline stages:
  1. KNN blocking (reads prebuilt HNSW index)
  2. Type guard (skip cross-type pairs — G3)
  3. Scoring cascade (JW + cosine, auto-accept/reject/borderline)
  4. LLM clustering of borderline components (if llm_low < llm_high)
  5. Leiden clustering on all match edges
  6. Betweenness cleanup — remove bridge edges + re-cluster (G4)

Setting llm_low >= llm_high disables the LLM tier (Stage 4).
Setting betweenness_threshold to None disables bridge cleanup (Stage 6).
Setting type_guard=False disables type filtering (Stage 2).
"""

import json
import logging
import sqlite3
import time
from collections import defaultdict

from .blocking import block, leiden_cluster
from .jaro_winkler import jaro_winkler

log = logging.getLogger(__name__)

# ── GBNF Grammar ──────────────────────────────────────────────────

GBNF_ER_CLUSTER_NUM = (
    'root       ::= "{" ws "\\"groups\\"" ws ":" ws "[" ws group-list ws "]" ws "}" \n'
    'group-list ::= group (ws "," ws group)* \n'
    'group      ::= "[" ws int-list ws "]" \n'
    'int-list   ::= integer (ws "," ws integer)* \n'
    "integer    ::= [1-9] [0-9]* \n"
    "ws         ::= [ \\t\\n]* \n"
)

SYSTEM_PROMPT = (
    "You are an entity resolution expert. Given a numbered list of entity names, "
    "group the ones that refer to the same real-world entity. Return groups as "
    "lists of numbers. Entities with no match go in their own singleton group.\n\n"
    "You MUST respond with valid JSON only. No additional text, explanation, or markdown.\n\n"
    "Example:\n"
    "Input:\n"
    '  1. Sony VAIO VPC-EB15FM/BI 15.5" Notebook PC\n'
    "  2. Sony VAIO VPCEB15FM/BI Notebook\n"
    '  3. Acer Aspire 5100-5033 15.4" Notebook\n'
    "  4. Acer Aspire AS5100-5033 Laptop\n"
    "  5. Canon PowerShot SD1200 IS Digital Camera\n\n"
    "Output:\n"
    '  {"groups": [[1, 2], [3, 4], [5]]}'
)

USER_PROMPT = (
    "Group these entities by whether they refer to the same thing. "
    "Use their numbers, not names.\n\n"
    "{candidate_list}\n\n"
    "Respond with JSON only."
)

MAX_COMPONENT_SIZE = 20


def run(
    conn: sqlite3.Connection,
    model_name: str,
    *,
    k: int = 10,
    dist_threshold: float = 0.4,
    jw_weight: float = 0.3,
    llm_low: float = 0.3,
    llm_high: float = 0.7,
    type_guard: bool = True,
    betweenness_threshold: float | None = None,
) -> tuple[dict[str, int], dict]:
    """Run the full ER pipeline on a prebuilt embedded DB.

    Returns (entity_id -> cluster_id, stats).

    Args:
        type_guard: If True, skip candidate pairs where both entities have
            known types and the types differ. For DeepMatcher datasets, the
            "source" field (a/b) acts as type — cross-source pairs are the
            only ones that CAN be true matches, so same-source pairs are skipped.
        betweenness_threshold: If set, after Leiden clustering compute edge
            betweenness and remove bridge edges above this threshold, then
            re-cluster. None = disabled.
    """
    # Stage 1: KNN blocking
    t_block = time.perf_counter()
    id_map, name_map, source_map, candidate_pairs = block(conn, k, dist_threshold)
    blocking_time = time.perf_counter() - t_block

    # Stage 2: Type guard (G3)
    t_type = time.perf_counter()
    n_type_filtered = 0
    if type_guard:
        filtered_pairs: dict[tuple[int, int], float] = {}
        for (r1, r2), dist in candidate_pairs.items():
            s1 = source_map.get(r1)
            s2 = source_map.get(r2)
            # DeepMatcher convention: "source" = which table the record came from.
            # Records from the SAME source table are already deduplicated —
            # they can never be a true match. Only CROSS-source pairs can match.
            # e.g., Abt product a_42 can match Buy product b_117,
            #       but a_42 can never match a_99 (both from Abt).
            #
            # For KG NER data, swap this: same entity_type CAN match,
            # different types should be skipped (person ≠ location).
            if s1 and s2 and s1 == s2:
                n_type_filtered += 1
                continue
            filtered_pairs[(r1, r2)] = dist
        candidate_pairs = filtered_pairs
    type_guard_time = time.perf_counter() - t_type

    # Stage 3: Scoring cascade
    t_score = time.perf_counter()
    match_edges: list[tuple[str, str, float]] = []
    borderline_pairs: list[tuple[int, int]] = []
    n_auto_accepted = 0
    n_auto_rejected = 0

    for (r1, r2), cosine_dist in candidate_pairs.items():
        n1 = name_map[r1]
        n2 = name_map[r2]
        cosine_sim = 1.0 - cosine_dist

        if n1 == n2:
            match_edges.append((id_map[r1], id_map[r2], 1.0))
            n_auto_accepted += 1
            continue
        if n1.lower() == n2.lower():
            match_edges.append((id_map[r1], id_map[r2], 0.9))
            n_auto_accepted += 1
            continue

        jw = jaro_winkler(n1.lower(), n2.lower())
        score = jw_weight * jw + (1.0 - jw_weight) * cosine_sim

        if score >= llm_high:
            match_edges.append((id_map[r1], id_map[r2], score))
            n_auto_accepted += 1
        elif score >= llm_low and llm_low < llm_high:
            borderline_pairs.append((r1, r2))
        else:
            n_auto_rejected += 1

    scoring_time = time.perf_counter() - t_score

    # Stage 4: LLM clustering of borderline components
    components = _connected_components(borderline_pairs)
    components = _split_oversized(components, MAX_COMPONENT_SIZE)

    comp_sizes = [len(c) for c in components if len(c) >= 2]
    avg_comp_size = sum(comp_sizes) / len(comp_sizes) if comp_sizes else 0.0
    comp_size_var = sum((s - avg_comp_size) ** 2 for s in comp_sizes) / len(comp_sizes) if len(comp_sizes) > 1 else 0.0

    llm_calls = 0
    llm_time = 0.0

    for component in components:
        if len(component) < 2:
            continue

        comp_list = sorted(component)
        num_to_rid = {i + 1: rid for i, rid in enumerate(comp_list)}
        candidate_list = "\n".join(f"  {i + 1}. {name_map[rid]}" for i, rid in enumerate(comp_list))

        prompt = USER_PROMPT.format(candidate_list=candidate_list)
        max_tokens = 30 * len(comp_list)

        t0 = time.perf_counter()
        result = conn.execute(
            "SELECT muninn_chat(?, ?, ?, ?, ?)",
            (model_name, prompt, GBNF_ER_CLUSTER_NUM, max_tokens, SYSTEM_PROMPT),
        ).fetchone()[0]
        llm_time += time.perf_counter() - t0
        llm_calls += 1

        groups = _parse_cluster_result(result, num_to_rid, id_map)
        for group in groups:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    match_edges.append((group[i], group[j], 0.85))

    log.info(
        "Pipeline: %d edges (%d accept, %d borderline, %d reject, %d type-filtered), "
        "%d LLM calls (%.2fs), %d components",
        len(match_edges),
        n_auto_accepted,
        len(borderline_pairs),
        n_auto_rejected,
        n_type_filtered,
        llm_calls,
        llm_time,
        len(comp_sizes),
    )

    # Stage 5: Leiden clustering
    t_leiden = time.perf_counter()
    all_entity_ids = list(id_map.values())
    clusters = leiden_cluster(conn, all_entity_ids, match_edges)
    leiden_time = time.perf_counter() - t_leiden

    # Stage 6: Betweenness cleanup (G4)
    t_between = time.perf_counter()
    n_bridges_removed = 0
    if betweenness_threshold is not None and match_edges:
        clusters, n_bridges_removed = _betweenness_cleanup(conn, all_entity_ids, match_edges, betweenness_threshold)
    betweenness_time = time.perf_counter() - t_between

    stats = {
        "params": {
            "k": k,
            "dist_threshold": dist_threshold,
            "jw_weight": jw_weight,
            "llm_low": llm_low,
            "llm_high": llm_high,
            "max_component_size": MAX_COMPONENT_SIZE,
            "type_guard": type_guard,
            "betweenness_threshold": betweenness_threshold,
        },
        "total_pairs": len(candidate_pairs) + n_type_filtered,
        "type_filtered": n_type_filtered,
        "scored_pairs": len(candidate_pairs),
        "auto_accepted": n_auto_accepted,
        "borderline_pairs": len(borderline_pairs),
        "auto_rejected": n_auto_rejected,
        "n_components": len(comp_sizes),
        "avg_component_size": round(avg_comp_size, 2),
        "component_size_variance": round(comp_size_var, 2),
        "llm_calls": llm_calls,
        "bridges_removed": n_bridges_removed,
        "timing": {
            "blocking_s": round(blocking_time, 3),
            "type_guard_s": round(type_guard_time, 3),
            "scoring_s": round(scoring_time, 3),
            "llm_s": round(llm_time, 3),
            "leiden_s": round(leiden_time, 3),
            "betweenness_s": round(betweenness_time, 3),
        },
    }
    return clusters, stats


# ── Betweenness Cleanup (G4) ─────────────────────────────────────


def _betweenness_cleanup(
    conn: sqlite3.Connection,
    all_entity_ids: list[str],
    match_edges: list[tuple[str, str, float]],
    threshold: float,
) -> tuple[dict[str, int], int]:
    """Remove high-betweenness bridge edges and re-cluster.

    Uses graph_edge_betweenness TVF for surgical edge removal — only
    the specific bridge edge is removed, not all edges touching the
    bridge node. This preserves valid edges of nodes that happen to
    be on bridge paths.

    Returns (new_clusters, n_edges_removed).
    """
    # The _match_edges table was populated by leiden_cluster.
    betweenness_rows = conn.execute(
        "SELECT src, dst, centrality FROM graph_edge_betweenness"
        " WHERE edge_table = '_match_edges'"
        "   AND src_col = 'src'"
        "   AND dst_col = 'dst'"
        "   AND direction = 'both'"
    ).fetchall()

    # Build lookup of high-betweenness edges (normalise to min/max order)
    bridge_edges: set[tuple[str, str]] = set()
    for src, dst, bc in betweenness_rows:
        if bc > threshold:
            bridge_edges.add((min(src, dst), max(src, dst)))

    if not bridge_edges:
        log.info("Betweenness cleanup: no bridge edges above threshold %.4f", threshold)
        return leiden_cluster(conn, all_entity_ids, match_edges), 0

    # Remove bridge edges
    pruned_edges = [
        (s, d, w) for s, d, w in match_edges if (min(s, d), max(s, d)) not in bridge_edges
    ]

    log.info(
        "Betweenness cleanup: %d bridge edges removed (threshold=%.4f), %d→%d edges",
        len(bridge_edges),
        threshold,
        len(match_edges),
        len(pruned_edges),
    )

    clusters = leiden_cluster(conn, all_entity_ids, pruned_edges)
    return clusters, len(bridge_edges)


# ── Helpers ───────────────────────────────────────────────────────


def _connected_components(pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Find connected components in an undirected pair graph via BFS."""
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in pairs:
        adj[a].append(b)
        adj[b].append(a)

    visited: set[int] = set()
    components: list[list[int]] = []
    for node in adj:
        if node in visited:
            continue
        component: list[int] = []
        queue = [node]
        while queue:
            n = queue.pop()
            if n in visited:
                continue
            visited.add(n)
            component.append(n)
            queue.extend(adj[n])
        components.append(component)
    return components


def _split_oversized(components: list[list[int]], max_size: int) -> list[list[int]]:
    """Split components larger than max_size into chunks."""
    result: list[list[int]] = []
    for comp in components:
        if len(comp) <= max_size:
            result.append(comp)
        else:
            for i in range(0, len(comp), max_size):
                result.append(comp[i : i + max_size])
    return result


def _parse_cluster_result(
    result: str,
    num_to_rid: dict[int, int],
    id_map: dict[int, str],
) -> list[list[str]]:
    """Parse numbered cluster result into groups of entity_ids."""
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        log.warning("Failed to parse cluster result: %s", result[:100])
        return []

    if "groups" not in parsed or not isinstance(parsed["groups"], list):
        return []

    groups: list[list[str]] = []
    for group in parsed["groups"]:
        if not isinstance(group, list) or len(group) < 2:
            continue
        entity_ids: list[str] = []
        for num in group:
            num_int = int(num) if isinstance(num, (int, float, str)) else None
            if num_int and num_int in num_to_rid:
                entity_ids.append(id_map[num_to_rid[num_int]])
        if len(entity_ids) >= 2:
            groups.append(entity_ids)
    return groups
