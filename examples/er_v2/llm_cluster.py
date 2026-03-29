"""LLM-cluster ER pipeline: HNSW blocking + tiered matching + LLM batch clustering + Leiden.

Uses three matching tiers:
  1. Confident matches (score > llm_high)      -> auto-accept
  2. Clear non-matches (score < llm_low)        -> auto-reject
  3. Borderline pairs (llm_low <= score <= llm_high) -> routed to LLM

Borderline pairs are grouped into connected components and sent as batch
clustering calls to muninn_chat() with GBNF_ER_CLUSTER_NUM grammar.
Each LLM call processes one component, returning numbered groups.

Note: setting llm_low >= llm_high disables the LLM tier entirely,
making this pipeline functionally identical to string-only.
"""

import json
import logging
import sqlite3
import time
from collections import defaultdict

from .blocking import block, leiden_cluster
from .jaro_winkler import jaro_winkler

log = logging.getLogger(__name__)

# ── GBNF Grammar: Numbered Cluster Format ─────────────────────────
# LLM returns group indices instead of reproducing entity names verbatim.
# e.g., {"groups": [[1, 3], [2], [4, 5]]}

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

MAX_COMPONENT_SIZE = 20  # Cap to prevent output truncation


def run(
    conn: sqlite3.Connection,
    model_name: str,
    *,
    k: int = 10,
    dist_threshold: float = 0.4,
    jw_weight: float = 0.4,
    llm_low: float = 0.3,
    llm_high: float = 0.7,
) -> tuple[dict[str, int], dict]:
    """Run matching + clustering on a prebuilt embedded DB.

    Returns (entity_id -> cluster_id, stats).

    The conn must be a prebuilt DB from blocking.open_prep_db() containing
    entities and entity_vecs tables. Embedding is NOT done here.

    Stages:
      1. KNN blocking (reads prebuilt HNSW index)
      2. Scoring cascade (JW + cosine, auto-accept/reject/borderline)
      3. LLM clustering of borderline components (if llm_low < llm_high)
      4. Leiden clustering on all edges
    """
    # Stage 1: KNN blocking
    t_block = time.perf_counter()
    id_map, name_map, candidate_pairs = block(conn, k, dist_threshold)
    blocking_time = time.perf_counter() - t_block

    # Stage 2: Scoring cascade
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

    # Stage 3: LLM clustering of borderline components
    components = _connected_components(borderline_pairs)
    components = _split_oversized(components, MAX_COMPONENT_SIZE)

    # Component size statistics
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
        "LLM-cluster: %d edges, %d auto-accept, %d borderline, %d auto-reject, "
        "%d LLM calls (%.2fs), %d components (avg %.1f)",
        len(match_edges),
        n_auto_accepted,
        len(borderline_pairs),
        n_auto_rejected,
        llm_calls,
        llm_time,
        len(comp_sizes),
        avg_comp_size,
    )

    # Stage 4: Leiden clustering
    t_leiden = time.perf_counter()
    all_entity_ids = list(id_map.values())
    clusters = leiden_cluster(conn, all_entity_ids, match_edges)
    leiden_time = time.perf_counter() - t_leiden

    stats = {
        "params": {
            "k": k,
            "dist_threshold": dist_threshold,
            "jw_weight": jw_weight,
            "llm_low": llm_low,
            "llm_high": llm_high,
            "max_component_size": MAX_COMPONENT_SIZE,
        },
        "total_pairs": len(candidate_pairs),
        "auto_accepted": n_auto_accepted,
        "borderline_pairs": len(borderline_pairs),
        "auto_rejected": n_auto_rejected,
        "n_components": len(comp_sizes),
        "avg_component_size": round(avg_comp_size, 2),
        "component_size_variance": round(comp_size_var, 2),
        "llm_calls": llm_calls,
        "timing": {
            "blocking_s": round(blocking_time, 3),
            "scoring_s": round(scoring_time, 3),
            "llm_s": round(llm_time, 3),
            "leiden_s": round(leiden_time, 3),
        },
    }
    return clusters, stats


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
