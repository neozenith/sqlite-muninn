"""String-only ER pipeline: HNSW blocking + Jaro-Winkler/cosine cascade + Leiden.

No LLM calls. The matching cascade uses three tiers:
  1. Exact name match       -> score 1.0
  2. Case-insensitive match -> score 0.9
  3. JW + cosine combined   -> jw_weight * JW + (1 - jw_weight) * cosine_sim

Pairs above `match_threshold` become edges for Leiden clustering.
"""

import logging
import sqlite3

from .blocking import embed_and_block, leiden_cluster
from .datasets import Entity
from .jaro_winkler import jaro_winkler
from .models import DEFAULT_EMBED_MODEL

log = logging.getLogger(__name__)


def run(
    conn: sqlite3.Connection,
    entities: list[Entity],
    *,
    k: int = 10,
    dist_threshold: float = 0.4,
    match_threshold: float = 0.5,
    jw_weight: float = 0.4,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
) -> tuple[dict[str, int], dict]:
    """Run string-only ER pipeline. Returns (entity_id -> cluster_id, stats)."""
    id_map, name_map, candidate_pairs = embed_and_block(conn, entities, k, dist_threshold, embed_model_name)

    match_edges: list[tuple[str, str, float]] = []
    for (r1, r2), cosine_dist in candidate_pairs.items():
        n1 = name_map[r1]
        n2 = name_map[r2]
        cosine_sim = 1.0 - cosine_dist

        if n1 == n2:
            score = 1.0
        elif n1.lower() == n2.lower():
            score = 0.9
        else:
            jw = jaro_winkler(n1.lower(), n2.lower())
            score = jw_weight * jw + (1.0 - jw_weight) * cosine_sim

        if score > match_threshold:
            match_edges.append((id_map[r1], id_map[r2], score))

    log.info(
        "String matching: %d/%d pairs pass threshold %.2f", len(match_edges), len(candidate_pairs), match_threshold
    )

    clusters = leiden_cluster(conn, entities, match_edges)

    stats = {
        "embed_model": embed_model_name,
        "params": {
            "k": k,
            "dist_threshold": dist_threshold,
            "match_threshold": match_threshold,
            "jw_weight": jw_weight,
        },
        "total_pairs": len(candidate_pairs),
        "matched_pairs": len(match_edges),
    }
    return clusters, stats
