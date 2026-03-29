"""Analyse false positives and false negatives from the string-only pipeline.

Runs the pipeline, compares predicted clusters to ground truth, and
categorises FP/FN pairs with their match scores to reveal systematic failure modes.

Usage:
  uv run examples/er_v2/analyse_errors.py --dataset abt-buy --limit 500
  uv run examples/er_v2/analyse_errors.py --dataset amazon-google
"""

import argparse
import logging
from collections import defaultdict

from .blocking import embed_and_block
from .datasets import DATASETS, load_dataset
from .jaro_winkler import jaro_winkler
from .models import create_db

log = logging.getLogger(__name__)


def _cluster_pairs(clusters: dict[str, int]) -> set[tuple[str, str]]:
    """Build set of all (a, b) pairs within the same cluster (a < b)."""
    by_cluster: dict[int, list[str]] = defaultdict(list)
    for eid, cid in clusters.items():
        by_cluster[cid].append(eid)
    pairs: set[tuple[str, str]] = set()
    for members in by_cluster.values():
        members.sort()
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    return pairs


def analyse(dataset_slug: str, limit: int | None = None) -> None:
    cfg = DATASETS[dataset_slug]
    entities, gold = load_dataset(cfg, limit)
    entity_names = {e.id: e.name for e in entities}

    conn = create_db()

    # Run blocking to get candidate pairs and scores
    id_map, name_map, candidate_pairs = embed_and_block(conn, entities, k=10, dist_threshold=0.15)

    # Compute match scores for all candidate pairs
    pair_scores: dict[tuple[str, str], dict] = {}
    for (r1, r2), cosine_dist in candidate_pairs.items():
        eid1, eid2 = id_map[r1], id_map[r2]
        pair = (min(eid1, eid2), max(eid1, eid2))
        n1, n2 = name_map[r1], name_map[r2]
        cosine_sim = 1.0 - cosine_dist
        jw = jaro_winkler(n1.lower(), n2.lower())
        combined = 0.4 * jw + 0.6 * cosine_sim
        pair_scores[pair] = {
            "n1": n1,
            "n2": n2,
            "cosine_sim": round(cosine_sim, 4),
            "jaro_winkler": round(jw, 4),
            "combined": round(combined, 4),
        }

    # Run string-only pipeline on a fresh set of tables in the same connection
    from .models import cleanup_pipeline_tables
    from .string_only import run as run_string_only

    cleanup_pipeline_tables(conn)
    predicted, _stats = run_string_only(conn, entities, dist_threshold=0.15, match_threshold=0.9)

    # Build pair sets
    pred_pairs = _cluster_pairs(predicted)
    gold_pairs = _cluster_pairs(gold)

    fp = pred_pairs - gold_pairs
    fn = gold_pairs - pred_pairs
    tp = pred_pairs & gold_pairs

    print(f"\n  === Error Analysis: {cfg.display_name} ({len(entities)} entities) ===")
    print(f"  True Positives:   {len(tp)}")
    print(f"  False Positives:  {len(fp)}")
    print(f"  False Negatives:  {len(fn)}")
    print(f"  Precision:        {len(tp) / (len(tp) + len(fp)):.4f}" if (len(tp) + len(fp)) > 0 else "  Precision: N/A")
    print(f"  Recall:           {len(tp) / (len(tp) + len(fn)):.4f}" if (len(tp) + len(fn)) > 0 else "  Recall: N/A")

    # Categorise FPs
    print(f"\n  ── False Positives ({len(fp)}) ──")
    print("  These pairs were merged but should NOT have been.\n")

    fp_scored = []
    for a, b in sorted(fp):
        score_info = pair_scores.get((a, b), {})
        fp_scored.append((a, b, score_info))

    # Bin by score range
    fp_bins: dict[str, list] = {
        "0.90-1.00 (very high)": [],
        "0.80-0.90 (high)": [],
        "0.70-0.80 (medium)": [],
        "< 0.70 (low/transitive)": [],
    }
    for a, b, info in fp_scored:
        score = info.get("combined", 0)
        if score >= 0.90:
            fp_bins["0.90-1.00 (very high)"].append((a, b, info))
        elif score >= 0.80:
            fp_bins["0.80-0.90 (high)"].append((a, b, info))
        elif score >= 0.70:
            fp_bins["0.70-0.80 (medium)"].append((a, b, info))
        else:
            fp_bins["< 0.70 (low/transitive)"].append((a, b, info))

    for bin_name, items in fp_bins.items():
        if not items:
            continue
        print(f"  Score {bin_name}: {len(items)} FPs")
        for a, b, info in items[:5]:
            n1 = entity_names.get(a, a)[:60]
            n2 = entity_names.get(b, b)[:60]
            cs = info.get("cosine_sim", "?")
            jw_s = info.get("jaro_winkler", "?")
            comb = info.get("combined", "?")
            print(f"    {n1}")
            print(f"    {n2}")
            print(f"    cos={cs} jw={jw_s} combined={comb}")
            print()
        if len(items) > 5:
            print(f"    ... and {len(items) - 5} more\n")

    # Categorise FNs
    print(f"\n  ── False Negatives ({len(fn)}) ──")
    print("  These pairs should have been merged but were NOT.\n")

    fn_in_candidates = []
    fn_not_in_candidates = []
    for a, b in sorted(fn):
        if (a, b) in pair_scores:
            fn_in_candidates.append((a, b, pair_scores[(a, b)]))
        else:
            fn_in_candidates_name = entity_names.get(a, a), entity_names.get(b, b)
            fn_not_in_candidates.append((a, b, fn_in_candidates_name))

    print(f"  In candidate set (blocked but rejected by threshold): {len(fn_in_candidates)}")
    for a, b, info in fn_in_candidates[:10]:
        n1 = entity_names.get(a, a)[:60]
        n2 = entity_names.get(b, b)[:60]
        cs = info.get("cosine_sim", "?")
        jw_s = info.get("jaro_winkler", "?")
        comb = info.get("combined", "?")
        print(f"    {n1}")
        print(f"    {n2}")
        print(f"    cos={cs} jw={jw_s} combined={comb}")
        print()
    if len(fn_in_candidates) > 10:
        print(f"    ... and {len(fn_in_candidates) - 10} more\n")

    print(f"\n  NOT in candidate set (missed by HNSW blocking): {len(fn_not_in_candidates)}")
    for _a, _b, (n1, n2) in fn_not_in_candidates[:10]:
        print(f"    {n1[:60]}")
        print(f"    {n2[:60]}")
        print()
    if len(fn_not_in_candidates) > 10:
        print(f"    ... and {len(fn_not_in_candidates) - 10} more\n")

    # Summary
    total_fn = len(fn)
    blocked_fn = len(fn_in_candidates)
    missed_fn = len(fn_not_in_candidates)
    print("\n  ── Summary ──")
    print(f"  FP: {len(fp)} pairs incorrectly merged")
    print(f"  FN: {total_fn} pairs missed")
    print(f"    - {blocked_fn} were candidates but below threshold (fixable with lower mt)")
    print(f"    - {missed_fn} never entered candidate set (need better embeddings or higher k/dist)")
    if total_fn > 0:
        print(f"    - Blocking recall: {blocked_fn / total_fn:.1%} of FNs were at least candidates")

    conn.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Analyse ER false positives and false negatives")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    analyse(args.dataset, args.limit)


if __name__ == "__main__":
    main()
