"""Shared evaluation metrics for KG benchmark treatments.

Provides entity-level micro F1, triple F1, and B-Cubed F1 for use by
kg_extract (NER), kg_re (relation extraction), and kg_resolve (entity resolution).
"""

import logging
from collections import defaultdict

log = logging.getLogger(__name__)


def entity_micro_f1(
    predicted: list[tuple[int, int, str]],
    gold: list[tuple[int, int, str]],
) -> dict[str, float]:
    """Compute entity-level micro F1 via span+type exact match.

    Args:
        predicted: List of (start, end, label) tuples from the model.
        gold: List of (start, end, label) gold-standard tuples.

    Returns:
        Dict with keys: precision, recall, f1.
    """
    pred_set = set(predicted)
    gold_set = set(gold)

    if not pred_set and not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def triple_f1(
    predicted: list[tuple[str, str, str]],
    gold: list[tuple[str, str, str]],
) -> dict[str, float]:
    """Compute triple-level F1 via strict (subject, predicate, object) match.

    Args:
        predicted: List of (subject, predicate, object) tuples from the model.
        gold: List of (subject, predicate, object) gold-standard tuples.

    Returns:
        Dict with keys: precision, recall, f1.
    """
    pred_set = set(predicted)
    gold_set = set(gold)

    if not pred_set and not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def bcubed_f1(
    predicted_clusters: dict[str, int],
    gold_clusters: dict[str, int],
) -> dict[str, float]:
    """Compute B-Cubed F1 for clustering evaluation.

    Each dict maps element ID to cluster ID. Only elements present in both
    dicts are evaluated (intersection).

    Args:
        predicted_clusters: {element_id: predicted_cluster_id}
        gold_clusters: {element_id: gold_cluster_id}

    Returns:
        Dict with keys: precision, recall, f1.
    """
    common = set(predicted_clusters.keys()) & set(gold_clusters.keys())
    if not common:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Build reverse index: cluster_id -> set of elements
    pred_by_cluster: dict[int, set[str]] = defaultdict(set)
    gold_by_cluster: dict[int, set[str]] = defaultdict(set)
    for elem in common:
        pred_by_cluster[predicted_clusters[elem]].add(elem)
        gold_by_cluster[gold_clusters[elem]].add(elem)

    # Per-element precision and recall
    total_precision = 0.0
    total_recall = 0.0

    for elem in common:
        pred_cluster_members = pred_by_cluster[predicted_clusters[elem]] & common
        gold_cluster_members = gold_by_cluster[gold_clusters[elem]] & common

        # Elements sharing both predicted and gold cluster with this element
        shared = pred_cluster_members & gold_cluster_members

        total_precision += len(shared) / len(pred_cluster_members) if pred_cluster_members else 0.0
        total_recall += len(shared) / len(gold_cluster_members) if gold_cluster_members else 0.0

    n = len(common)
    precision = total_precision / n
    recall = total_recall / n
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
