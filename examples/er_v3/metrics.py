"""Entity resolution evaluation metrics: B-Cubed F1 and Pairwise F1."""

from collections import defaultdict


def bcubed_f1(predicted: dict[str, int], gold: dict[str, int]) -> dict[str, float]:
    """B-Cubed F1 for clustering evaluation (primary ER metric).

    Each dict maps element_id -> cluster_id. Only elements in both dicts are evaluated.
    """
    common = set(predicted.keys()) & set(gold.keys())
    if not common:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_by_cluster: dict[int, set[str]] = defaultdict(set)
    gold_by_cluster: dict[int, set[str]] = defaultdict(set)
    for elem in common:
        pred_by_cluster[predicted[elem]].add(elem)
        gold_by_cluster[gold[elem]].add(elem)

    total_p = 0.0
    total_r = 0.0
    for elem in common:
        pred_members = pred_by_cluster[predicted[elem]] & common
        gold_members = gold_by_cluster[gold[elem]] & common
        shared = pred_members & gold_members
        total_p += len(shared) / len(pred_members) if pred_members else 0.0
        total_r += len(shared) / len(gold_members) if gold_members else 0.0

    n = len(common)
    p = total_p / n
    r = total_r / n
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def pairwise_f1(predicted: dict[str, int], gold: dict[str, int]) -> dict[str, float]:
    """Pairwise F1 for clustering evaluation (secondary ER metric).

    Evaluates whether pairs of elements in the same predicted cluster
    are also in the same gold cluster.
    """
    common = sorted(set(predicted.keys()) & set(gold.keys()))
    if len(common) < 2:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_by_cluster: dict[int, set[str]] = defaultdict(set)
    gold_by_cluster: dict[int, set[str]] = defaultdict(set)
    for elem in common:
        pred_by_cluster[predicted[elem]].add(elem)
        gold_by_cluster[gold[elem]].add(elem)

    def _pairs(cluster_map: dict[int, set[str]]) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for members in cluster_map.values():
            members_list = sorted(members)
            for i in range(len(members_list)):
                for j in range(i + 1, len(members_list)):
                    pairs.add((members_list[i], members_list[j]))
        return pairs

    pred_pairs = _pairs(pred_by_cluster)
    gold_pairs = _pairs(gold_by_cluster)

    if not pred_pairs and not gold_pairs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    tp = len(pred_pairs & gold_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
