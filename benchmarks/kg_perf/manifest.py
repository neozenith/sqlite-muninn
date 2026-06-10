"""Permutation manifest: enumerate (strategy × filter × query) and report status."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.kg_perf.constants import DEFAULT_DB, RESULTS_DIR
from benchmarks.kg_perf.strategies import STRATEGIES
from benchmarks.kg_perf.workload import QUERY_SHAPES, Filter, QuerySpec, Workload, filter_widths


def all_permutations(db_path: Path = DEFAULT_DB) -> list[dict[str, object]]:
    completed = _completed_ids()
    perms: list[dict[str, object]] = []
    for strategy_name in STRATEGIES:
        for flt in filter_widths():
            for qry in QUERY_SHAPES:
                wl = Workload(db_path=db_path, filter=flt, query=qry)
                pid = f"{strategy_name}__{wl.slug}"
                perms.append(
                    {
                        "permutation_id": pid,
                        "strategy": strategy_name,
                        "workload": wl,
                        "label": f"{strategy_name:24s} | {wl.slug}",
                        "done": pid in completed,
                        "sort_key": (_filter_cost(flt), _query_cost(qry), strategy_name),
                    }
                )
    return perms


def _completed_ids() -> set[str]:
    if not RESULTS_DIR.exists():
        return set()
    seen: set[str] = set()
    for path in RESULTS_DIR.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                seen.add(json.loads(line)["permutation_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def _filter_cost(flt: Filter) -> int:
    """Smaller filter set = cheaper. Used for sort_key ordering."""
    rank = 0 if flt.project_id else 2
    rank += 0 if flt.days else 1
    return rank


def _query_cost(q: QuerySpec) -> int:
    if q.metric == "degree":
        return 0
    if q.metric == "node_betweenness":
        return 2
    return 3


def find_permutation(perm_id: str) -> dict[str, object] | None:
    for p in all_permutations():
        if p["permutation_id"] == perm_id:
            return p
    return None
