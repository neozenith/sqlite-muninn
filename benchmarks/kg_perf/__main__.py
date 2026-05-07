"""kg_perf CLI — manifest / run / compare.

Usage:
    uv run -m benchmarks.kg_perf manifest
    uv run -m benchmarks.kg_perf manifest --missing --commands
    uv run -m benchmarks.kg_perf run --id baseline__sessions_demo__p-all_t-all__node_betweenness_k3_d2_m2
    uv run -m benchmarks.kg_perf run-all --strategy baseline
    uv run -m benchmarks.kg_perf compare
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import cast

from benchmarks.kg_perf.bench import append_record, fidelity_against, time_one
from benchmarks.kg_perf.constants import RESULTS_DIR
from benchmarks.kg_perf.manifest import all_permutations, find_permutation
from benchmarks.kg_perf.strategies import STRATEGIES
from benchmarks.kg_perf.workload import Workload

log = logging.getLogger(__name__)


def cmd_manifest(args: argparse.Namespace) -> None:
    perms = all_permutations()
    if args.missing:
        perms = [p for p in perms if not p["done"]]
    if args.done:
        perms = [p for p in perms if p["done"]]
    if args.strategy:
        perms = [p for p in perms if p["strategy"] == args.strategy]
    perms.sort(key=lambda p: cast(tuple, p["sort_key"]))
    if args.limit:
        perms = perms[: args.limit]

    if args.commands:
        for p in perms:
            print(f"uv run -m benchmarks.kg_perf run --id {p['permutation_id']}")
        return

    done = sum(1 for p in perms if p["done"])
    print(f"=== Manifest ({done}/{len(perms)} done) ===\n")
    for p in perms:
        flag = "[DONE]" if p["done"] else "[MISS]"
        print(f"  {flag} {p['label']}")


def cmd_run(args: argparse.Namespace) -> None:
    perm = find_permutation(args.id)
    if perm is None:
        print(f"unknown permutation_id: {args.id}", file=sys.stderr)
        sys.exit(2)
    workload = cast(Workload, perm["workload"])
    strategy = STRATEGIES[cast(str, perm["strategy"])]()
    record = time_one(strategy, workload)
    out = append_record(record)
    print(
        f"[{record['strategy']}] {workload.slug}  "
        f"p50={record['wall_ms']['p50']:.2f}ms  p95={record['wall_ms']['p95']:.2f}ms  "
        f"-> {out}"
    )


def cmd_run_all(args: argparse.Namespace) -> None:
    perms = all_permutations()
    if args.strategy:
        perms = [p for p in perms if p["strategy"] == args.strategy]
    if args.missing:
        perms = [p for p in perms if not p["done"]]
    perms.sort(key=lambda p: cast(tuple, p["sort_key"]))
    if args.limit:
        perms = perms[: args.limit]
    for p in perms:
        workload = cast(Workload, p["workload"])
        strategy = STRATEGIES[cast(str, p["strategy"])]()
        record = time_one(strategy, workload)
        append_record(record)
        sig = record["signature"]
        print(
            f"[{record['strategy']:18s}] {workload.slug:60s}  "
            f"p50={record['wall_ms']['p50']:7.2f}ms  "
            f"nodes={sig['node_count']:4d}  edges={sig['edge_count']:4d}  "
            f"sig={sig['node_hash']}/{sig['edge_hash']}"
        )


def cmd_compare(_: argparse.Namespace) -> None:
    """Most-recent run per (strategy, workload); show speedup vs baseline + Jaccard."""
    latest: dict[tuple[str, str], dict] = {}
    for path in RESULTS_DIR.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            wl_key = rec["permutation_id"].split("__", 1)[1]
            key = (rec["strategy"], wl_key)
            prev = latest.get(key)
            if prev is None or rec["timestamp"] > prev["timestamp"]:
                latest[key] = rec

    by_workload: dict[str, list[dict]] = defaultdict(list)
    for (_strategy, wl_key), rec in latest.items():
        by_workload[wl_key].append(rec)

    headers = f"{'workload':62s} {'strategy':14s} {'p50_ms':>9s} {'speedup':>8s}  fidelity"
    print(headers)
    print("-" * len(headers))
    for wl_key in sorted(by_workload):
        records = by_workload[wl_key]
        # Baseline first, then others sorted by p50.
        records.sort(key=lambda r: (0 if r["strategy"] == "baseline" else 1, r["wall_ms"]["p50"]))
        baseline = next((r for r in records if r["strategy"] == "baseline"), None)
        for r in records:
            p50 = r["wall_ms"]["p50"]
            if baseline:
                speedup = baseline["wall_ms"]["p50"] / p50 if p50 > 0 else float("inf")
                fid = fidelity_against(r, baseline)
                fid_str = " ".join(f"{k}={v:.2f}" for k, v in fid.items())
            else:
                speedup = 1.0
                fid_str = "(no baseline)"
            print(f"{wl_key:62s} {r['strategy']:14s} {p50:9.2f} {speedup:7.2f}x  {fid_str}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="benchmarks.kg_perf")
    p.set_defaults(func=_help(p))
    sub = p.add_subparsers(dest="cmd", required=False)

    m = sub.add_parser("manifest", help="List permutations and their done/missing status")
    m.add_argument("--missing", action="store_true")
    m.add_argument("--done", action="store_true")
    m.add_argument("--commands", action="store_true", help="Emit runnable commands")
    m.add_argument("--strategy", default=None)
    m.add_argument("--limit", type=int, default=None)
    m.set_defaults(func=cmd_manifest)

    r = sub.add_parser("run", help="Run a single permutation by id")
    r.add_argument("--id", required=True)
    r.set_defaults(func=cmd_run)

    ra = sub.add_parser("run-all", help="Run every permutation (or filter via flags)")
    ra.add_argument("--strategy", default=None)
    ra.add_argument("--missing", action="store_true")
    ra.add_argument("--limit", type=int, default=None)
    ra.set_defaults(func=cmd_run_all)

    c = sub.add_parser("compare", help="Tabulate strategy results, speedup vs baseline, fidelity")
    c.set_defaults(func=cmd_compare)
    return p


def _help(parser: argparse.ArgumentParser):
    def _print_help(_: argparse.Namespace) -> None:
        parser.print_help()

    return _print_help


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
