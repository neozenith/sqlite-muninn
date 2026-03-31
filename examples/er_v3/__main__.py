"""ER Benchmark v3 — Uses muninn_extract_er() C function.

Manifest-driven, same parameter space as er_v2 for direct comparison.

Usage:
  uv run -m examples.er_v3 manifest [--missing] [--commands]
  uv run -m examples.er_v3 run --dataset amazon-google --dist 0.10 \\
    --jw-weight 0.3 --llm-high 0.90 --borderline-delta 0.0
  uv run -m examples.er_v3 analyse
"""

import argparse
import json
import logging
import time
from pathlib import Path

from .datasets import DATASETS
from .pipeline import run_er
from .registry import (
    RESULTS_DIR,
    permutation_id,
    permutation_manifest,
    print_manifest,
)

log = logging.getLogger(__name__)


def _save_result(perm_id: str, result: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result["permutation_id"] = perm_id
    result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = RESULTS_DIR / f"{perm_id}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log.info("Saved: %s", path.name)
    return path


def _load_results() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    results = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            results.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return results


def cmd_manifest(args: argparse.Namespace) -> None:
    entries = permutation_manifest()
    print_manifest(
        entries,
        missing=args.missing,
        done=args.done,
        sort=args.sort,
        limit=args.limit,
        commands=args.commands,
        force=args.force,
    )


def cmd_run(args: argparse.Namespace) -> None:
    ds_slug = args.dataset
    dist = args.dist
    jw_weight = args.jw_weight
    llm_high = args.llm_high
    borderline_delta = args.borderline_delta
    betweenness = args.betweenness_threshold

    # Derive match_threshold from the implicit formula
    match_threshold = 1.0 - dist + borderline_delta

    perm_id = permutation_id(
        ds_slug,
        dist_threshold=dist,
        jw_weight=jw_weight,
        llm_high=llm_high,
        borderline_delta=borderline_delta,
    )
    out_path = RESULTS_DIR / f"{perm_id}.json"

    if out_path.exists() and not args.force:
        raise SystemExit(f"Result exists: {out_path.name}. Use --force to overwrite.")

    log.info("Running: %s", perm_id)

    result = run_er(
        ds_slug,
        k=args.k,
        dist_threshold=dist,
        jw_weight=jw_weight,
        borderline_delta=borderline_delta,
        edge_betweenness_threshold=betweenness,
    )

    print(f"\n  ── {perm_id} ──")
    print(f"  Dataset:    {result['dataset']} ({result['n_entities']} entities)")
    print(f"  Params:     dist={dist} jw={jw_weight} hi={llm_high} mt={match_threshold:.2f} Δ={borderline_delta}")
    bc = result
    print(f"  B-Cubed:    F1={bc['bcubed_f1']:.4f}  P={bc['bcubed_precision']:.4f}  R={bc['bcubed_recall']:.4f}")
    print(f"  Pairwise:   F1={bc['pairwise_f1']:.4f}  P={bc['pairwise_precision']:.4f}  R={bc['pairwise_recall']:.4f}")
    print(f"  Wall clock: {result['elapsed_s']:.2f}s")

    _save_result(perm_id, result)


def cmd_analyse(_args: argparse.Namespace) -> None:
    results = _load_results()
    if not results:
        log.warning("No results in %s", RESULTS_DIR)
        return

    rows = sorted(
        results,
        key=lambda r: (
            r["dataset"],
            r.get("params", {}).get("dist_threshold", 0),
            r.get("params", {}).get("jw_weight", 0),
        ),
    )

    print(f"\n  {'=' * 100}")
    print(
        f"  {'Dataset':<14} {'dist':>5} {'jw':>5} {'mt':>5} {'B³F1':>7} {'B³P':>7} {'B³R':>7} {'PwF1':>7} {'Time':>7}"
    )
    print(f"  {'-' * 100}")

    for r in rows:
        p = r.get("params", {})
        dist = p.get("dist_threshold", "?")
        jw = p.get("jw_weight", "?")
        delta = p.get("borderline_delta", 0)
        mt = round(1.0 - (dist if isinstance(dist, float) else 0) + delta, 2)
        print(
            f"  {r['dataset']:<14} {dist:>5} {jw:>5} {mt:>5}"
            f" {r['bcubed_f1']:>7.4f} {r['bcubed_precision']:>7.4f} {r['bcubed_recall']:>7.4f}"
            f" {r['pairwise_f1']:>7.4f} {r['elapsed_s']:>6.1f}s"
        )

    print(f"  {'=' * 100}")
    print(f"\n  {len(rows)} results from {RESULTS_DIR}/")


def _help(p: argparse.ArgumentParser):
    def _print_help(_: argparse.Namespace) -> None:
        p.print_help()

    return _print_help


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="examples.er_v3",
        description="ER v3 — muninn_extract_er() benchmark",
    )
    parser.set_defaults(func=_help(parser))
    sub = parser.add_subparsers(dest="command", required=False)

    p = sub.add_parser("manifest", help="Show permutation status")
    p.add_argument("--missing", action="store_true")
    p.add_argument("--done", action="store_true")
    p.add_argument("--sort", choices=["size", "name"], default="size")
    p.add_argument("--limit", type=int)
    p.add_argument("--commands", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("run", help="Run a single permutation")
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--dist", type=float, required=True)
    p.add_argument("--jw-weight", type=float, required=True)
    p.add_argument("--llm-high", type=float, required=True)
    p.add_argument("--borderline-delta", type=float, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--betweenness-threshold", type=float, default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyse", help="Print comparison table")
    p.set_defaults(func=cmd_analyse)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
