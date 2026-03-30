"""Entity Resolution Benchmark v2 — CLI entry point.

Unified pipeline: all runs go through llm_cluster with configurable thresholds.
Setting borderline_delta=0 disables LLM (equivalent to string-only).

Usage:
  uv run -m examples.er_v2 manifest [--missing] [--commands]
  uv run -m examples.er_v2 sweep --layer 1 [--commands]
  uv run -m examples.er_v2 run --dataset abt-buy --dist 0.10 --jw-weight 0.5 --llm-high 0.95 --borderline-delta 0.05
  uv run -m examples.er_v2 analyse
"""

import argparse
import json
import logging
import time
from pathlib import Path

from .datasets import DATASETS, load_dataset
from .metrics import bcubed_f1, pairwise_f1
from .models import CHAT_MODELS, register_chat_model
from .registry import (
    CHAT_MODEL,
    EMBED_MODEL,
    RESULTS_DIR,
    permutation_id,
    permutation_manifest,
    print_manifest,
)

log = logging.getLogger(__name__)


# ── Result I/O ────────────────────────────────────────────────────


def _save_result(
    perm_id: str,
    dataset: str,
    n_entities: int,
    bc: dict[str, float],
    pw: dict[str, float],
    elapsed: float,
    pipeline_stats: dict,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "permutation_id": perm_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataset,
        "embed_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
        "n_entities": n_entities,
        "bcubed_f1": bc["f1"],
        "bcubed_precision": bc["precision"],
        "bcubed_recall": bc["recall"],
        "pairwise_f1": pw["f1"],
        "pairwise_precision": pw["precision"],
        "pairwise_recall": pw["recall"],
        "elapsed_s": round(elapsed, 3),
        **pipeline_stats,
    }
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
            log.warning("Skipping malformed result: %s", p.name)
    return results


# ── Commands ──────────────────────────────────────────────────────


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
    """Run a single benchmark permutation."""
    ds_slug = args.dataset
    dist = args.dist
    jw_weight = args.jw_weight
    llm_high = args.llm_high
    borderline_delta = args.borderline_delta
    llm_low = round(llm_high - borderline_delta, 4)

    if ds_slug not in DATASETS:
        raise SystemExit(f"Unknown dataset: {ds_slug}. Available: {', '.join(DATASETS)}")

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

    # Load ground truth (dataset download + parsing)
    cfg = DATASETS[ds_slug]
    _entities, gold = load_dataset(cfg, limit=None)

    # Open prebuilt embedded DB (preps on first use ~30-50s, then instant)
    from .blocking import open_prep_db

    conn = open_prep_db(ds_slug, EMBED_MODEL)

    # Register chat model (needed if borderline_delta > 0)
    need_llm = borderline_delta > 0
    if need_llm:
        chat_model = CHAT_MODELS[CHAT_MODEL]
        register_chat_model(conn, chat_model)

    # Run pipeline (blocking + matching + clustering — no embedding)
    from .llm_cluster import run as run_pipeline

    type_guard = args.type_guard
    betweenness = args.betweenness_threshold

    t0 = time.perf_counter()
    predicted, stats = run_pipeline(
        conn,
        model_name=CHAT_MODEL,
        dist_threshold=dist,
        jw_weight=jw_weight,
        llm_low=llm_low,
        llm_high=llm_high,
        type_guard=type_guard,
        betweenness_threshold=betweenness,
    )
    elapsed = time.perf_counter() - t0

    # Evaluate
    bc = bcubed_f1(predicted, gold)
    pw = pairwise_f1(predicted, gold)

    # Report
    n_entities = len(predicted)
    llm_calls = stats.get("llm_calls", 0)
    bridges = stats.get("bridges_removed", 0)
    print(f"\n  ── {perm_id} ──")
    print(f"  Dataset:    {cfg.display_name} ({n_entities} entities)")
    print(f"  Params:     dist={dist} jw={jw_weight} hi={llm_high} lo={llm_low} Δ={borderline_delta}")
    if type_guard:
        print(f"  Type guard: ON ({stats.get('type_filtered', 0)} pairs filtered)")
    if betweenness is not None:
        print(f"  Betweenness: threshold={betweenness} ({bridges} bridges removed)")
    print(f"  B-Cubed:    F1={bc['f1']:.4f}  P={bc['precision']:.4f}  R={bc['recall']:.4f}")
    print(f"  Pairwise:   F1={pw['f1']:.4f}  P={pw['precision']:.4f}  R={pw['recall']:.4f}")
    print(
        f"  Pairs:      {stats.get('total_pairs', 0)} total"
        f", {stats.get('type_filtered', 0)} type-filtered"
        f", {stats.get('auto_accepted', 0)} accept"
        f", {stats.get('borderline_pairs', 0)} borderline"
        f", {stats.get('auto_rejected', 0)} reject"
    )
    print(f"  LLM calls:  {llm_calls}")
    print(f"  Wall clock: {elapsed:.2f}s (embedding precomputed)")

    # Save
    _save_result(perm_id, ds_slug, n_entities, bc, pw, elapsed, stats)
    conn.close()


def cmd_analyse(_args: argparse.Namespace) -> None:
    """Print comparison table from all accumulated results."""
    results = _load_results()
    if not results:
        log.warning("No results found in %s", RESULTS_DIR)
        return

    rows = sorted(
        results,
        key=lambda r: (
            r["dataset"],
            r.get("params", {}).get("dist_threshold", 0),
            r.get("params", {}).get("jw_weight", 0),
            r.get("params", {}).get("llm_high", 0),
            r.get("params", {}).get("llm_low", 0),
        ),
    )

    print(f"\n  {'=' * 130}")
    print(
        f"  {'Dataset':<14} {'dist':>5} {'jw':>5} {'hi':>5} {'lo':>5}"
        f" {'B³F1':>7} {'B³P':>7} {'B³R':>7} {'PwF1':>7}"
        f" {'pairs':>6} {'accept':>6} {'border':>6} {'reject':>6} {'LLM#':>5} {'Time':>7}"
    )
    print(f"  {'-' * 130}")

    for r in rows:
        p = r.get("params", {})
        ds = r["dataset"]
        dist = p.get("dist_threshold", "?")
        jw = p.get("jw_weight", "?")
        hi = p.get("llm_high", "?")
        lo = p.get("llm_low", "?")
        llm_calls = r.get("llm_calls", 0)
        print(
            f"  {ds:<14} {dist:>5} {jw:>5} {hi:>5} {lo:>5}"
            f" {r['bcubed_f1']:>7.4f} {r['bcubed_precision']:>7.4f} {r['bcubed_recall']:>7.4f}"
            f" {r['pairwise_f1']:>7.4f}"
            f" {r.get('total_pairs', 0):>6} {r.get('auto_accepted', 0):>6}"
            f" {r.get('borderline_pairs', 0):>6} {r.get('auto_rejected', 0):>6}"
            f" {llm_calls:>5} {r['elapsed_s']:>6.1f}s"
        )

    print(f"  {'=' * 130}")
    print(f"\n  {len(rows)} results from {RESULTS_DIR}/")


def cmd_errors(args: argparse.Namespace) -> None:
    """Analyse FP/FN failure modes for a dataset."""
    from .analyse_errors import analyse

    analyse(args.dataset, args.limit)


# ── CLI Parser ────────────────────────────────────────────────────


def _help(p: argparse.ArgumentParser):
    def _print_help(_: argparse.Namespace) -> None:
        p.print_help()

    return _print_help


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="examples.er_v2",
        description="ER Benchmark v2 — Unified pipeline parameter exploration",
    )
    parser.set_defaults(func=_help(parser))
    sub = parser.add_subparsers(dest="command", required=False)

    # manifest (Layer 1 only — dist sweep)
    p = sub.add_parser("manifest", help="Show Layer 1 (dist sweep) permutations")
    p.add_argument("--missing", action="store_true")
    p.add_argument("--done", action="store_true")
    p.add_argument("--sort", choices=["size", "name"], default="size")
    p.add_argument("--limit", type=int)
    p.add_argument("--commands", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_manifest)

    # run (single permutation — all 4 params explicit)
    p = sub.add_parser("run", help="Run a single benchmark permutation")
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--dist", type=float, required=True, help="Distance threshold")
    p.add_argument("--jw-weight", type=float, required=True, help="JW vs cosine weight (1.0=pure JW, 0.0=pure cosine)")
    p.add_argument("--llm-high", type=float, required=True, help="Auto-accept threshold (1.0=nothing accepted)")
    p.add_argument("--borderline-delta", type=float, required=True, help="LLM window width (0.0=no LLM)")
    p.add_argument("--type-guard", action="store_true", default=True, help="Enable type/source guard (default: on)")
    p.add_argument("--no-type-guard", action="store_false", dest="type_guard", help="Disable type/source guard")
    p.add_argument(
        "--betweenness-threshold", type=float, default=None, help="Bridge removal threshold (default: disabled)"
    )
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_run)

    # analyse
    p = sub.add_parser("analyse", help="Print comparison table from results")
    p.set_defaults(func=cmd_analyse)

    # errors
    p = sub.add_parser("errors", help="Analyse FP/FN failure modes")
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_errors)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
