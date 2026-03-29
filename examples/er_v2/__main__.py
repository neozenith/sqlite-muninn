"""Entity Resolution Benchmark v2 — CLI entry point.

Usage:
  uv run -m examples.er_v2 manifest [--missing] [--commands] [--limit N]
  uv run -m examples.er_v2 run --dataset abt-buy --pipeline string-only [--limit 100]
  uv run -m examples.er_v2 run --dataset abt-buy --pipeline llm-cluster --model Qwen3.5-4B [--limit 100]
  uv run -m examples.er_v2 analyse
"""

import argparse
import json
import logging
import time
from pathlib import Path

from .datasets import DATASETS, load_dataset
from .metrics import bcubed_f1, pairwise_f1
from .models import CHAT_MODELS, DEFAULT_EMBED_MODEL, EMBED_MODELS, create_db, register_chat_model
from .registry import DEFAULTS, LIMITS, PIPELINES, RESULTS_DIR, permutation_id, permutation_manifest, print_manifest

log = logging.getLogger(__name__)


# ── Result I/O ────────────────────────────────────────────────────


def _save_result(
    perm_id: str,
    dataset: str,
    pipeline: str,
    model: str,
    limit: int | None,
    n_entities: int,
    bc: dict[str, float],
    pw: dict[str, float],
    elapsed: float,
    pipeline_stats: dict,
) -> Path:
    """Save benchmark result as JSON. Returns file path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "permutation_id": perm_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataset,
        "pipeline": pipeline,
        "model": model,
        "limit": limit,
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
    """Load all JSON results from RESULTS_DIR."""
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
    pipeline = args.pipeline
    model_name = args.model or "-"
    limit = args.limit
    embed_model_name = args.embed_model

    # Tuning params (defaults from registry.DEFAULTS)
    k = args.k
    dist = args.dist
    llm_low = args.llm_low
    llm_high = args.llm_high

    if ds_slug not in DATASETS:
        raise SystemExit(f"Unknown dataset: {ds_slug}. Available: {', '.join(DATASETS)}")
    if pipeline not in PIPELINES:
        raise SystemExit(f"Unknown pipeline: {pipeline}. Available: {', '.join(PIPELINES)}")
    if pipeline == "llm-cluster" and model_name == "-":
        raise SystemExit("--model is required for llm-cluster pipeline")
    if pipeline == "llm-cluster" and model_name not in CHAT_MODELS:
        raise SystemExit(f"Unknown model: {model_name}. Available: {', '.join(CHAT_MODELS)}")

    perm_id = permutation_id(
        ds_slug,
        pipeline,
        model_name,
        limit,
        embed_model=embed_model_name,
        k=k,
        dist_threshold=dist,
        llm_low=llm_low,
        llm_high=llm_high,
    )
    out_path = RESULTS_DIR / f"{perm_id}.json"

    if out_path.exists() and not args.force:
        raise SystemExit(f"Result already exists: {out_path.name}. Use --force to overwrite.")

    log.info("Running: %s", perm_id)

    # Load dataset
    cfg = DATASETS[ds_slug]
    entities, gold = load_dataset(cfg, limit)

    # Create DB
    conn = create_db(embed_model_name=embed_model_name)

    # Run pipeline
    predicted: dict[str, int] = {}
    stats: dict = {}
    t0 = time.perf_counter()

    if pipeline == "string-only":
        from .string_only import run as run_string_only

        predicted, stats = run_string_only(
            conn, entities, k=k, dist_threshold=dist, match_threshold=llm_high, embed_model_name=embed_model_name
        )
    else:  # llm-cluster
        from .llm_cluster import run as run_llm_cluster

        chat_model = CHAT_MODELS[model_name]
        register_chat_model(conn, chat_model)
        predicted, stats = run_llm_cluster(
            conn,
            entities,
            model_name=model_name,
            k=k,
            dist_threshold=dist,
            llm_low=llm_low,
            llm_high=llm_high,
            embed_model_name=embed_model_name,
        )

    elapsed = time.perf_counter() - t0

    # Evaluate
    bc = bcubed_f1(predicted, gold)
    pw = pairwise_f1(predicted, gold)

    # Report
    llm_calls = stats.get("llm_calls", 0)
    params = stats.get("params", {})
    print(f"\n  ── {perm_id} ──")
    print(f"  Dataset:    {cfg.display_name} ({len(entities)} entities)")
    print(f"  Pipeline:   {pipeline}" + (f" ({model_name})" if model_name != "-" else ""))
    print(f"  Params:     {params}")
    print(f"  B-Cubed:    F1={bc['f1']:.4f}  P={bc['precision']:.4f}  R={bc['recall']:.4f}")
    print(f"  Pairwise:   F1={pw['f1']:.4f}  P={pw['precision']:.4f}  R={pw['recall']:.4f}")
    print(f"  Pairs:      {stats.get('total_pairs', 0)} total", end="")
    if "borderline_pairs" in stats:
        print(
            f", {stats['auto_accepted']} auto-accept"
            f", {stats['borderline_pairs']} borderline"
            f", {stats['auto_rejected']} auto-reject"
        )
        print(
            f"  Components: {stats['n_components']}"
            f" (avg {stats['avg_component_size']:.1f}, var {stats['component_size_variance']:.1f})"
        )
    else:
        print(f", {stats.get('matched_pairs', 0)} matched")
    print(f"  LLM calls:  {llm_calls}")
    print(f"  Wall clock: {elapsed:.2f}s")

    # Save
    _save_result(perm_id, ds_slug, pipeline, model_name, limit, len(entities), bc, pw, elapsed, stats)
    conn.close()


def cmd_analyse(_args: argparse.Namespace) -> None:
    """Print comparison table from all accumulated results."""
    results = _load_results()
    if not results:
        log.warning("No results found in %s", RESULTS_DIR)
        return

    rows = sorted(results, key=lambda r: (r.get("limit") or 99999, r["dataset"], r["pipeline"], r["model"]))

    # Baselines: string-only with default params B³ F1 per (dataset, limit)
    baselines: dict[tuple[str, int | None], float] = {}
    for r in rows:
        if r["pipeline"] == "string-only" and not _has_custom_params(r):
            baselines[(r["dataset"], r.get("limit"))] = r["bcubed_f1"]

    print(f"\n  {'=' * 130}")
    print(
        f"  {'Dataset':<14} {'Pipeline':<14} {'Model':<14} {'Limit':>6} {'N':>6} "
        f"{'B³ F1':>8} {'B³ P':>8} {'B³ R':>8} {'PW F1':>8} {'LLM#':>5} {'Time':>8}  {'Tuning'}"
    )
    print(f"  {'-' * 130}")

    for r in rows:
        ds = r["dataset"]
        pipeline = r["pipeline"]
        model = r["model"]
        limit = r.get("limit")
        limit_str = str(limit) if limit else "full"
        baseline = baselines.get((ds, limit), 0.0)
        delta = (
            f"({r['bcubed_f1'] - baseline:+.3f})"
            if not (pipeline == "string-only" and not _has_custom_params(r))
            else ""
        )
        tuning = _tuning_summary(r)
        llm_calls = r.get("llm_calls", 0)
        print(
            f"  {ds:<14} {pipeline:<14} {model:<14} {limit_str:>6} {r['n_entities']:>6} "
            f"{r['bcubed_f1']:>8.4f} {r['bcubed_precision']:>8.4f} {r['bcubed_recall']:>8.4f} "
            f"{r['pairwise_f1']:>8.4f} {llm_calls:>5} {r['elapsed_s']:>7.1f}s {delta}  {tuning}"
        )

    print(f"  {'=' * 130}")

    # Summary stats
    n_datasets = len({r["dataset"] for r in rows})
    n_done = len(rows)
    entries = permutation_manifest()
    n_total = len(entries)
    print(f"\n  {n_done}/{n_total}+ results across {n_datasets} dataset(s) from {RESULTS_DIR}/")


def _has_custom_params(r: dict) -> bool:
    """Check if a result has non-default tuning params."""
    params = r.get("params", {})
    if not params:
        return False
    for key, default in DEFAULTS.items():
        # match_threshold maps to llm_high for string-only
        param_key = "match_threshold" if key == "llm_high" and "match_threshold" in params else key
        if param_key in params and params[param_key] != default:
            return True
    return False


def _tuning_summary(r: dict) -> str:
    """Return a compact string of non-default tuning params, or empty."""
    params = r.get("params", {})
    if not params:
        return ""
    parts: list[str] = []
    if params.get("k", DEFAULTS["k"]) != DEFAULTS["k"]:
        parts.append(f"k={params['k']}")
    dt = params.get("dist_threshold", DEFAULTS["dist_threshold"])
    if dt != DEFAULTS["dist_threshold"]:
        parts.append(f"dist={dt}")
    mt = params.get("match_threshold")
    if mt is not None and mt != DEFAULTS["llm_high"]:
        parts.append(f"mt={mt}")
    lo = params.get("llm_low")
    if lo is not None and lo != DEFAULTS["llm_low"]:
        parts.append(f"lo={lo}")
    hi = params.get("llm_high")
    if hi is not None and hi != DEFAULTS["llm_high"]:
        parts.append(f"hi={hi}")
    return " ".join(parts)


# ── CLI Parser ────────────────────────────────────────────────────


def cmd_errors(args: argparse.Namespace) -> None:
    """Analyse FP/FN failure modes for a dataset."""
    from .analyse_errors import analyse

    analyse(args.dataset, args.limit)


def _help(p: argparse.ArgumentParser):
    def _print_help(_: argparse.Namespace) -> None:
        p.print_help()

    return _print_help


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="examples.er_v2",
        description="Entity Resolution Benchmark v2 — Manifest-driven permutation runner",
    )
    parser.set_defaults(func=_help(parser))
    sub = parser.add_subparsers(dest="command", required=False)

    # manifest
    p = sub.add_parser("manifest", help="Show permutation status")
    p.add_argument("--missing", action="store_true", help="Show only incomplete permutations")
    p.add_argument("--done", action="store_true", help="Show only completed permutations")
    p.add_argument("--sort", choices=["size", "name"], default="size", help="Sort order")
    p.add_argument("--limit", type=int, help="Limit to first N entries")
    p.add_argument("--commands", action="store_true", help="Print runnable self-commands")
    p.add_argument("--force", action="store_true", help="Append --force to generated commands")
    p.set_defaults(func=cmd_manifest)

    # run
    p = sub.add_parser("run", help="Run a single benchmark permutation")
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()), help="Dataset to benchmark")
    p.add_argument("--pipeline", required=True, choices=PIPELINES, help="Pipeline implementation")
    p.add_argument("--model", choices=list(CHAT_MODELS.keys()), help="Chat model (required for llm-cluster)")
    p.add_argument("--limit", type=int, choices=[x for x in LIMITS if x], help="Max entities (omit for full)")
    p.add_argument(
        "--embed-model",
        choices=list(EMBED_MODELS.keys()),
        default=DEFAULT_EMBED_MODEL,
        help=f"Embedding model (default: {DEFAULT_EMBED_MODEL})",
    )
    p.add_argument("--k", type=int, default=DEFAULTS["k"], help=f"KNN neighbors (default: {DEFAULTS['k']})")
    p.add_argument(
        "--dist",
        type=float,
        default=DEFAULTS["dist_threshold"],
        help=f"Max cosine distance (default: {DEFAULTS['dist_threshold']})",
    )
    p.add_argument(
        "--llm-low",
        type=float,
        default=DEFAULTS["llm_low"],
        help=f"LLM borderline floor (default: {DEFAULTS['llm_low']})",
    )
    p.add_argument(
        "--llm-high",
        type=float,
        default=DEFAULTS["llm_high"],
        help=f"LLM auto-accept ceiling (default: {DEFAULTS['llm_high']})",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing result")
    p.set_defaults(func=cmd_run)

    # analyse
    p = sub.add_parser("analyse", help="Print comparison table from accumulated results")
    p.set_defaults(func=cmd_analyse)

    # errors
    p = sub.add_parser("errors", help="Analyse FP/FN failure modes for a dataset")
    p.add_argument("--dataset", required=True, choices=list(DATASETS.keys()), help="Dataset to analyse")
    p.add_argument("--limit", type=int, default=None, help="Max entities (omit for full)")
    p.set_defaults(func=cmd_errors)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
