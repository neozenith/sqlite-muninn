"""Permutation registry: datasets x pipelines x models x limits.

Generates the full cross-product of benchmark configurations with build status.
"""

from pathlib import Path

from .datasets import DATASETS
from .models import CHAT_MODELS, DEFAULT_EMBED_MODEL

RESULTS_DIR = Path(__file__).resolve().parent / "results"

PIPELINES = ["string-only", "llm-cluster"]
LIMITS: list[int | None] = [100, 500, 1000, None]  # None = full dataset


# Default tuning values — used to determine which params appear in the slug.
# dist_threshold=0.15 and llm_high=0.9 validated by grid search (2026-03-28):
#   Abt-Buy:       0.654 -> 0.799 (+22%)
#   Amazon-Google:  0.853 -> 0.973 (+14%)
#   DBLP-ACM:      0.937 -> 0.993 (+6%)
DEFAULTS = {"k": 10, "dist_threshold": 0.15, "llm_low": 0.3, "llm_high": 0.9}


def permutation_id(
    dataset: str,
    pipeline: str,
    model: str,
    limit: int | None,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    k: int = DEFAULTS["k"],
    dist_threshold: float = DEFAULTS["dist_threshold"],
    llm_low: float = DEFAULTS["llm_low"],
    llm_high: float = DEFAULTS["llm_high"],
) -> str:
    """Build a unique permutation slug.

    Only non-default tuning params are included in the slug, so the default
    manifest permutations keep their existing IDs. Manual runs with custom
    params get longer, distinguishable filenames.
    """
    limit_str = str(limit) if limit else "full"
    slug = f"{dataset}__{pipeline}__{model}__{limit_str}"

    # Append non-default tuning params as k=V segments
    tuning: list[str] = []
    if embed_model != DEFAULT_EMBED_MODEL:
        tuning.append(f"emb-{embed_model}")
    if k != DEFAULTS["k"]:
        tuning.append(f"k{k}")
    if dist_threshold != DEFAULTS["dist_threshold"]:
        tuning.append(f"dist{dist_threshold:.2f}")
    if llm_low != DEFAULTS["llm_low"]:
        tuning.append(f"lo{llm_low:.2f}")
    if llm_high != DEFAULTS["llm_high"]:
        tuning.append(f"hi{llm_high:.2f}")

    if tuning:
        slug += "__" + "_".join(tuning)
    return slug


def permutation_manifest(output_dir: Path | None = None) -> list[dict]:
    """Build manifest of all permutations with build status.

    Returns list of dicts with: permutation_id, dataset, pipeline, model, limit,
    done, output_path, sort_key, label.
    """
    out = output_dir or RESULTS_DIR
    entries: list[dict] = []

    for ds_slug, ds_cfg in DATASETS.items():
        for pipeline in PIPELINES:
            models = list(CHAT_MODELS.keys()) if pipeline == "llm-cluster" else ["-"]
            for model in models:
                for limit in LIMITS:
                    perm_id = permutation_id(ds_slug, pipeline, model, limit)
                    out_path = out / f"{perm_id}.json"
                    done = out_path.exists()

                    # Sort key: limit first (cheapest), then dataset, pipeline, model
                    sort_limit = limit if limit else 99999
                    entries.append(
                        {
                            "permutation_id": perm_id,
                            "dataset": ds_slug,
                            "pipeline": pipeline,
                            "model": model,
                            "limit": limit,
                            "done": done,
                            "output_path": out_path,
                            "sort_key": (sort_limit, ds_slug, pipeline, model),
                            "label": f"{ds_cfg.display_name} / {pipeline} / {model} / {limit or 'full'}",
                        }
                    )

    return entries


def print_manifest(
    entries: list[dict],
    *,
    missing: bool = False,
    done: bool = False,
    sort: str = "size",
    limit: int | None = None,
    commands: bool = False,
    force: bool = False,
) -> None:
    """Print manifest with filtering, sorting, and command generation."""
    if missing:
        entries = [e for e in entries if not e["done"]]
    if done:
        entries = [e for e in entries if e["done"]]

    if sort == "name":
        entries = sorted(entries, key=lambda e: e["permutation_id"])
    else:
        entries = sorted(entries, key=lambda e: e["sort_key"])

    if limit is not None:
        entries = entries[:limit]

    if commands:
        force_suffix = " --force" if force else ""
        for e in entries:
            limit_part = f" --limit {e['limit']}" if e["limit"] else ""
            model_part = f" --model {e['model']}" if e["model"] != "-" else ""
            print(
                f"uv run -m examples.er_v2 run"
                f" --dataset {e['dataset']}"
                f" --pipeline {e['pipeline']}"
                f"{model_part}"
                f"{limit_part}"
                f"{force_suffix}"
            )
        return

    total = len(entries)
    total_done = sum(1 for e in entries if e["done"])
    print(f"\n  === Manifest ({total_done}/{total}) ===\n")
    for e in entries:
        marker = "DONE" if e["done"] else "MISS"
        print(f"  [{marker}] {e['permutation_id']:<55s} {e['label']}")
    print()
