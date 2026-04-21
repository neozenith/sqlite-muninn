"""Permutation registry for er_v3 — mirrors er_v2 parameter space.

Uses the same sweep ranges so results are directly comparable.
"""

from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

DATASETS_LIST = ["amazon-google", "dblp-acm"]

DIST_RANGE = [round(0.05 + i * 0.05, 2) for i in range(8)]  # 0.05 – 0.40
JW_RANGE = [round(1.0 - i * 0.05, 2) for i in range(21)]  # 1.0 – 0.0
LLM_HIGH_RANGE = [round(1.0 - i * 0.05, 2) for i in range(5)]  # 1.0 – 0.80 step 0.05
DELTA_RANGE = [0.0, 0.01]


def permutation_id(
    dataset: str,
    *,
    dist_threshold: float,
    jw_weight: float,
    llm_high: float,
    borderline_delta: float,
) -> str:
    llm_low = round(llm_high - borderline_delta, 4)
    return f"{dataset}__dist{dist_threshold:.2f}_jw{jw_weight:.2f}_hi{llm_high:.2f}_lo{llm_low:.2f}"


def permutation_manifest(output_dir: Path | None = None) -> list[dict]:
    out = output_dir or RESULTS_DIR
    entries: list[dict] = []

    for dist in DIST_RANGE:
        for jw in JW_RANGE:
            for hi in LLM_HIGH_RANGE:
                for delta in DELTA_RANGE:
                    for ds_slug in DATASETS_LIST:
                        perm_id = permutation_id(
                            ds_slug,
                            dist_threshold=dist,
                            jw_weight=jw,
                            llm_high=hi,
                            borderline_delta=delta,
                        )
                        out_path = out / f"{perm_id}.json"
                        llm_low = round(hi - delta, 4)
                        entries.append(
                            {
                                "permutation_id": perm_id,
                                "dataset": ds_slug,
                                "dist_threshold": dist,
                                "jw_weight": jw,
                                "llm_high": hi,
                                "borderline_delta": delta,
                                "llm_low": llm_low,
                                "done": out_path.exists(),
                                "output_path": out_path,
                                "sort_key": (dist, -jw, -hi, delta, ds_slug),
                                "label": f"{ds_slug} dist={dist} jw={jw} hi={hi} Δ={delta}",
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
            print(
                f"uv run -m examples.er_v3 run"
                f" --dataset {e['dataset']}"
                f" --dist {e['dist_threshold']}"
                f" --jw-weight {e['jw_weight']}"
                f" --llm-high {e['llm_high']}"
                f" --borderline-delta {e['borderline_delta']}"
                f"{force_suffix}"
            )
        return

    total = len(entries)
    total_done = sum(1 for e in entries if e["done"])
    print(f"\n  === Manifest ({total_done}/{total}) ===\n")
    for e in entries:
        marker = "DONE" if e["done"] else "MISS"
        print(f"  [{marker}] {e['permutation_id']:<65s} {e['label']}")
    print()
