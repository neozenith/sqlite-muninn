"""Prep module: download entity resolution benchmark datasets.

Downloads ER datasets with ground truth for Pairwise F1 and B-Cubed F1 evaluation.

Tier 1 (tiny, zero-setup): Febrl 1/4 via recordlinkage package
Tier 2 (small): DBLP-ACM, Affiliations (Leipzig direct download)

Source: docs/plans/entity_resolution_benchmarks.md
"""

import logging

from benchmarks.harness.common import KG_DIR

log = logging.getLogger(__name__)

KNOWN_ER_DATASETS = ["febrl1", "febrl4"]


def print_status():
    """Print status of ER benchmark datasets."""
    er_dir = KG_DIR / "er"

    print("=== ER Dataset Status ===\n")
    print(f"  {'DATASET':<16s}   {'DIR':<28s}   {'STATUS'}")
    print(f"  {'-' * 16}   {'-' * 28}   {'-' * 12}")

    for ds_name in KNOWN_ER_DATASETS:
        ds_dir = er_dir / ds_name
        if ds_dir.exists() and any(ds_dir.iterdir()):
            print(f"  {ds_name:<16s}   {f'er/{ds_name}/':<28s}   READY")
        elif ds_dir.exists():
            print(f"  {ds_name:<16s}   {f'er/{ds_name}/':<28s}   EMPTY")
        else:
            print(f"  {ds_name:<16s}   {f'er/{ds_name}/':<28s}   MISSING")

    # Show any extra datasets
    if er_dir.exists():
        for ds_dir in sorted(er_dir.iterdir()):
            if ds_dir.is_dir() and ds_dir.name not in KNOWN_ER_DATASETS:
                has_files = any(ds_dir.iterdir())
                status = "READY" if has_files else "EMPTY"
                print(f"  {ds_dir.name:<16s}   {f'er/{ds_dir.name}/':<28s}   {status} (extra)")

    print()


def prep_er_datasets(dataset=None, status_only=False, force=False):
    """Download ER benchmark datasets.

    Args:
        dataset: Specific dataset to download (e.g., 'febrl1'). If None, downloads all Tier 1.
        status_only: If True, show status and return.
        force: If True, re-download datasets even if they exist.
    """
    if status_only:
        print_status()
        return

    er_dir = KG_DIR / "er"
    er_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_prep = [dataset] if dataset else KNOWN_ER_DATASETS

    for ds_name in datasets_to_prep:
        ds_dir = er_dir / ds_name

        if ds_dir.exists() and any(ds_dir.iterdir()) and not force:
            log.info("  Dataset %s: already exists (use --force to re-download)", ds_name)
            continue

        ds_dir.mkdir(parents=True, exist_ok=True)

        if ds_name.startswith("febrl"):
            _prep_febrl(ds_name, ds_dir)
        else:
            log.warning("  Dataset %s: download not yet implemented", ds_name)

    log.info("ER dataset prep complete. Datasets in %s", er_dir)


def _prep_febrl(dataset_name, output_dir):
    """Download Febrl dataset via recordlinkage package and save as parquet."""
    try:
        from recordlinkage.datasets import load_febrl1, load_febrl4
    except ImportError:
        log.warning("  recordlinkage package not available — skipping %s", dataset_name)
        log.warning("  Install with: uv pip install recordlinkage")
        return

    if dataset_name == "febrl1":
        df = load_febrl1()
        out_path = output_dir / "febrl1.parquet"
        df.to_parquet(out_path)
        log.info("  %s: saved %d records to %s", dataset_name, len(df), out_path)

    elif dataset_name == "febrl4":
        df_a, df_b = load_febrl4()
        path_a = output_dir / "febrl4_a.parquet"
        path_b = output_dir / "febrl4_b.parquet"
        df_a.to_parquet(path_a)
        df_b.to_parquet(path_b)
        log.info("  %s: saved %d + %d records to %s", dataset_name, len(df_a), len(df_b), output_dir)

    else:
        log.warning("  Unknown Febrl dataset: %s", dataset_name)
