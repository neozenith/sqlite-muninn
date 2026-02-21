"""CLI entry point: `python -m benchmarks.harness.cli`

Four subcommands:
    prep        — Download models, texts, build caches
    manifest    — Show benchmark completion status
    benchmark   — Run a single benchmark permutation
    analyse     — Generate charts + tables from results
"""

import argparse
import logging
import sys

from benchmarks.harness.prep.kg_datasets import KG_PREP_TASKS

log = logging.getLogger(__name__)


def _cmd_prep(args):
    """Handle the 'prep' subcommand."""
    target = args.prep_target

    if target is None:
        print("Usage: benchmarks.harness.cli prep {vectors,texts,kg-chunks,kg,all}")
        print("Run 'benchmarks.harness.cli prep --help' for details.")
        sys.exit(1)

    status_only = getattr(args, "status", False)
    force = getattr(args, "force", False)

    if target in ("vectors", "all"):
        from benchmarks.harness.prep.vectors import prep_vectors

        prep_vectors(
            only_model=getattr(args, "model", None),
            only_dataset=getattr(args, "dataset", None),
            status_only=status_only,
            force=force,
        )

    if target in ("texts", "all"):
        from benchmarks.harness.prep.texts import prep_texts

        prep_texts(
            book_id=getattr(args, "book_id", None),
            random_book=getattr(args, "random", False),
            category=getattr(args, "category", "economics"),
            list_cached=getattr(args, "list_cached", False),
            status_only=status_only,
            force=force,
        )

    if target in ("kg-chunks", "all"):
        from benchmarks.harness.prep.kg_chunks import prep_kg_chunks

        prep_kg_chunks(
            book_id=getattr(args, "book_id", None),
            status_only=status_only,
            force=force,
        )

    if target in ("kg", "all"):
        from benchmarks.harness.prep.kg_datasets import prep_kg_datasets

        prep_kg_datasets(
            dataset=getattr(args, "dataset", None),
            status_only=status_only,
            force=force,
        )


def _print_category_summary(status):
    """Print a summary table of available categories."""
    cats = {}
    for s in status:
        cat = s["category"]
        if cat not in cats:
            cats[cat] = {"total": 0, "done": 0}
        cats[cat]["total"] += 1
        if s["done"]:
            cats[cat]["done"] += 1

    print("Available categories:\n")
    print(f"  {'CATEGORY':<20s} {'TOTAL':>6s} {'DONE':>6s} {'MISSING':>8s}")
    print(f"  {'-' * 20} {'-' * 6} {'-' * 6} {'-' * 8}")
    for cat in sorted(cats.keys()):
        total = cats[cat]["total"]
        done = cats[cat]["done"]
        missing = total - done
        print(f"  {cat:<20s} {total:>6d} {done:>6d} {missing:>8d}")
    print()


def _sort_entries(entries, sort_mode):
    """Sort entries within a single category by the given mode."""
    if sort_mode == "name":
        return sorted(entries, key=lambda s: s["permutation_id"])
    # Default: sort by size (sort_key tuple, ascending)
    return sorted(entries, key=lambda s: s["sort_key"])


def _cmd_manifest(args):
    """Handle the 'manifest' subcommand."""
    from benchmarks.harness.registry import permutation_status

    status = permutation_status()

    # If --category given without a value, list available categories
    category_filter = getattr(args, "category", None)
    if category_filter == "":
        _print_category_summary(status)
        return

    # Apply filters
    if category_filter:
        matched = [s for s in status if s["category"] == category_filter]
        if not matched:
            all_cats = sorted({s["category"] for s in status})
            log.error("Unknown category: %s. Available: %s", category_filter, ", ".join(all_cats))
            sys.exit(1)
        status = matched
    if args.missing:
        status = [s for s in status if not s["done"]]
    if args.done:
        status = [s for s in status if s["done"]]

    sort_mode = getattr(args, "sort", "size")

    # Group by category, then sort within each group
    by_category = {}
    for s in status:
        by_category.setdefault(s["category"], []).append(s)
    for category in by_category:
        by_category[category] = _sort_entries(by_category[category], sort_mode)

    # Flatten sorted entries (preserving category order) for --limit
    flat_entries = []
    for category in sorted(by_category.keys()):
        flat_entries.extend(by_category[category])

    limit = getattr(args, "limit", None)
    if limit is not None:
        flat_entries = flat_entries[:limit]
        # Rebuild by_category from the limited flat list
        by_category = {}
        for s in flat_entries:
            by_category.setdefault(s["category"], []).append(s)

    if args.commands:
        # Print runnable commands (sorted within category, categories in alpha order)
        force_suffix = " --force" if getattr(args, "force", False) else ""
        for s in flat_entries:
            print(f"uv run -m benchmarks.harness.cli benchmark --id {s['permutation_id']}{force_suffix}")
        return

    total_done = sum(1 for s in flat_entries if s["done"])
    total = len(flat_entries)

    print(f"=== Benchmark Manifest ({total_done}/{total} done) ===\n")

    for category in sorted(by_category.keys()):
        entries = by_category[category]
        cat_done = sum(1 for e in entries if e["done"])
        print(f"{category.upper()} ({cat_done}/{len(entries)}):")
        for e in entries:
            marker = "DONE" if e["done"] else "MISS"
            print(f"  [{marker}] {e['permutation_id']:60s} {e['label']}")
        print()


def _cmd_benchmark(args):
    """Handle the 'benchmark' subcommand."""
    from benchmarks.harness.harness import run_treatment
    from benchmarks.harness.registry import filter_permutations

    perms = filter_permutations(permutation_id=args.id)
    if not perms:
        log.error("No permutation found with ID: %s", args.id)
        sys.exit(1)

    treatment = perms[0]
    force = getattr(args, "force", False)
    run_treatment(treatment, force=force)


def _cmd_analyse(args):
    """Handle the 'analyse' subcommand."""
    from benchmarks.harness.analysis.aggregator import run_aggregation
    from benchmarks.harness.common import RESULTS_DIR

    category = getattr(args, "category", None)
    render_docs = getattr(args, "render_docs", False)

    run_aggregation(results_dir=RESULTS_DIR, category=category, render_docs=render_docs)


def main():
    parser = argparse.ArgumentParser(
        prog="benchmarks.harness.cli",
        description="Unified benchmark CLI: prep, manifest, benchmark, analyse",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # ── prep ──────────────────────────────────────────────────────
    prep_parser = subparsers.add_parser("prep", help="Download models, texts, build caches")
    prep_subs = prep_parser.add_subparsers(dest="prep_target", help="Prep target")

    # prep vectors
    prep_vectors_p = prep_subs.add_parser("vectors", help="Download embedding models + build .npy caches")
    prep_vectors_p.add_argument("--status", action="store_true", help="Show cache status without downloading")
    prep_vectors_p.add_argument("--force", action="store_true", help="Re-create caches even if they exist")
    prep_vectors_p.add_argument("--model", help="Only prep this embedding model (e.g., MiniLM)")
    prep_vectors_p.add_argument("--dataset", help="Only prep this dataset (e.g., ag_news)")

    # prep texts
    prep_texts_p = prep_subs.add_parser(
        "texts",
        help="Download Gutenberg texts for benchmark corpora",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  prep texts                          # Download default corpus (Wealth of Nations)\n"
            "  prep texts --random                 # Download a random economics book\n"
            "  prep texts --random --category law  # Download a random law book\n"
            "  prep texts --book-id 3300           # Download Wealth of Nations specifically\n"
            "  prep texts --list                   # Show all cached texts\n"
        ),
    )
    prep_texts_p.add_argument("--status", action="store_true", help="Show cache status without downloading")
    prep_texts_p.add_argument("--force", action="store_true", help="Re-download texts even if cached")
    prep_texts_p.add_argument("--book-id", type=int, help="Download a specific Gutenberg book by ID")
    prep_texts_p.add_argument("--random", action="store_true", help="Download a random book from the category")
    prep_texts_p.add_argument(
        "--category", default="economics", help="Gutenberg subject category for --random (default: economics)"
    )
    prep_texts_p.add_argument(
        "--list", action="store_true", dest="list_cached", help="List all cached texts with metadata"
    )

    # prep kg-chunks
    prep_kg_p = prep_subs.add_parser("kg-chunks", help="Create KG chunk databases from cached texts")
    prep_kg_p.add_argument("--status", action="store_true", help="Show chunk database status")
    prep_kg_p.add_argument("--force", action="store_true", help="Re-create chunk databases even if they exist")
    prep_kg_p.add_argument("--book-id", type=int, help="Gutenberg book ID")

    # prep kg (unified NER + RE + ER)
    kg_task_ids = [t.task_id for t in KG_PREP_TASKS]
    prep_kg_ds_p = prep_subs.add_parser("kg", help="Download KG benchmark datasets (NER, RE, ER)")
    prep_kg_ds_p.add_argument("--status", action="store_true", help="Show KG dataset download status")
    prep_kg_ds_p.add_argument("--force", action="store_true", help="Re-download datasets even if they exist")
    prep_kg_ds_p.add_argument("--dataset", choices=kg_task_ids, help="Specific dataset to download")

    # prep all
    prep_all_p = prep_subs.add_parser("all", help="Prep everything (vectors + texts + kg-chunks + kg)")
    prep_all_p.add_argument("--status", action="store_true", help="Show status of all prep targets")
    prep_all_p.add_argument("--force", action="store_true", help="Force re-creation of all prep targets")

    # ── manifest ──────────────────────────────────────────────────
    manifest_parser = subparsers.add_parser("manifest", help="Show benchmark completion status")
    manifest_parser.add_argument("--missing", action="store_true", help="Only show missing benchmarks")
    manifest_parser.add_argument("--done", action="store_true", help="Only show completed benchmarks")
    manifest_parser.add_argument(
        "--category",
        nargs="?",
        const="",
        default=None,
        help="Filter by category (omit value to list available categories). "
        "Categories: vss, graph, centrality, community, graph_vt, "
        "kg-extract, kg-re, kg-resolve, kg-graphrag, node2vec",
    )
    manifest_parser.add_argument("--commands", action="store_true", help="Print runnable commands")
    manifest_parser.add_argument(
        "--force",
        action="store_true",
        help="When used with --commands, append --force to each generated benchmark command",
    )
    manifest_parser.add_argument(
        "--sort",
        choices=["size", "name"],
        default="size",
        help="Sort order: 'size' (ascending by scaling dimension, default) or 'name' (alphabetical by ID)",
    )
    manifest_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit output to the first N entries (cheapest/smallest first when sorted by size)",
    )

    # ── benchmark ─────────────────────────────────────────────────
    bench_parser = subparsers.add_parser("benchmark", help="Run a single benchmark permutation")
    bench_parser.add_argument("--id", required=True, help="Permutation ID to run")
    bench_parser.add_argument(
        "--force",
        action="store_true",
        help="Skip 30s countdown when re-running a benchmark with existing results",
    )

    # ── analyse ───────────────────────────────────────────────────
    analyse_parser = subparsers.add_parser("analyse", help="Generate charts + tables from results")
    analyse_parser.add_argument("--category", help="Only analyse this category")
    analyse_parser.add_argument("--render-docs", action="store_true", help="Also render Jinja2 doc templates")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if args.command == "prep":
        _cmd_prep(args)
    elif args.command == "manifest":
        _cmd_manifest(args)
    elif args.command == "benchmark":
        _cmd_benchmark(args)
    elif args.command == "analyse":
        _cmd_analyse(args)


if __name__ == "__main__":
    main()
