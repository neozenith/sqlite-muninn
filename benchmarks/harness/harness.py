"""Benchmark execution harness.

Executes a single Treatment:
1. Creates RESULTS_DIR / treatment.permutation_id / db.sqlite
2. Opens SQLite connection, loads muninn
3. Records RSS before, calls setup + run + teardown with timing
4. Merges common metrics + treatment metrics into JSONL record
5. Appends record to RESULTS_DIR / {category}_{variant}.jsonl
"""

import logging
import os
import sys
import time
from pathlib import Path

from benchmarks.harness.common import (
    RESULTS_DIR,
    load_muninn,
    peak_rss_mb,
    platform_info,
    write_jsonl,
)
from benchmarks.harness.treatments.base import Treatment

log = logging.getLogger(__name__)


def _handle_existing_db(db_path: Path, force: bool = False) -> None:
    """Handle an existing db.sqlite file before running a benchmark.

    Default: warn, 30s countdown, then delete.
    With force=True: delete immediately, no countdown.
    """
    if not db_path.exists():
        return

    if force:
        log.info("  --force: deleting existing %s", db_path)
        db_path.unlink()
        return

    log.warning("  WARNING: %s already exists", db_path)
    log.warning("  Will delete in 30 seconds. Press Ctrl+C to cancel.")
    for remaining in range(30, 0, -1):
        sys.stderr.write(f"\r  Deleting in {remaining}s... (Ctrl+C to cancel) ")
        sys.stderr.flush()
        time.sleep(1)
    sys.stderr.write("\r  Deleting existing results...                       \n")
    sys.stderr.flush()
    db_path.unlink()


def run_treatment(treatment: Treatment, results_dir: Path | None = None, force: bool = False) -> dict:
    """Execute a single benchmark treatment and write results.

    Args:
        treatment: The Treatment instance to execute.
        results_dir: Override for RESULTS_DIR (useful in tests).
        force: If True, delete existing db.sqlite immediately without countdown.

    Returns:
        The complete metrics record that was written to JSONL.
    """
    results_dir = results_dir or RESULTS_DIR

    # Create the permutation output directory
    perm_dir = results_dir / treatment.permutation_id
    perm_dir.mkdir(parents=True, exist_ok=True)
    db_path = perm_dir / "db.sqlite"

    # Handle existing results
    _handle_existing_db(db_path, force=force)

    log.info("Running: %s", treatment.label)
    log.info("  DB: %s", db_path)

    # Open connection and load extension
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        load_muninn(conn)
    except Exception:
        log.warning("  Could not load muninn extension — treatment may handle its own extensions")

    # Record RSS before
    rss_before = peak_rss_mb()

    # Setup
    t0 = time.perf_counter()
    try:
        setup_metrics = treatment.setup(conn, db_path) or {}
    except Exception as e:
        log.error("  Setup failed: %s", e)
        conn.close()
        raise
    wall_time_setup_ms = (time.perf_counter() - t0) * 1000

    # Run
    t0 = time.perf_counter()
    try:
        run_metrics = treatment.run(conn) or {}
    except Exception as e:
        log.error("  Run failed: %s", e)
        treatment.teardown(conn)
        conn.close()
        raise
    wall_time_run_ms = (time.perf_counter() - t0) * 1000

    # Teardown
    treatment.teardown(conn)

    # Record RSS after and DB size
    rss_after = peak_rss_mb()
    conn.close()

    db_size_bytes = os.path.getsize(str(db_path)) if db_path.exists() else 0

    # Build the complete record
    info = platform_info()
    record = {
        # Identity
        "permutation_id": treatment.permutation_id,
        "category": treatment.category,
        # Common metrics
        "wall_time_setup_ms": round(wall_time_setup_ms, 3),
        "wall_time_run_ms": round(wall_time_run_ms, 3),
        "peak_rss_mb": round(rss_after, 1),
        "rss_delta_mb": round(max(0, rss_after - rss_before), 1),
        "db_size_bytes": db_size_bytes,
        # Platform
        "timestamp": info["timestamp"],
        "platform": info["platform"],
        "python_version": info["python_version"],
    }

    # Merge treatment params and metrics
    record.update(treatment.params_dict())
    record.update(setup_metrics)
    record.update(run_metrics)

    # Determine JSONL file path: {category}_{variant}.jsonl
    # The variant is derived from permutation_id minus the category prefix
    perm_id = treatment.permutation_id
    if perm_id.startswith(treatment.category + "_"):
        variant = perm_id[len(treatment.category) + 1 :]
    else:
        variant = perm_id
    jsonl_path = results_dir / f"{treatment.category}_{variant}.jsonl"

    write_jsonl(jsonl_path, record)

    log.info("  Setup: %.1f ms, Run: %.1f ms, DB: %d bytes", wall_time_setup_ms, wall_time_run_ms, db_size_bytes)
    log.info("  Results: %s", jsonl_path)

    return record
