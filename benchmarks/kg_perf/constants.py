"""Paths and defaults for the kg_perf harness."""

from __future__ import annotations

import subprocess
from pathlib import Path
from shlex import split

_run = lambda cmd: subprocess.check_output(split(cmd), text=True).strip()  # noqa: E731

PROJECT_ROOT = Path(_run("git rev-parse --show-toplevel"))

DEFAULT_DB = PROJECT_ROOT / "viz" / "frontend" / "public" / "demos" / "sessions_demo.db"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "kg_perf" / "results"
MUNINN_PATH = PROJECT_ROOT / "muninn"

REPETITIONS = 5
WARMUP = 1
