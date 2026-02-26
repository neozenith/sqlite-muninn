"""Configuration for the muninn visualization server."""

import os
from pathlib import Path

# Project root is two levels up from viz/server/
VIZ_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = VIZ_ROOT.parent

# Extension path — look for build artifact first, then fall back to root-level
EXTENSION_PATH = str(PROJECT_ROOT / "build" / "muninn")

# Default database path — the Wealth of Nations KG
DEFAULT_DB_PATH = str(PROJECT_ROOT / "benchmarks" / "kg" / "3300.db")
DB_PATH = os.environ.get("MUNINN_DB_PATH", DEFAULT_DB_PATH)

# Demo databases directory (where manifest.json and built .db files live)
DEMOS_DIR = VIZ_ROOT / "frontend" / "public" / "demos"

# Server ports
DEFAULT_PORT = 8200
