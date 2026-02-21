"""Tests for Makefile.refactored targets."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BENCHMARKS_ROOT = PROJECT_ROOT / "benchmarks"


def test_makefile_refactored_exists():
    makefile = BENCHMARKS_ROOT / "Makefile.refactored"
    assert makefile.exists(), "benchmarks/Makefile.refactored should exist"


def test_makefile_has_required_targets():
    makefile = BENCHMARKS_ROOT / "Makefile.refactored"
    content = makefile.read_text(encoding="utf-8")

    required_targets = [
        "help",
        "prep",
        "manifest",
        "benchmark-vss",
        "benchmark-graph",
        "benchmark-all",
        "analyse",
        "clean",
    ]

    for target in required_targets:
        assert f"{target}:" in content, f"Makefile.refactored should have target: {target}"


def test_makefile_uses_cli():
    """Makefile targets should invoke the harness CLI."""
    makefile = BENCHMARKS_ROOT / "Makefile.refactored"
    content = makefile.read_text(encoding="utf-8")
    assert "benchmarks.harness.cli" in content
