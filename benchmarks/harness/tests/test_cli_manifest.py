"""Tests for the manifest subcommand."""

import subprocess
import sys


class TestManifestCLI:
    def test_manifest_runs(self):
        """The manifest subcommand should run without errors."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "manifest"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Benchmark Manifest" in result.stdout

    def test_manifest_with_category(self):
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "manifest", "--category", "vss"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "VSS" in result.stdout

    def test_manifest_missing_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "manifest", "--missing"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        # All should be MISS since we haven't run any benchmarks
        assert "MISS" in result.stdout

    def test_manifest_commands_flag(self):
        """--commands should output runnable benchmark commands."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.harness.cli",
                "manifest",
                "--missing",
                "--category",
                "vss",
                "--commands",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "benchmarks.harness.cli benchmark --id" in result.stdout
        # Should have one line per missing permutation
        lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
        assert len(lines) > 0

    def test_manifest_done_flag(self):
        """--done should only show completed benchmarks (no MISS entries)."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "manifest", "--done"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        # --done should never show MISS entries
        assert "MISS" not in result.stdout

    def test_manifest_limit(self):
        """--limit N should restrict output to N entries."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "manifest", "--missing", "--limit", "3", "--commands"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
        assert len(lines) == 3

    def test_manifest_limit_display_mode(self):
        """--limit N in display mode should restrict visible entries."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "manifest", "--missing", "--limit", "2"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        # Should show "2 done" total
        assert "(0/2 done)" in result.stdout

    def test_manifest_category_no_value_lists_categories(self):
        """--category without a value should list available categories."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "manifest", "--category"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Available categories" in result.stdout
        assert "vss" in result.stdout
        assert "graph" in result.stdout
        assert "graph_vt" in result.stdout

    def test_manifest_invalid_category_errors(self):
        """--category with an invalid value should error with available categories."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "manifest", "--category", "nonexistent"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    def test_manifest_commands_with_force(self):
        """--commands --force should append --force to each generated command."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.harness.cli",
                "manifest",
                "--missing",
                "--category",
                "vss",
                "--commands",
                "--force",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
        assert len(lines) > 0
        for line in lines:
            assert line.endswith("--force"), f"Expected --force suffix: {line}"

    def test_manifest_commands_without_force(self):
        """--commands without --force should NOT append --force."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.harness.cli",
                "manifest",
                "--missing",
                "--category",
                "vss",
                "--commands",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
        assert len(lines) > 0
        for line in lines:
            assert not line.endswith("--force"), f"Unexpected --force suffix: {line}"

    def test_manifest_limit_one_command(self):
        """--limit 1 --commands should give exactly one runnable command."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "manifest", "--missing", "--limit", "1", "--commands"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
        assert len(lines) == 1
        assert "benchmark --id" in lines[0]
