"""Tests for the benchmark subcommand + subprocess isolation."""

import subprocess
import sys


class TestBenchmarkCLI:
    def test_unknown_id_exits_with_error(self):
        """An invalid permutation ID should exit with error."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "benchmark", "--id", "nonexistent_id_12345"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    def test_missing_id_flag_errors(self):
        """Omitting --id should show usage error."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "benchmark"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    def test_help_flag(self):
        """--help should work."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "benchmark", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "--id" in result.stdout
        assert "--force" in result.stdout
