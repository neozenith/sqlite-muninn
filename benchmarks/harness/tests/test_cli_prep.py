"""Tests for the prep subcommand CLI structure."""

import subprocess
import sys


class TestPrepCLI:
    def test_prep_help(self):
        """prep --help should list sub-subcommands."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "prep", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "vectors" in result.stdout
        assert "texts" in result.stdout
        assert "kg-chunks" in result.stdout
        assert "kg" in result.stdout
        assert "all" in result.stdout

    def test_prep_no_subcommand_shows_usage(self):
        """prep without a target should show usage and exit with error."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "prep"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0

    def test_prep_vectors_help(self):
        """prep vectors --help should show --status and --force flags."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "prep", "vectors", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "--status" in result.stdout
        assert "--force" in result.stdout
        assert "--model" in result.stdout
        assert "--dataset" in result.stdout

    def test_prep_texts_help_has_examples(self):
        """prep texts --help should show examples."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "prep", "texts", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "--status" in result.stdout
        assert "--force" in result.stdout
        assert "--book-id" in result.stdout
        assert "--random" in result.stdout
        assert "--category" in result.stdout
        assert "--list" in result.stdout
        assert "Examples:" in result.stdout

    def test_prep_kg_chunks_help(self):
        """prep kg-chunks --help should show --status and --force."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "prep", "kg-chunks", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "--status" in result.stdout
        assert "--force" in result.stdout
        assert "--book-id" in result.stdout

    def test_prep_kg_help(self):
        """prep kg --help should show --status, --force, and --dataset."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "prep", "kg", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "--status" in result.stdout
        assert "--force" in result.stdout
        assert "--dataset" in result.stdout

    def test_prep_all_help(self):
        """prep all --help should show --status and --force."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "prep", "all", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "--status" in result.stdout
        assert "--force" in result.stdout

    def test_prep_vectors_status(self):
        """prep vectors --status should show cache status table."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "prep", "vectors", "--status"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Vector Cache Status" in result.stdout

    def test_prep_texts_status(self):
        """prep texts --status should show text cache status."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "prep", "texts", "--status"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Text Cache Status" in result.stdout

    def test_prep_texts_list(self):
        """prep texts --list should show cached texts."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "prep", "texts", "--list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        # Should show either cached texts or "No cached texts"
        assert "Cached Gutenberg Texts" in result.stdout or "No cached texts" in result.stdout

    def test_prep_kg_chunks_status(self):
        """prep kg-chunks --status should show chunk DB status."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "prep", "kg-chunks", "--status"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "KG Chunk Database Status" in result.stdout

    def test_prep_kg_status(self):
        """prep kg --status should show KG dataset status."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "prep", "kg", "--status"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert "KG Dataset Status" in result.stdout

    def test_prep_all_status(self):
        """prep all --status should show status for all targets."""
        result = subprocess.run(
            [sys.executable, "-m", "benchmarks.harness.cli", "prep", "all", "--status"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert "Vector Cache Status" in result.stdout
        assert "Text Cache Status" in result.stdout
        assert "KG Chunk Database Status" in result.stdout
        assert "KG Dataset Status" in result.stdout
