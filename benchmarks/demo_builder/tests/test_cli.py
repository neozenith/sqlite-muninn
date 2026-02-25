"""Tests for CLI subcommand parsing and info modes."""

from __future__ import annotations

import subprocess
from shlex import split

from benchmarks.demo_builder.constants import PROJECT_ROOT


def _run(cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(split(cmd), capture_output=True, text=True, cwd=PROJECT_ROOT)


def test_help_exits_with_error_code() -> None:
    result = _run("uv run -m benchmarks.demo_builder")
    assert result.returncode == 1


def test_help_flag() -> None:
    result = _run("uv run -m benchmarks.demo_builder --help")
    assert result.returncode == 0
    assert "manifest" in result.stdout
    assert "build" in result.stdout
    assert "list-books" in result.stdout
    assert "list-models" in result.stdout


def test_list_models() -> None:
    result = _run("uv run -m benchmarks.demo_builder list-models")
    assert result.returncode == 0
    assert "MiniLM" in result.stdout
    assert "NomicEmbed" in result.stdout


def test_list_books() -> None:
    result = _run("uv run -m benchmarks.demo_builder list-books")
    assert result.returncode == 0
    assert "Available Books" in result.stdout


def test_manifest() -> None:
    result = _run("uv run -m benchmarks.demo_builder manifest --output-folder wasm/assets")
    assert result.returncode == 0
    assert "Demo DB Manifest" in result.stdout


def test_manifest_missing_commands() -> None:
    result = _run("uv run -m benchmarks.demo_builder manifest --output-folder wasm/assets --missing --commands")
    assert result.returncode == 0
    # Output is either commands or nothing (if all are built)
    if result.stdout.strip():
        assert "benchmarks.demo_builder build" in result.stdout
