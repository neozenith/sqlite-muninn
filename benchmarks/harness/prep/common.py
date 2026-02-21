"""Shared helpers for prep modules: formatting, JSONL I/O."""

import json
from pathlib import Path


def fmt_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of dicts as newline-delimited JSON."""
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    return sum(1 for line in path.read_text(encoding="utf-8").strip().split("\n") if line.strip())
