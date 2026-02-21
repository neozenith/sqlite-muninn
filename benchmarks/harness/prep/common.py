"""Shared helpers for prep modules: formatting, JSONL I/O, span conversion."""

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


# ── Span conversion utilities ────────────────────────────────────


def bio_to_spans(tokens: list[str], tags: list[str]) -> list[dict]:
    """Convert BIO-tagged tokens to character-offset entity spans.

    Args:
        tokens: List of word tokens.
        tags: List of BIO tags (B-PER, I-PER, O, etc.).

    Returns:
        List of {"start": int, "end": int, "label": str, "surface": str} dicts.
    """
    spans = []
    current_start = None
    current_label = None
    char_offset = 0
    token_starts = []

    # Compute character offsets for each token (space-separated)
    for token in tokens:
        token_starts.append(char_offset)
        char_offset += len(token) + 1  # +1 for space

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            # Close previous span if open
            if current_start is not None:
                surface = " ".join(tokens[current_start:i])
                spans.append(
                    {
                        "start": token_starts[current_start],
                        "end": token_starts[i - 1] + len(tokens[i - 1]),
                        "label": current_label,
                        "surface": surface,
                    }
                )
            current_start = i
            current_label = tag[2:]
        elif tag.startswith("I-"):
            # Continue current span (if label matches)
            if current_start is None or tag[2:] != current_label:
                # Malformed BIO: I without matching B — treat as B
                if current_start is not None:
                    surface = " ".join(tokens[current_start:i])
                    spans.append(
                        {
                            "start": token_starts[current_start],
                            "end": token_starts[i - 1] + len(tokens[i - 1]),
                            "label": current_label,
                            "surface": surface,
                        }
                    )
                current_start = i
                current_label = tag[2:]
        else:
            # O tag — close current span
            if current_start is not None:
                surface = " ".join(tokens[current_start:i])
                spans.append(
                    {
                        "start": token_starts[current_start],
                        "end": token_starts[i - 1] + len(tokens[i - 1]),
                        "label": current_label,
                        "surface": surface,
                    }
                )
                current_start = None
                current_label = None

    # Close trailing span
    if current_start is not None:
        surface = " ".join(tokens[current_start:])
        spans.append(
            {
                "start": token_starts[current_start],
                "end": token_starts[-1] + len(tokens[-1]),
                "label": current_label,
                "surface": surface,
            }
        )

    return spans


def io_to_spans(tokens: list[str], tags: list[str]) -> list[dict]:
    """Convert IO-tagged tokens (no B/I distinction) to character-offset entity spans.

    Few-NERD uses this scheme: each token is either "O" or a fine-grained type
    like "person-actor". Consecutive tokens with the same label form one span.

    Args:
        tokens: List of word tokens.
        tags: List of IO tags (e.g., "O", "person-actor", "location-city").

    Returns:
        List of {"start": int, "end": int, "label": str, "surface": str} dicts.
    """
    spans = []
    current_start = None
    current_label = None
    char_offset = 0
    token_starts = []

    for token in tokens:
        token_starts.append(char_offset)
        char_offset += len(token) + 1

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag == "O":
            if current_start is not None:
                surface = " ".join(tokens[current_start:i])
                spans.append(
                    {
                        "start": token_starts[current_start],
                        "end": token_starts[i - 1] + len(tokens[i - 1]),
                        "label": current_label,
                        "surface": surface,
                    }
                )
                current_start = None
                current_label = None
        elif tag != current_label:
            # New entity or label change
            if current_start is not None:
                surface = " ".join(tokens[current_start:i])
                spans.append(
                    {
                        "start": token_starts[current_start],
                        "end": token_starts[i - 1] + len(tokens[i - 1]),
                        "label": current_label,
                        "surface": surface,
                    }
                )
            current_start = i
            current_label = tag
        # else: same label continues, keep extending

    if current_start is not None:
        surface = " ".join(tokens[current_start:])
        spans.append(
            {
                "start": token_starts[current_start],
                "end": token_starts[-1] + len(tokens[-1]),
                "label": current_label,
                "surface": surface,
            }
        )

    return spans
