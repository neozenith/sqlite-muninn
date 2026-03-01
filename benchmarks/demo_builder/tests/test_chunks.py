"""Tests for the _chunk_text function in phases/chunks.py."""

from __future__ import annotations

import pytest

from benchmarks.demo_builder.constants import EMBEDDING_MODELS, NER_RE_CHUNK_CHARS_MAX
from benchmarks.demo_builder.phases.chunks import _chunk_text


def test_single_short_text() -> None:
    """Text shorter than chunk_chars produces a single chunk."""
    text = "Hello world."
    chunks = _chunk_text(text, chunk_chars=100)
    assert len(chunks) == 1
    assert chunks[0] == "Hello world."


def test_chunks_respect_max_size() -> None:
    """Every chunk is <= chunk_chars in length."""
    text = "word " * 500  # ~2500 chars
    chunks = _chunk_text(text, chunk_chars=200)
    for chunk in chunks:
        assert len(chunk) <= 200


def test_overlap_between_consecutive_chunks() -> None:
    """Consecutive chunks share overlapping content."""
    # Use text without sentence boundaries to prevent snapping
    text = "abcdefghij" * 100  # 1000 chars, no sentence boundaries
    chunks = _chunk_text(text, chunk_chars=200, overlap_frac=0.2)
    assert len(chunks) > 1
    # With 200 char chunks and 20% overlap (40 chars), consecutive chunks
    # should share ~40 chars at the boundary
    for i in range(len(chunks) - 1):
        tail = chunks[i][-30:]  # Last 30 chars of current chunk
        assert tail in chunks[i + 1], f"No overlap found between chunk {i} and {i + 1}"


def test_full_text_coverage() -> None:
    """All content from the original text appears in at least one chunk."""
    sentences = [f"Sentence number {i}." for i in range(50)]
    text = " ".join(sentences)
    chunks = _chunk_text(text, chunk_chars=200)
    joined = " ".join(chunks)
    # Every sentence should appear in the joined output
    for sentence in sentences:
        assert sentence in joined, f"Missing sentence: {sentence}"


def test_sentence_boundary_snapping() -> None:
    """Chunks prefer to end at sentence boundaries."""
    # Create text where sentence boundaries are clear
    text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    # chunk_chars just big enough for ~2 sentences, should snap at '. '
    chunks = _chunk_text(text, chunk_chars=40)
    # Most chunks (except possibly the last) should end with '.' or similar
    for chunk in chunks[:-1]:
        assert chunk.endswith(".") or chunk.endswith("!") or chunk.endswith("?"), (
            f"Chunk did not snap to sentence boundary: '{chunk}'"
        )


def test_empty_text_returns_empty() -> None:
    """Empty text produces no chunks."""
    assert _chunk_text("", chunk_chars=100) == []


def test_whitespace_only_returns_empty() -> None:
    """Whitespace-only text produces no chunks (stripped to empty)."""
    assert _chunk_text("   \n\n  ", chunk_chars=100) == []


@pytest.mark.parametrize("chunk_chars", [100, 500, 1000, 4096])
def test_various_chunk_sizes(chunk_chars: int) -> None:
    """Chunking works correctly at different size parameters."""
    text = ("The quick brown fox jumps over the lazy dog. " * 20 + "\n") * 10
    chunks = _chunk_text(text, chunk_chars=chunk_chars)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk) <= chunk_chars
        assert len(chunk) > 0


def test_no_empty_chunks_in_output() -> None:
    """The output never contains empty strings."""
    text = "\n\n\nHello.\n\n\nWorld.\n\n\n"
    chunks = _chunk_text(text, chunk_chars=50)
    for chunk in chunks:
        assert chunk.strip() != ""


def test_chunk_chars_capped_by_ner_re_limit() -> None:
    """When a model's chunk_chars exceeds NER_RE_CHUNK_CHARS_MAX, effective size is capped."""
    text = "word " * 2000  # ~10000 chars
    for name, info in EMBEDDING_MODELS.items():
        effective = min(info["chunk_chars"], NER_RE_CHUNK_CHARS_MAX)
        chunks = _chunk_text(text, chunk_chars=effective)
        for chunk in chunks:
            assert len(chunk) <= effective, f"{name}: chunk of {len(chunk)} chars exceeds effective limit {effective}"
