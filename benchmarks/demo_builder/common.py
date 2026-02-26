"""Shared utility functions used across the demo_builder package.

This module is imported by phases.py (which has ML deps), so top-level
numpy/spacy/struct imports are acceptable here — they are only loaded
when phases.py is imported, which happens inside _cmd_build().
"""

from __future__ import annotations

import logging
import sqlite3
import struct
from typing import TYPE_CHECKING

import numpy as np
import spacy.tokens

from benchmarks.demo_builder.constants import (
    BOOK_ID_TO_DATASET,
    EMBEDDING_MODELS,
    MUNINN_PATH,
    VECTORS_DIR,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)


# ── Vector packing ────────────────────────────────────────────────


def pack_vector(v: np.ndarray | list[float]) -> bytes:
    """Pack a float list/array into a float32 BLOB for SQLite."""
    if isinstance(v, np.ndarray):
        return bytes(v.astype(np.float32).tobytes())
    return struct.pack(f"{len(v)}f", *v)


# ── Chunk vector loading ─────────────────────────────────────────


def load_chunk_vectors(
    book_id: int,
    num_chunks: int,
    model_name: str,
    st_model: SentenceTransformer,
    chunk_texts: list[str],
) -> np.ndarray:
    """Load cached chunk embeddings from .npy or compute on the fly.

    Checks VECTORS_DIR for a cached {model}_{dataset}_docs.npy file (produced by
    the benchmark harness prep-vectors step). If unavailable, computes embeddings
    using the provided SentenceTransformer model.
    """
    dim = EMBEDDING_MODELS[model_name]["dim"]

    # Try cached .npy using the dataset name mapping
    dataset_name = BOOK_ID_TO_DATASET.get(book_id)
    if dataset_name:
        npy_path = VECTORS_DIR / f"{model_name}_{dataset_name}_docs.npy"
        if npy_path.exists():
            vectors = np.load(str(npy_path))
            if vectors.shape[0] != num_chunks:
                log.warning(
                    "  Cached vector count mismatch: %d vectors vs %d chunks "
                    "(chunk sizes changed). Recomputing...",
                    vectors.shape[0],
                    num_chunks,
                )
            elif vectors.shape[1] != dim:
                log.warning(
                    "  Cached vector dim mismatch: %d vs expected %d. Recomputing...",
                    vectors.shape[1],
                    dim,
                )
            else:
                log.info(
                    "  Loaded cached vectors (%d x %d) from %s",
                    vectors.shape[0],
                    vectors.shape[1],
                    npy_path.name,
                )
                return vectors

    # Compute embeddings on the fly
    log.info("  Computing %d chunk embeddings with %s (dim=%d)...", num_chunks, model_name, dim)
    vectors = st_model.encode(chunk_texts, show_progress_bar=True, normalize_embeddings=True)
    vectors = vectors.astype(np.float32)
    assert vectors.shape == (num_chunks, dim), (
        f"Computed vector shape mismatch: {vectors.shape} vs expected ({num_chunks}, {dim})"
    )
    return vectors


# ── Jaro-Winkler similarity (pure Python) ────────────────────────


def jaro_winkler(s1: str, s2: str) -> float:
    """Compute Jaro-Winkler similarity between two strings."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0

    s1_matched = [False] * len1
    s2_matched = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        lo = max(0, i - match_dist)
        hi = min(i + match_dist + 1, len2)
        for j in range(lo, hi):
            if s2_matched[j] or s1[i] != s2[j]:
                continue
            s1_matched[i] = True
            s2_matched[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matched[i]:
            continue
        while not s2_matched[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3.0

    # Winkler prefix bonus (up to 4 chars)
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1.0 - jaro)


# ── GLiNER char-span to spaCy token-span converter ───────────────


def char_span_to_token_span(doc: spacy.tokens.Doc, char_start: int, char_end: int) -> tuple[int, int] | None:
    """Convert character offsets to spaCy token indices (inclusive start, exclusive end).

    Returns None if the span doesn't align with token boundaries.
    """
    span = doc.char_span(char_start, char_end, alignment_mode="expand")
    if span is None:
        return None
    return (span.start, span.end)


# ── Extension loading ─────────────────────────────────────────────


def load_muninn(conn: sqlite3.Connection) -> None:
    """Load the muninn extension into a SQLite connection."""
    conn.enable_load_extension(True)
    conn.load_extension(MUNINN_PATH)


# ── Size formatting ──────────────────────────────────────────────


def fmt_size(size: int | float) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
