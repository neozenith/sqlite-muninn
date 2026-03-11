"""Shared utility functions used across the demo_builder package.

This module is imported by phases.py (which has ML deps), so top-level
numpy/spacy/struct imports are acceptable here — they are only loaded
when phases.py is imported, which happens inside _cmd_build().
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import huggingface_hub.constants
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
                    "  Cached vector count mismatch: %d vectors vs %d chunks (chunk sizes changed). Recomputing...",
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


# ── Progress tracking ────────────────────────────────────────────


def _fmt_elapsed(secs: float) -> str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class ProgressTracker:
    """Rolling-window speed, elapsed time, and ETA tracker.

    Call update(n) for every INPUT item or batch processed — this drives
    the speed, ETA, and done/total counters. Optionally call record_output(n)
    to track produced OUTPUT units (e.g., entities, relations, chunks); these
    are shown in report() but never influence speed or ETA calculations.

    Call should_log() to check whether a new window of `window` INPUT items
    has been crossed — it returns True and captures a speed sample when the
    threshold is hit. Call report() to get a formatted progress string.

    Speed is computed over the last completed window interval (not
    cumulative), giving a responsive view of current throughput.
    """

    def __init__(self, total: int, window: int = 200, min_interval_s: float = 0.0) -> None:
        self._total = total
        self._window = window
        self._min_interval_s = min_interval_s  # if > 0, also log when this many seconds elapsed
        self._done = 0
        self._start = time.monotonic()
        self._last_log_at = 0
        self._last_log_time = self._start
        self._speed: float = 0.0  # input items/sec from last window interval
        self._output_total: int = 0  # cumulative output units produced
        self._output_last_log: int = 0  # output total at last window boundary
        self._output_window: int = 0  # output units in the last completed window

    def update(self, n: int = 1) -> None:
        """Advance the INPUT counter by n. Drives speed, ETA, and done/total."""
        self._done += n

    def record_output(self, n: int) -> None:
        """Record n OUTPUT units produced since the last record_output call.

        Output counts are shown in report() for informational purposes only —
        they never affect speed or ETA, which are always input-based.
        """
        self._output_total += n

    def should_log(self) -> bool:
        """Return True (and capture speed + output window samples) every `window` INPUT items,
        or when min_interval_s has elapsed (useful for slow per-item operations like LLM calls)."""
        time_trigger = False
        if self._min_interval_s > 0 and self._done > self._last_log_at:
            time_trigger = (time.monotonic() - self._last_log_time) >= self._min_interval_s
        if self._done - self._last_log_at >= self._window or time_trigger:
            now = time.monotonic()
            items_in_window = self._done - self._last_log_at
            time_in_window = now - self._last_log_time
            if time_in_window > 0:
                self._speed = items_in_window / time_in_window
            self._output_window = self._output_total - self._output_last_log
            self._output_last_log = self._output_total
            self._last_log_at = self._done
            self._last_log_time = now
            return True
        return False

    def report(self) -> str:
        """Return a formatted progress string: in:done/total | speed | elapsed | ETA | out:total (+window)."""
        now = time.monotonic()
        elapsed = now - self._start
        done = self._done
        remaining = max(0, self._total - done)
        # Speed and ETA are always based on INPUT units — output rate is irrelevant.
        speed = self._speed if self._speed > 0 else (done / elapsed if elapsed > 0 else 0.0)
        if remaining == 0:
            eta_str = "done"
        elif speed > 0:
            eta_str = _fmt_elapsed(remaining / speed)
        else:
            eta_str = "?"
        return (
            f"{done:,}/{self._total:,}  |  {speed:.1f} in/s  |  elapsed:{_fmt_elapsed(elapsed)}"
            f"  |  ETA:{eta_str}  |  out:{self._output_total:,} (+{self._output_window:,})"
        )


# ── Offline model loading ─────────────────────────────────────────


@contextmanager
def offline_mode() -> Generator[None, None, None]:
    """Patch huggingface_hub into offline mode for the duration of the block.

    os.environ["HF_HUB_OFFLINE"] is ineffective after import — the constant
    is frozen at module load time. We patch the module dict directly so that
    is_offline_mode() returns True, forcing local_files_only=True throughout
    transformers' AutoTokenizer/AutoModel loading chain.
    """
    saved = huggingface_hub.constants.HF_HUB_OFFLINE
    huggingface_hub.constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        huggingface_hub.constants.HF_HUB_OFFLINE = saved


def _read_backbone(snapshot_path: str) -> str:
    """Read the backbone encoder repo ID from a GLiNER or GLiREL snapshot config.

    Both store the backbone as "model_name" in their config JSON.
    The backbone repo must be downloaded separately — it is not bundled.
    """
    for config_name in ("gliner_config.json", "glirel_config.json", "config.json"):
        config_file = Path(snapshot_path) / config_name
        if config_file.exists():
            config = json.loads(config_file.read_text(encoding="utf-8"))
            if backbone := config.get("model_name"):
                return str(backbone)
    raise ValueError(f"Could not find model_name in any config at {snapshot_path}")


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
