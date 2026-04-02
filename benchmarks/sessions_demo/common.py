"""Shared utility functions for the sessions_demo package.

Contains only the utilities that sessions_demo phases actually need.
This module is independent of demo_builder.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import huggingface_hub.constants
import spacy.tokens

log = logging.getLogger(__name__)

# ── Path constants ────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MUNINN_PATH = str(PROJECT_ROOT / "build" / "muninn")


# ── Extension loading ─────────────────────────────────────────────


def load_muninn(conn: sqlite3.Connection) -> None:
    """Load the muninn extension into a SQLite connection."""
    conn.enable_load_extension(True)
    conn.load_extension(MUNINN_PATH)


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
        self._min_interval_s = min_interval_s
        self._done = 0
        self._start = time.monotonic()
        self._last_log_at = 0
        self._last_log_time = self._start
        self._speed: float = 0.0
        self._output_total: int = 0
        self._output_last_log: int = 0
        self._output_window: int = 0

    def update(self, n: int = 1) -> None:
        """Advance the INPUT counter by n."""
        self._done += n

    def record_output(self, n: int) -> None:
        """Record n OUTPUT units produced since the last record_output call."""
        self._output_total += n

    def should_log(self) -> bool:
        """Return True (and capture speed + output window samples) every `window` INPUT items,
        or when min_interval_s has elapsed."""
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
        """Return a formatted progress string."""
        now = time.monotonic()
        elapsed = now - self._start
        done = self._done
        remaining = max(0, self._total - done)
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
    is frozen at module load time. We patch the module dict directly.
    """
    saved = huggingface_hub.constants.HF_HUB_OFFLINE
    huggingface_hub.constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        huggingface_hub.constants.HF_HUB_OFFLINE = saved


# ── GLiNER char-span to spaCy token-span converter ───────────────


def char_span_to_token_span(doc: spacy.tokens.Doc, char_start: int, char_end: int) -> tuple[int, int] | None:
    """Convert character offsets to spaCy token indices (inclusive start, exclusive end).

    Returns None if the span doesn't align with token boundaries.
    """
    span = doc.char_span(char_start, char_end, alignment_mode="expand")
    if span is None:
        return None
    return (span.start, span.end)
