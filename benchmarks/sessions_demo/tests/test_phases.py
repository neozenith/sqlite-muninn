"""Unit tests for sessions_demo phases and constants.

Tests cover pure logic that does not require the muninn extension,
ML models, or real JSONL files.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator

import pytest

from typing import Any

from benchmarks.sessions_demo.cache import _compute_event_costs, _message_kind, model_family_from_id
from benchmarks.sessions_demo.constants import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    PHASE_NAMES,
)
from benchmarks.sessions_demo.phases import Phase, default_phases
from benchmarks.sessions_demo.phases.chunks import PhaseChunks, _split_into_chunks

# ── _split_into_chunks ────────────────────────────────────────────


def test_split_empty_string() -> None:
    assert _split_into_chunks("", 1200, 100) == []


def test_split_short_text_below_min() -> None:
    result = _split_into_chunks("hi", 1200, 100)
    assert result == [("hi", 0)]


def test_split_single_chunk() -> None:
    text = "A" * 500
    result = _split_into_chunks(text, 1200, 100)
    assert len(result) == 1
    assert result[0] == (text, 0)


def test_split_respects_max_chars() -> None:
    para_a = "A" * 700
    para_b = "B" * 700
    text = f"{para_a}\n\n{para_b}"
    result = _split_into_chunks(text, 1200, 100)
    # Each paragraph is 700 chars; combined 1402 > max 1200 → must split
    assert len(result) == 2
    assert para_a in result[0][0]
    assert para_b in result[1][0]


def test_split_merges_tiny_trailing_chunk() -> None:
    big = "A" * 600
    tiny = "x" * 50  # below CHUNK_MIN_CHARS=100
    text = f"{big}\n\n{tiny}"
    result = _split_into_chunks(text, 1200, 100)
    # Tiny trailing chunk should be merged into previous chunk
    assert len(result) == 1
    assert tiny in result[0][0]


def test_split_preserves_offsets() -> None:
    para_a = "First paragraph."
    sep = "\n\n"
    para_b = "Second paragraph, long enough " + "x" * 200
    text = para_a + sep + para_b
    result = _split_into_chunks(text, 1200, 10)
    assert result[0][1] == 0  # first chunk always starts at 0


# ── PhaseChunks._type_filter ──────────────────────────────────────


def test_type_filter_human_alias() -> None:
    phase = PhaseChunks(message_types=["human"])
    sql, params = phase._type_filter()
    assert "msg_kind = 'human'" in sql
    assert params == []


def test_type_filter_single_type() -> None:
    phase = PhaseChunks(message_types=["assistant"])
    sql, params = phase._type_filter()
    assert "IN (?)" in sql
    assert params == ["assistant"]


def test_type_filter_multiple_types() -> None:
    phase = PhaseChunks(message_types=["user", "assistant"])
    sql, params = phase._type_filter()
    assert "IN (?,?)" in sql
    assert params == ["user", "assistant"]


# ── default_phases structure ──────────────────────────────────────


def test_default_phases_count() -> None:
    phases = default_phases()
    assert len(phases) == 13


def test_default_phases_names_match_constants() -> None:
    phases = default_phases()
    names = [p.name for p in phases]
    assert names == PHASE_NAMES


def test_default_phases_all_have_name() -> None:
    for phase in default_phases():
        assert isinstance(phase.name, str)
        assert len(phase.name) > 0


def test_default_phases_are_callable() -> None:
    for phase in default_phases():
        assert callable(phase)


def test_default_phases_legacy_models() -> None:
    phases = default_phases(legacy_models=True)
    assert len(phases) == 13
    names = [p.name for p in phases]
    assert names == PHASE_NAMES


# ── Phase ABC inheritance ─────────────────────────────────────────


def test_local_phases_are_phase_subclasses() -> None:
    from benchmarks.sessions_demo.phases.chunks import PhaseChunks
    from benchmarks.sessions_demo.phases.chunks_vec import PhaseChunksVec
    from benchmarks.sessions_demo.phases.ingest import PhaseIngest

    assert issubclass(PhaseIngest, Phase)
    assert issubclass(PhaseChunks, Phase)
    assert issubclass(PhaseChunksVec, Phase)


# ── PhaseChunks.is_stale with in-memory DB ───────────────────────


@pytest.fixture()  # type: ignore[untyped-decorator]
def conn_with_events() -> Generator[sqlite3.Connection, None, None]:
    """In-memory SQLite with events + event_message_chunks tables."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            event_type TEXT,
            msg_kind TEXT,
            message_content TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE event_message_chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            chunk_offset INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    yield conn
    conn.close()


def test_is_stale_true_when_no_events_chunked(conn_with_events: sqlite3.Connection) -> None:
    conn_with_events.execute(
        "INSERT INTO events(id, event_type, msg_kind, message_content) VALUES (1, 'user', 'human', 'hello world')"
    )
    conn_with_events.commit()
    phase = PhaseChunks(message_types=["human"])
    assert phase.is_stale(conn_with_events) is True


def test_is_stale_false_when_all_events_chunked(conn_with_events: sqlite3.Connection) -> None:
    conn_with_events.execute(
        "INSERT INTO events(id, event_type, msg_kind, message_content) VALUES (1, 'user', 'human', 'hello world')"
    )
    conn_with_events.execute(
        "INSERT INTO event_message_chunks(event_id, text, chunk_offset) VALUES (1, 'hello world', 0)"
    )
    conn_with_events.commit()
    phase = PhaseChunks(message_types=["human"])
    assert phase.is_stale(conn_with_events) is False


def test_is_stale_true_when_table_missing() -> None:
    conn = sqlite3.connect(":memory:")
    phase = PhaseChunks(message_types=["human"])
    assert phase.is_stale(conn) is True
    conn.close()


# ── Constants sanity checks ───────────────────────────────────────


# ── _message_kind ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "event_type,is_meta,content,expected",
    [
        ("user", False, "hello", "human"),
        ("user", False, [{"type": "text", "text": "hi"}], "user_text"),
        ("user", True, "injected", "meta"),
        ("user", False, [{"type": "tool_result", "content": "ok"}], "tool_result"),
        ("user", False, "<task-notification>\n<task-id>x</task-id>", "task_notification"),
        ("assistant", False, [{"type": "tool_use", "name": "Bash"}], "tool_use"),
        ("assistant", False, [{"type": "text", "text": "answer"}], "assistant_text"),
        ("assistant", False, [{"type": "thinking", "thinking": "pondering"}], "thinking"),
        ("progress", False, None, "other"),
        ("system", False, None, "other"),
    ],
)
def test_message_kind(event_type: str, is_meta: bool, content: Any, expected: str) -> None:
    assert _message_kind(event_type, is_meta, content) == expected


# ── Cost helpers ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("claude-opus-4-6", "opus"),
        ("claude-sonnet-4-6", "sonnet"),
        ("claude-haiku-4-5-20251001", "haiku"),
        ("gpt-4o", "unknown"),
        (None, "unknown"),
    ],
)
def test_model_family_from_id(model_id: str | None, expected: str) -> None:
    assert model_family_from_id(model_id) == expected


def test_compute_event_costs_sonnet_matches_formula() -> None:
    # sonnet: input=3.0, output=15.0, cache_read=0.3, cache_write=3.75 per Mtok
    rate, billable, cost = _compute_event_costs("claude-sonnet-4-6", 1000, 500, 2000, 400)
    # billable = 1000 + 500*5 + 2000*0.1 + 400*1.25 = 1000 + 2500 + 200 + 500 = 4200
    assert rate == 3.0
    assert billable == pytest.approx(4200.0)
    # cost = 4200 * 3.0 / 1e6 = 0.0126
    assert cost == pytest.approx(0.0126, rel=1e-6)


def test_compute_event_costs_unknown_model_returns_zero() -> None:
    assert _compute_event_costs(None, 1000, 500, 2000, 400) == (0.0, 0.0, 0.0)
    assert _compute_event_costs("gpt-4o", 1000, 500, 2000, 400) == (0.0, 0.0, 0.0)


# ── Constants sanity checks ───────────────────────────────────────


def test_chunk_max_chars_safe_for_gliner() -> None:
    # Must be ≤ 1200 per scripts/kg_chunk_size_fix.py empirical finding
    assert CHUNK_MAX_CHARS <= 1200


def test_chunk_min_less_than_max() -> None:
    assert CHUNK_MIN_CHARS < CHUNK_MAX_CHARS
