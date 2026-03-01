"""Phase 3: Chunks — split event messages into searchable chunks with FTS."""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

from benchmarks.demo_builder.common import ProgressTracker
from benchmarks.sessions_demo.constants import CHUNK_MAX_CHARS, CHUNK_MIN_CHARS

if TYPE_CHECKING:
    from benchmarks.sessions_demo.build import PhaseContext

log = logging.getLogger(__name__)


def _split_into_chunks(text: str, max_chars: int, min_chars: int) -> list[tuple[str, int]]:
    """Split text into paragraph-based chunks with offset tracking.

    Tries to split on paragraph boundaries (double newlines). If a paragraph
    exceeds max_chars, splits on single newlines, then on sentence boundaries.
    Returns list of (chunk_text, char_offset) tuples.
    """
    if not text or len(text) < min_chars:
        return [(text, 0)] if text else []

    chunks: list[tuple[str, int]] = []
    paragraphs = text.split("\n\n")

    offset = 0
    current_chunk = ""
    current_offset = 0

    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            # Account for the \n\n separator
            offset += 2
            continue

        if not current_chunk:
            current_chunk = para
            current_offset = offset
        elif len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk += "\n\n" + para
        else:
            # Emit current chunk if it's big enough
            if len(current_chunk) >= min_chars:
                chunks.append((current_chunk, current_offset))
            current_chunk = para
            current_offset = offset

        # Move offset past this paragraph + separator
        offset += len(para) + (2 if i < len(paragraphs) - 1 else 0)

    # Emit final chunk
    if current_chunk:
        if len(current_chunk) >= min_chars or not chunks:
            chunks.append((current_chunk, current_offset))
        elif chunks:
            # Merge tiny trailing chunk into previous
            prev_text, prev_offset = chunks[-1]
            chunks[-1] = (prev_text + "\n\n" + current_chunk, prev_offset)

    return chunks


class PhaseChunks:
    """Split event message_content into chunks with FTS5 index.

    Creates event_message_chunks table with chunk_id, event_id, text,
    and chunk_offset. Creates FTS5 index on chunk text.
    """

    @property
    def name(self) -> str:
        return "chunks"

    def is_stale(self, conn: sqlite3.Connection) -> bool:
        """Return True if any events with content have not been chunked yet."""
        try:
            pending = conn.execute("""
                SELECT COUNT(*) FROM events e
                WHERE e.message_content IS NOT NULL
                  AND e.message_content != ''
                  AND e.id NOT IN (SELECT DISTINCT event_id FROM event_message_chunks)
            """).fetchone()[0]
            return pending > 0
        except sqlite3.OperationalError:
            return True

    def restore_ctx(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        """Restore chunk counts from DB when phase is skipped."""
        try:
            total = conn.execute("SELECT count(*) FROM event_message_chunks").fetchone()[0]
            ctx.chunks_created = 0  # nothing created this run
            ctx.num_chunks = total
        except sqlite3.OperationalError:
            pass

    def setup(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS event_message_chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                chunk_offset INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_event_id
            ON event_message_chunks(event_id)
        """)

        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS event_message_chunks_fts
            USING fts5(text, content=event_message_chunks, content_rowid=chunk_id)
        """)

        # Triggers to keep FTS in sync
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON event_message_chunks BEGIN
                INSERT INTO event_message_chunks_fts(rowid, text)
                VALUES (new.chunk_id, new.text);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON event_message_chunks BEGIN
                INSERT INTO event_message_chunks_fts(event_message_chunks_fts, rowid, text)
                VALUES('delete', old.chunk_id, old.text);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON event_message_chunks BEGIN
                INSERT INTO event_message_chunks_fts(event_message_chunks_fts, rowid, text)
                VALUES('delete', old.chunk_id, old.text);
                INSERT INTO event_message_chunks_fts(rowid, text)
                VALUES (new.chunk_id, new.text);
            END
        """)
        conn.commit()
        log.info("Created event_message_chunks + FTS5 tables")

    def run(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        # Count events needing chunking for progress estimate
        total_events = conn.execute("""
            SELECT COUNT(*) FROM events e
            WHERE e.message_content IS NOT NULL
              AND e.message_content != ''
              AND e.id NOT IN (SELECT DISTINCT event_id FROM event_message_chunks)
        """).fetchone()[0]
        log.info("Chunking %d events (max_chars=%d)", total_events, CHUNK_MAX_CHARS)

        if total_events == 0:
            ctx.chunks_created = 0
            return

        cursor = conn.execute("""
            SELECT e.id, e.message_content
            FROM events e
            WHERE e.message_content IS NOT NULL
              AND e.message_content != ''
              AND e.id NOT IN (SELECT DISTINCT event_id FROM event_message_chunks)
        """)

        total_chunks = 0
        events_processed = 0
        tracker = ProgressTracker(total_events)
        for row in cursor:
            event_id = row[0]
            text = row[1]

            chunks = _split_into_chunks(text, CHUNK_MAX_CHARS, CHUNK_MIN_CHARS)
            for chunk_text, chunk_offset in chunks:
                conn.execute(
                    "INSERT INTO event_message_chunks (event_id, text, chunk_offset) VALUES (?, ?, ?)",
                    (event_id, chunk_text, chunk_offset),
                )
                total_chunks += 1

            events_processed += 1
            tracker.update()
            if tracker.should_log():
                log.info("  Chunked %s  (%d chunks so far)", tracker.report(), total_chunks)

        conn.commit()
        ctx.chunks_created = total_chunks
        ctx.num_chunks = total_chunks  # alias for demo_builder KG phase compat
        log.info("Created %d chunks from %d events", total_chunks, events_processed)

    def teardown(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        # ── viz-compatible chunks table ───────────────────────────────
        # Demo_builder KG phases (NER, RE, entity_resolution) read from a
        # table named `chunks` with (chunk_id, text). We create a real table
        # (not a VIEW) because PhaseNER declares entities.chunk_id with
        # FOREIGN KEY REFERENCES chunks(chunk_id), and SQLite's FK enforcement
        # cannot resolve foreign keys against views — only real tables.
        conn.execute("CREATE TABLE IF NOT EXISTS chunks (chunk_id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
        conn.execute("INSERT OR IGNORE INTO chunks SELECT chunk_id, text FROM event_message_chunks")

        # chunks_fts is the viz-expected FTS companion for chunks_vec.
        # Mirrors demo_builder: points at the `chunks` real table.
        # Rebuilt once here rather than via triggers (build pipeline, not incremental KG).
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content=chunks, content_rowid=chunk_id)"
        )
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        conn.commit()
        log.info("Created chunks table and chunks_fts FTS5 (%d entries)", ctx.num_chunks)

    def __call__(self, conn: sqlite3.Connection, ctx: PhaseContext) -> None:
        self.setup(conn, ctx)
        self.run(conn, ctx)
        self.teardown(conn, ctx)
