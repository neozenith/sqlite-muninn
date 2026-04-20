"""CacheManager — incremental JSONL ingestion into a SQLite cache.

Discovers Claude Code session JSONL files from ~/.claude/projects/,
parses events, builds parent-child edge relationships, and maintains
aggregate project/session tables.

Copied from introspect_sessions.py CacheManager with the schema inlined
for self-containment.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.sessions_demo.constants import PROJECTS_PATH, SCHEMA_VERSION

log = logging.getLogger(__name__)

# ── Schema DDL ────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Source files table: tracks all parsed JSONL files
CREATE TABLE IF NOT EXISTS source_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT UNIQUE NOT NULL,
    mtime REAL NOT NULL,
    size_bytes INTEGER NOT NULL,
    line_count INTEGER NOT NULL,
    last_ingested_at TEXT NOT NULL,
    project_id TEXT NOT NULL,
    session_id TEXT,  -- NULL for orphan agent files
    file_type TEXT NOT NULL CHECK (file_type IN ('main_session', 'subagent', 'agent_root'))
);

CREATE INDEX IF NOT EXISTS idx_source_files_project ON source_files(project_id);
CREATE INDEX IF NOT EXISTS idx_source_files_session ON source_files(session_id);
CREATE INDEX IF NOT EXISTS idx_source_files_mtime ON source_files(mtime);

-- Projects table: aggregated project metadata
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT UNIQUE NOT NULL,
    first_activity TEXT,
    last_activity TEXT,
    session_count INTEGER DEFAULT 0,
    event_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_projects_last_activity ON projects(last_activity);

-- Sessions table: aggregated session metadata
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    first_timestamp TEXT,
    last_timestamp TEXT,
    event_count INTEGER DEFAULT 0,
    subagent_count INTEGER DEFAULT 0,
    total_input_tokens INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    total_cache_read_tokens INTEGER DEFAULT 0,
    total_cache_creation_tokens INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,
    UNIQUE(project_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_id);
CREATE INDEX IF NOT EXISTS idx_sessions_last_timestamp ON sessions(last_timestamp);

-- Events table: all parsed events
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT,
    parent_uuid TEXT,
    prompt_id TEXT,  -- joins subagent first events to parent tool_result/tool_use chain
    event_type TEXT NOT NULL,
    msg_kind TEXT,  -- derived: human|user_text|assistant_text|tool_use|tool_result|thinking|meta|task_notification
    timestamp TEXT,
    timestamp_local TEXT,
    session_id TEXT,
    project_id TEXT NOT NULL,
    is_sidechain INTEGER DEFAULT 0,
    agent_id TEXT,
    agent_slug TEXT,
    message_role TEXT,
    message_content TEXT,  -- Plain text for FTS
    message_content_json TEXT,  -- Original JSON structure
    model_id TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_creation_tokens INTEGER DEFAULT 0,
    cache_5m_tokens INTEGER DEFAULT 0,
    token_rate REAL DEFAULT 0.0,         -- input $/Mtok for this event's model
    billable_tokens REAL DEFAULT 0.0,    -- weighted input-equivalent token count
    total_cost_usd REAL DEFAULT 0.0,     -- pre-computed at ingest
    source_file_id INTEGER NOT NULL REFERENCES source_files(id) ON DELETE CASCADE,
    line_number INTEGER NOT NULL,
    raw_json TEXT NOT NULL               -- intentionally empty; source-of-truth is the JSONL file
);

CREATE INDEX IF NOT EXISTS idx_events_uuid ON events(uuid);
CREATE INDEX IF NOT EXISTS idx_events_parent_uuid ON events(parent_uuid);
CREATE INDEX IF NOT EXISTS idx_events_prompt_id ON events(prompt_id);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_project ON events(project_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_msg_kind ON events(msg_kind);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_source_file ON events(source_file_id);
CREATE INDEX IF NOT EXISTS idx_events_project_session ON events(project_id, session_id);
CREATE INDEX IF NOT EXISTS idx_events_session_type ON events(session_id, event_type);
CREATE INDEX IF NOT EXISTS idx_events_session_uuid ON events(session_id, uuid);

-- Covering index for analytical GROUP BY queries
CREATE INDEX IF NOT EXISTS idx_events_covering ON events(
    timestamp, project_id, session_id, model_id,
    input_tokens, output_tokens,
    cache_read_tokens, cache_creation_tokens, total_cost_usd
);

-- FTS5 virtual table for full-text search on message content
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    message_content,
    content='events',
    content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
    INSERT INTO events_fts(rowid, message_content) VALUES (new.id, new.message_content);
END;

CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
    INSERT INTO events_fts(events_fts, rowid, message_content) VALUES('delete', old.id, old.message_content);
END;

CREATE TRIGGER IF NOT EXISTS events_au AFTER UPDATE ON events BEGIN
    INSERT INTO events_fts(events_fts, rowid, message_content) VALUES('delete', old.id, old.message_content);
    INSERT INTO events_fts(rowid, message_content) VALUES (new.id, new.message_content);
END;

-- Cache metadata table
CREATE TABLE IF NOT EXISTS cache_metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Event edges table: parent-child relationships for graph traversal
CREATE TABLE IF NOT EXISTS event_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    event_uuid TEXT NOT NULL,
    parent_event_uuid TEXT NOT NULL,
    source_file_id INTEGER NOT NULL REFERENCES source_files(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_event_edges_forward ON event_edges(project_id, session_id, event_uuid);
CREATE INDEX IF NOT EXISTS idx_event_edges_reverse ON event_edges(project_id, session_id, parent_event_uuid);
CREATE INDEX IF NOT EXISTS idx_event_edges_source_file ON event_edges(source_file_id);

CREATE INDEX IF NOT EXISTS idx_source_files_project_session ON source_files(project_id, session_id);

-- =====================================================================
-- Dimensional aggregation table (star schema)
-- =====================================================================
-- Pre-aggregated measures at multiple time granularities in a single
-- table, discriminated by the `granularity` column. Maintained
-- incrementally via refresh_aggregates_for_range() after each ingest.
-- Grain: (granularity, time_bucket, project_id, session_id, model_id).
-- session_id / model_id use '' sentinel for NULL (SQLite PK treats
-- NULLs as non-equal which would break uniqueness).
-- =====================================================================

CREATE TABLE IF NOT EXISTS agg (
    granularity TEXT NOT NULL,  -- 'hourly', 'daily', 'weekly', 'monthly'
    time_bucket TEXT NOT NULL,
    project_id TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    model_id TEXT NOT NULL DEFAULT '',
    event_count INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    billable_tokens REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (granularity, time_bucket, project_id, session_id, model_id)
);
CREATE INDEX IF NOT EXISTS idx_agg_granularity_time
    ON agg(granularity, time_bucket);
CREATE INDEX IF NOT EXISTS idx_agg_granularity_project_time
    ON agg(granularity, project_id, time_bucket);
"""


class CacheManager:
    """Manages the SQLite cache for session data."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def destroy(self) -> None:
        """Delete the database file entirely for a clean rebuild.

        Preferred over DROP TABLE when the DB contains virtual tables
        (HNSW, FTS5) that require extension modules to drop.
        """
        log.info("Destroying database: %s", self.db_path)
        self.close()
        if self.db_path.exists():
            self.db_path.unlink()
            # Also remove WAL/SHM files if present
            for suffix in ("-wal", "-shm"):
                wal = self.db_path.parent / (self.db_path.name + suffix)
                if wal.exists():
                    wal.unlink()

    def init_schema(self) -> None:
        """Initialize database schema.

        If the DB already exists with a stale schema version, destroys it
        first to avoid CREATE INDEX failures on missing columns.
        """
        if self.db_path.exists() and self.needs_rebuild():
            log.info("Schema version mismatch — destroying stale database")
            self.destroy()
        log.info("Initializing cache schema...")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.execute(
            "INSERT OR REPLACE INTO cache_metadata (key, value) VALUES (?, ?)",
            ("schema_version", SCHEMA_VERSION),
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO cache_metadata (key, value) VALUES (?, ?)",
            ("created_at", datetime.now(UTC).isoformat()),
        )
        self.conn.commit()
        log.info("Cache initialized at %s", self.db_path)

    def needs_rebuild(self) -> bool:
        """Check if the schema version requires a rebuild."""
        try:
            row = self.conn.execute("SELECT value FROM cache_metadata WHERE key = 'schema_version'").fetchone()
            if row is None:
                return True
            return bool(row[0] != SCHEMA_VERSION)
        except sqlite3.OperationalError:
            return True

    def clear(self) -> None:
        """Clear all cached data. Safe to call even if tables don't exist."""
        log.info("Clearing cache...")
        tables_to_clear = [
            "agg",
            "event_edges",
            "events",
            "sessions",
            "projects",
            "source_files",
            "events_fts",
        ]
        for table in tables_to_clear:
            try:
                self.conn.execute(f"DELETE FROM {table}")  # noqa: S608
            except sqlite3.OperationalError:
                pass
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO cache_metadata (key, value) VALUES (?, ?)",
                ("last_cleared_at", datetime.now(UTC).isoformat()),
            )
        except sqlite3.OperationalError:
            pass
        self.conn.commit()
        log.info("Cache cleared")

    def get_status(self) -> dict[str, Any]:
        """Get cache status information."""
        cursor = self.conn.cursor()

        file_count = cursor.execute("SELECT COUNT(*) FROM source_files").fetchone()[0]
        project_count = cursor.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        session_count = cursor.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        event_count = cursor.execute("SELECT COUNT(*) FROM events").fetchone()[0]

        edge_count = 0
        try:
            edge_count = cursor.execute("SELECT COUNT(*) FROM event_edges").fetchone()[0]
        except sqlite3.OperationalError:
            pass

        created_at = cursor.execute("SELECT value FROM cache_metadata WHERE key = 'created_at'").fetchone()
        last_update = cursor.execute("SELECT value FROM cache_metadata WHERE key = 'last_update_at'").fetchone()

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "db_path": str(self.db_path),
            "db_size_bytes": db_size,
            "db_size_mb": round(db_size / (1024 * 1024), 2),
            "source_files": file_count,
            "projects": project_count,
            "sessions": session_count,
            "events": event_count,
            "event_edges": edge_count,
            "created_at": created_at[0] if created_at else None,
            "last_update_at": last_update[0] if last_update else None,
        }

    def discover_files(self, projects_path: Path | None = None) -> list[dict[str, Any]]:
        """Discover all JSONL files and classify them."""
        if projects_path is None:
            projects_path = PROJECTS_PATH

        files: list[dict[str, Any]] = []

        if not projects_path.exists():
            return files

        for project_dir in projects_path.iterdir():
            if not project_dir.is_dir():
                continue

            project_id = project_dir.name

            for jsonl_file in project_dir.rglob("*.jsonl"):
                rel_path = jsonl_file.relative_to(project_dir)
                parts = rel_path.parts

                file_info: dict[str, Any] = {
                    "filepath": str(jsonl_file),
                    "project_id": project_id,
                    "session_id": None,
                    "file_type": "unknown",
                }

                if len(parts) == 1:
                    filename = parts[0]
                    if filename.startswith("agent-"):
                        file_info["file_type"] = "agent_root"
                    else:
                        session_id = filename.replace(".jsonl", "")
                        file_info["session_id"] = session_id
                        file_info["file_type"] = "main_session"

                elif len(parts) >= 2 and "subagents" in parts:
                    session_id = parts[0]
                    file_info["session_id"] = session_id
                    file_info["file_type"] = "subagent"

                elif len(parts) == 2:
                    session_id = parts[0]
                    file_info["session_id"] = session_id
                    file_info["file_type"] = "subagent"

                files.append(file_info)

        return files

    def get_files_needing_update(self, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter files to only those that need updating."""
        cursor = self.conn.cursor()
        needs_update: list[dict[str, Any]] = []

        for file_info in files:
            filepath = file_info["filepath"]

            try:
                stat = os.stat(filepath)
                current_mtime = stat.st_mtime
                current_size = stat.st_size
            except OSError:
                continue

            cached = cursor.execute(
                "SELECT mtime, size_bytes FROM source_files WHERE filepath = ?",
                (filepath,),
            ).fetchone()

            if cached is None:
                file_info["mtime"] = current_mtime
                file_info["size_bytes"] = current_size
                file_info["reason"] = "new"
                needs_update.append(file_info)
            elif cached["mtime"] != current_mtime or cached["size_bytes"] != current_size:
                file_info["mtime"] = current_mtime
                file_info["size_bytes"] = current_size
                file_info["reason"] = "modified"
                needs_update.append(file_info)

        return needs_update

    def ingest_file(self, file_info: dict[str, Any]) -> int:
        """Ingest a single JSONL file into the cache. Returns event count."""
        filepath = file_info["filepath"]
        project_id = file_info["project_id"]
        session_id = file_info.get("session_id")
        file_type = file_info["file_type"]
        mtime = file_info["mtime"]
        size_bytes = file_info["size_bytes"]

        log.debug("Ingesting %s", filepath)

        cursor = self.conn.cursor()

        # Delete existing data for this file (if re-ingesting)
        existing = cursor.execute("SELECT id FROM source_files WHERE filepath = ?", (filepath,)).fetchone()
        if existing:
            cursor.execute("DELETE FROM event_edges WHERE source_file_id = ?", (existing[0],))
            cursor.execute("DELETE FROM events WHERE source_file_id = ?", (existing[0],))
            cursor.execute("DELETE FROM source_files WHERE id = ?", (existing[0],))

        # Parse file and count lines
        events_data: list[dict[str, Any]] = []
        line_count = 0
        detected_session_id = session_id

        try:
            with open(filepath, encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line_count = line_num
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # For agent_root files, extract sessionId from content
                    if file_type == "agent_root" and detected_session_id is None:
                        detected_session_id = raw.get("sessionId")

                    event = _parse_event_for_cache(raw, project_id, detected_session_id, line_num)
                    if event:
                        events_data.append(event)

        except (FileNotFoundError, PermissionError) as e:
            log.warning("Could not read %s: %s", filepath, e)
            return 0

        # Insert source file record
        cursor.execute(
            """INSERT INTO source_files
               (filepath, mtime, size_bytes, line_count, last_ingested_at,
                project_id, session_id, file_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                filepath,
                mtime,
                size_bytes,
                line_count,
                datetime.now(UTC).isoformat(),
                project_id,
                detected_session_id,
                file_type,
            ),
        )
        source_file_id = cursor.lastrowid

        # Insert events
        for event in events_data:
            cursor.execute(
                """INSERT INTO events
                   (uuid, parent_uuid, prompt_id, event_type, msg_kind,
                    timestamp, timestamp_local,
                    session_id, project_id, is_sidechain, agent_id, agent_slug,
                    message_role,
                    message_content, message_content_json, model_id,
                    input_tokens, output_tokens, cache_read_tokens,
                    cache_creation_tokens, cache_5m_tokens,
                    token_rate, billable_tokens, total_cost_usd,
                    source_file_id, line_number, raw_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["uuid"],
                    event["parent_uuid"],
                    event["prompt_id"],
                    event["event_type"],
                    event["msg_kind"],
                    event["timestamp"],
                    event["timestamp_local"],
                    detected_session_id,
                    project_id,
                    event["is_sidechain"],
                    event["agent_id"],
                    event["agent_slug"],
                    event["message_role"],
                    event["message_content"],
                    event["message_content_json"],
                    event["model_id"],
                    event["input_tokens"],
                    event["output_tokens"],
                    event["cache_read_tokens"],
                    event["cache_creation_tokens"],
                    event["cache_5m_tokens"],
                    event["token_rate"],
                    event["billable_tokens"],
                    event["total_cost_usd"],
                    source_file_id,
                    event["line_number"],
                    event["raw_json"],
                ),
            )

        # Insert event edges for parent-child relationships
        for event in events_data:
            if event["uuid"] and event["parent_uuid"]:
                cursor.execute(
                    """INSERT INTO event_edges
                       (project_id, session_id, event_uuid, parent_event_uuid,
                        source_file_id)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        project_id,
                        detected_session_id,
                        event["uuid"],
                        event["parent_uuid"],
                        source_file_id,
                    ),
                )

        return len(events_data)

    def rebuild_aggregates(self) -> None:
        """Rebuild projects and sessions tables from events."""
        log.info("Rebuilding aggregate tables...")
        cursor = self.conn.cursor()

        # Rebuild projects
        cursor.execute("DELETE FROM projects")
        cursor.execute("""
            INSERT INTO projects (project_id, first_activity, last_activity, session_count, event_count)
            SELECT
                project_id,
                MIN(timestamp) as first_activity,
                MAX(timestamp) as last_activity,
                COUNT(DISTINCT session_id) as session_count,
                COUNT(*) as event_count
            FROM events
            WHERE timestamp IS NOT NULL
            GROUP BY project_id
        """)

        # Rebuild sessions
        cursor.execute("DELETE FROM sessions")
        cursor.execute("""
            INSERT INTO sessions (
                session_id, project_id, first_timestamp, last_timestamp,
                event_count, subagent_count,
                total_input_tokens, total_output_tokens,
                total_cache_read_tokens, total_cache_creation_tokens
            )
            SELECT
                session_id,
                project_id,
                MIN(timestamp) as first_timestamp,
                MAX(timestamp) as last_timestamp,
                COUNT(*) as event_count,
                COUNT(DISTINCT agent_id) - 1 as subagent_count,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(cache_read_tokens) as total_cache_read_tokens,
                SUM(cache_creation_tokens) as total_cache_creation_tokens
            FROM events
            WHERE session_id IS NOT NULL
            GROUP BY project_id, session_id
        """)

        # Calculate costs per-event using model family pricing
        cursor.execute("""
            UPDATE sessions SET total_cost_usd = (
                SELECT COALESCE(SUM(
                    CASE
                        WHEN e.model_id LIKE '%opus%' THEN
                            (e.input_tokens / 1000000.0) * 15.0 +
                            (e.output_tokens / 1000000.0) * 75.0 +
                            (e.cache_read_tokens / 1000000.0) * 1.5 +
                            (e.cache_creation_tokens / 1000000.0) * 18.75
                        WHEN e.model_id LIKE '%sonnet%' THEN
                            (e.input_tokens / 1000000.0) * 3.0 +
                            (e.output_tokens / 1000000.0) * 15.0 +
                            (e.cache_read_tokens / 1000000.0) * 0.3 +
                            (e.cache_creation_tokens / 1000000.0) * 3.75
                        WHEN e.model_id LIKE '%haiku%' THEN
                            (e.input_tokens / 1000000.0) * 1.0 +
                            (e.output_tokens / 1000000.0) * 5.0 +
                            (e.cache_read_tokens / 1000000.0) * 0.1 +
                            (e.cache_creation_tokens / 1000000.0) * 1.25
                        ELSE
                            (e.input_tokens / 1000000.0) * 15.0 +
                            (e.output_tokens / 1000000.0) * 75.0 +
                            (e.cache_read_tokens / 1000000.0) * 1.5 +
                            (e.cache_creation_tokens / 1000000.0) * 18.75
                    END
                ), 0)
                FROM events e
                WHERE e.session_id = sessions.session_id
                  AND e.project_id = sessions.project_id
            )
        """)

        self.conn.commit()
        log.info("Aggregate tables rebuilt")

    # ------------------------------------------------------------------
    # Dimensional aggregates (agg_{hourly,daily,weekly,monthly})
    # ------------------------------------------------------------------

    _AGG_BUCKET_EXPRS: dict[str, str] = {
        "hourly": "strftime('%Y-%m-%dT%H:00:00', timestamp)",
        "daily": "date(timestamp)",
        "weekly": "date(timestamp, 'weekday 0', '-6 days')",
        "monthly": "strftime('%Y-%m-01', timestamp)",
    }

    def refresh_aggregates_for_range(
        self,
        start_bucket: str | None = None,
        end_bucket: str | None = None,
    ) -> dict[str, int]:
        """Refresh the agg table for all granularities in a time range (or fully)."""
        cursor = self.conn.cursor()
        counts: dict[str, int] = {}
        for granularity, bucket_expr in self._AGG_BUCKET_EXPRS.items():
            if start_bucket is None or end_bucket is None:
                cursor.execute("DELETE FROM agg WHERE granularity = ?", (granularity,))
                range_clause = ""
                range_params: tuple[str, ...] = ()
            else:
                cursor.execute(
                    "DELETE FROM agg WHERE granularity = ? AND time_bucket >= ? AND time_bucket <= ?",
                    (granularity, start_bucket, end_bucket),
                )
                range_clause = f"AND {bucket_expr} BETWEEN ? AND ?"
                range_params = (start_bucket, end_bucket)

            cursor.execute(
                f"""
                INSERT INTO agg (
                    granularity, time_bucket, project_id, session_id, model_id,
                    event_count,
                    input_tokens, output_tokens,
                    cache_read_tokens, cache_creation_tokens,
                    total_cost_usd, billable_tokens
                )
                SELECT
                    '{granularity}',
                    {bucket_expr} AS time_bucket,
                    project_id,
                    COALESCE(session_id, ''),
                    COALESCE(model_id, ''),
                    COUNT(*),
                    COALESCE(SUM(input_tokens), 0),
                    COALESCE(SUM(output_tokens), 0),
                    COALESCE(SUM(cache_read_tokens), 0),
                    COALESCE(SUM(cache_creation_tokens), 0),
                    COALESCE(SUM(total_cost_usd), 0.0),
                    COALESCE(SUM(billable_tokens), 0.0)
                FROM events
                WHERE timestamp IS NOT NULL
                  {range_clause}
                GROUP BY {bucket_expr}, project_id,
                         COALESCE(session_id, ''), COALESCE(model_id, '')
                """,  # noqa: S608
                range_params,
            )
            counts[granularity] = cursor.rowcount
        self.conn.commit()
        return counts

    def build_cross_agent_edges(self, session_id: str, project_id: str) -> int:
        """Bridge subagent-first events to parent tool_use events via prompt_id.

        Subagent JSONL first events have ``parentUuid=null``. Claude Code writes
        the same ``promptId`` onto both (a) the subagent's first user event and
        (b) the parent session's ``tool_result`` event, and the tool_result's
        ``parentUuid`` points at the ``tool_use`` that spawned the agent. We
        walk that chain and materialize a synthetic edge so graph traversal
        reaches across the main-session / subagent file boundary.

        Returns the number of new bridge edges inserted.
        """
        cursor = self.conn.cursor()

        subagent_starts = cursor.execute(
            """
            SELECT e.uuid, e.prompt_id, e.agent_id, e.source_file_id
            FROM events e
            WHERE e.session_id = ?
              AND e.parent_uuid IS NULL
              AND e.agent_id IS NOT NULL
              AND e.is_sidechain = 1
              AND e.prompt_id IS NOT NULL
            """,
            (session_id,),
        ).fetchall()

        created = 0
        for start in subagent_starts:
            exists = cursor.execute(
                "SELECT 1 FROM event_edges WHERE session_id = ? AND event_uuid = ?",
                (session_id, start["uuid"]),
            ).fetchone()
            if exists:
                continue

            parent_tool_use = cursor.execute(
                """
                SELECT tool_result.parent_uuid AS tool_use_uuid
                FROM events tool_result
                WHERE tool_result.session_id = ?
                  AND tool_result.agent_id IS NULL
                  AND tool_result.msg_kind = 'tool_result'
                  AND tool_result.prompt_id = ?
                  AND tool_result.parent_uuid IS NOT NULL
                LIMIT 1
                """,
                (session_id, start["prompt_id"]),
            ).fetchone()

            if parent_tool_use and parent_tool_use["tool_use_uuid"]:
                cursor.execute(
                    """
                    INSERT INTO event_edges
                        (project_id, session_id, event_uuid, parent_event_uuid, source_file_id)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        session_id,
                        start["uuid"],
                        parent_tool_use["tool_use_uuid"],
                        start["source_file_id"],
                    ),
                )
                created += 1

        if created:
            self.conn.commit()
        return created

    def update(self, projects_path: Path | None = None) -> dict[str, Any]:
        """Perform incremental update of the cache. Returns counts."""
        if projects_path is None:
            projects_path = PROJECTS_PATH

        log.info("Starting incremental cache update...")

        all_files = self.discover_files(projects_path)
        log.info("Discovered %d total files", len(all_files))

        files_to_update = self.get_files_needing_update(all_files)
        log.info("Found %d files needing update", len(files_to_update))

        if not files_to_update:
            log.info("Cache is up to date")
            return {"files_updated": 0, "events_added": 0}

        # Ingest files, tracking which sessions were touched so we can scope
        # the bridge-edge build and the dimensional aggregate refresh.
        total_events = 0
        affected_sessions: dict[str, str] = {}  # session_id -> project_id
        for file_info in files_to_update:
            events_added = self.ingest_file(file_info)
            total_events += events_added
            log.debug("  %s: %d events (%s)", file_info["filepath"], events_added, file_info.get("reason", "new"))
            sid = file_info.get("session_id")
            if sid:
                affected_sessions[sid] = file_info["project_id"]

        self.conn.commit()

        # Build cross-agent bridge edges for every touched session
        total_bridges = 0
        for sid, pid in affected_sessions.items():
            total_bridges += self.build_cross_agent_edges(sid, pid)
        if total_bridges:
            log.info("Created %d cross-agent bridge edges", total_bridges)

        # Rebuild projects+sessions roll-ups (cheap)
        self.rebuild_aggregates()

        # Dimensional agg_* tables: full rebuild when empty, otherwise
        # incremental by the timestamp window of ingested sessions.
        agg_empty = self.conn.execute("SELECT COUNT(*) FROM agg").fetchone()[0] == 0
        if agg_empty:
            log.info("Dimensional aggregates empty — doing full cold rebuild")
            self.refresh_aggregates_for_range()
        elif affected_sessions:
            session_ids = list(affected_sessions.keys())
            placeholders = ",".join("?" * len(session_ids))
            row = self.conn.execute(
                f"""
                SELECT MIN(timestamp), MAX(timestamp)
                FROM events
                WHERE timestamp IS NOT NULL
                  AND session_id IN ({placeholders})
                """,  # noqa: S608
                tuple(session_ids),
            ).fetchone()
            if row and row[0]:
                self.refresh_aggregates_for_range(str(row[0]), str(row[1]))

        self.conn.execute(
            "INSERT OR REPLACE INTO cache_metadata (key, value) VALUES (?, ?)",
            ("last_update_at", datetime.now(UTC).isoformat()),
        )
        self.conn.commit()

        log.info("Updated %d files, %d events", len(files_to_update), total_events)
        return {"files_updated": len(files_to_update), "events_added": total_events}


# ── Pricing / cost helpers ────────────────────────────────────────
#
# Costs are computed at ingestion time and stored directly on each event so
# downstream queries (and the agg_* dimensional tables) can SUM them without
# per-row CASE expressions. All model families share the same relative
# multipliers (output 5×, cache_read 0.1×, cache_write 1.25×), so token_rate
# alone is sufficient to reconstruct a family's full price shape.

PRICING: dict[str, dict[str, float]] = {
    "opus": {
        "input": 15.0,
        "output": 75.0,
        "cache_read_multiplier": 0.1,
        "cache_write_multiplier": 1.25,
    },
    "sonnet": {
        "input": 3.0,
        "output": 15.0,
        "cache_read_multiplier": 0.1,
        "cache_write_multiplier": 1.25,
    },
    "haiku": {
        "input": 1.0,
        "output": 5.0,
        "cache_read_multiplier": 0.1,
        "cache_write_multiplier": 1.25,
    },
}


def model_family_from_id(model_id: str | None) -> str:
    """Extract model family (opus/sonnet/haiku) from a full model ID string."""
    if model_id is None:
        return "unknown"
    model_lower = model_id.lower()
    for family in ("opus", "sonnet", "haiku"):
        if family in model_lower:
            return family
    return "unknown"


def _compute_event_costs(
    model_id: str | None,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
) -> tuple[float, float, float]:
    """Compute (token_rate, billable_tokens, total_cost_usd) for an event.

    Returns (0.0, 0.0, 0.0) for unknown models so the row still inserts cleanly.
    """
    family = model_family_from_id(model_id)
    pricing = PRICING.get(family)
    if pricing is None:
        return 0.0, 0.0, 0.0

    token_rate = pricing["input"]
    output_mult = pricing["output"] / pricing["input"]  # always 5.0
    cache_read_mult = pricing["cache_read_multiplier"]  # always 0.1
    cache_write_mult = pricing["cache_write_multiplier"]  # always 1.25

    billable = (
        input_tokens
        + output_tokens * output_mult
        + cache_read_tokens * cache_read_mult
        + cache_creation_tokens * cache_write_mult
    )
    return token_rate, round(billable, 4), round(billable * token_rate / 1_000_000, 8)


# ── Event parsing helpers (module-level functions) ────────────────


def _first_content_block_type(content: Any) -> str | None:
    """Derive the content shape for a message's content field.

    Returns:
        'string'      — raw string content (typically human-typed prompts)
        'text'        — list whose first block has type='text'
        'tool_use'    — list whose first block has type='tool_use'
        'tool_result' — list whose first block has type='tool_result'
        'thinking'    — list whose first block has type='thinking'
        other str     — first block's type value (catch-all)
        None          — content is None or empty list
    """
    if content is None:
        return None
    if isinstance(content, str):
        return "string"
    if isinstance(content, list) and content and isinstance(content[0], dict):
        return content[0].get("type")
    return None


def _extract_text_content(content: Any) -> str:
    """Extract plain text from message content for FTS."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    parts.append(block.get("thinking", ""))
                elif block.get("type") == "tool_use":
                    parts.append(f"[tool: {block.get('name', '')}]")
                    inp = block.get("input", {})
                    if isinstance(inp, dict):
                        for v in inp.values():
                            if isinstance(v, str):
                                parts.append(v[:500])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(filter(None, parts))

    return ""


def _message_kind(event_type: str, is_meta: bool, content: Any) -> str:
    """Classify an event into one of 9 fine-grained message kinds.

    Kinds:
        human              — user, not meta, string content (actual typed prompts)
        task_notification  — user, not meta, string starting with <task-notification>
        tool_result        — user, not meta, tool_result list
        user_text          — user, not meta, text/other list
        meta               — user, isMeta=true (system-injected context)
        assistant_text     — assistant, text list
        thinking           — assistant, thinking list
        tool_use           — assistant, tool_use list
        other              — progress / system / queue-operation / etc.
    """
    fct = _first_content_block_type(content)
    if event_type == "user":
        if is_meta:
            return "meta"
        if fct == "string":
            if isinstance(content, str) and content.lstrip().startswith("<task-notification>"):
                return "task_notification"
            return "human"
        if fct == "tool_result":
            return "tool_result"
        return "user_text"
    if event_type == "assistant":
        if fct == "thinking":
            return "thinking"
        if fct == "tool_use":
            return "tool_use"
        return "assistant_text"
    return "other"


def _parse_event_for_cache(
    raw: dict[str, Any],
    project_id: str,
    session_id: str | None,
    line_number: int,
) -> dict[str, Any] | None:
    """Parse a raw event dict for cache insertion."""
    event_type = raw.get("type", "")

    # Skip file-history-snapshot events
    if event_type == "file-history-snapshot":
        return None

    timestamp = raw.get("timestamp")
    uuid = raw.get("uuid")
    parent_uuid = raw.get("parentUuid")
    prompt_id = raw.get("promptId")
    is_sidechain = raw.get("isSidechain", False)
    agent_id = raw.get("agentId")
    agent_slug = raw.get("slug")

    is_meta = bool(raw.get("isMeta"))

    message = raw.get("message", {}) or {}
    message_role = message.get("role") if isinstance(message, dict) else None
    message_content_raw = message.get("content") if isinstance(message, dict) else None
    model_id = message.get("model") if isinstance(message, dict) else None

    # Sanitize: drop `signature` from thinking blocks (large base64 token, useless for analytics)
    if isinstance(message_content_raw, list):
        message_content_raw = [
            {k: v for k, v in block.items() if k != "signature"}
            if isinstance(block, dict) and block.get("type") == "thinking" and "signature" in block
            else block
            for block in message_content_raw
        ]

    msg_kind = _message_kind(event_type, is_meta, message_content_raw)
    message_content_text = _extract_text_content(message_content_raw)

    usage = message.get("usage", {}) if isinstance(message, dict) else {}
    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    cache_read_tokens = usage.get("cache_read_input_tokens", 0) or 0
    cache_creation_tokens = usage.get("cache_creation_input_tokens", 0) or 0
    cache_creation = usage.get("cache_creation", {}) or {}
    cache_5m_tokens = cache_creation.get("ephemeral_5m_input_tokens", 0) or 0

    # Pre-compute cost fields at ingest so agg_* rollups are a plain SUM().
    token_rate, billable_tokens, total_cost_usd = _compute_event_costs(
        model_id, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens
    )

    timestamp_local = None
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            local_dt = dt.astimezone()
            timestamp_local = local_dt.isoformat()
        except (ValueError, TypeError):
            pass

    return {
        "uuid": uuid,
        "parent_uuid": parent_uuid,
        "prompt_id": prompt_id,
        "event_type": event_type,
        "msg_kind": msg_kind,
        "timestamp": timestamp,
        "timestamp_local": timestamp_local,
        "is_sidechain": 1 if is_sidechain else 0,
        "agent_id": agent_id,
        "agent_slug": agent_slug,
        "message_role": message_role,
        "message_content": message_content_text,
        "message_content_json": json.dumps(message_content_raw) if message_content_raw else None,
        "model_id": model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_5m_tokens": cache_5m_tokens,
        "token_rate": token_rate,
        "billable_tokens": billable_tokens,
        "total_cost_usd": total_cost_usd,
        "line_number": line_number,
        # raw_json intentionally empty — the source-of-truth for the raw
        # payload is the JSONL file on disk (source_files.filepath + line_number).
        # Storing a duplicate copy costs 2+ GB and leaks thinking-block signatures.
        "raw_json": "",
    }
