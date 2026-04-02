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
    event_type TEXT NOT NULL,
    msg_kind TEXT,  -- derived: human|user_text|assistant_text|tool_use|tool_result|thinking|meta|task_notification|other
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
    source_file_id INTEGER NOT NULL REFERENCES source_files(id) ON DELETE CASCADE,
    line_number INTEGER NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_uuid ON events(uuid);
CREATE INDEX IF NOT EXISTS idx_events_parent_uuid ON events(parent_uuid);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_project ON events(project_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_msg_kind ON events(msg_kind);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_source_file ON events(source_file_id);
CREATE INDEX IF NOT EXISTS idx_events_project_session ON events(project_id, session_id);
CREATE INDEX IF NOT EXISTS idx_events_session_type ON events(session_id, event_type);
CREATE INDEX IF NOT EXISTS idx_events_session_uuid ON events(session_id, uuid);

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
                   (uuid, parent_uuid, event_type, msg_kind,
                    timestamp, timestamp_local,
                    session_id, project_id, is_sidechain, agent_id, agent_slug,
                    message_role,
                    message_content, message_content_json, model_id,
                    input_tokens, output_tokens, cache_read_tokens,
                    cache_creation_tokens, cache_5m_tokens,
                    source_file_id, line_number, raw_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event["uuid"],
                    event["parent_uuid"],
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

        total_events = 0
        for file_info in files_to_update:
            events_added = self.ingest_file(file_info)
            total_events += events_added
            log.debug("  %s: %d events (%s)", file_info["filepath"], events_added, file_info.get("reason", "new"))

        self.conn.commit()

        self.rebuild_aggregates()

        self.conn.execute(
            "INSERT OR REPLACE INTO cache_metadata (key, value) VALUES (?, ?)",
            ("last_update_at", datetime.now(UTC).isoformat()),
        )
        self.conn.commit()

        log.info("Updated %d files, %d events", len(files_to_update), total_events)
        return {"files_updated": len(files_to_update), "events_added": total_events}


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


def _derive_msg_kind(
    event_type: str,
    message_role: str | None,
    is_meta: bool,
    first_block_type: str | None,
    content_raw: Any,
) -> str:
    """Derive msg_kind from event attributes.

    Returns one of 9 categories:
        human             — user event, string content, not meta (real human prompts)
        user_text         — user event, text block content, not meta
        meta              — user event with isMeta=true (system-injected wrappers)
        tool_result       — user event whose first content block is tool_result
        tool_use          — assistant event whose first content block is tool_use
        assistant_text    — assistant event whose first content block is text
        thinking          — assistant event whose first content block is thinking
        task_notification — user event whose content starts with <task-notification>
        other             — everything else (progress, system, queue-operation, etc.)
    """
    if event_type == "user" and message_role == "user":
        if is_meta:
            return "meta"
        if first_block_type == "string":
            # Check for task_notification (string content starting with <task-notification>)
            if isinstance(content_raw, str) and content_raw.strip().startswith("<task-notification>"):
                return "task_notification"
            return "human"
        if first_block_type == "text":
            return "user_text"
        if first_block_type == "tool_result":
            return "tool_result"
        return "other"
    if event_type == "assistant" and message_role == "assistant":
        if first_block_type == "tool_use":
            return "tool_use"
        if first_block_type == "thinking":
            return "thinking"
        if first_block_type == "text":
            return "assistant_text"
        return "other"
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
    is_sidechain = raw.get("isSidechain", False)
    agent_id = raw.get("agentId")
    agent_slug = raw.get("slug")

    is_meta = bool(raw.get("isMeta"))

    message = raw.get("message", {}) or {}
    message_role = message.get("role") if isinstance(message, dict) else None
    message_content_raw = message.get("content") if isinstance(message, dict) else None
    model_id = message.get("model") if isinstance(message, dict) else None

    first_block_type = _first_content_block_type(message_content_raw)
    message_content_text = _extract_text_content(message_content_raw)

    # Derive msg_kind — a single column replacing the is_meta + first_content_block_type combo.
    # 9 categories matching the introspect_sessions.py schema.
    msg_kind = _derive_msg_kind(event_type, message_role, is_meta, first_block_type, message_content_raw)

    usage = message.get("usage", {}) if isinstance(message, dict) else {}
    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    cache_read_tokens = usage.get("cache_read_input_tokens", 0) or 0
    cache_creation_tokens = usage.get("cache_creation_input_tokens", 0) or 0
    cache_creation = usage.get("cache_creation", {}) or {}
    cache_5m_tokens = cache_creation.get("ephemeral_5m_input_tokens", 0) or 0

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
        "line_number": line_number,
        "raw_json": json.dumps(raw),
    }
