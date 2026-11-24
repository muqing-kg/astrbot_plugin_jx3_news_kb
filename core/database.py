"""SQLite storage for JX3 announcements, activities and reminders."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS announcements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    catid TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT '',
    url TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    content_text TEXT NOT NULL DEFAULT '',
    raw_json TEXT NOT NULL DEFAULT '{}',
    content_hash TEXT NOT NULL,
    published_at TEXT NOT NULL,
    updated_at_source TEXT NOT NULL,
    activity_extracted_at TEXT,
    activity_extract_mode TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    modified_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_announcements_date ON announcements(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_announcements_type ON announcements(type);

CREATE TABLE IF NOT EXISTS announcement_revisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    announcement_id INTEGER NOT NULL REFERENCES announcements(id) ON DELETE CASCADE,
    revision_no INTEGER NOT NULL,
    title TEXT NOT NULL,
    content_text TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    updated_at_source TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    UNIQUE(announcement_id, content_hash)
);

CREATE TABLE IF NOT EXISTS activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    announcement_id INTEGER NOT NULL REFERENCES announcements(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    action TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    start_time TEXT,
    end_time TEXT,
    item_expiry TEXT,
    item_name TEXT NOT NULL DEFAULT '',
    explanation TEXT NOT NULL DEFAULT '',
    evidence TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.0,
    reminder_enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    modified_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_activities_end ON activities(end_time);
CREATE INDEX IF NOT EXISTS idx_activities_expiry ON activities(item_expiry);

CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id INTEGER NOT NULL REFERENCES activities(id) ON DELETE CASCADE,
    target_type TEXT NOT NULL CHECK(target_type IN ('group', 'private')),
    target_id TEXT NOT NULL,
    scheduled_at TEXT NOT NULL,
    message_text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending', 'sent', 'cancelled', 'failed')),
    sent_at TEXT,
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    UNIQUE(activity_id, target_type, target_id, scheduled_at)
);

CREATE INDEX IF NOT EXISTS idx_reminders_pending ON reminders(status, scheduled_at);

CREATE TABLE IF NOT EXISTS deleted_fingerprints (
    fingerprint TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    deleted_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS fetch_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    finished_at TEXT,
    success INTEGER NOT NULL DEFAULT 0,
    fetch_limit INTEGER NOT NULL,
    returned_count INTEGER NOT NULL DEFAULT 0,
    inserted_count INTEGER NOT NULL DEFAULT 0,
    revised_count INTEGER NOT NULL DEFAULT 0,
    skipped_count INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT ''
);
"""


class Database:
    """Small synchronous SQLite wrapper; all public callers run it in asyncio."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            # Written by older builds, never read back; drop whatever is left.
            conn.execute("DROP TABLE IF EXISTS reminder_logs")
            stored = conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'version'"
            ).fetchone()
            try:
                version = int(stored["value"]) if stored else 0
            except (TypeError, ValueError):
                version = 0
            if version < 2:
                self._drop_legacy_tables(conn)
            if version < 3:
                self._add_extract_mode(conn)
            conn.execute(
                """
                INSERT INTO schema_meta(key, value) VALUES('version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _drop_legacy_tables(conn: sqlite3.Connection) -> None:
        """v1 stored retrieval chunks and a FTS index; nothing reads them now."""
        for table in ("chunks_fts", "chunks"):
            conn.execute(f"DROP TABLE IF EXISTS {table}")

    @staticmethod
    def _add_extract_mode(conn: sqlite3.Connection) -> None:
        """v2 has no record of which extractor produced an announcement's rows."""
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(announcements)")
        }
        if "activity_extract_mode" not in columns:
            conn.execute(
                "ALTER TABLE announcements "
                "ADD COLUMN activity_extract_mode TEXT NOT NULL DEFAULT ''"
            )

    def count(self, table: str) -> int:
        if not table.replace("_", "").isalnum():
            raise ValueError("invalid table name")
        with self.connect() as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
