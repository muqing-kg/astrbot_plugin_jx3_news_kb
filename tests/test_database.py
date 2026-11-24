import sqlite3

from core.database import SCHEMA_VERSION, Database


def _tables(db: Database) -> set[str]:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {row["name"] for row in rows}


def test_initialize_creates_core_tables(tmp_path):
    db = Database(tmp_path / "plugin.db")
    db.initialize()

    tables = _tables(db)
    assert {
        "announcements",
        "announcement_revisions",
        "activities",
        "reminders",
        "deleted_fingerprints",
        "fetch_logs",
    } <= tables
    assert "chunks" not in tables
    assert "chunks_fts" not in tables
    assert "reminder_logs" not in tables

    with db.connect() as conn:
        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()["value"]
    assert int(version) == SCHEMA_VERSION


def test_initialize_drops_legacy_chunk_tables(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta(key, value) VALUES ('version', '1');
        CREATE TABLE chunks (id INTEGER PRIMARY KEY, content TEXT);
        CREATE TABLE chunks_fts (content TEXT);
        """
    )
    conn.commit()
    conn.close()

    db = Database(path)
    db.initialize()

    tables = _tables(db)
    assert "chunks" not in tables
    assert "chunks_fts" not in tables

    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO announcements(
                source_id, title, type, url, content_hash, published_at,
                updated_at_source
            ) VALUES ('1', '公告', '官方公告', 'https://example.com/1',
                      'hash-1', '2026-09-11T00:00:00+08:00',
                      '2026-09-11T00:00:00+08:00')
            """
        )

    # Re-initialising on an already-migrated database keeps the new data.
    db.initialize()
    assert db.count("announcements") == 1


def test_initialize_adds_extract_mode_to_v2_database(tmp_path):
    path = tmp_path / "v2.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta(key, value) VALUES ('version', '2');
        CREATE TABLE announcements (
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
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            modified_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );
        INSERT INTO announcements(
            source_id, title, url, content_hash, published_at, updated_at_source
        ) VALUES ('1', '老公告', 'https://example.com/old', 'h',
                  '2026-09-11T00:00:00+08:00', '2026-09-11T00:00:00+08:00');
        """
    )
    conn.commit()
    conn.close()

    db = Database(path)
    db.initialize()

    with db.connect() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(announcements)")}
        mode = conn.execute(
            "SELECT activity_extract_mode FROM announcements"
        ).fetchone()[0]
    assert "activity_extract_mode" in columns
    assert mode == ""

def test_initialize_drops_the_unused_reminder_log_table(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE reminder_logs (id INTEGER PRIMARY KEY);")
    conn.commit()
    conn.close()

    db = Database(path)
    db.initialize()

    assert "reminder_logs" not in _tables(db)
