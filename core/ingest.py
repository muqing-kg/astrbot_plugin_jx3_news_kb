"""Ingestion, revision history and hard deletion."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .database import Database
from .text import (
    content_fingerprint,
    parse_date,
    strip_html,
    unix_to_iso,
)


class IngestService:
    def __init__(self, database: Database, timezone_name: str = "Asia/Shanghai") -> None:
        self.db = database
        self.timezone_name = timezone_name

    async def ingest_items(self, items: list[dict[str, Any]]) -> dict[str, int]:
        """Ingest raw JX3API records."""
        result = {"returned": len(items), "inserted": 0, "revised": 0, "skipped": 0}
        with self.db.connect() as conn:
            for item in items:
                outcome = self._ingest_item(conn, item)
                result[outcome] += 1
        return result

    def _ingest_item(self, conn: sqlite3.Connection, item: dict[str, Any]) -> str:
        desc = item.get("desc") or {}
        url = str(item.get("url") or desc.get("url") or "").strip()
        title = str(item.get("title") or desc.get("title") or "").strip()
        if not url or not title:
            return "skipped"

        content_text = strip_html(desc.get("content"))
        description = strip_html(desc.get("description"))
        fingerprint = content_fingerprint(url, title, content_text)
        deleted = conn.execute(
            "SELECT 1 FROM deleted_fingerprints WHERE fingerprint = ? OR url = ?",
            (fingerprint, url),
        ).fetchone()
        if deleted:
            return "skipped"

        announcement_date = parse_date(str(item.get("date") or ""))
        # Source timestamps win; the date-only fallback exists for rows whose
        # record carries no unix time at all. An unreadable date stays empty
        # rather than becoming a 1970 placeholder that later reads as real.
        published_source = unix_to_iso(desc.get("inputtime"), self.timezone_name)
        published_at = published_source or (
            f"{announcement_date}T00:00:00+08:00" if announcement_date else ""
        )
        updated_at_source = unix_to_iso(
            desc.get("updatetime"), self.timezone_name
        ) or published_at
        source_id = str(desc.get("id") or url)
        catid = str(item.get("catid") or desc.get("catid") or "")
        announcement_type = str(item.get("type") or "")
        raw_json = json.dumps(item, ensure_ascii=False, separators=(",", ":"))

        existing = conn.execute(
            "SELECT id, content_hash, content_text FROM announcements WHERE url = ?",
            (url,),
        ).fetchone()

        if existing is None:
            cursor = conn.execute(
                """
                INSERT INTO announcements(
                    source_id, catid, title, type, url, description, content_text,
                    raw_json, content_hash, published_at, updated_at_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id, catid, title, announcement_type, url, description,
                    content_text, raw_json, fingerprint, published_at,
                    updated_at_source,
                ),
            )
            self._insert_revision(
                conn, int(cursor.lastrowid), 1, title, content_text, raw_json,
                fingerprint, updated_at_source,
            )
            return "inserted"

        announcement_id = int(existing["id"])
        if existing["content_hash"] == fingerprint:
            return "skipped"

        # A later fetch that comes back without a body is a failed read, not a
        # content change: writing it would blank text that is already stored.
        if str(existing["content_text"] or "").strip() and not content_text.strip():
            return "skipped"

        # A row stored without a body (the API had no content yet) is filled in
        # place so the revision history stays one entry per real content change.
        backfill = not str(existing["content_text"] or "").strip() and bool(
            content_text.strip()
        )
        if not backfill:
            old_revision = conn.execute(
                """
                SELECT id FROM announcement_revisions
                WHERE announcement_id = ? AND content_hash = ?
                """,
                (announcement_id, fingerprint),
            ).fetchone()
            if old_revision is None:
                next_no = int(
                    conn.execute(
                        """
                        SELECT COALESCE(MAX(revision_no), 0) + 1
                        FROM announcement_revisions WHERE announcement_id = ?
                        """,
                        (announcement_id,),
                    ).fetchone()[0]
                )
                self._insert_revision(
                    conn, announcement_id, next_no, title, content_text, raw_json,
                    fingerprint, updated_at_source,
                )

        conn.execute(
            """
            UPDATE announcements
            SET source_id = ?, catid = ?, title = ?, type = ?, description = ?,
                content_text = ?, raw_json = ?, content_hash = ?,
                published_at = COALESCE(NULLIF(?, ''), published_at),
                updated_at_source = ?,
                modified_at = datetime('now', 'localtime')
            WHERE id = ?
            """,
            (
                source_id, catid, title, announcement_type, description,
                content_text, raw_json, fingerprint, published_source,
                updated_at_source, announcement_id,
            ),
        )
        return "revised"

    def _insert_revision(
        self,
        conn: sqlite3.Connection,
        announcement_id: int,
        revision_no: int,
        title: str,
        content_text: str,
        raw_json: str,
        content_hash: str,
        updated_at_source: str,
    ) -> int:
        cursor = conn.execute(
            """
            INSERT INTO announcement_revisions(
                announcement_id, revision_no, title, content_text, raw_json,
                content_hash, updated_at_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                announcement_id, revision_no, title, content_text, raw_json,
                content_hash, updated_at_source,
            ),
        )
        return int(cursor.lastrowid)

    def hard_delete_announcement(self, announcement_id: int) -> bool:
        """Delete one announcement and all related records, leaving a tombstone."""
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT title, url, content_hash FROM announcements WHERE id = ?",
                (announcement_id,),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                "DELETE FROM announcements WHERE id = ?", (announcement_id,)
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO deleted_fingerprints(fingerprint, url, title)
                VALUES (?, ?, ?)
                """,
                (row["content_hash"], row["url"], row["title"]),
            )
        return True
