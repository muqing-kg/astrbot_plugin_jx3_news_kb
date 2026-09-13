"""Ingestion, revision history, chunking and hard deletion."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .database import Database
from .text import chunk_text, content_fingerprint, parse_date, strip_html, unix_to_iso


class IngestService:
    def __init__(self, database: Database, timezone_name: str = "Asia/Shanghai") -> None:
        self.db = database
        self.timezone_name = timezone_name

    async def ingest_items(
        self,
        items: list[dict[str, Any]],
        embedding_provider: Any | None = None,
    ) -> dict[str, int]:
        """Ingest raw JX3API records. Embedding indexing remains best effort."""
        result = {"returned": len(items), "inserted": 0, "revised": 0, "skipped": 0}
        with self.db.connect() as conn:
            for item in items:
                outcome = self._ingest_item(conn, item)
                result[outcome] += 1
        if embedding_provider is not None:
            await self.index_missing_embeddings(embedding_provider)
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
        published_at = unix_to_iso(
            desc.get("inputtime"), self.timezone_name
        ) or f"{announcement_date or '1970-01-01'}T00:00:00+08:00"
        updated_at_source = unix_to_iso(
            desc.get("updatetime"), self.timezone_name
        ) or published_at
        source_id = str(desc.get("id") or url)
        catid = str(item.get("catid") or desc.get("catid") or "")
        announcement_type = str(item.get("type") or "")
        raw_json = json.dumps(item, ensure_ascii=False, separators=(",", ":"))

        existing = conn.execute(
            "SELECT id, content_hash FROM announcements WHERE url = ?",
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
            announcement_id = int(cursor.lastrowid)
            revision_id = self._insert_revision(
                conn, announcement_id, 1, title, content_text, raw_json,
                fingerprint, updated_at_source,
            )
            self._insert_chunks(
                conn, announcement_id, revision_id, title, announcement_date,
                announcement_type, url, content_text,
            )
            return "inserted"

        announcement_id = int(existing["id"])
        if existing["content_hash"] == fingerprint:
            return "skipped"

        old_revision = conn.execute(
            """
            SELECT id, revision_no FROM announcement_revisions
            WHERE announcement_id = ? AND content_hash = ?
            """,
            (announcement_id, fingerprint),
        ).fetchone()
        if old_revision:
            revision_id = int(old_revision["id"])
        else:
            next_no = int(
                conn.execute(
                    """
                    SELECT COALESCE(MAX(revision_no), 0) + 1
                    FROM announcement_revisions WHERE announcement_id = ?
                    """,
                    (announcement_id,),
                ).fetchone()[0]
            )
            revision_id = self._insert_revision(
                conn, announcement_id, next_no, title, content_text, raw_json,
                fingerprint, updated_at_source,
            )

        conn.execute(
            """
            UPDATE announcements
            SET source_id = ?, catid = ?, title = ?, type = ?, description = ?,
                content_text = ?, raw_json = ?, content_hash = ?,
                updated_at_source = ?, modified_at = datetime('now', 'localtime')
            WHERE id = ?
            """,
            (
                source_id, catid, title, announcement_type, description,
                content_text, raw_json, fingerprint, updated_at_source,
                announcement_id,
            ),
        )
        self._insert_chunks(
            conn, announcement_id, revision_id, title, announcement_date,
            announcement_type, url, content_text,
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

    def _insert_chunks(
        self,
        conn: sqlite3.Connection,
        announcement_id: int,
        revision_id: int,
        title: str,
        announcement_date: str,
        announcement_type: str,
        url: str,
        content_text: str,
    ) -> None:
        chunks = chunk_text(content_text)
        for index, content in enumerate(chunks):
            cursor = conn.execute(
                """
                INSERT INTO chunks(
                    announcement_id, revision_id, chunk_index, title,
                    announcement_date, announcement_type, url, content
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    announcement_id, revision_id, index, title, announcement_date,
                    announcement_type, url, content,
                ),
            )
            chunk_id = int(cursor.lastrowid)
            conn.execute(
                """
                INSERT INTO chunks_fts(rowid, title, content, announcement_id, chunk_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_id, title, content, announcement_id, chunk_id),
            )

    async def index_missing_embeddings(self, embedding_provider: Any) -> int:
        """Index chunks missing vectors; failures never affect database ingestion."""
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT id, content FROM chunks WHERE embedding IS NULL ORDER BY id"
            ).fetchall()
        if not rows:
            return 0
        vectors = await embedding_provider.get_embeddings_batch(
            [row["content"] for row in rows]
        )
        if len(vectors) != len(rows):
            raise RuntimeError("向量接口返回的向量数量与预期不符")
        dim = len(vectors[0])
        with self.db.connect() as conn:
            for row, vector in zip(rows, vectors, strict=True):
                if len(vector) != dim:
                    raise RuntimeError("向量接口返回的向量维度不一致")
                conn.execute(
                    """
                    UPDATE chunks
                    SET embedding = ?, embedding_dim = ?,
                        embedding_updated_at = datetime('now', 'localtime')
                    WHERE id = ?
                    """,
                    (json.dumps(vector).encode("utf-8"), dim, row["id"]),
                )
        return len(rows)

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
