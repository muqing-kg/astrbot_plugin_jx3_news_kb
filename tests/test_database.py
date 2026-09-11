from core.database import Database


def test_initialize_creates_tables_and_rebuilds_fts(tmp_path):
    db = Database(tmp_path / "plugin.db")
    db.initialize()

    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO announcements(
                source_id, catid, title, type, url, description, content_text,
                raw_json, content_hash, published_at, updated_at_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "1", "24581", "测试公告", "官方公告", "https://example.com/1",
                "摘要", "正文", "{}", "hash-1", "2026-09-11T00:00:00+08:00",
                "2026-09-11T00:00:00+08:00",
            ),
        )
        announcement_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO announcement_revisions(
                announcement_id, revision_no, title, content_text, raw_json,
                content_hash, updated_at_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (announcement_id, 1, "测试公告", "正文", "{}", "hash-1", "2026-09-11"),
        )
        revision_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            """
            INSERT INTO chunks(
                announcement_id, revision_id, chunk_index, title,
                announcement_date, announcement_type, url, content
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (announcement_id, revision_id, 0, "测试公告", "2026-09-11", "官方公告",
             "https://example.com/1", "正文内容"),
        )

    assert db.fts_rebuild() == 1
    with db.connect() as conn:
        hit = conn.execute(
            "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ?",
            ('"正文内容"',),
        ).fetchone()
    assert hit is not None
    assert hit[0] == 1
