import asyncio

from core.database import Database
from core.ingest import IngestService


def make_item(content="第一段内容", title="测试公告"):
    return {
        "catid": "24581",
        "date": "2026/09/11",
        "title": title,
        "type": "官方公告",
        "url": "https://example.com/1",
        "desc": {
            "id": "1",
            "catid": "2458",
            "title": title,
            "description": "<p>摘要</p>",
            "inputtime": "1789012800",
            "updatetime": "1789012800",
            "content": f"<p>{content}</p>",
        },
    }


def make_push(date="09/11"):
    return {
        "title": "测试公告",
        "type": "官方公告",
        "url": "https://example.com/1",
        "date": date,
    }


def make_service(tmp_path) -> tuple[Database, IngestService]:
    db = Database(tmp_path / "plugin.db")
    db.initialize()
    return db, IngestService(db)


def test_ingest_deduplicates_by_url(tmp_path):
    db, service = make_service(tmp_path)

    first = asyncio.run(service.ingest_items([make_item()]))
    second = asyncio.run(service.ingest_items([make_item()]))

    assert first == {"returned": 1, "inserted": 1, "revised": 0, "skipped": 0}
    assert second["skipped"] == 1
    assert db.count("announcements") == 1


def test_changed_content_appends_revision(tmp_path):
    db, service = make_service(tmp_path)

    asyncio.run(service.ingest_items([make_item("原始内容")]))
    result = asyncio.run(service.ingest_items([make_item("修订后的内容")]))

    assert result["revised"] == 1
    assert db.count("announcement_revisions") == 2


def test_push_writes_row_without_body(tmp_path):
    db, service = make_service(tmp_path)

    assert service.ingest_push(make_push()) == "inserted"
    with db.connect() as conn:
        row = conn.execute(
            "SELECT title, type, content_text, published_at, source_id "
            "FROM announcements"
        ).fetchone()
    assert row["title"] == "测试公告"
    assert row["type"] == "官方公告"
    assert row["content_text"] == ""
    assert row["published_at"].startswith("2026-09-11")
    assert row["source_id"] == "https://example.com/1"
    assert db.count("announcement_revisions") == 1

    # The server only pushes an announcement once, so a repeat is noise.
    assert service.ingest_push(make_push()) == "skipped"
    assert db.count("announcements") == 1


def test_fetch_backfills_push_row_without_new_revision(tmp_path):
    db, service = make_service(tmp_path)

    service.ingest_push(make_push())
    result = asyncio.run(service.ingest_items([make_item("补上的正文")]))

    assert result["revised"] == 1
    assert db.count("announcement_revisions") == 1
    with db.connect() as conn:
        row = conn.execute("SELECT content_text FROM announcements").fetchone()
    assert row["content_text"] == "补上的正文"


def test_empty_body_from_later_fetch_keeps_stored_text(tmp_path):
    db, service = make_service(tmp_path)

    asyncio.run(service.ingest_items([make_item("原有正文")]))
    result = asyncio.run(service.ingest_items([make_item("")]))

    assert result["skipped"] == 1
    assert db.count("announcement_revisions") == 1
    with db.connect() as conn:
        row = conn.execute("SELECT content_text FROM announcements").fetchone()
    assert row["content_text"] == "原有正文"


def test_hard_delete_blocks_reingestion(tmp_path):
    db, service = make_service(tmp_path)
    asyncio.run(service.ingest_items([make_item()]))
    announcement_id = db.count("announcements")

    assert service.hard_delete_announcement(announcement_id) is True
    assert db.count("announcements") == 0
    assert db.count("deleted_fingerprints") == 1

    result = asyncio.run(service.ingest_items([make_item()]))
    assert result["skipped"] == 1
    assert db.count("announcements") == 0


def test_unreadable_date_stays_empty_instead_of_1970(tmp_path):
    db, service = make_service(tmp_path)
    item = make_item()
    item["date"] = "待定"
    item["desc"].pop("inputtime")

    asyncio.run(service.ingest_items([item]))

    with db.connect() as conn:
        assert conn.execute("SELECT published_at FROM announcements").fetchone()[0] == ""
