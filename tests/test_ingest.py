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


def test_ingest_deduplicates_by_url(tmp_path):
    db = Database(tmp_path / "plugin.db")
    db.initialize()
    service = IngestService(db)

    first = asyncio.run(service.ingest_items([make_item()]))
    second = asyncio.run(service.ingest_items([make_item()]))

    assert first == {"returned": 1, "inserted": 1, "revised": 0, "skipped": 0}
    assert second["skipped"] == 1
    assert db.count("announcements") == 1
    assert db.count("chunks") == 1


def test_changed_content_appends_revision_and_chunks(tmp_path):
    db = Database(tmp_path / "plugin.db")
    db.initialize()
    service = IngestService(db)

    asyncio.run(service.ingest_items([make_item("原始内容")]))
    result = asyncio.run(service.ingest_items([make_item("修订后的内容")]))

    assert result["revised"] == 1
    assert db.count("announcement_revisions") == 2
    assert db.count("chunks") == 2


def test_hard_delete_blocks_reingestion(tmp_path):
    db = Database(tmp_path / "plugin.db")
    db.initialize()
    service = IngestService(db)
    asyncio.run(service.ingest_items([make_item()]))
    announcement_id = db.count("announcements")

    assert service.hard_delete_announcement(announcement_id) is True
    assert db.count("announcements") == 0
    assert db.count("chunks") == 0
    assert db.count("deleted_fingerprints") == 1

    result = asyncio.run(service.ingest_items([make_item()]))
    assert result["skipped"] == 1
    assert db.count("announcements") == 0
