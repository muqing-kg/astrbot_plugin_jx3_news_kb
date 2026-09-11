import asyncio

from core.database import Database
from core.ingest import IngestService
from core.qa import QAService
from core.search import SearchService


def make_item(title, content):
    return {
        "catid": "24581",
        "date": "2026/09/11",
        "title": title,
        "type": "官方公告",
        "url": f"https://example.com/{title}",
        "desc": {
            "id": title,
            "title": title,
            "description": "摘要",
            "inputtime": "1789012800",
            "updatetime": "1789012800",
            "content": f"<p>{content}</p>",
        },
    }


def test_fulltext_search_returns_grounded_chunks(tmp_path):
    db = Database(tmp_path / "plugin.db")
    db.initialize()
    asyncio.run(
        IngestService(db).ingest_items(
            [
                make_item("联动公告", "神风舟签到活动内容"),
                make_item("维护公告", "例行维护内容"),
            ]
        )
    )
    results = asyncio.run(SearchService(db).search("神风舟 签到", top_k=3))

    assert results
    assert "神风舟" in results[0]["content"]
    assert results[0]["url"].endswith("联动公告")


def test_search_without_results_is_empty(tmp_path):
    db = Database(tmp_path / "plugin.db")
    db.initialize()
    assert asyncio.run(SearchService(db).search("完全不相关关键词")) == []


def test_qa_fallback_without_llm(tmp_path):
    db = Database(tmp_path / "plugin.db")
    db.initialize()
    asyncio.run(IngestService(db).ingest_items([make_item("联动公告", "神风舟签到活动内容")]))
    qa = QAService(SearchService(db), context=None)

    handled, answer = asyncio.run(qa.answer("神风舟活动怎么参与"))
    assert handled is True
    assert "神风舟" in answer
