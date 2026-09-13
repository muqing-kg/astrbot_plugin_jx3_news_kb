"""Tests for the Plugin Page web API route handlers."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.activities import ActivityService
from core.database import Database
from core.ingest import IngestService
from core.scheduler import ReminderTargets, SchedulerService
from web_api import routes

FULL_GROUP_SESSION = "fake:GroupMessage:10001"


class FakePlugin:
    """Duck-typed plugin exposing what the route handlers need."""

    def __init__(self, tmp_path):
        self.db = Database(tmp_path / "test.sqlite3")
        self.db.initialize()
        self.ingest = IngestService(self.db)
        self.scheduler = SchedulerService(self.db, self.ingest, ActivityService(self.db))
        self.config = {"daily_fetch_time": "00:00"}
        self.embedding_provider = None
        self.fetch_calls: list[int] = []

    async def fetch_and_ingest(self, limit: int) -> dict[str, Any]:
        self.fetch_calls.append(limit)
        return {"success": True, "limit": limit, "returned": 0, "inserted": 0,
                "revised": 0, "skipped": 0, "error": "", "finished_at": ""}

    async def extract_and_schedule(self) -> int:
        return 0

    async def probe_providers(self) -> dict[str, Any]:
        return {
            "embedding": {"available": False, "provider_id": ""},
            "reranker": {"available": False, "provider_id": ""},
        }

    def reminder_target_count(self) -> int:
        return 0

    def background_alive(self) -> bool:
        return True

    def valid_slot_keys(self) -> set[tuple[int, str]]:
        return self.scheduler.valid_slot_keys(
            ReminderTargets.from_config(self.config)
        )


def _insert_announcement(
    plugin: FakePlugin, title="版本更新公告", url="https://example.com/a"
) -> int:
    """Create one announcement through the real ingest path (chunks + FTS)."""
    item = {
        "catid": "24581",
        "date": "2026/08/13",
        "title": title,
        "type": "官方公告",
        "url": url,
        "desc": {
            "id": "1",
            "title": title,
            "description": "<p>摘要</p>",
            "inputtime": "1785991200",
            "updatetime": "1785991200",
            "content": "<p>正文内容</p>",
        },
    }
    asyncio.run(plugin.ingest.ingest_items([item]))
    with plugin.db.connect() as conn:
        return int(
            conn.execute(
                "SELECT id FROM announcements WHERE url = ?", (url,)
            ).fetchone()[0]
        )


@pytest.fixture()
def plugin(tmp_path):
    return FakePlugin(tmp_path)


def test_stats_counts_and_next_fetch(plugin):
    _insert_announcement(plugin)
    data = asyncio.run(routes.handle_stats(plugin, {}, {}))
    assert data["counts"]["announcements"] == 1
    assert data["next_fetch_at"].endswith("00:00:00+08:00")
    assert data["embedding_available"] is False
    assert data["reminder_enabled"] is True
    assert data["reminder_targets"] == 0  # FakePlugin whitelist is empty
    assert data["upcoming_reminders"] == []


def test_announcement_list_search_and_pagination(plugin):
    for index in range(25):
        _insert_announcement(
            plugin,
            title=f"公告{index}",
            url=f"https://example.com/{index}",
        )
    page1 = asyncio.run(routes.handle_announcements(plugin, {"page": "1"}, {}))
    assert page1["total"] == 25
    assert len(page1["items"]) == 20
    assert page1["items"][0]["title"] == "公告24"

    found = asyncio.run(
        routes.handle_announcements(plugin, {"q": "公告1"}, {})
    )
    assert found["total"] == 11  # 公告1 and 公告10..公告19

    typed = asyncio.run(
        routes.handle_announcements(plugin, {"type": "官方公告"}, {})
    )
    assert typed["total"] == 25
    assert "官方公告" in typed["types"]


def test_detail_includes_revisions_activities_reminders(plugin):
    announcement_id = _insert_announcement(plugin)
    with plugin.db.connect() as conn:
        conn.execute(
            """
            INSERT INTO activities(
                announcement_id, name, action, category, end_time
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (announcement_id, "签到领券", "使用", "free_coupon",
             "2099-09-17T07:00:00+08:00"),
        )
    plugin.scheduler.create_pending_reminders(
        ReminderTargets(sessions=[FULL_GROUP_SESSION])
    )

    data = asyncio.run(routes.handle_announcement_detail(
        plugin, {}, {}, str(announcement_id)
    ))
    assert data["announcement"]["title"] == "版本更新公告"
    assert data["announcement"]["raw_json"]["desc"]["id"] == "1"
    assert data["chunk_count"] == 1
    assert data["chunks"][0]["chunk_index"] == 0
    assert "正文内容" in data["chunks"][0]["content"]
    assert "activities" not in data
    assert "reminders" not in data

    missing = asyncio.run(routes.handle_announcement_detail(plugin, {}, {}, "999"))
    assert missing[1] == 404


def test_delete_requires_confirmation_flag(plugin):
    announcement_id = _insert_announcement(plugin)
    missing = asyncio.run(routes.handle_announcement_delete(
        plugin, {}, {}, str(announcement_id)
    ))
    assert missing[1] == 400
    wrong = asyncio.run(routes.handle_announcement_delete(
        plugin, {}, {"confirm": "DELETE"}, str(announcement_id)
    ))
    assert wrong[1] == 400
    assert plugin.db.count("announcements") == 1


def test_delete_removes_everything_and_tombstones(plugin):
    announcement_id = _insert_announcement(plugin)
    with plugin.db.connect() as conn:
        conn.execute(
            """
            INSERT INTO activities(
                announcement_id, name, action, category, end_time
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (announcement_id, "签到领券", "使用", "free_coupon",
             "2099-09-17T07:00:00+08:00"),
        )
    plugin.scheduler.create_pending_reminders(
        ReminderTargets(sessions=[FULL_GROUP_SESSION, "fake:FriendMessage:20001"])
    )

    data = asyncio.run(routes.handle_announcement_delete(
        plugin, {}, {"confirm": True}, str(announcement_id)
    ))
    assert data == {"deleted": True, "id": announcement_id, "cancelled_reminders": 4}

    assert plugin.db.count("announcements") == 0
    assert plugin.db.count("chunks") == 0
    assert plugin.db.count("activities") == 0
    assert plugin.db.count("reminders") == 0
    with plugin.db.connect() as conn:
        tombstone = conn.execute(
            "SELECT url, title FROM deleted_fingerprints"
        ).fetchone()
    assert tombstone["url"] == "https://example.com/a"

    again = asyncio.run(routes.handle_announcement_delete(
        plugin, {}, {"confirm": True}, str(announcement_id)
    ))
    assert again[1] == 404


def _insert_future_activity(plugin, name="签到领券", end_time="2099-09-17T07:00:00+08:00") -> int:
    announcement_id = _insert_announcement(plugin)
    with plugin.db.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO activities(
                announcement_id, name, action, category, end_time, item_name
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (announcement_id, name, "使用", "free_coupon", end_time, "校服拓印券"),
        )
        return int(cursor.lastrowid)


def test_activities_lists_only_ongoing(plugin):
    activity_id = _insert_future_activity(plugin, name="进行中的活动")
    _insert_future_activity(plugin, name="已结束的活动", end_time="2020-01-01T00:00:00+08:00")

    data = asyncio.run(routes.handle_activities(plugin, {}, {}))
    names = [item["name"] for item in data["items"]]
    assert names == ["进行中的活动"]
    assert data["items"][0]["id"] == activity_id
    assert data["items"][0]["announcement_title"] == "版本更新公告"


def test_activity_delete_removes_reminders_without_confirmation(plugin):
    activity_id = _insert_future_activity(plugin)
    plugin.scheduler.create_pending_reminders(
        ReminderTargets(sessions=[FULL_GROUP_SESSION])
    )
    assert plugin.db.count("reminders") == 2

    data = asyncio.run(routes.handle_activity_delete(
        plugin, {}, {}, str(activity_id)
    ))
    assert data == {"deleted": True, "id": activity_id}
    assert plugin.db.count("activities") == 0
    assert plugin.db.count("reminders") == 0

    again = asyncio.run(routes.handle_activity_delete(
        plugin, {}, {}, str(activity_id)
    ))
    assert again[1] == 404


def test_fetch_route_validates_limit(plugin):
    bad = asyncio.run(routes.handle_fetch(plugin, {}, {"limit": 0}))
    assert bad[1] == 400
    ok = asyncio.run(routes.handle_fetch(plugin, {}, {"limit": 10}))
    assert ok["success"] is True
    assert plugin.fetch_calls == [10]


def test_stats_upcoming_shows_only_next_48h(plugin):
    from datetime import datetime as dt, timedelta as td
    from zoneinfo import ZoneInfo as ZI

    tz = ZI("Asia/Shanghai")
    fixed_now = dt(2026, 9, 13, 15, 0, tzinfo=tz)

    announcement_id = _insert_announcement(plugin)
    deadline = (fixed_now + td(days=1)).replace(
        hour=20, minute=0, second=0, microsecond=0
    )
    activity_id = _insert_future_activity(
        plugin, end_time=deadline.isoformat(timespec="seconds")
    )
    plugin.config["reminder_days_before"] = 1
    # Pin the clock: deadline tomorrow 20:00 (Monday evening, not maintenance).
    plugin.scheduler.now = lambda: fixed_now
    plugin.scheduler.create_pending_reminders(
        ReminderTargets(
            sessions=[FULL_GROUP_SESSION], days_before=1, send_time="10:00"
        )
    )
    # A slot far in the future and a stale row: neither is "recent".
    with plugin.db.connect() as conn:
        conn.execute(
            """
            INSERT INTO reminders(
                activity_id, target_type, target_id, scheduled_at, message_text
            ) VALUES (?, 'group', 'fake:GroupMessage:10001',
                      '2026-10-02T10:00:00+08:00', 'far'),
                     (?, 'group', 'fake:GroupMessage:10001',
                      '2026-09-01T10:00:00+08:00', 'stale')
            """,
            (activity_id, activity_id),
        )

    data = asyncio.run(routes.handle_stats(plugin, {}, {}))
    entries = [
        (row["name"], row["scheduled_at"][:16]) for row in data["upcoming_reminders"]
    ]
    # Only what will actually be pushed within the next 48 hours: tomorrow's
    # daily slot and the urgent last call. Far-future and stale rows hidden.
    assert entries == [
        ("签到领券", "2026-09-14T10:00"),
        ("签到领券", "2026-09-14T19:00"),
    ]
    assert data["pending_reminders"] == 4


def test_rebuild_routes(plugin):
    _insert_announcement(plugin)
    fts = asyncio.run(routes.handle_rebuild_fts(plugin, {}, {}))
    assert fts == {"rebuilt": True, "chunks": 1}

    embedding = asyncio.run(routes.handle_rebuild_embeddings(plugin, {}, {}))
    assert embedding[1] == 400


def test_logs_route(plugin):
    asyncio.run(plugin.scheduler.fetch_and_ingest(_FakeClient(), limit=10))
    data = asyncio.run(routes.handle_logs(plugin, {}, {}))
    assert len(data["logs"]) == 1
    assert data["logs"][0]["success"] == 1


class _FakeClient:
    async def fetch(self, limit: int):
        return []


def test_reminders_route_filters_by_status(plugin):
    announcement_id = _insert_announcement(plugin)
    with plugin.db.connect() as conn:
        conn.execute(
            """
            INSERT INTO activities(
                announcement_id, name, action, category, end_time
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (announcement_id, "签到领券", "使用", "free_coupon",
             "2099-09-17T07:00:00+08:00"),
        )
    plugin.scheduler.create_pending_reminders(
        ReminderTargets(sessions=[FULL_GROUP_SESSION])
    )
    pending = asyncio.run(routes.handle_reminders(plugin, {}, {}))
    assert pending["status"] == "pending"
    assert len(pending["items"]) == 2
    assert pending["items"][0]["activity_name"] == "签到领券"

    bad = asyncio.run(routes.handle_reminders(plugin, {"status": "nope"}, {}))
    assert bad[1] == 400


def test_routes_module_has_no_core_import():
    """routes.py is executed inside the plugin package at runtime; a top-level
    core import breaks AstrBot's package loading (regression guard for an
    environment difference local tests cannot catch)."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "web_api" / "routes.py"
    ).read_text(encoding="utf-8")
    for line in source.splitlines():
        if line.startswith(("import ", "from ")):
            assert not line.startswith(("from core", "import core")), line
