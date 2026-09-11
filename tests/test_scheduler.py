"""Tests for fetch-limit choice, reminder scheduling and dispatch."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from core.activities import ActivityService
from core.database import Database
from core.ingest import IngestService
from core.jx3api import NewsClient
from core.scheduler import ReminderTargets, SchedulerService

TZ = ZoneInfo("Asia/Shanghai")


def _make_scheduler(tmp_path, **kwargs) -> SchedulerService:
    database = Database(tmp_path / "test.sqlite3")
    database.initialize()
    ingest = IngestService(database)
    activities = ActivityService(database)
    defaults = {
        "database": database,
        "ingest_service": ingest,
        "activity_service": activities,
    }
    defaults.update(kwargs)
    return SchedulerService(**defaults)


def _insert_announcement(db: Database, published="2026-08-13T10:00:00+08:00") -> int:
    with db.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO announcements(
                source_id, title, type, url, content_text, raw_json,
                content_hash, published_at, updated_at_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "1", "版本更新公告", "官方公告", f"https://example.com/{published}",
                "内容", "{}", f"hash-{published}", published, published,
            ),
        )
        return int(cursor.lastrowid)


def _insert_activity(
    db: Database,
    announcement_id: int,
    end_time: str = "",
    item_expiry: str = "",
    start_time: str = "",
) -> int:
    with db.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO activities(
                announcement_id, name, action, category, start_time,
                end_time, item_expiry
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                announcement_id, "签到领校服拓印券", "使用免费校服拓印券",
                "free_coupon", start_time, end_time, item_expiry,
            ),
        )
        return int(cursor.lastrowid)


class _FakeClient:
    def __init__(self, items: list[dict[str, Any]] | None = None, error: Exception | None = None):
        self.items = items or []
        self.error = error
        self.requested: int | None = None

    async def fetch(self, limit: int) -> list[dict[str, Any]]:
        self.requested = limit
        if self.error:
            raise self.error
        return self.items


def test_pick_limit_on_empty_database(tmp_path):
    scheduler = _make_scheduler(tmp_path, initial_fetch_limit=50, daily_fetch_limit=10)
    assert scheduler.pick_fetch_limit() == 50


def test_pick_limit_daily_after_recent_fetch(tmp_path):
    scheduler = _make_scheduler(tmp_path, initial_fetch_limit=50, daily_fetch_limit=10)
    _insert_announcement(scheduler.db)
    with scheduler.db.connect() as conn:
        conn.execute(
            """
            INSERT INTO fetch_logs(
                started_at, finished_at, success, fetch_limit
            ) VALUES (datetime('now', 'localtime', '-2 hours'), datetime('now', 'localtime'), 1, 10)
            """
        )
    assert scheduler.pick_fetch_limit() == 10


def test_pick_limit_catchup_after_downtime(tmp_path):
    scheduler = _make_scheduler(
        tmp_path, daily_fetch_limit=10, catchup_fetch_limit=50
    )
    _insert_announcement(scheduler.db)
    with scheduler.db.connect() as conn:
        conn.execute(
            """
            INSERT INTO fetch_logs(
                started_at, finished_at, success, fetch_limit
            ) VALUES (datetime('now', 'localtime', '-3 days'), datetime('now', 'localtime', '-3 days'), 1, 10)
            """
        )
    assert scheduler.pick_fetch_limit() == 50


@pytest.mark.asyncio()
async def test_fetch_and_ingest_logs_success_and_error(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    ok = await scheduler.fetch_and_ingest(_FakeClient([]), limit=10)
    assert ok["success"] and ok["limit"] == 10

    failed = await scheduler.fetch_and_ingest(
        _FakeClient(error=RuntimeError("boom")), limit=10
    )
    assert not failed["success"]
    assert "boom" in failed["error"]

    with scheduler.db.connect() as conn:
        logs = conn.execute(
            "SELECT success, error FROM fetch_logs ORDER BY id"
        ).fetchall()
    assert [int(row["success"]) for row in logs] == [1, 0]
    assert "boom" in logs[1]["error"]


TARGETS = ReminderTargets(
    sessions=["fake:GroupMessage:10001", "fake:FriendMessage:20001"],
    days_before=1,
    send_time="10:00",
)


def test_reminder_scheduled_one_day_before_at_send_time(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    # Fixed "now": 2026-09-15 09:00 Asia/Shanghai.
    now = datetime(2026, 9, 15, 9, 0, tzinfo=TZ)
    announcement_id = _insert_announcement(scheduler.db)
    _insert_activity(
        scheduler.db, announcement_id, end_time="2026-09-17T07:00:00+08:00"
    )

    with scheduler.db.connect() as conn:
        row = conn.execute("SELECT * FROM activities").fetchone()
    moment = scheduler._scheduled_at(row, TARGETS, now)
    assert moment == datetime(2026, 9, 16, 10, 0, tzinfo=TZ)


def test_reminder_skips_slot_missed_beyond_grace(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    now = datetime(2026, 9, 16, 20, 0, tzinfo=TZ)
    announcement_id = _insert_announcement(scheduler.db)
    _insert_activity(
        scheduler.db, announcement_id, end_time="2026-09-17T07:00:00+08:00"
    )
    with scheduler.db.connect() as conn:
        row = conn.execute("SELECT * FROM activities").fetchone()
    # Slot (2026-09-16 10:00) was missed by 10 hours, beyond the 6h grace.
    assert scheduler._scheduled_at(row, TARGETS, now) is None


def test_reminder_slot_within_grace_is_kept(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    now = datetime(2026, 9, 16, 10, 20, tzinfo=TZ)
    announcement_id = _insert_announcement(scheduler.db)
    _insert_activity(
        scheduler.db, announcement_id, end_time="2026-09-17T07:00:00+08:00"
    )
    with scheduler.db.connect() as conn:
        row = conn.execute("SELECT * FROM activities").fetchone()
    assert scheduler._scheduled_at(row, TARGETS, now) == datetime(
        2026, 9, 16, 10, 0, tzinfo=TZ
    )


def test_short_window_reminds_thirty_minutes_before_start(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    now = datetime(2026, 9, 15, 9, 0, tzinfo=TZ)
    announcement_id = _insert_announcement(scheduler.db)
    _insert_activity(
        scheduler.db,
        announcement_id,
        start_time="2026-09-15T20:00:00+08:00",
        end_time="2026-09-15T21:00:00+08:00",
    )
    with scheduler.db.connect() as conn:
        row = conn.execute("SELECT * FROM activities").fetchone()
    assert scheduler._scheduled_at(row, TARGETS, now) == datetime(
        2026, 9, 15, 19, 30, tzinfo=TZ
    )


def test_expired_deadline_is_never_scheduled(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    now = datetime(2026, 9, 15, 9, 0, tzinfo=TZ)
    announcement_id = _insert_announcement(scheduler.db)
    _insert_activity(
        scheduler.db, announcement_id, end_time="2026-09-14T07:00:00+08:00"
    )
    with scheduler.db.connect() as conn:
        row = conn.execute("SELECT * FROM activities").fetchone()
    assert scheduler._scheduled_at(row, TARGETS, now) is None


def test_item_expiry_takes_priority_over_end_time(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    now = datetime(2026, 9, 15, 9, 0, tzinfo=TZ)
    announcement_id = _insert_announcement(scheduler.db)
    _insert_activity(
        scheduler.db,
        announcement_id,
        end_time="2026-09-20T23:59:00+08:00",
        item_expiry="2026-09-17T07:00:00+08:00",
    )
    with scheduler.db.connect() as conn:
        row = conn.execute("SELECT * FROM activities").fetchone()
    assert scheduler._scheduled_at(row, TARGETS, now) == datetime(
        2026, 9, 16, 10, 0, tzinfo=TZ
    )


def test_create_pending_reminders_is_idempotent(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    announcement_id = _insert_announcement(scheduler.db)
    _insert_activity(
        scheduler.db, announcement_id, end_time="2099-09-17T07:00:00+08:00"
    )
    first = scheduler.create_pending_reminders(TARGETS)
    second = scheduler.create_pending_reminders(TARGETS)
    assert first == 2  # one group + one private
    assert second == 0

    with scheduler.db.connect() as conn:
        rows = conn.execute(
            "SELECT target_type, target_id, message_text FROM reminders ORDER BY id"
        ).fetchall()
    assert {row["target_type"] for row in rows} == {"group", "private"}
    assert {row["target_id"] for row in rows} == {
        "fake:GroupMessage:10001",
        "fake:FriendMessage:20001",
    }
    assert all("【签到领校服拓印券 到期提醒】" in row["message_text"] for row in rows)
    assert all("签到领校服拓印券" in row["message_text"] for row in rows)


def test_create_pending_reminders_without_targets(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    announcement_id = _insert_announcement(scheduler.db)
    _insert_activity(
        scheduler.db, announcement_id, end_time="2099-09-17T07:00:00+08:00"
    )
    assert scheduler.create_pending_reminders(ReminderTargets(sessions=[])) == 0


def test_from_config_collects_full_session_addresses():
    targets = ReminderTargets.from_config(
        {
            "whitelist_groups": [
                "aiocqhttp:GroupMessage:10001",
                "10002",
            ],
            "whitelist_users": [
                "qqofficial:FriendMessage:20001",
                "not-a-session",
                "aiocqhttp:WrongType:1",
            ],
            "reminder_days_before": 2,
            "reminder_send_time": "09:30",
        }
    )
    assert targets.sessions == [
        "aiocqhttp:GroupMessage:10001",
        "qqofficial:FriendMessage:20001",
    ]
    assert targets.days_before == 2
    assert targets.send_time == "09:30"
    assert targets.as_pairs() == [
        ("group", "aiocqhttp:GroupMessage:10001"),
        ("private", "qqofficial:FriendMessage:20001"),
    ]


def test_parse_session_address_variants():
    from core.scheduler import extract_session_id, parse_session_address

    assert parse_session_address("aiocqhttp:GroupMessage:123") == (
        "group",
        "aiocqhttp:GroupMessage:123",
    )
    assert parse_session_address("webchat:FriendMessage:abc ") == (
        "private",
        "webchat:FriendMessage:abc",
    )
    assert parse_session_address("10086") is None
    assert parse_session_address("bad:Type:1") is None
    assert parse_session_address("a:GroupMessage:") is None
    assert extract_session_id("aiocqhttp:GroupMessage:123") == "123"
    assert extract_session_id("10086") == "10086"
    assert extract_session_id("") == ""


def test_due_reminders_and_mark(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    announcement_id = _insert_announcement(scheduler.db)
    _insert_activity(
        scheduler.db, announcement_id, end_time="2099-09-17T07:00:00+08:00"
    )
    scheduler.create_pending_reminders(TARGETS)
    # Force a due slot in the past.
    with scheduler.db.connect() as conn:
        conn.execute(
            "UPDATE reminders SET scheduled_at = datetime('now', 'localtime', '-1 hour')"
        )
    due = scheduler.due_reminders()
    assert len(due) == 2

    scheduler.mark_reminder(int(due[0]["id"]), "sent")
    scheduler.mark_reminder(int(due[1]["id"]), "failed", error="send timeout")

    with scheduler.db.connect() as conn:
        statuses = conn.execute(
            "SELECT status FROM reminders ORDER BY id"
        ).fetchall()
        logs = conn.execute("SELECT success, error FROM reminder_logs").fetchall()
    assert [row["status"] for row in statuses] == ["sent", "failed"]
    assert sorted(int(row["success"]) for row in logs) == [0, 1]
    assert scheduler.due_reminders() == []


def test_cancel_reminders_for_announcement(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    announcement_id = _insert_announcement(scheduler.db)
    _insert_activity(
        scheduler.db, announcement_id, end_time="2099-09-17T07:00:00+08:00"
    )
    scheduler.create_pending_reminders(TARGETS)
    cancelled = scheduler.cancel_reminders_for_announcement(announcement_id)
    assert cancelled == 2


def test_merge_reminder_texts_combines_same_day_items():
    from core.scheduler import merge_reminder_texts

    single = "【活动A 到期提醒】\n待办：使用\n截止：2026-09-17 07:00"
    assert merge_reminder_texts([single]) == single

    merged = merge_reminder_texts(
        [
            "【活动A 到期提醒】\n待办：使用\n截止：2026-09-17 07:00",
            "【活动B 到期提醒】\n待办：领取\n截止：2026-09-17 07:00",
        ]
    )
    assert merged.startswith("【今日到期提醒 · 共 2 项】")
    assert "【活动A 到期提醒】" in merged
    assert "【活动B 到期提醒】" in merged
    # Items are separated by a blank line, no divider lines.
    assert "———" not in merged
    assert "【今日到期提醒 · 共 2 项】\n\n【活动A 到期提醒】" in merged


def test_prune_logs_keeps_recent_only(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    announcement_id = _insert_announcement(scheduler.db)
    activity_id = _insert_activity(
        scheduler.db, announcement_id, end_time="2099-09-17T07:00:00+08:00"
    )
    with scheduler.db.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO reminders(
                activity_id, target_type, target_id, scheduled_at, message_text
            ) VALUES (?, 'group', 'fake:GroupMessage:10001',
                      '2026-01-01T10:00:00+08:00', 'text')
            """,
            (activity_id,),
        )
        reminder_id = cursor.lastrowid
        conn.execute(
            """
            INSERT INTO fetch_logs(started_at, finished_at, success, fetch_limit)
            VALUES (datetime('now', 'localtime', '-10 days'),
                    datetime('now', 'localtime', '-10 days'), 1, 10)
            """
        )
        conn.execute(
            """
            INSERT INTO fetch_logs(started_at, finished_at, success, fetch_limit)
            VALUES (datetime('now', 'localtime'), datetime('now', 'localtime'), 1, 10)
            """
        )
        conn.execute(
            """
            INSERT INTO reminder_logs(
                reminder_id, activity_name, target_type, target_id,
                scheduled_at, sent_at, success
            ) VALUES (?, 'a', 'group', '1', '2026-01-01',
                      datetime('now', 'localtime', '-10 days'), 1)
            """,
            (reminder_id,),
        )

    pruned = scheduler.prune_logs()
    assert pruned == {"fetch_logs": 1, "reminder_logs": 1}

    with scheduler.db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM fetch_logs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM reminder_logs").fetchone()[0] == 0


def test_status_snapshot_next_run(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    snapshot = scheduler.status_snapshot("00:00")
    assert snapshot["next_fetch_at"].endswith("00:00:00+08:00")
    assert snapshot["last_fetch"] is None
