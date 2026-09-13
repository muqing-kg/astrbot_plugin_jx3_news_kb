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
NOW = datetime(2026, 9, 15, 9, 0, tzinfo=TZ)


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


def test_slots_countdown_and_urgent(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    # Sunday 20:00 deadline: countdown D-1/D-0 plus the urgent slot.
    deadline = datetime(2026, 9, 16, 20, 0, tzinfo=TZ)
    targets = ReminderTargets(
        sessions=["fake:GroupMessage:10001"], days_before=1, send_time="10:00"
    )
    slots = scheduler._slots_for(deadline, targets, NOW)
    moments = [m for m, _, _ in slots]
    assert moments == [
        datetime(2026, 9, 15, 10, 0, tzinfo=TZ),
        datetime(2026, 9, 16, 10, 0, tzinfo=TZ),
        datetime(2026, 9, 16, 19, 0, tzinfo=TZ),
    ]
    assert [(kind, days) for _, kind, days in slots] == [
        ("countdown", 1),
        ("countdown", 0),
        ("urgent", 0),
    ]


def test_slots_countdown_multi_day(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    deadline = datetime(2026, 9, 20, 20, 0, tzinfo=TZ)  # Sunday evening
    targets = ReminderTargets(
        sessions=["fake:GroupMessage:10001"], days_before=3, send_time="10:00"
    )
    slots = scheduler._slots_for(deadline, targets, NOW)
    moments = [(m.strftime("%m-%d %H:%M"), kind, days) for m, kind, days in slots]
    assert moments == [
        ("09-17 10:00", "countdown", 3),
        ("09-18 10:00", "countdown", 2),
        ("09-19 10:00", "countdown", 1),
        ("09-20 10:00", "countdown", 0),
        ("09-20 19:00", "urgent", 0),
    ]


def test_slots_maintenance_evening_replaces_urgent(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    # Thursday 07:00 deadline: dies with the maintenance window.
    deadline = datetime(2026, 9, 17, 7, 0, tzinfo=TZ)
    targets = ReminderTargets(
        sessions=["fake:GroupMessage:10001"], days_before=3, send_time="10:00"
    )
    now = datetime(2026, 9, 13, 9, 0, tzinfo=TZ)
    slots = scheduler._slots_for(deadline, targets, now)
    moments = [(m.strftime("%m-%d %H:%M"), kind) for m, kind, _ in slots]
    assert moments == [
        ("09-14 10:00", "countdown"),
        ("09-15 10:00", "countdown"),
        ("09-16 10:00", "countdown"),
        ("09-16 21:00", "evening"),
    ]


def test_slots_urgent_supersedes_earlier_daily_slot(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    # Deadline 10:30: urgent (09:30) is earlier than the daily slot (10:00).
    deadline = datetime(2026, 9, 16, 10, 30, tzinfo=TZ)
    targets = ReminderTargets(
        sessions=["fake:GroupMessage:10001"], days_before=1, send_time="10:00"
    )
    slots = scheduler._slots_for(deadline, targets, NOW)
    moments = [(m.strftime("%m-%d %H:%M"), kind) for m, kind, _ in slots]
    assert moments == [
        ("09-15 10:00", "countdown"),
        ("09-16 09:30", "urgent"),
    ]


def test_slots_expired_deadline(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    deadline = datetime(2026, 9, 14, 20, 0, tzinfo=TZ)
    assert scheduler._slots_for(deadline, TARGETS, NOW) == []


def test_create_pending_uses_item_expiry_over_end_time(tmp_path):
    """deadline = item_expiry || end_time: the item expiry wins when both exist."""
    scheduler = _make_scheduler(tmp_path)
    announcement_id = _insert_announcement(scheduler.db)
    _insert_activity(
        scheduler.db,
        announcement_id,
        end_time="2099-12-31T23:59:00+08:00",
        item_expiry="2099-09-17T07:00:00+08:00",
    )
    scheduler.create_pending_reminders(TARGETS)

    with scheduler.db.connect() as conn:
        rows = conn.execute(
            "SELECT scheduled_at, message_text FROM reminders"
        ).fetchall()
    # Reminders anchor to the September item deadline, not the December end.
    assert len(rows) == 4
    assert all("2099-09-16" in row["scheduled_at"] for row in rows)
    assert all("【签到领校服拓印券 到期提醒】" in row["message_text"] for row in rows)
    assert all("2099-12-31" not in row["message_text"] for row in rows)


def test_slots_evening_disabled(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    deadline = datetime(2026, 9, 17, 7, 0, tzinfo=TZ)
    targets = ReminderTargets(
        sessions=["fake:GroupMessage:10001"],
        days_before=1,
        send_time="10:00",
        evening_enabled=False,
    )
    slots = scheduler._slots_for(deadline, targets, NOW)
    assert [kind for _, kind, _ in slots] == ["countdown"]


def test_slots_urgent_disabled(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    deadline = datetime(2026, 9, 16, 20, 0, tzinfo=TZ)
    targets = ReminderTargets(
        sessions=["fake:GroupMessage:10001"],
        days_before=1,
        send_time="10:00",
        urgent_enabled=False,
    )
    slots = scheduler._slots_for(deadline, targets, NOW)
    assert [kind for _, kind, _ in slots] == ["countdown", "countdown"]


def test_slots_slot_missed_beyond_grace_is_dropped(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    deadline = datetime(2026, 9, 17, 7, 0, tzinfo=TZ)
    # D-1 slot was 09-16 10:00, now is 20:00 — 10 hours late, beyond grace.
    now = datetime(2026, 9, 16, 20, 0, tzinfo=TZ)
    targets = ReminderTargets(
        sessions=["fake:GroupMessage:10001"], days_before=1, send_time="10:00"
    )
    slots = scheduler._slots_for(deadline, targets, now)
    # Only the evening slot (21:00, still ahead) survives.
    assert [(m.strftime("%H:%M"), kind) for m, kind, _ in slots] == [
        ("21:00", "evening")
    ]


def test_slots_slot_within_grace_is_kept(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    deadline = datetime(2026, 9, 17, 7, 0, tzinfo=TZ)
    now = datetime(2026, 9, 16, 10, 20, tzinfo=TZ)
    targets = ReminderTargets(
        sessions=["fake:GroupMessage:10001"], days_before=1, send_time="10:00"
    )
    slots = scheduler._slots_for(deadline, targets, now)
    assert any(
        m == datetime(2026, 9, 16, 10, 0, tzinfo=TZ) and kind == "countdown"
        for m, kind, _ in slots
    )


def test_remaining_text_variants(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    deadline = datetime(2026, 9, 20, 20, 0, tzinfo=TZ)
    assert (
        scheduler._remaining_text("countdown", 3, deadline, False, 60)
        == "剩余时间：3 天，该活动将于9月20日结束"
    )
    assert (
        scheduler._remaining_text("countdown", 3, deadline, True, 60)
        == "剩余时间：3 天，该道具将于9月20日到期"
    )
    assert (
        scheduler._remaining_text("countdown", 1, deadline, False, 60)
        == "剩余时间：1 天，该活动将于明天结束"
    )
    assert (
        scheduler._remaining_text("countdown", 0, deadline, False, 60)
        == "剩余时间：该活动将于今天结束（20:00）"
    )
    assert (
        scheduler._remaining_text("evening", 0, deadline, False, 60)
        == "剩余时间：该活动将于明天结束（明早 20:00）"
    )
    assert (
        scheduler._remaining_text("urgent", 0, deadline, False, 60)
        == "剩余时间：该活动将于1 小时后结束"
    )
    assert (
        scheduler._remaining_text("urgent", 0, deadline, True, 30)
        == "剩余时间：该道具将于30 分钟后到期"
    )
    assert (
        scheduler._remaining_text("urgent", 0, deadline, True, 90)
        == "剩余时间：该道具将于1 小时 30 分钟后到期"
    )


def test_create_pending_reminders_is_idempotent(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    announcement_id = _insert_announcement(scheduler.db)
    # 2099-09-17 is a Thursday; 07:00 is inside the maintenance window, so
    # D-1 + evening slots exist (urgent suppressed).
    _insert_activity(
        scheduler.db, announcement_id, end_time="2099-09-17T07:00:00+08:00"
    )
    first = scheduler.create_pending_reminders(TARGETS)
    second = scheduler.create_pending_reminders(TARGETS)
    assert first == 4  # 2 slots x 2 targets
    # Re-running replaces future slots instead of accumulating duplicates.
    assert second == 4
    with scheduler.db.connect() as conn:
        total = int(conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0])
    assert total == 4

    with scheduler.db.connect() as conn:
        rows = conn.execute(
            "SELECT target_type, message_text FROM reminders ORDER BY id"
        ).fetchall()
    assert {row["target_type"] for row in rows} == {"group", "private"}
    assert all("【签到领校服拓印券 结束提醒】" in row["message_text"] for row in rows)
    assert all("待办事项：" in row["message_text"] for row in rows)
    assert all("剩余时间：" in row["message_text"] for row in rows)
    assert any("明早 07:00" in row["message_text"] for row in rows)
    assert all("券/道具消失" not in row["message_text"] for row in rows)


def test_create_pending_reminders_without_targets(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    announcement_id = _insert_announcement(scheduler.db)
    _insert_activity(
        scheduler.db, announcement_id, end_time="2099-09-17T07:00:00+08:00"
    )
    assert scheduler.create_pending_reminders(ReminderTargets(sessions=[])) == 0


def test_create_pending_reminders_refresh_on_config_change(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    announcement_id = _insert_announcement(scheduler.db)
    _insert_activity(
        scheduler.db, announcement_id, end_time="2099-09-17T07:00:00+08:00"
    )
    scheduler.create_pending_reminders(TARGETS)

    # User moves the daily push time; the next full pass must replace the
    # stale future slots instead of adding a second set.
    new_targets = ReminderTargets(
        sessions=["fake:GroupMessage:10001", "fake:FriendMessage:20001"],
        days_before=1,
        send_time="11:30",
    )
    scheduler.create_pending_reminders(new_targets)

    with scheduler.db.connect() as conn:
        rows = conn.execute(
            "SELECT scheduled_at, status FROM reminders ORDER BY id"
        ).fetchall()
    assert len(rows) == 4  # replaced, not duplicated
    assert all("11:30" in row["scheduled_at"] or "21:00" in row["scheduled_at"]
               for row in rows)
    assert all(row["status"] == "pending" for row in rows)


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
            "reminder_evening_time": "20:30",
            "reminder_urgent_minutes": 30,
        }
    )
    assert targets.sessions == [
        "aiocqhttp:GroupMessage:10001",
        "qqofficial:FriendMessage:20001",
    ]
    assert targets.days_before == 2
    assert targets.send_time == "09:30"
    assert targets.evening_enabled is True
    assert targets.evening_time == "20:30"
    assert targets.urgent_minutes == 30
    assert targets.as_pairs() == [
        ("group", "aiocqhttp:GroupMessage:10001"),
        ("private", "qqofficial:FriendMessage:20001"),
    ]


def test_from_config_clamps_and_disables():
    targets = ReminderTargets.from_config({"reminder_days_before": 10})
    assert targets.days_before == 7

    off_switch = ReminderTargets.from_config({"reminder_urgent_enabled": False})
    assert off_switch.urgent_minutes == 0

    zero_minutes = ReminderTargets.from_config({"reminder_urgent_minutes": 0})
    assert zero_minutes.urgent_minutes == 0

    # Missing evening key falls back to the default time, not to "disabled".
    default_evening = ReminderTargets.from_config({})
    assert default_evening.evening_time == "21:00"
    assert default_evening.evening_enabled is True

    empty_evening = ReminderTargets.from_config({"reminder_evening_time": ""})
    assert empty_evening.evening_time == ""


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


def test_merge_dedupes_same_source_link():
    from core.scheduler import merge_reminder_texts

    url = "https://jx3.xoyo.com/announce/gg.html?id=1"
    merged = merge_reminder_texts(
        [
            f"【活动A 到期提醒】\n待办：使用\n链接：{url}",
            f"【活动B 到期提醒】\n待办：领取\n链接：{url}",
        ]
    )
    assert merged.count("链接：") == 1
    assert merged.rfind("链接：") > merged.rfind("【活动B 到期提醒】")
    assert "待办：使用\n\n【活动B" in merged  # link line removed from item A

    other = "https://jx3.xoyo.com/announce/gg.html?id=2"
    mixed = merge_reminder_texts(
        [
            f"【活动A 到期提醒】\n待办：使用\n链接：{url}",
            f"【活动B 到期提醒】\n待办：领取\n链接：{other}",
        ]
    )
    # Different sources keep their own link lines.
    assert mixed.count("链接：") == 2


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


def test_due_reminders_and_mark(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    announcement_id = _insert_announcement(scheduler.db)
    _insert_activity(
        scheduler.db, announcement_id, end_time="2099-09-17T07:00:00+08:00"
    )
    scheduler.create_pending_reminders(TARGETS)
    # Force every slot due in the past, keeping timestamps distinct so the
    # UNIQUE(activity, target, scheduled_at) constraint is not violated.
    # Timestamps are computed in the scheduler timezone (never the machine's).
    with scheduler.db.connect() as conn:
        ids = [
            row["id"]
            for row in conn.execute("SELECT id FROM reminders").fetchall()
        ]
        for index, reminder_id in enumerate(ids, start=1):
            past = (scheduler.now() - timedelta(minutes=index)).isoformat(
                timespec="seconds"
            )
            conn.execute(
                "UPDATE reminders SET scheduled_at = ? WHERE id = ?",
                (past, reminder_id),
            )
    due = scheduler.due_reminders()
    assert len(due) == 4

    scheduler.mark_reminder(int(due[0]["id"]), "sent")
    scheduler.mark_reminder(int(due[1]["id"]), "failed", error="send timeout")

    with scheduler.db.connect() as conn:
        statuses = [
            row["status"]
            for row in conn.execute(
                "SELECT status FROM reminders ORDER BY id"
            ).fetchall()
        ]
        logs = conn.execute("SELECT success, error FROM reminder_logs").fetchall()
    assert sorted(statuses) == sorted(["sent", "failed", "pending", "pending"])
    assert sorted(int(row["success"]) for row in logs) == [0, 1]
    # The two untouched rows are still due.
    assert len(scheduler.due_reminders()) == 2


def test_cancel_reminders_for_announcement(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    announcement_id = _insert_announcement(scheduler.db)
    _insert_activity(
        scheduler.db, announcement_id, end_time="2099-09-17T07:00:00+08:00"
    )
    scheduler.create_pending_reminders(TARGETS)
    cancelled = scheduler.cancel_reminders_for_announcement(announcement_id)
    assert cancelled == 4


def test_status_snapshot_next_run(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    snapshot = scheduler.status_snapshot("00:00")
    assert snapshot["next_fetch_at"].endswith("00:00:00+08:00")
    assert snapshot["last_fetch"] is None
