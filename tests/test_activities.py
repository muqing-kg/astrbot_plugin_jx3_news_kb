"""Tests for activity filtering, time parsing and reminder rendering."""

from __future__ import annotations

import sqlite3

import pytest

from core.activities import (
    ActivityService,
    is_reminder_worthy,
    parse_local_datetime,
)


def test_free_coupon_with_discount_mention_in_evidence_is_kept():
    """校服任选拓印券的原文里提到“角色折扣券”，仍属于免费券，应保留。"""
    activity = {
        "name": "签到领校服任选拓印券",
        "action": "使用免费校服拓印券",
        "category": "free_coupon",
        "item_name": "校服任选拓印券",
        "end_time": "2026-09-17 07:00:00",
        "evidence": "使用角色折扣券可半价拓印，活动期间签到免费领取校服拓印券",
        "confidence": 0.9,
    }
    worthy, reason = is_reminder_worthy(activity)
    assert worthy, reason


def test_discount_voucher_is_excluded():
    activity = {
        "name": "外装拓印五折券发放",
        "action": "领取外装拓印券",
        "category": "free_coupon",
        "item_name": "外装拓印五折券",
        "end_time": "2026-09-17 07:00:00",
        "confidence": 0.9,
    }
    worthy, reason = is_reminder_worthy(activity)
    assert not worthy
    assert reason == "discount_item"


def test_testserver_and_offline_and_questionnaire_excluded():
    for name in ("测试服通宝激励", "线下嘉年华活动", "有奖问卷调查"):
        activity = {
            "name": name,
            "category": "game_activity",
            "end_time": "2026-09-17 07:00:00",
            "confidence": 0.9,
        }
        worthy, _ = is_reminder_worthy(activity)
        assert not worthy, name


def test_no_deadline_excluded():
    activity = {
        "name": "版本玩法调整说明",
        "category": "game_activity",
        "confidence": 0.9,
    }
    worthy, reason = is_reminder_worthy(activity)
    assert not worthy
    assert reason == "no_deadline"


def test_low_confidence_excluded():
    activity = {
        "name": "活动奖励",
        "category": "reward_claim",
        "end_time": "2026-09-17 07:00:00",
        "confidence": 0.3,
    }
    worthy, reason = is_reminder_worthy(activity)
    assert not worthy
    assert reason == "low_confidence"


def test_parse_local_datetime_variants():
    assert (
        parse_local_datetime("9月17日07:00", "2026-08-13")
        == "2026-09-17T07:00:00+08:00"
    )
    assert (
        parse_local_datetime("9月17日晚上9点30分", "2026-08-13")
        == "2026-09-17T21:30:00+08:00"
    )
    assert (
        parse_local_datetime("9月17日24:00", "2026-08-13")
        == "2026-09-18T00:00:00+08:00"
    )
    assert (
        parse_local_datetime("2026-09-17T07:00:00+08:00")
        == "2026-09-17T07:00:00+08:00"
    )
    assert parse_local_datetime("") == ""
    assert parse_local_datetime("待定") == ""


@pytest.fixture()
def db(tmp_path):
    from core.database import Database

    database = Database(tmp_path / "test.sqlite3")
    database.initialize()
    return database


def _insert_announcement(db, title="版本更新公告", content="活动内容", published="2026-08-13T10:00:00+08:00"):
    with db.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO announcements(
                source_id, title, type, url, content_text, raw_json,
                content_hash, published_at, updated_at_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "1", title, "官方公告", f"https://jx3.xoyo.com/announce/{title}",
                content, "{}", f"hash-{title}-{published}", published, published,
            ),
        )
        return int(cursor.lastrowid)


@pytest.mark.asyncio()
async def test_reextract_updates_in_place_without_id_change(db):
    """Re-extracting the same announcement must keep activity ids stable so
    the reminders UNIQUE constraint keeps preventing duplicates."""
    announcement_id = _insert_announcement(db)
    payload_first = (
        '{"activities":[{"name":"签到领校服拓印券","action":"使用","category":"free_coupon",'
        '"end_time":"9月17日07:00","item_name":"校服任选拓印券","confidence":0.9}]}'
    )
    service = ActivityService(db)
    await service.extract_for_announcements(_FakeProvider(payload_first), [announcement_id])

    with db.connect() as conn:
        first_id = conn.execute("SELECT id FROM activities").fetchone()["id"]
        first_count = conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0]

    # Same announcement revised: LLM returns a changed deadline this time.
    payload_second = (
        '{"activities":[{"name":"签到领校服拓印券","action":"使用","category":"free_coupon",'
        '"end_time":"9月18日07:00","item_name":"校服任选拓印券","confidence":0.9}]}'
    )
    await service.extract_for_announcements(_FakeProvider(payload_second), [announcement_id])

    with db.connect() as conn:
        row = conn.execute("SELECT id, end_time FROM activities").fetchone()
    assert first_count == 1
    assert row["id"] == first_id  # stable id
    assert row["end_time"] == "2026-09-18T07:00:00+08:00"


@pytest.mark.asyncio()
async def test_reextract_drops_removed_activities(db):
    announcement_id = _insert_announcement(db)
    payload_first = (
        '{"activities":['
        '{"name":"签到领校服拓印券","action":"使用","category":"free_coupon",'
        '"end_time":"9月17日07:00","item_name":"校服任选拓印券","confidence":0.9},'
        '{"name":"纸鸢载愿","action":"参与","category":"game_activity",'
        '"end_time":"9月20日22:00","confidence":0.9}]}'
    )
    service = ActivityService(db)
    await service.extract_for_announcements(_FakeProvider(payload_first), [announcement_id])
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0] == 2

    # Revision keeps only one of the two activities.
    payload_second = (
        '{"activities":[{"name":"签到领校服拓印券","action":"使用","category":"free_coupon",'
        '"end_time":"9月17日07:00","item_name":"校服任选拓印券","confidence":0.9}]}'
    )
    await service.extract_for_announcements(_FakeProvider(payload_second), [announcement_id])
    with db.connect() as conn:
        names = [r["name"] for r in conn.execute("SELECT name FROM activities").fetchall()]
    assert names == ["签到领校服拓印券"]

class _FakeProvider:
    def __init__(self, payload: str):
        self.payload = payload

    async def text_chat(self, prompt: str, system_prompt: str = "", **kwargs):
        class _Response:
            completion_text = self.payload

        return _Response()


@pytest.mark.asyncio()
async def test_extract_skips_unworthy_and_stores_times(db):
    from core.ingest import IngestService

    payload = (
        '{"activities":['
        '{"name":"签到领校服拓印券","action":"使用免费校服拓印券",'
        '"category":"free_coupon","end_time":"9月17日07:00",'
        '"item_name":"校服任选拓印券","evidence":"活动期间签到免费领取",'
        '"confidence":0.9},'
        '{"name":"外装拓印五折券","action":"领取","category":"free_coupon",'
        '"end_time":"9月17日07:00","item_name":"外装拓印五折券",'
        '"confidence":0.9}'
        "]}"
    )
    provider = _FakeProvider(payload)
    announcement_id = _insert_announcement(db)

    service = ActivityService(db)
    created = await service.extract_for_announcements(provider, [announcement_id])
    assert created == 1

    with db.connect() as conn:
        row = conn.execute("SELECT * FROM activities").fetchone()
        assert row["name"] == "签到领校服拓印券"
        assert row["end_time"] == "2026-09-17T07:00:00+08:00"


def test_reminder_message_template(db):
    announcement_id = _insert_announcement(db)
    with db.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO activities(
                announcement_id, name, action, category, end_time,
                item_name, explanation
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                announcement_id, "签到领校服拓印券", "使用免费校服拓印券",
                "free_coupon", "2026-09-17T07:00:00+08:00", "校服任选拓印券",
                "券到期后会消失",
            ),
        )
        activity_id = int(cursor.lastrowid)

    with db.connect() as conn:
        row = conn.execute(
            """
            SELECT a.*, date(an.published_at) AS announcement_date,
                   an.title, an.url
            FROM activities a JOIN announcements an ON an.id = a.announcement_id
            WHERE a.id = ?
            """,
            (activity_id,),
        ).fetchone()

    service = ActivityService(db)
    message = service.reminder_message(
        row, "剩余时间：1 天，该活动将于明天结束"
    )
    assert "【签到领校服拓印券 结束提醒】" in message
    assert "待办事项：使用免费校服拓印券" in message
    assert "相关物品：校服任选拓印券" in message
    assert "说明：券到期后会消失" in message
    assert "剩余时间：1 天，该活动将于明天结束" in message
    assert "链接：" in message
    assert "券/道具消失" not in message
    assert "活动：" not in message

    item_row = dict(row)
    item_row["item_expiry"] = "2026-09-17T07:00:00+08:00"
    item_message = service.reminder_message(
        item_row, "剩余时间：1 天，该道具将于明天到期"
    )
    assert "【签到领校服拓印券 到期提醒】" in item_message
    assert "剩余时间：1 天，该道具将于明天到期" in item_message


def _touch_announcement(db, announcement_id: int) -> None:
    """Mark the row as changed so the next pass re-extracts it."""
    with db.connect() as conn:
        conn.execute(
            "UPDATE announcements SET modified_at = datetime('now', 'localtime', '+1 minute')"
            " WHERE id = ?",
            (announcement_id,),
        )


@pytest.mark.asyncio()
async def test_regex_reread_retires_rows_it_no_longer_finds(db):
    """A second pattern read replaces the first instead of stacking on it."""
    announcement_id = _insert_announcement(
        db, title="中秋活动", content="活动时间：9月20日7:00—9月25日7:00"
    )
    service = ActivityService(db)
    assert await service.extract_for_announcements(None, [announcement_id]) == 1

    with db.connect() as conn:
        conn.execute(
            "UPDATE announcements SET content_text = ? WHERE id = ?",
            (
                "活动时间：9月20日7:00—9月25日7:00\n"
                "活动时间：10月1日7:00—10月8日7:00",
                announcement_id,
            ),
        )
    _touch_announcement(db, announcement_id)

    assert await service.extract_for_announcements(None, [announcement_id]) == 2
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT name, end_time FROM activities WHERE announcement_id = ?"
            " ORDER BY end_time",
            (announcement_id,),
        ).fetchall()
    # Two windows, no leftover row from the first read.
    assert [r["end_time"] for r in rows] == [
        "2026-09-25T07:00:00+08:00",
        "2026-10-08T07:00:00+08:00",
    ]
    assert [r["name"] for r in rows] == ["中秋活动（9月25日）", "中秋活动（10月8日）"]


@pytest.mark.asyncio()
async def test_missing_provider_keeps_existing_activities(db):
    """No LLM must not delete what a previous run already found."""
    announcement_id = _insert_announcement(
        db, content="活动时间：9月20日7:00—9月25日7:00"
    )
    payload = (
        '{"activities":[{"name":"签到领校服拓印券","action":"使用","category":"free_coupon",'
        '"end_time":"9月17日07:00","item_name":"校服任选拓印券","confidence":0.9}]}'
    )
    service = ActivityService(db)
    await service.extract_for_announcements(_FakeProvider(payload), [announcement_id])

    _touch_announcement(db, announcement_id)
    created = await service.extract_for_announcements(None, [announcement_id])

    assert created == 0
    with db.connect() as conn:
        rows = conn.execute("SELECT name, end_time FROM activities").fetchall()
        mode = conn.execute(
            "SELECT activity_extract_mode FROM announcements WHERE id = ?",
            (announcement_id,),
        ).fetchone()[0]
    # The pattern would have found a different window; the model's read stands.
    assert [(r["name"], r["end_time"]) for r in rows] == [
        ("签到领校服拓印券", "2026-09-17T07:00:00+08:00")
    ]
    assert mode == "llm"


@pytest.mark.asyncio()
async def test_regex_fallback_reads_date_range(db):
    announcement_id = _insert_announcement(
        db,
        title="中秋活动",
        content="活动时间：9月20日7:00—10月8日维护前\n参与即可获得奖励。",
        published="2026-09-11T10:00:00+08:00",
    )
    service = ActivityService(db)
    created = await service.extract_for_announcements(None, [announcement_id])

    assert created == 1
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM activities").fetchone()
    # "活动时间：" is a label, not a name: the announcement title is used.
    assert row["name"] == "中秋活动"
    assert row["start_time"] == "2026-09-20T07:00:00+08:00"
    assert row["end_time"] == "2026-10-08T07:00:00+08:00"


@pytest.mark.asyncio()
async def test_regex_keeps_each_window_of_one_announcement(db):
    announcement_id = _insert_announcement(
        db,
        title="双周活动",
        content=(
            "活动时间：9月20日7:00—9月27日7:00\n"
            "活动时间：10月1日7:00—10月8日7:00\n"
        ),
        published="2026-09-11T10:00:00+08:00",
    )
    service = ActivityService(db)
    assert await service.extract_for_announcements(None, [announcement_id]) == 2
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT name, end_time FROM activities ORDER BY end_time"
        ).fetchall()
    assert [r["end_time"] for r in rows] == [
        "2026-09-27T07:00:00+08:00",
        "2026-10-08T07:00:00+08:00",
    ]
    # Same fallback title twice: the reminder titles must stay tellable apart.
    assert [r["name"] for r in rows] == ["双周活动（9月27日）", "双周活动（10月8日）"]


@pytest.mark.asyncio()
async def test_regex_reads_window_without_times(db):
    """Announcements often write "9月20日—9月25日" with no clock time."""
    announcement_id = _insert_announcement(
        db,
        title="无时刻活动",
        content="活动时间：9月20日—9月25日，期间每日可领取。",
        published="2026-09-11T10:00:00+08:00",
    )
    service = ActivityService(db)
    assert await service.extract_for_announcements(None, [announcement_id]) == 1
    with db.connect() as conn:
        row = conn.execute("SELECT start_time, end_time FROM activities").fetchone()
    assert row["start_time"] == "2026-09-20T00:00:00+08:00"
    assert row["end_time"] == "2026-09-25T23:59:00+08:00"


@pytest.mark.asyncio()
async def test_regex_miss_marks_the_announcement_as_read(db):
    """No model and no pattern hit: the attempt is recorded so it can be pruned."""
    announcement_id = _insert_announcement(db, content="本公告没有任何日期区间。")
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO activities(announcement_id, name, category, end_time)"
            " VALUES (?, '旧活动', 'game_activity', '2026-09-17T07:00:00+08:00')",
            (announcement_id,),
        )

    service = ActivityService(db)
    assert await service.extract_for_announcements(None, [announcement_id]) == 0

    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0] == 0
        stamped, mode = conn.execute(
            "SELECT activity_extracted_at, activity_extract_mode"
            " FROM announcements WHERE id = ?",
            (announcement_id,),
        ).fetchone()
    assert stamped
    assert mode == "regex"


def test_regex_treats_maintenance_words_as_seven_am():
    from core.activities import regex_extract

    acts = regex_extract(
        "版本更新公告", "活动时间：9月20日维护后—10月8日维护前", "2026-09-18"
    )
    assert len(acts) == 1
    assert acts[0].start_time.startswith("2026-09-20T07:00")
    assert acts[0].end_time.startswith("2026-10-08T07:00")


def test_regex_places_a_december_window_in_the_previous_year():
    from core.activities import regex_extract

    acts = regex_extract("元旦活动", "活动时间：12月28日—1月6日", "2027-01-03")
    assert len(acts) == 1
    assert acts[0].start_time.startswith("2026-12-28")
    assert acts[0].end_time.startswith("2027-01-06")


def test_regex_rolls_a_january_window_into_the_next_year():
    from core.activities import regex_extract

    acts = regex_extract("元旦活动", "活动时间：1月6日—1月20日", "2026-12-28")
    assert len(acts) == 1
    assert acts[0].start_time.startswith("2027-01-06")
    assert acts[0].end_time.startswith("2027-01-20")


def test_confidence_accepts_non_numeric_values():
    from core.activities import _to_confidence, is_reminder_worthy

    assert _to_confidence(0.9) == 0.9
    assert _to_confidence("0.7") == 0.7
    assert _to_confidence("high") == 0.0
    assert _to_confidence(None) == 0.0

    activity = {
        "name": "签到活动",
        "action": "领取",
        "category": "login_reward",
        "end_time": "2026-09-17T07:00:00+08:00",
        "confidence": "high",
    }
    # An unreadable confidence filters the item instead of raising.
    assert is_reminder_worthy(activity) == (False, "low_confidence")


async def test_same_name_in_one_batch_updates_instead_of_twinning(db):
    payload = (
        '{"activities":['
        '{"name":"签到活动","action":"领取","category":"login_reward",'
        '"start_time":"2026-09-13T00:00","end_time":"2026-09-17T07:00","confidence":0.9},'
        '{"name":"签到活动","action":"领取","category":"login_reward",'
        '"start_time":"2026-09-18T00:00","end_time":"2026-09-25T07:00","confidence":0.9}]}'
    )
    announcement_id = _insert_announcement(db)
    service = ActivityService(db)
    await service.extract_for_announcements(_FakeProvider(payload), [announcement_id])

    with db.connect() as conn:
        rows = conn.execute("SELECT name, end_time FROM activities").fetchall()
    assert len(rows) == 1
    assert rows[0]["end_time"].startswith("2026-09-25")


def test_regex_ignores_a_placeholder_publication_year():
    from datetime import datetime

    from core.activities import regex_extract
    from zoneinfo import ZoneInfo

    acts = regex_extract("活动", "活动时间：9月20日—9月25日", "1970-01-01")
    year = datetime.now(ZoneInfo("Asia/Shanghai")).year
    assert len(acts) == 1
    assert acts[0].start_time.startswith(f"{year}-09-20")


def test_pending_rows_skip_announcements_without_a_body(db):
    body_less = _insert_announcement(db, title="待补正文公告", content="")
    _insert_announcement(db, content="活动时间：9月20日—9月25日")

    rows = ActivityService(db)._pending_rows(None, False)

    assert body_less not in [row["id"] for row in rows]
