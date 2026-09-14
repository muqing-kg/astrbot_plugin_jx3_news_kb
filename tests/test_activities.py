"""Tests for activity filtering, time parsing and reminder rendering."""

from __future__ import annotations

import sqlite3

import pytest

from core.activities import (
    ActivityService,
    is_reminder_worthy,
    parse_local_datetime,
)
from core.qa import clean_markdown


def test_clean_markdown_strips_headings_bold_and_code():
    raw = (
        "### 一、联动活动\n"
        "1. **渡厄问心**：参与活动\n"
        "2. 联动签到`第一期`\n"
        "**来源：2026-09-10 版本更新公告**"
    )
    cleaned = clean_markdown(raw)
    assert "###" not in cleaned
    assert "**" not in cleaned
    assert "`" not in cleaned
    assert "一、联动活动" in cleaned
    assert "渡厄问心：参与活动" in cleaned
    assert "联动签到第一期" in cleaned


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
