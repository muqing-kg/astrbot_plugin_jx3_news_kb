"""LLM activity extraction and reminder scheduling."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .database import Database


EXTRACT_SYSTEM_PROMPT = """你从剑网3官方公告中抽取“可提醒的到期事项”。
只输出 JSON，不输出解释。格式：
{"activities":[
  {
    "name":"活动名称",
    "action":"需要玩家完成的动作",
    "category":"game_activity|free_coupon|free_item|reward_claim|other",
    "start_time":"YYYY-MM-DDTHH:MM:SS+08:00 或空字符串",
    "end_time":"YYYY-MM-DDTHH:MM:SS+08:00 或空字符串",
    "item_expiry":"YYYY-MM-DDTHH:MM:SS+08:00 或空字符串",
    "item_name":"券/道具名称，可空",
    "explanation":"到期影响和原因",
    "evidence":"关键原文",
    "confidence":0.0
  }
]}

必须排除：测试服、线下活动、问卷、折扣券/优惠券/几折促销、处罚、维护、BUG修复、纯玩法调整、没有明确截止时间的内容。
必须保留：游戏内活动、签到、代币收集、代币兑换、奖励领取、免费券、免费道具、券或道具消失时间、明确奖励领取期。
日期不足时按公告日期补全年份。时间必须使用中国时区 ISO 8601。
"""


@dataclass(slots=True)
class ExtractedActivity:
    name: str
    action: str = ""
    category: str = "other"
    start_time: str = ""
    end_time: str = ""
    item_expiry: str = ""
    item_name: str = ""
    explanation: str = ""
    evidence: str = ""
    confidence: float = 0.0


EXCLUDED_KEYWORDS = (
    "测试服", "测试资格", "线下", "问卷", "折扣", "优惠", "满减",
    "处罚", "封停", "冻结", "维护", "修复", "bug", "BUG",
)

# Only numeric discounts ("8.5折", "五折") are treated as discount items; a
# bare "折" must not match unrelated characters inside a voucher name.
DISCOUNT_ITEM_PATTERN = re.compile(r"折扣|优惠|满减|半价|[0-9一二三四五六七八九]+(?:\.[0-9]+)?折")

EXCLUDED_CATEGORIES = (
    "test", "offline", "questionnaire", "discount", "punishment", "maintenance",
)


def _title_block(activity: dict[str, Any]) -> str:
    """Name/action/item fields only. Evidence quotes raw announcement text,
    which may legitimately mention e.g. a discount coupon next to a free one."""
    return "\n".join(
        str(activity.get(key, "") or "")
        for key in ("name", "action", "item_name")
    )


def is_reminder_worthy(activity: dict[str, Any]) -> tuple[bool, str]:
    title = _title_block(activity).lower()
    category = str(activity.get("category", "other")).lower()
    if any(keyword.lower() in title for keyword in EXCLUDED_KEYWORDS):
        return False, "excluded_keyword"
    if any(word in category for word in EXCLUDED_CATEGORIES):
        return False, "excluded_category"
    item_name = str(activity.get("item_name", ""))
    if DISCOUNT_ITEM_PATTERN.search(item_name):
        return False, "discount_item"
    deadline = activity.get("end_time") or activity.get("item_expiry")
    if not deadline and not activity.get("start_time"):
        return False, "no_deadline"
    if not deadline and category not in ("game_activity", "reward_claim"):
        return False, "no_deadline_category"
    confidence = float(activity.get("confidence", 0.0))
    if confidence < 0.55:
        return False, "low_confidence"
    return True, "ok"


def parse_local_datetime(value: str, fallback_date: str = "") -> str:
    """Parse Chinese/ISO times into Asia/Shanghai ISO 8601."""
    value = str(value or "").strip()
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        return parsed.isoformat(timespec="seconds")
    except ValueError:
        pass

    fallback = None
    if fallback_date:
        try:
            fallback = datetime.fromisoformat(fallback_date).date()
        except ValueError:
            fallback = None

    # Missing year is common in announcements; use the publication year.
    year = fallback.year if fallback else datetime.now(ZoneInfo("Asia/Shanghai")).year
    match = re.search(
        r"(\d{1,2})月(\d{1,2})日(?:\s*)?"
        r"(?:凌晨|早上|上午|中午|下午|晚上)?\s*(\d{1,2})(?:[:：点](\d{1,2}))?",
        value,
    )
    if not match:
        date_only = re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", value)
        if not date_only:
            return ""
        hour, minute = (0, 0)
        year, month, day = map(int, date_only.groups())
    else:
        month, day, hour_text, minute_text = match.groups()
        month, day = int(month), int(day)
        hour = int(hour_text)
        minute = int(minute_text or 0)
        if "下午" in value and hour < 12:
            hour += 12
        if "晚上" in value and hour < 12:
            hour += 12
        if "凌晨" in value and hour == 12:
            hour = 0
        # “24:00” is commonly used as the end of that day.
        if hour == 24:
            base = datetime(year, month, day, 0, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
            return (base + timedelta(days=1)).isoformat(timespec="seconds")

    return datetime(
        year, month, day, hour, minute, tzinfo=ZoneInfo("Asia/Shanghai")
    ).isoformat(timespec="seconds")


class ActivityService:
    def __init__(self, database: Database, timezone_name: str = "Asia/Shanghai") -> None:
        self.db = database
        self.timezone_name = timezone_name

    async def extract_for_announcements(
        self,
        provider: Any | None,
        announcement_ids: list[int] | None = None,
    ) -> int:
        """Extract from new/changed announcements; return created activity count."""
        with self.db.connect() as conn:
            if announcement_ids:
                marks = ",".join("?" for _ in announcement_ids)
                rows = conn.execute(
                    f"""
                    SELECT id, title, date(published_at) AS announcement_date, content_text
                    FROM announcements
                    WHERE id IN ({marks})
                    """,
                    announcement_ids,
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, title, date(published_at) AS announcement_date, content_text
                    FROM announcements
                    WHERE activity_extracted_at IS NULL
                       OR activity_extracted_at < modified_at
                    ORDER BY published_at DESC
                    LIMIT 50
                    """
                ).fetchall()

        created = 0
        for row in rows:
            activities = await self._extract_one(provider, row)
            with self.db.connect() as conn:
                conn.execute(
                    "DELETE FROM activities WHERE announcement_id = ?", (row["id"],)
                )
                for activity in activities:
                    conn.execute(
                        """
                        INSERT INTO activities(
                            announcement_id, name, action, category, start_time,
                            end_time, item_expiry, item_name, explanation,
                            evidence, confidence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["id"], activity.name, activity.action, activity.category,
                            activity.start_time, activity.end_time, activity.item_expiry,
                            activity.item_name, activity.explanation, activity.evidence,
                            activity.confidence,
                        ),
                    )
                    created += 1
                conn.execute(
                    """
                    UPDATE announcements
                    SET activity_extracted_at = datetime('now', 'localtime')
                    WHERE id = ?
                    """,
                    (row["id"],),
                )
        return created

    async def _extract_one(
        self, provider: Any | None, row: Any
    ) -> list[ExtractedActivity]:
        if provider is None:
            return []
        content = str(row["content_text"] or "")[:16000]
        prompt = (
            f"公告ID：{row['id']}\n公告日期：{row['announcement_date']}\n"
            f"标题：{row['title']}\n正文：\n{content}"
        )
        response = await provider.text_chat(prompt=prompt, system_prompt=EXTRACT_SYSTEM_PROMPT)
        text = str(getattr(response, "completion_text", "") or "").strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

        activities: list[ExtractedActivity] = []
        for item in payload.get("activities", []) if isinstance(payload, dict) else []:
            if not isinstance(item, dict):
                continue
            worthy, _ = is_reminder_worthy(item)
            if not worthy:
                continue
            activities.append(
                ExtractedActivity(
                    name=str(item.get("name", "")).strip(),
                    action=str(item.get("action", "")).strip(),
                    category=str(item.get("category", "other")).strip(),
                    start_time=parse_local_datetime(
                        item.get("start_time"), str(row["announcement_date"] or "")
                    ),
                    end_time=parse_local_datetime(
                        item.get("end_time"), str(row["announcement_date"] or "")
                    ),
                    item_expiry=parse_local_datetime(
                        item.get("item_expiry"), str(row["announcement_date"] or "")
                    ),
                    item_name=str(item.get("item_name", "")).strip(),
                    explanation=str(item.get("explanation", "")).strip(),
                    evidence=str(item.get("evidence", "")).strip(),
                    confidence=float(item.get("confidence", 0.0)),
                )
            )
        return [item for item in activities if item.name]

    def reminder_message(self, row: Any) -> str:
        deadline = row["item_expiry"] or row["end_time"]
        deadline_type = "券/道具消失" if row["item_expiry"] else "活动/领取截止"
        item_text = f"\n相关物品：{row['item_name']}" if row["item_name"] else ""
        explanation = f"\n说明：{row['explanation']}" if row["explanation"] else ""
        return (
            f"【剑网3到期提醒】\n"
            f"活动：{row['name']}\n"
            f"待办：{row['action'] or '请及时处理'}\n"
            f"{deadline_type}：{format_deadline(deadline)}{item_text}{explanation}\n"
            f"来源：{row['announcement_date']}《{row['title']}》\n"
            f"链接：{row['url']}"
        )


def format_deadline(value: str | None) -> str:
    """Render an ISO timestamp as 'YYYY-MM-DD HH:MM'; pass through on failure."""
    value = str(value or "").strip()
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value
