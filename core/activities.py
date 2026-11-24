"""Activity extraction (LLM first, date-range pattern as fallback)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .database import Database

logger = logging.getLogger(__name__)


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

# Fallback extractor. Most announcements state the window as
# "9月20日7:00—10月8日维护前", and a good share write "维护后" on the opening
# date as well. A miss leaves the announcement marked regex so a working LLM
# re-reads it later, instead of guessing a deadline.
DATE_RANGE_PATTERN = re.compile(
    r"(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*日"
    r"\s*(?:凌晨|早上|上午|中午|下午|晚上)?\s*(?:(\d{1,2})\s*[:：]\s*(\d{2}))?"
    r"\s*(维护前|维护后|更新后)?\s*"
    r"(?:—|–|－|-|~|～|至|到)+\s*"
    r"(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*日"
    r"\s*(?:凌晨|早上|上午|中午|下午|晚上)?\s*(?:(\d{1,2})\s*[:：]\s*(\d{2}))?"
    r"\s*(维护前|维护后|更新后)?"
)

REGEX_CONFIDENCE = 0.6
# "维护前" means the window closes when the server goes down; JX3 maintenance
# starts at 5:00 and finishes around 7:00, so 7:00 of that day is the deadline.
MAINTENANCE_END_HOUR = 7


def _make_moment(
    year: int, month: int, day: int, hour: int, minute: int, tz: ZoneInfo
) -> datetime | None:
    try:
        if hour >= 24:
            base = datetime(year, month, day, tzinfo=tz) + timedelta(days=1)
            return base.replace(hour=hour - 24, minute=minute)
        return datetime(year, month, day, hour, minute, tzinfo=tz)
    except ValueError:
        return None


def _name_for(text: str, index: int, fallback: str) -> str:
    """Reuse the line introducing the window ("活动名称：xxx") when it is short."""
    start = text.rfind("\n", 0, index) + 1
    line = text[start:index].strip(" \t\u3000：:·、-—")
    if (
        not line
        or len(line) > 30
        or line.endswith(("时间", "日期", "期间"))
        or DATE_RANGE_PATTERN.search(line)
    ):
        return fallback
    return line


def regex_extract(
    title: str,
    content_text: str,
    announcement_date: str = "",
    timezone_name: str = "Asia/Shanghai",
) -> list[ExtractedActivity]:
    """Pull dated windows out of raw announcement text without an LLM."""
    text = str(content_text or "")
    if not text:
        return []
    tz = ZoneInfo(timezone_name)
    fallback_day: date | None = None
    try:
        fallback_day = date.fromisoformat(str(announcement_date or "")[:10])
    except ValueError:
        fallback_day = None
    # Old rows carry a 1970 placeholder; treating it as the publication year
    # would schedule every window into the past.
    if fallback_day is not None and fallback_day.year < 2000:
        fallback_day = None
    fallback_year = fallback_day.year if fallback_day else datetime.now(tz).year
    fallback_name = str(title or "").strip() or "活动"

    found: dict[str, ExtractedActivity] = {}
    for match in DATE_RANGE_PATTERN.finditer(text):
        (
            start_year, smonth, sday, shour, sminute, stail,
            end_year, emonth, eday, ehour, eminute, tail,
        ) = match.groups()
        start_value = int(start_year or fallback_year)
        if shour:
            start_hour, start_minute = int(shour), int(sminute or 0)
        elif stail:
            start_hour, start_minute = MAINTENANCE_END_HOUR, 0
        else:
            start_hour, start_minute = 0, 0
        start = _make_moment(
            start_value, int(smonth), int(sday), start_hour, start_minute, tz
        )
        # The pattern has no year. Taking the publication year is right for the
        # usual case, but an early-January announcement about "12月28日" would
        # land eleven months in the future, and a late-December announcement
        # about "1月6日" would land eleven months in the past. Only a gap that
        # large is a year boundary; a window a few weeks out is normal.
        if start is not None and fallback_day is not None:
            gap = (start.date() - fallback_day).days
            if gap > 300:
                start_value -= 1
            elif gap < -30:
                start_value += 1
            if start_value != int(start_year or fallback_year):
                start = _make_moment(
                    start_value, int(smonth), int(sday), start_hour, start_minute, tz
                )
        if start is None:
            continue
        if ehour:
            end_hour, end_minute = int(ehour), int(eminute or 0)
        elif tail:
            end_hour, end_minute = MAINTENANCE_END_HOUR, 0
        else:
            end_hour, end_minute = 23, 59
        end_value = int(end_year or start_value or fallback_year)
        end = _make_moment(end_value, int(emonth), int(eday), end_hour, end_minute, tz)
        if end is None or end <= start:
            end = _make_moment(
                end_value + 1, int(emonth), int(eday), end_hour, end_minute, tz
            )
        if end is None or end <= start:
            continue

        activity = ExtractedActivity(
            name=_name_for(text, match.start(), fallback_name),
            action="查看公告并在截止前完成领取或参与",
            category="game_activity",
            start_time=start.isoformat(timespec="seconds"),
            end_time=end.isoformat(timespec="seconds"),
            explanation="公告写明的活动结束时间，过期后无法继续参与。",
            evidence=match.group(0).strip(),
            confidence=REGEX_CONFIDENCE,
        )
        worthy, _ = is_reminder_worthy(
            {
                "name": activity.name,
                "action": activity.action,
                "item_name": activity.item_name,
                "category": activity.category,
                "start_time": activity.start_time,
                "end_time": activity.end_time,
                "confidence": activity.confidence,
            }
        )
        if worthy:
            # Keyed by window, not just name: one announcement often lists
            # several windows that share the same leading label.
            found.setdefault(f"{activity.name}|{activity.end_time}", activity)

    # Two windows of one announcement may fall back to the same title; keep
    # the reminder titles distinguishable.
    seen: dict[str, int] = {}
    for activity in found.values():
        seen[activity.name] = seen.get(activity.name, 0) + 1
    for activity in found.values():
        if seen[activity.name] > 1:
            end = datetime.fromisoformat(activity.end_time)
            activity.name = f"{activity.name}（{end.month}月{end.day}日）"
    return list(found.values())


def _to_confidence(value: Any) -> float:
    """Model output occasionally carries null or prose in ``confidence``."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
    confidence = _to_confidence(activity.get("confidence"))
    if confidence < 0.55:
        return False, "low_confidence"
    return True, "ok"


def parse_local_datetime(
    value: str,
    fallback_date: str = "",
    timezone_name: str = "Asia/Shanghai",
    end_of_day: bool = True,
) -> str:
    """Parse Chinese/ISO times into an ISO 8601 string in the plugin timezone.

    A value that carries no clock time is read as the end of that day (or its
    start, when ``end_of_day`` is false): "10月8日" as a deadline means 23:59,
    not midnight, or the deadline would look expired a full day early. Never
    raises: a malformed value becomes an empty string so one bad row cannot
    abort a batch.
    """
    tz = ZoneInfo(timezone_name)
    text = str(value or "").strip()
    if not text:
        return ""
    default_hour, default_minute = (23, 59) if end_of_day else (0, 0)

    # "维护前"/"维护后"指当天开服时刻，取维护结束的 7:00。
    if "维护" in text or "更新后" in text:
        default_hour, default_minute = MAINTENANCE_END_HOUR, 0

    iso = re.search(
        r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})(?:[T\s](\d{1,2}):(\d{2}))?",
        text,
    )
    if iso:
        year, month, day, hour, minute = iso.groups()
        try:
            return datetime(
                int(year), int(month), int(day),
                int(hour) if hour else default_hour,
                int(minute) if hour else default_minute,
                tzinfo=tz,
            ).isoformat(timespec="seconds")
        except ValueError:
            return ""

    fallback = None
    if fallback_date:
        try:
            fallback = datetime.fromisoformat(fallback_date).date()
        except ValueError:
            fallback = None

    # Missing year is common in announcements; use the publication year.
    year = fallback.year if fallback else datetime.now(tz).year
    match = re.search(
        r"(\d{1,2})月(\d{1,2})日(?:\s*)?"
        r"(?:凌晨|早上|上午|中午|下午|晚上)?\s*(?:(\d{1,2})(?:[:：点](\d{1,2}))?)?",
        text,
    )
    if not match:
        return ""
    month, day, hour_text, minute_text = match.groups()
    if hour_text:
        hour, minute = int(hour_text), int(minute_text or 0)
        if "下午" in text and hour < 12:
            hour += 12
        if "晚上" in text and hour < 12:
            hour += 12
        if "凌晨" in text and hour == 12:
            hour = 0
    else:
        hour, minute = default_hour, default_minute
    try:
        if hour >= 24:
            # "24:00" 是当天结束的常见写法。
            base = datetime(
                year, int(month), int(day), tzinfo=tz
            ) + timedelta(days=1)
            return base.replace(hour=hour - 24, minute=minute).isoformat(
                timespec="seconds"
            )
        return datetime(
            year, int(month), int(day), hour, minute, tzinfo=tz
        ).isoformat(timespec="seconds")
    except ValueError:
        return ""


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
        rows = self._pending_rows(announcement_ids, retry_regex=provider is not None)
        created = 0
        for row in rows:
            activities = await self._extract_one(provider, row)
            # None means the LLM was unavailable or unusable: fall back to the
            # pattern, and never treat that as "this announcement has nothing".
            authoritative = activities is not None
            mode = "llm"
            if activities is None:
                if row["activity_extract_mode"] == "llm":
                    # A pattern read must never overwrite what the model found.
                    continue
                mode = "regex"
                activities = regex_extract(
                    row["title"], row["content_text"],
                    str(row["announcement_date"] or ""), self.timezone_name,
                )
            if not activities and not authoritative:
                logger.debug(
                    "公告 %s 无可用抽取结果（模式=%s），保留既有活动", row["id"], mode
                )
                continue
            created += self._store(row, activities, mode, authoritative)
        return created

    def _pending_rows(
        self, announcement_ids: list[int] | None, retry_regex: bool
    ) -> list[Any]:
        with self.db.connect() as conn:
            if announcement_ids:
                marks = ",".join("?" for _ in announcement_ids)
                rows = conn.execute(
                    f"""
                    SELECT id, title, date(published_at) AS announcement_date, content_text,
                           activity_extract_mode
                    FROM announcements
                    WHERE id IN ({marks})
                    """,
                    announcement_ids,
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, title, date(published_at) AS announcement_date, content_text,
                           activity_extract_mode
                    FROM announcements
                    WHERE (
                            activity_extracted_at IS NULL
                         OR activity_extracted_at < modified_at
                         OR (? AND activity_extract_mode <> 'llm')
                        )
                      AND TRIM(COALESCE(content_text, '')) <> ''
                    ORDER BY activity_extracted_at IS NULL DESC, published_at DESC
                    LIMIT 50
                    """,
                    (1 if retry_regex else 0,),
                ).fetchall()
        return rows

    def _store(
        self,
        row: Any,
        activities: list[ExtractedActivity],
        mode: str,
        authoritative: bool,
    ) -> int:
        created = 0
        with self.db.connect() as conn:
            # Re-extraction updates existing rows in place so activity ids stay
            # stable: a changing id would invalidate the reminders UNIQUE
            # constraint and let duplicate reminders pile up.
            existing = {
                r["name"]: r["id"]
                for r in conn.execute(
                    "SELECT id, name FROM activities WHERE announcement_id = ?",
                    (row["id"],),
                ).fetchall()
            }
            for activity in activities:
                values = (
                    row["id"], activity.name, activity.action, activity.category,
                    activity.start_time, activity.end_time, activity.item_expiry,
                    activity.item_name, activity.explanation, activity.evidence,
                    activity.confidence,
                )
                old_id = existing.get(activity.name)
                if old_id is not None:
                    conn.execute(
                        """
                        UPDATE activities
                        SET action = ?, category = ?,
                            start_time = ?, end_time = ?, item_expiry = ?,
                            item_name = ?, explanation = ?, evidence = ?,
                            confidence = ?, modified_at =
                                datetime('now', 'localtime')
                        WHERE id = ?
                        """,
                        (*values[2:], old_id),
                    )
                else:
                    cursor = conn.execute(
                        """
                        INSERT INTO activities(
                            announcement_id, name, action, category, start_time,
                            end_time, item_expiry, item_name, explanation,
                            evidence, confidence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        values,
                    )
                    # A second entry reusing the name updates this row instead
                    # of inserting a twin that later passes never touch.
                    existing[activity.name] = int(cursor.lastrowid)
                    created += 1
            # Retiring rows is a privilege of the latest full read. A pattern
            # miss must not delete what the LLM found, but a previous pattern
            # read has nothing worth protecting: leaving its rows behind would
            # duplicate them whenever a window is added or its label changes.
            if authoritative or row["activity_extract_mode"] != "llm":
                keep = [activity.name for activity in activities]
                if keep:
                    marks = ",".join("?" for _ in keep)
                    conn.execute(
                        f"DELETE FROM activities WHERE announcement_id = ? "
                        f"AND name NOT IN ({marks})",
                        (row["id"], *keep),
                    )
                else:
                    conn.execute(
                        "DELETE FROM activities WHERE announcement_id = ?",
                        (row["id"],),
                    )
            conn.execute(
                """
                UPDATE announcements
                SET activity_extracted_at = datetime('now', 'localtime'),
                    activity_extract_mode = ?
                WHERE id = ?
                """,
                (mode, row["id"]),
            )
        return created

    async def _extract_one(
        self, provider: Any | None, row: Any
    ) -> list[ExtractedActivity] | None:
        """``None`` means the LLM could not be used; ``[]`` means it found none."""
        if provider is None:
            return None
        content = str(row["content_text"] or "")[:16000]
        prompt = (
            f"公告ID：{row['id']}\n公告日期：{row['announcement_date']}\n"
            f"标题：{row['title']}\n正文：\n{content}"
        )
        try:
            response = await provider.text_chat(
                prompt=prompt, system_prompt=EXTRACT_SYSTEM_PROMPT
            )
        except Exception as exc:  # noqa: BLE001 - one bad call must not stop the batch
            logger.warning("公告 %s 活动抽取调用失败：%s", row["id"], exc)
            return None
        text = str(getattr(response, "completion_text", "") or "").strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            logger.warning("公告 %s 活动抽取未返回 JSON", row["id"])
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("公告 %s 活动抽取 JSON 解析失败", row["id"])
            return None

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
                        item.get("start_time"), str(row["announcement_date"] or ""),
                        self.timezone_name, end_of_day=False,
                    ),
                    end_time=parse_local_datetime(
                        item.get("end_time"), str(row["announcement_date"] or ""),
                        self.timezone_name,
                    ),
                    item_expiry=parse_local_datetime(
                        item.get("item_expiry"), str(row["announcement_date"] or ""),
                        self.timezone_name,
                    ),
                    item_name=str(item.get("item_name", "")).strip(),
                    explanation=str(item.get("explanation", "")).strip(),
                    evidence=str(item.get("evidence", "")).strip(),
                    confidence=_to_confidence(item.get("confidence")),
                )
            )
        return [item for item in activities if item.name]

    def reminder_message(self, row: Any, remaining: str = "") -> str:
        """Render one reminder; ``remaining`` is the ``剩余时间：`` line."""
        suffix = "到期提醒" if row["item_expiry"] else "结束提醒"
        lines = [
            f"【{row['name']} {suffix}】",
            f"待办事项：{row['action'] or '请及时处理'}",
        ]
        if row["item_name"]:
            lines.append(f"相关物品：{row['item_name']}")
        if row["explanation"]:
            lines.append(f"说明：{row['explanation']}")
        if remaining:
            lines.append(remaining)
        lines.append(f"链接：{row['url']}")
        return "\n".join(lines)
