"""Scheduled fetch, activity extraction, reminder creation and dispatch."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .activities import ActivityService
from .database import Database
from .ingest import IngestService
from .jx3api import NewsClient

logger = logging.getLogger(__name__)

# A gap longer than this since the last successful fetch means announcements
# were missed and a large catch-up fetch is needed.
CATCHUP_THRESHOLD = timedelta(hours=36)
# A scheduled slot reached within this grace period is still dispatched by the
# reminder loop; only older slots are treated as expired and dropped.
DISPATCH_GRACE = timedelta(hours=6)
# Fetch and reminder logs older than this are pruned by the daily job.
LOG_RETENTION_DAYS = 7

# Activities ending on Monday/Thursday morning die with the weekly maintenance;
# the last useful reminder is the previous evening.
MAINTENANCE_WEEKDAYS = (0, 3)
MAINTENANCE_DEADLINE_BEFORE = dtime(12, 0)

LINK_PREFIX = "链接："


def merge_reminder_texts(texts: list[str]) -> str:
    """Combine several reminder messages for one target into a single message.

    Each item keeps its own ``【活动名 到期提醒】`` heading; items are
    separated by a blank line under one summary header. When every item
    comes from the same source announcement, the link line is written once
    at the end instead of being repeated per item.
    """
    if len(texts) <= 1:
        return texts[0] if texts else ""

    bodies: list[str] = []
    links: list[str] = []
    for text in texts:
        body: list[str] = []
        link = ""
        for line in text.strip().splitlines():
            if line.startswith(LINK_PREFIX):
                link = line[len(LINK_PREFIX):].strip()
            else:
                body.append(line)
        bodies.append("\n".join(body).strip())
        links.append(link)

    unique_links = list(dict.fromkeys(link for link in links if link))
    if len(unique_links) == 1:
        sections = list(bodies)
        sections[-1] += f"\n{LINK_PREFIX}{unique_links[0]}"
    else:
        sections = [
            body + (f"\n{LINK_PREFIX}{link}" if link else "")
            for body, link in zip(bodies, links, strict=True)
        ]

    header = f"【今日到期提醒 · 共 {len(sections)} 项】"
    return header + "\n\n" + "\n\n".join(sections)


@dataclass(slots=True)
class ReminderTargets:
    """Whitelist-derived reminder recipients as full session addresses.

    Entries follow AstrBot's unified_msg_origin format
    ``platform:MessageType:session_id`` (e.g. ``aiocqhttp:GroupMessage:123``).
    """

    sessions: list[str]
    days_before: int = 1
    send_time: str = "10:00"
    evening_enabled: bool = True
    evening_time: str = "21:00"
    urgent_enabled: bool = True
    urgent_minutes: int = 60

    def as_pairs(self) -> list[tuple[str, str]]:
        pairs = []
        for session in self.sessions:
            scope = "group" if ":GroupMessage:" in session else "private"
            pairs.append((scope, session))
        return pairs

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ReminderTargets:
        raw_items = [str(item) for item in (config.get("whitelist_groups") or [])]
        raw_items += [str(item) for item in (config.get("whitelist_users") or [])]
        sessions = []
        for item in raw_items:
            parsed = parse_session_address(item)
            if parsed is not None:
                sessions.append(parsed[1])
        try:
            days_before = max(0, min(7, int(config.get("reminder_days_before", 1))))
        except (TypeError, ValueError):
            days_before = 1
        send_time = str(config.get("reminder_send_time") or "10:00")
        evening_enabled = bool(config.get("reminder_evening_enabled", True))
        evening_time = str(config.get("reminder_evening_time") or "").strip()
        urgent_enabled = bool(config.get("reminder_urgent_enabled", True))
        try:
            urgent_minutes = int(config.get("reminder_urgent_minutes", 60))
        except (TypeError, ValueError):
            urgent_minutes = 60
        if urgent_minutes < 0:
            urgent_minutes = 0
        if not urgent_enabled:
            urgent_minutes = 0
        return cls(
            sessions=sessions,
            days_before=days_before,
            send_time=send_time,
            evening_enabled=evening_enabled,
            evening_time=evening_time,
            urgent_minutes=urgent_minutes,
        )


MESSAGE_TYPE_TO_SCOPE = {
    "GroupMessage": "group",
    "FriendMessage": "private",
}


def parse_session_address(value: str) -> tuple[str, str] | None:
    """Parse ``platform:MessageType:session_id`` into (scope, normalized address)."""
    parts = str(value or "").strip().split(":", 2)
    if len(parts) != 3:
        return None
    platform, message_type, session_id = (part.strip() for part in parts)
    scope = MESSAGE_TYPE_TO_SCOPE.get(message_type)
    if not platform or not session_id or scope is None:
        return None
    return scope, f"{platform}:{message_type}:{session_id}"


def extract_session_id(value: str) -> str:
    """Session id from a full address, or the value itself when it is a bare id."""
    parts = str(value or "").strip().split(":", 2)
    if len(parts) == 3 and parts[1].strip() in MESSAGE_TYPE_TO_SCOPE:
        return parts[2].strip()
    return str(value or "").strip()


class SchedulerService:
    def __init__(
        self,
        database: Database,
        ingest_service: IngestService,
        activity_service: ActivityService,
        timezone_name: str = "Asia/Shanghai",
        initial_fetch_limit: int = 50,
        daily_fetch_limit: int = 10,
        catchup_fetch_limit: int = 50,
    ) -> None:
        self.db = database
        self.ingest = ingest_service
        self.activities = activity_service
        self.timezone = ZoneInfo(timezone_name)
        self.initial_fetch_limit = max(1, int(initial_fetch_limit))
        self.daily_fetch_limit = max(1, int(daily_fetch_limit))
        self.catchup_fetch_limit = max(1, int(catchup_fetch_limit))

    def now(self) -> datetime:
        return datetime.now(self.timezone)

    def pick_fetch_limit(self) -> int:
        """Initial install and long downtime catch up with a large fetch."""
        with self.db.connect() as conn:
            if int(conn.execute("SELECT COUNT(*) FROM announcements").fetchone()[0]) == 0:
                return self.initial_fetch_limit
            row = conn.execute(
                "SELECT MAX(finished_at) FROM fetch_logs WHERE success = 1"
            ).fetchone()
        last = row[0] if row else None
        if not last:
            return self.catchup_fetch_limit
        try:
            finished = datetime.fromisoformat(str(last))
        except ValueError:
            return self.catchup_fetch_limit
        if finished.tzinfo is None:
            finished = finished.replace(tzinfo=self.timezone)
        else:
            finished = finished.astimezone(self.timezone)
        if self.now() - finished > CATCHUP_THRESHOLD:
            return self.catchup_fetch_limit
        return self.daily_fetch_limit

    async def fetch_and_ingest(
        self,
        client: NewsClient,
        embedding_provider: Any | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Fetch once, ingest, and write a fetch log entry."""
        chosen = self.pick_fetch_limit() if limit is None else max(1, int(limit))
        started_at = self.now()
        success = False
        error = ""
        returned = inserted = revised = skipped = 0
        try:
            items = await client.fetch(chosen)
            result = await self.ingest.ingest_items(items, embedding_provider)
            returned = int(result.get("returned", 0))
            inserted = int(result.get("inserted", 0))
            revised = int(result.get("revised", 0))
            skipped = int(result.get("skipped", 0))
            success = True
        except Exception as exc:  # noqa: BLE001 - logged and surfaced via fetch_logs
            error = str(exc)[:500]
            logger.error("fetch failed: %s", error)
        finished_at = self.now()
        self._log_fetch(
            started_at, finished_at, success, chosen, returned, inserted,
            revised, skipped, error,
        )
        return {
            "success": success,
            "limit": chosen,
            "returned": returned,
            "inserted": inserted,
            "revised": revised,
            "skipped": skipped,
            "error": error,
            "finished_at": finished_at.isoformat(timespec="seconds"),
        }

    def _log_fetch(
        self,
        started_at: datetime,
        finished_at: datetime,
        success: bool,
        fetch_limit: int,
        returned: int,
        inserted: int,
        revised: int,
        skipped: int,
        error: str,
    ) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO fetch_logs(
                    started_at, finished_at, success, fetch_limit,
                    returned_count, inserted_count, revised_count,
                    skipped_count, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    started_at.isoformat(timespec="seconds"),
                    finished_at.isoformat(timespec="seconds"),
                    int(success), fetch_limit, returned, inserted, revised,
                    skipped, error,
                ),
            )

    def _parse_send_time(self, send_time: str) -> tuple[int, int]:
        try:
            hour_text, minute_text = send_time.split(":", 1)
            hour, minute = int(hour_text), int(minute_text)
        except (ValueError, AttributeError):
            return 10, 0
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return 10, 0
        return hour, minute

    def _slots_for(
        self, deadline: datetime, targets: ReminderTargets, now: datetime
    ) -> list[tuple[datetime, str, int]]:
        """All reminder moments for one deadline as (moment, kind, days).

        Kinds: ``countdown`` (days counts down to 0), ``evening`` (the night
        before a Monday/Thursday morning deadline) and ``urgent`` (minutes
        before the deadline).
        """
        if deadline <= now:
            return []
        maintenance = (
            deadline.weekday() in MAINTENANCE_WEEKDAYS
            and deadline.time() < MAINTENANCE_DEADLINE_BEFORE
        )
        slots: list[tuple[datetime, str, int]] = []

        for days in range(targets.days_before, -1, -1):
            day = deadline.date() - timedelta(days=days)
            hour, minute = self._parse_send_time(targets.send_time)
            moment = datetime(
                day.year, day.month, day.day, hour, minute, tzinfo=deadline.tzinfo
            )
            if moment >= deadline:
                continue
            kept = self._keep(moment, now)
            if kept:
                slots.append((kept, "countdown", days))

        if maintenance and targets.evening_enabled and targets.evening_time:
            hour, minute = self._parse_send_time(targets.evening_time)
            day = deadline.date() - timedelta(days=1)
            moment = datetime(
                day.year, day.month, day.day, hour, minute, tzinfo=deadline.tzinfo
            )
            kept = self._keep(moment, now)
            if kept:
                slots.append((kept, "evening", 0))

        if (
            targets.urgent_enabled
            and targets.urgent_minutes > 0
            and not maintenance
        ):
            moment = deadline - timedelta(minutes=targets.urgent_minutes)
            kept = self._keep(moment, now)
            if kept:
                slots.append((kept, "urgent", 0))

        # An urgent slot earlier than the same-day countdown slot replaces it.
        urgent_by_date = {m.date(): m for m, kind, _ in slots if kind == "urgent"}
        slots = [
            (m, kind, days)
            for m, kind, days in slots
            if not (
                kind == "countdown"
                and m.date() in urgent_by_date
                and urgent_by_date[m.date()] < m
            )
        ]

        unique: list[tuple[datetime, str, int]] = []
        seen: set[str] = set()
        for moment, kind, days in sorted(slots, key=lambda s: s[0]):
            key = moment.isoformat()
            if key in seen:
                continue
            seen.add(key)
            unique.append((moment, kind, days))
        return unique

    @staticmethod
    def _humanize_minutes(minutes: int) -> str:
        if minutes < 60:
            return f"{minutes} 分钟"
        hours, rest = divmod(minutes, 60)
        if rest == 0:
            return f"{hours} 小时"
        return f"{hours} 小时 {rest} 分钟"

    def _remaining_text(
        self,
        kind: str,
        days: int,
        deadline: datetime,
        is_item: bool,
        urgent_minutes: int,
    ) -> str:
        """The ``剩余时间：`` line, worded per reminder kind and target class."""
        noun = "该道具" if is_item else "该活动"
        verb = "到期" if is_item else "结束"
        if kind == "countdown":
            if days == 0:
                return f"剩余时间：{noun}将于今天{verb}（{deadline:%H:%M}）"
            if days == 1:
                return f"剩余时间：1 天，{noun}将于明天{verb}"
            return (
                f"剩余时间：{days} 天，{noun}将于"
                f"{deadline.month}月{deadline.day}日{verb}"
            )
        if kind == "evening":
            return f"剩余时间：{noun}将于明天{verb}（明早 {deadline:%H:%M}）"
        return (
            f"剩余时间：{noun}将于{self._humanize_minutes(urgent_minutes)}后{verb}"
        )

    @staticmethod
    def _keep(moment: datetime, now: datetime) -> datetime | None:
        """Keep future slots and slots just reached within the dispatch grace."""
        if moment > now:
            return moment
        if now - moment <= DISPATCH_GRACE:
            return moment
        return None

    def create_pending_reminders(
        self, targets: ReminderTargets, announcement_ids: list[int] | None = None
    ) -> int:
        """Insert reminder rows for activities; idempotent via UNIQUE."""
        now = self.now()
        pairs = targets.as_pairs()
        if not pairs:
            return 0

        query = """
            SELECT a.id, a.start_time, a.end_time, a.item_expiry,
                   a.name, a.action, a.category, a.item_name, a.explanation,
                   an.title, an.url, date(an.published_at) AS announcement_date
            FROM activities a
            JOIN announcements an ON an.id = a.announcement_id
        """
        params: list[Any] = []
        if announcement_ids:
            marks = ",".join("?" for _ in announcement_ids)
            query += f" WHERE a.announcement_id IN ({marks})"
            params = list(announcement_ids)

        with self.db.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            created = 0
            for row in rows:
                deadline_text = row["item_expiry"] or row["end_time"]
                if not deadline_text:
                    continue
                try:
                    deadline = datetime.fromisoformat(deadline_text)
                except ValueError:
                    continue
                is_item = bool(row["item_expiry"])
                for moment, kind, days in self._slots_for(deadline, targets, now):
                    remaining = self._remaining_text(
                        kind, days, deadline, is_item, targets.urgent_minutes
                    )
                    message = self.activities.reminder_message(row, remaining)
                    for target_type, target_id in pairs:
                        cursor = conn.execute(
                            """
                            INSERT OR IGNORE INTO reminders(
                                activity_id, target_type, target_id,
                                scheduled_at, message_text
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                row["id"], target_type, target_id,
                                moment.isoformat(timespec="seconds"), message,
                            ),
                        )
                        created += int(cursor.rowcount > 0)
        return created

    def due_reminders(self, limit: int = 50) -> list[dict[str, Any]]:
        now = self.now().isoformat(timespec="seconds")
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT r.id, r.activity_id, r.target_type, r.target_id,
                       r.scheduled_at, r.message_text
                FROM reminders r
                WHERE r.status = 'pending' AND r.scheduled_at <= ?
                ORDER BY r.scheduled_at
                LIMIT ?
                """,
                (now, max(1, int(limit))),
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_reminder(
        self, reminder_id: int, status: str, error: str = ""
    ) -> None:
        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE reminders
                SET status = ?, error = ?, sent_at = datetime('now', 'localtime')
                WHERE id = ?
                """,
                (status, error[:500], reminder_id),
            )
            if status in ("sent", "failed", "cancelled"):
                conn.execute(
                    """
                    INSERT INTO reminder_logs(
                        reminder_id, activity_name, target_type, target_id,
                        scheduled_at, success, error
                    )
                    SELECT r.id, a.name, r.target_type, r.target_id,
                           r.scheduled_at, ?, ?
                    FROM reminders r
                    JOIN activities a ON a.id = r.activity_id
                    WHERE r.id = ?
                    """,
                    (1 if status == "sent" else 0, error[:500], reminder_id),
                )

    def cancel_reminders_for_announcement(self, announcement_id: int) -> int:
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE reminders SET status = 'cancelled'
                WHERE status = 'pending' AND activity_id IN (
                    SELECT id FROM activities WHERE announcement_id = ?
                )
                """,
                (announcement_id,),
            )
            return int(cursor.rowcount)

    def next_run_at(self, daily_fetch_time: str) -> str:
        """Next daily fetch wall-clock time, for status display."""
        hour, minute = self._parse_send_time(daily_fetch_time)
        now = self.now()
        moment = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if moment <= now:
            moment += timedelta(days=1)
        return moment.isoformat(timespec="seconds")

    def prune_logs(self, retention_days: int = LOG_RETENTION_DAYS) -> dict[str, int]:
        """Drop fetch/reminder logs older than the retention window."""
        cutoff = (self.now() - timedelta(days=retention_days)).isoformat(
            timespec="seconds"
        )
        with self.db.connect() as conn:
            fetch = int(
                conn.execute(
                    "DELETE FROM fetch_logs WHERE started_at < ?", (cutoff,)
                ).rowcount
            )
            reminder = int(
                conn.execute(
                    "DELETE FROM reminder_logs WHERE sent_at < ?", (cutoff,)
                ).rowcount
            )
        return {"fetch_logs": fetch, "reminder_logs": reminder}

    def status_snapshot(self, daily_fetch_time: str) -> dict[str, Any]:
        with self.db.connect() as conn:
            last = conn.execute(
                """
                SELECT started_at, finished_at, success, fetch_limit,
                       returned_count, inserted_count, revised_count,
                       skipped_count, error
                FROM fetch_logs ORDER BY id DESC LIMIT 1
                """
            ).fetchone()
        return {
            "next_fetch_at": self.next_run_at(daily_fetch_time),
            "last_fetch": dict(last) if last else None,
        }


__all__ = [
    "CATCHUP_THRESHOLD",
    "DISPATCH_GRACE",
    "LOG_RETENTION_DAYS",
    "MAINTENANCE_DEADLINE_BEFORE",
    "MAINTENANCE_WEEKDAYS",
    "MESSAGE_TYPE_TO_SCOPE",
    "ReminderTargets",
    "SchedulerService",
    "extract_session_id",
    "merge_reminder_texts",
    "parse_session_address",
]
