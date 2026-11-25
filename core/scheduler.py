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
# Fetch and reminder logs older than this are pruned by the fetch job.
LOG_RETENTION_DAYS = 7
FINGERPRINT_RETENTION_DAYS = 60
# Re-scheduling keeps slots that fell due within this window: the dispatch loop
# runs once a minute, so a rebuild landing between due time and send must not
# delete the reminder that is about to go out.
RECENT_SLOT_GRACE = timedelta(minutes=10)

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

    header = f"【到期提醒 · 共 {len(sections)} 项】"
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
        # Missing key falls back to the default; an explicit empty value
        # still disables the evening reminder.
        evening_time = str(config.get("reminder_evening_time", "21:00") or "").strip()
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


def parse_hhmm(value: str, default: tuple[int, int]) -> tuple[int, int]:
    """Parse ``HH:MM``; junk input falls back to ``default``."""
    try:
        hour_text, minute_text = str(value).split(":", 1)
        hour, minute = int(hour_text), int(minute_text)
    except (ValueError, AttributeError):
        return default
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return default
    return hour, minute


class SchedulerService:
    def __init__(
        self,
        database: Database,
        ingest_service: IngestService,
        activity_service: ActivityService,
        timezone_name: str = "Asia/Shanghai",
        initial_fetch_limit: int = 50,
        fetch_limit: int = 10,
        catchup_fetch_limit: int = 50,
    ) -> None:
        self.db = database
        self.ingest = ingest_service
        self.activities = activity_service
        self.timezone = ZoneInfo(timezone_name)
        self.initial_fetch_limit = max(1, int(initial_fetch_limit))
        self.fetch_limit = max(1, int(fetch_limit))
        self.catchup_fetch_limit = max(1, int(catchup_fetch_limit))

    def now(self) -> datetime:
        return datetime.now(self.timezone)

    def parse_moment(self, value: Any) -> datetime | None:
        """Stored deadline as an aware datetime; a naive string takes the
        plugin timezone so a row written without an offset cannot abort the
        whole scheduling pass."""
        try:
            moment = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=self.timezone)
        return moment

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
        return self.fetch_limit

    async def fetch_and_ingest(
        self,
        client: NewsClient,
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
            result = await self.ingest.ingest_items(items)
            returned = int(result.get("returned", 0))
            inserted = int(result.get("inserted", 0))
            revised = int(result.get("revised", 0))
            skipped = int(result.get("skipped", 0))
            success = True
        except Exception as exc:  # noqa: BLE001 - logged and surfaced via fetch_logs
            error = str(exc)[:500]
            logger.error("抓取失败：%s", error)
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
        return parse_hhmm(send_time, (10, 0))

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
            if deadline > moment > now:
                slots.append((moment, "countdown", days))

        if maintenance and targets.evening_enabled and targets.evening_time:
            hour, minute = self._parse_send_time(targets.evening_time)
            day = deadline.date() - timedelta(days=1)
            moment = datetime(
                day.year, day.month, day.day, hour, minute, tzinfo=deadline.tzinfo
            )
            if moment > now:
                slots.append((moment, "evening", 0))

        if (
            targets.urgent_enabled
            and targets.urgent_minutes > 0
            and not maintenance
        ):
            moment = deadline - timedelta(minutes=targets.urgent_minutes)
            if moment > now:
                slots.append((moment, "urgent", 0))

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

    def valid_slot_keys(
        self, targets: ReminderTargets, now: datetime | None = None
    ) -> dict[tuple[int, str], str]:
        """(activity_id, scheduled_at) -> slot kind, valid under current windows."""
        if now is None:
            now = self.now()
        keys: dict[tuple[int, str], str] = {}
        with self.db.connect() as conn:
            rows = conn.execute(
                "SELECT id, item_expiry, end_time FROM activities"
            ).fetchall()
        for row in rows:
            deadline_text = row["item_expiry"] or row["end_time"]
            if not deadline_text:
                continue
            deadline = self.parse_moment(deadline_text)
            if deadline is None:
                continue
            for moment, kind, _ in self._slots_for(deadline, targets, now):
                keys[(row["id"], moment.isoformat(timespec="seconds"))] = kind
        return keys

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
            # Re-scheduling replaces pending slots so a config change takes
            # effect immediately. Only slots that are still in the future are
            # dropped: a slot that just came due must survive, or a rebuild in
            # the same minute as the slot (the hourly sync lands on :00) eats
            # it before the dispatch loop can send it.
            now_text = now.isoformat(timespec="seconds")
            cutoff = (now - RECENT_SLOT_GRACE).isoformat(timespec="seconds")
            if announcement_ids:
                marks = ",".join("?" for _ in announcement_ids)
                conn.execute(
                    f"""
                    DELETE FROM reminders
                    WHERE status = 'pending'
                      AND scheduled_at > ?
                      AND activity_id IN (
                          SELECT id FROM activities
                          WHERE announcement_id IN ({marks})
                      )
                    """,
                    [now_text, *announcement_ids],
                )
            else:
                conn.execute(
                    "DELETE FROM reminders WHERE status = 'pending' AND scheduled_at > ?",
                    (now_text,),
                )
            # Slots that came due while the plugin was down are expired, not
            # sent hours later: the rule is "no catch-up".
            conn.execute(
                """
                UPDATE reminders SET status = 'cancelled'
                WHERE status = 'pending' AND scheduled_at <= ?
                """,
                (cutoff,),
            )
            created = 0
            for row in rows:
                deadline_text = row["item_expiry"] or row["end_time"]
                if not deadline_text:
                    continue
                deadline = self.parse_moment(deadline_text)
                if deadline is None:
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
        """Record a dispatch outcome.

        Only a ``pending`` row may leave that state: a rebuild running while a
        send is in flight may already have cancelled the slot, and reviving it
        would report a send that the rules say must be dropped. A failure keeps
        the row pending so the next scan retries it inside the grace window.
        """
        with self.db.connect() as conn:
            if status == "pending":
                conn.execute(
                    "UPDATE reminders SET error = ? WHERE id = ? AND status = 'pending'",
                    (error[:500], reminder_id),
                )
                return
            conn.execute(
                """
                UPDATE reminders
                SET status = ?, error = ?, sent_at = ?
                WHERE id = ? AND status = 'pending'
                """,
                (
                    status,
                    error[:500],
                    self.now().isoformat(timespec="seconds"),
                    reminder_id,
                ),
            )

    def prune_logs(self, retention_days: int = LOG_RETENTION_DAYS) -> int:
        """Drop fetch logs older than the retention window."""
        cutoff = (self.now() - timedelta(days=retention_days)).isoformat(
            timespec="seconds"
        )
        with self.db.connect() as conn:
            return int(
                conn.execute(
                    "DELETE FROM fetch_logs WHERE started_at < ?", (cutoff,)
                ).rowcount
            )

    def prune_announcements(self, targets: ReminderTargets) -> dict[str, int]:
        """Drop announcements that are extracted and have nothing left to send.

        An announcement is spent once extraction is done and no reminder of it is
        still pending; with no destination configured nothing may be sent, so the
        caller's targets double as the guard against wiping the whole table.
        """
        if not targets.as_pairs():
            return {"announcements": 0, "fingerprints": 0}
        with self.db.connect() as conn:
            ids = [
                int(row["id"])
                for row in conn.execute(
                    """
                    SELECT an.id FROM announcements AS an
                    WHERE an.activity_extracted_at IS NOT NULL
                      AND an.activity_extracted_at <> ''
                      AND NOT EXISTS (
                          SELECT 1 FROM activities AS a
                          JOIN reminders AS r ON r.activity_id = a.id
                          WHERE a.announcement_id = an.id AND r.status = 'pending'
                      )
                    """
                ).fetchall()
            ]
        deleted = sum(1 for i in ids if self.ingest.hard_delete_announcement(i))

        stale = (self.now() - timedelta(days=FINGERPRINT_RETENTION_DAYS)).date().isoformat()
        with self.db.connect() as conn:
            fingerprints = int(
                conn.execute(
                    "DELETE FROM deleted_fingerprints WHERE date(deleted_at) < ?",
                    (stale,),
                ).rowcount
            )
        return {"announcements": deleted, "fingerprints": fingerprints}

    def status_snapshot(self, next_fetch_at: str) -> dict[str, Any]:
        """Status payload; the caller owns the schedule, so it passes the next run."""
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
            "next_fetch_at": next_fetch_at,
            "last_fetch": dict(last) if last else None,
        }


__all__ = [
    "CATCHUP_THRESHOLD",
    "FINGERPRINT_RETENTION_DAYS",
    "LOG_RETENTION_DAYS",
    "MAINTENANCE_DEADLINE_BEFORE",
    "MAINTENANCE_WEEKDAYS",
    "MESSAGE_TYPE_TO_SCOPE",
    "ReminderTargets",
    "SchedulerService",
    "merge_reminder_texts",
    "parse_session_address",
]
