"""Scheduled fetch, activity extraction, reminder creation and dispatch."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .activities import ActivityService, format_deadline
from .database import Database
from .ingest import IngestService
from .jx3api import NewsClient

logger = logging.getLogger(__name__)

# A gap longer than this since the last successful fetch means announcements
# were missed and a large catch-up fetch is needed.
CATCHUP_THRESHOLD = timedelta(hours=36)
SHORT_WINDOW = timedelta(hours=24)
SHORT_WINDOW_REMIND_BEFORE = timedelta(minutes=30)
# A scheduled slot reached within this grace period is still dispatched by the
# reminder loop; only older slots are treated as expired and dropped.
DISPATCH_GRACE = timedelta(hours=6)


@dataclass(slots=True)
class ReminderTargets:
    """Whitelist-derived reminder recipients as full session addresses.

    Entries follow AstrBot's unified_msg_origin format
    ``platform:MessageType:session_id`` (e.g. ``aiocqhttp:GroupMessage:123``).
    """

    sessions: list[str]
    days_before: int = 1
    send_time: str = "10:00"

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
            days_before = max(0, int(config.get("reminder_days_before", 1)))
        except (TypeError, ValueError):
            days_before = 1
        send_time = str(config.get("reminder_send_time") or "10:00")
        return cls(sessions=sessions, days_before=days_before, send_time=send_time)


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

    def _scheduled_at(
        self, row: Any, targets: ReminderTargets, now: datetime
    ) -> datetime | None:
        """Reminder time for one activity row, or None when not reminder-worthy."""
        end_time = row["end_time"] or ""
        item_expiry = row["item_expiry"] or ""
        start_time = row["start_time"] or ""
        deadline_text = item_expiry or end_time
        if not deadline_text:
            return None
        try:
            deadline = datetime.fromisoformat(deadline_text)
        except ValueError:
            return None
        if deadline <= now:
            return None

        # Short windows (e.g. a queue that opens for one hour) would be over
        # before a "one day ahead" reminder; remind shortly after it opens.
        if start_time and end_time:
            try:
                start = datetime.fromisoformat(start_time)
                end = datetime.fromisoformat(end_time)
                if end > start and end - start < SHORT_WINDOW:
                    moment = start - SHORT_WINDOW_REMIND_BEFORE
                    return self._keep(moment, now)
            except ValueError:
                pass

        hour, minute = self._parse_send_time(targets.send_time)
        send_day = deadline.date() - timedelta(days=targets.days_before)
        moment = datetime(
            send_day.year, send_day.month, send_day.day, hour, minute,
            tzinfo=deadline.tzinfo,
        )
        # The window between the scheduled slot and the deadline (e.g. the
        # default 10:00 slot for a 07:00 deadline) must still be in the future.
        if moment >= deadline:
            previous = send_day - timedelta(days=1)
            moment = moment.replace(year=previous.year, month=previous.month, day=previous.day)
        return self._keep(moment, now)

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
                moment = self._scheduled_at(row, targets, now)
                if moment is None:
                    continue
                message = self.activities.reminder_message(row)
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

    def status_snapshot(self, daily_fetch_time: str) -> dict[str, Any]:
        with self.db.connect() as conn:
            last = conn.execute(
                """
                SELECT started_at, finished_at, success, fetch_limit,
                       inserted_count, revised_count, error
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
    "SHORT_WINDOW",
    "SHORT_WINDOW_REMIND_BEFORE",
    "MESSAGE_TYPE_TO_SCOPE",
    "ReminderTargets",
    "SchedulerService",
    "extract_session_id",
    "format_deadline",
    "parse_session_address",
]
