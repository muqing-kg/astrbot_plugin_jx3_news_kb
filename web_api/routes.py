"""Plugin Page Web API handlers.

Handlers are framework-agnostic async callables taking
``(plugin, query, payload)`` and returning JSON-serializable data or a
``(data, status_code)`` tuple. main.py adapts them to AstrBot's
``context.register_web_api`` so tests can call them directly.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

PAGE_SIZE_DEFAULT = 20
PAGE_SIZE_MAX = 100


def _int_query(query: dict[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(query.get(key, default))
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


async def handle_stats(plugin: Any, query: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    db = plugin.db
    counts = {
        table: db.count(table)
        for table in ("announcements", "activities", "reminders")
    }
    with db.connect() as conn:
        pending = int(
            conn.execute(
                "SELECT COUNT(*) FROM reminders WHERE status = 'pending'"
            ).fetchone()[0]
        )
        scheduled_reminders = conn.execute(
            """
            SELECT r.id, r.activity_id, r.target_type, r.target_id,
                   r.scheduled_at, r.status,
                   a.name, an.title AS announcement_title, an.url,
                   date(an.published_at) AS announcement_date
            FROM reminders r
            JOIN activities a ON a.id = r.activity_id
            JOIN announcements an ON an.id = a.announcement_id
            WHERE r.status = 'pending'
            ORDER BY r.scheduled_at
            LIMIT 200
            """
        ).fetchall()

    # Only show what will actually be pushed soon: slots within the next
    # 48 hours (tomorrow's daily push, tonight's maintenance-evening push,
    # the urgent last-call). Anything further out is not "recent".
    valid_slots = plugin.valid_slot_keys()
    now_dt = plugin.scheduler.now()
    now_iso = now_dt.isoformat(timespec="seconds")
    window_end = now_dt + timedelta(hours=48)
    chosen: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in scheduled_reminders:
        key = (row["activity_id"], row["scheduled_at"])
        kind = valid_slots.get(key)
        if kind is None:
            continue
        if row["scheduled_at"] <= now_iso:
            chosen[(row["activity_id"], row["target_id"], "due")] = dict(row)
            continue
        slot_key = (row["activity_id"], row["target_id"], kind)
        if slot_key not in chosen or row["scheduled_at"] < chosen[slot_key]["scheduled_at"]:
            chosen[slot_key] = dict(row)
    upcoming_reminders = sorted(
        (
            row
            for row in chosen.values()
            if datetime.fromisoformat(row["scheduled_at"]) <= window_end
        ),
        key=lambda r: r["scheduled_at"],
    )

    snapshot = plugin.scheduler.status_snapshot(plugin.next_sync_at())
    return {
        "counts": counts,
        "pending_reminders": pending,
        "upcoming_reminders": upcoming_reminders,
        "last_fetch": snapshot["last_fetch"],
        "next_fetch_at": snapshot["next_fetch_at"],
        "reminder_enabled": bool(plugin.config.get("reminder_enabled", True)),
        "reminder_targets": int(plugin.reminder_target_count()),
        "scheduler_alive": bool(plugin.background_alive()),
        "socket_enabled": bool(plugin.config.get("socket_enabled", True)),
        "daily_full_pull_time": str(plugin.config.get("daily_full_pull_time") or "03:00"),
    }


async def handle_announcements(
    plugin: Any, query: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    page = _int_query(query, "page", 1, 1, 10_000)
    page_size = _int_query(query, "page_size", PAGE_SIZE_DEFAULT, 1, PAGE_SIZE_MAX)
    keyword = str(query.get("q") or "").strip()
    announcement_type = str(query.get("type") or "").strip()

    clauses: list[str] = []
    params: list[Any] = []
    if keyword:
        like = f"%{keyword.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')}%"
        clauses.append("(a.title LIKE ? ESCAPE '\\' OR a.content_text LIKE ? ESCAPE '\\')")
        params.extend([like, like])
    if announcement_type:
        clauses.append("a.type = ?")
        params.append(announcement_type)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    with plugin.db.connect() as conn:
        total = int(
            conn.execute(
                f"SELECT COUNT(*) FROM announcements a{where}", params
            ).fetchone()[0]
        )
        rows = conn.execute(
            f"""
            SELECT a.id, a.title, a.type, a.url, a.published_at,
                   a.updated_at_source, a.modified_at,
                   date(a.published_at) AS announcement_date,
                   (SELECT COUNT(*) FROM activities v WHERE v.announcement_id = a.id)
                       AS activity_count
            FROM announcements a{where}
            ORDER BY a.published_at DESC, a.id DESC
            LIMIT ? OFFSET ?
            """,
            [*params, page_size, (page - 1) * page_size],
        ).fetchall()
        types = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT type FROM announcements ORDER BY type"
            ).fetchall()
            if row[0]
        ]
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "types": types,
        "items": [dict(row) for row in rows],
    }


async def handle_announcement_detail(
    plugin: Any, query: dict[str, Any], payload: dict[str, Any], announcement_id: str
) -> tuple[dict[str, Any], int] | dict[str, Any]:
    try:
        announcement_id_int = int(announcement_id)
    except ValueError:
        return {"error": "无效的 ID"}, 400
    with plugin.db.connect() as conn:
        row = conn.execute(
            """
            SELECT id, source_id, title, type, url, description, content_text,
                   raw_json, content_hash, published_at, updated_at_source,
                   modified_at, date(published_at) AS announcement_date
            FROM announcements WHERE id = ?
            """,
            (announcement_id_int,),
        ).fetchone()
        if row is None:
            return {"error": "未找到该公告"}, 404
        revisions = conn.execute(
            """
            SELECT id, revision_no, title, updated_at_source, content_hash, created_at
            FROM announcement_revisions WHERE announcement_id = ?
            ORDER BY revision_no
            """,
            (announcement_id_int,),
        ).fetchall()
    detail = dict(row)
    detail["raw_json"] = _parse_raw_json(detail.get("raw_json"))
    return {
        "announcement": detail,
        "revisions": [dict(item) for item in revisions],
    }


def _parse_raw_json(value: Any) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return value


async def handle_announcement_delete(
    plugin: Any, query: dict[str, Any], payload: dict[str, Any], announcement_id: str
) -> tuple[dict[str, Any], int]:
    try:
        announcement_id_int = int(announcement_id)
    except ValueError:
        return {"error": "无效的 ID"}, 400
    if payload.get("confirm") is not True:
        return {"error": "缺少删除确认标志"}, 400
    # Reminders, activities and logs disappear through the foreign keys.
    deleted = plugin.ingest.hard_delete_announcement(announcement_id_int)
    if not deleted:
        return {"error": "未找到该公告"}, 404
    return {"deleted": True, "id": announcement_id_int}


async def handle_activities(
    plugin: Any, query: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    """Ongoing activities: deadline (item expiry first) still in the future."""
    now_iso = plugin.scheduler.now().isoformat(timespec="seconds")
    with plugin.db.connect() as conn:
        rows = conn.execute(
            """
            SELECT a.id, a.name, a.action, a.category, a.start_time, a.end_time,
                   a.item_expiry, a.item_name, a.explanation, a.confidence,
                   an.title AS announcement_title, an.url,
                   date(an.published_at) AS announcement_date
            FROM activities a
            JOIN announcements an ON an.id = a.announcement_id
            WHERE COALESCE(NULLIF(a.item_expiry, ''), NULLIF(a.end_time, '')) > ?
            ORDER BY COALESCE(NULLIF(a.item_expiry, ''), NULLIF(a.end_time, ''))
            LIMIT 200
            """,
            (now_iso,),
        ).fetchall()
        total = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM activities a
                WHERE COALESCE(NULLIF(a.item_expiry, ''), NULLIF(a.end_time, '')) > ?
                """,
                (now_iso,),
            ).fetchone()[0]
        )
    return {"items": [dict(row) for row in rows], "total": total}


async def handle_activity_delete(
    plugin: Any, query: dict[str, Any], payload: dict[str, Any], activity_id: str
) -> tuple[dict[str, Any], int]:
    """Remove one activity and its scheduled reminders immediately."""
    try:
        activity_id_int = int(activity_id)
    except ValueError:
        return {"error": "无效的 ID"}, 400
    with plugin.db.connect() as conn:
        row = conn.execute(
            "SELECT announcement_id FROM activities WHERE id = ?", (activity_id_int,)
        ).fetchone()
        if row is None:
            return {"error": "未找到该活动"}, 404
        conn.execute("DELETE FROM activities WHERE id = ?", (activity_id_int,))
        # Mark the announcement as read so the next extraction does not put
        # the row straight back while its text is unchanged.
        conn.execute(
            "UPDATE announcements SET activity_extracted_at = datetime('now', 'localtime')"
            " WHERE id = ?",
            (row["announcement_id"],),
        )
    return {"deleted": True, "id": activity_id_int}


async def handle_fetch(
    plugin: Any, query: dict[str, Any], payload: dict[str, Any]
) -> tuple[dict[str, Any], int] | dict[str, Any]:
    try:
        limit = int(payload.get("limit", 0) or 0)
    except (TypeError, ValueError):
        limit = 0
    if limit < 1 or limit > 50:
        return {"error": "同步条数必须在 1 到 50 之间"}, 400
    result = await plugin.run_sync(limit)
    if not result.get("success"):
        return {"error": result.get("error") or "同步失败", "detail": result}, 502
    return result


async def handle_logs(
    plugin: Any, query: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    limit = _int_query(query, "limit", 50, 1, 200)
    with plugin.db.connect() as conn:
        rows = conn.execute(
            """
            SELECT id, started_at, finished_at, success, fetch_limit,
                   returned_count, inserted_count, revised_count,
                   skipped_count, error
            FROM fetch_logs ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return {"logs": [dict(row) for row in rows]}


async def handle_reminders(
    plugin: Any, query: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    status = str(query.get("status") or "pending").strip()
    allowed = {"pending", "sent", "cancelled", "failed", "all"}
    if status not in allowed:
        return {"error": f"status 仅支持 {sorted(allowed)}"}, 400
    where = "" if status == "all" else " WHERE r.status = ?"
    params: list[Any] = [] if status == "all" else [status]
    with plugin.db.connect() as conn:
        rows = conn.execute(
            f"""
            SELECT r.id, r.target_type, r.target_id, r.scheduled_at, r.status,
                   r.message_text, a.name AS activity_name, a.end_time,
                   a.item_expiry, an.title AS announcement_title,
                   date(an.published_at) AS announcement_date
            FROM reminders r
            JOIN activities a ON a.id = r.activity_id
            JOIN announcements an ON an.id = a.announcement_id
            {where}
            ORDER BY r.scheduled_at DESC
            LIMIT 200
            """,
            params,
        ).fetchall()
    return {"status": status, "items": [dict(row) for row in rows]}


ROUTE_TABLE = [
    ("/stats", ["GET"], handle_stats),
    ("/announcements", ["GET"], handle_announcements),
    ("/announcements/<announcement_id>", ["GET"], handle_announcement_detail),
    ("/announcements/<announcement_id>/delete", ["POST"], handle_announcement_delete),
    ("/activities", ["GET"], handle_activities),
    ("/activities/<activity_id>/delete", ["POST"], handle_activity_delete),
    ("/fetch", ["POST"], handle_fetch),
    ("/logs", ["GET"], handle_logs),
    ("/reminders", ["GET"], handle_reminders),
]
