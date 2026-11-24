"""AstrBot plugin entry: JX3 announcement ingestion and activity reminders."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from astrbot.api.event import MessageChain, filter
from astrbot.api.star import Context, Star
from astrbot.api.web import error_response, json_response, request as web_request
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

try:
    # AstrBot loads this file as package ``data.plugins.<name>.main``.
    from .core.activities import ActivityService
    from .core.database import Database
    from .core.ingest import IngestService
    from .core.jx3api import NewsClient
    from .core.scheduler import (
        ReminderTargets,
        SchedulerService,
        merge_reminder_texts,
        parse_hhmm,
    )
    from .core.socket import DEFAULT_SOCKET_URL, PushClient
    from .web_api.routes import ROUTE_TABLE
except ImportError:  # imported as a top-level module (tests, direct run)
    from core.activities import ActivityService
    from core.database import Database
    from core.ingest import IngestService
    from core.jx3api import NewsClient
    from core.scheduler import (
        ReminderTargets,
        SchedulerService,
        merge_reminder_texts,
        parse_hhmm,
    )
    from core.socket import DEFAULT_SOCKET_URL, PushClient
    from web_api.routes import ROUTE_TABLE

try:
    # Official per-plugin logger: attaches the enricher filter (plugin_tag) and
    # the WebUI log queue handler; a raw logging.getLogger child of "astrbot"
    # would make that handler skip the record. Kept out of the import block
    # above so an AstrBot internal move degrades logging, not plugin loading.
    from astrbot.core.log import LogManager

    logger = LogManager.get_plugin_logger("jx3_news_kb")
except Exception:  # noqa: BLE001 - logging must never stop the plugin loading
    logger = logging.getLogger("astrbot.plugin.jx3_news_kb")

PLUGIN_NAME = "astrbot_plugin_jx3_news_kb"
REMINDER_LOOP_INTERVAL = 60
DEFAULT_SYNC_INTERVAL_MINUTES = 60
DEFAULT_DAILY_PULL_TIME = "03:00"
DEFAULT_DAILY_PULL_LIMIT = 50


def _as_int(config: Any, key: str, default: int) -> int:
    try:
        return int(config.get(key, default))
    except (TypeError, ValueError):
        return default


def _as_limit(config: Any, key: str, default: int) -> int:
    """Fetch limits outside 1..50 make every sync fail; clamp at the config edge."""
    return min(50, max(1, _as_int(config, key, default)))


class JX3NewsKBPlugin(Star):
    def __init__(self, context: Context, config: Any | None = None) -> None:
        super().__init__(context)
        self.config = config if isinstance(config, dict) else {}
        self.timezone = self._resolve_timezone()

        data_dir = Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME
        self.db = Database(data_dir / "knowledge_base.sqlite3")
        self.db.initialize()

        self.ingest = IngestService(self.db, self.timezone.key)
        self.activity_service = ActivityService(self.db, self.timezone.key)
        self.scheduler = SchedulerService(
            self.db,
            self.ingest,
            self.activity_service,
            timezone_name=self.timezone.key,
            initial_fetch_limit=_as_limit(self.config, "initial_fetch_limit", 50),
            fetch_limit=_as_limit(self.config, "fetch_limit", 10),
            catchup_fetch_limit=_as_limit(self.config, "catchup_fetch_limit", 50),
        )
        self.client = NewsClient(
            base_url=str(self.config.get("api_base_url") or "https://www.jx3api.com"),
            records_path=str(self.config.get("news_records_path") or "/news/records"),
            token=str(self.config.get("api_token") or ""),
            proxy=str(self.config.get("proxy") or ""),
        )
        self.push = PushClient(
            url=str(self.config.get("socket_url") or DEFAULT_SOCKET_URL),
            proxy=str(self.config.get("proxy") or ""),
            ping_seconds=_as_int(self.config, "socket_ping_seconds", 30),
            logger=logger,
        )
        self._background_tasks: list[asyncio.Task] = []
        self._sync_lock = asyncio.Lock()
        self._backfill_task: asyncio.Task | None = None
        # Day the daily full pull last ran; a sync straddling the configured
        # time must not push the pull to tomorrow.
        self._daily_done: date | None = None
        self._daily_cooldown_until: datetime | None = None
        self._backfill_pending = False
        self._register_routes()
        self._try_start_background_tasks()

    # ---------- configuration helpers ----------

    def _resolve_timezone(self) -> ZoneInfo:
        name = str(self.config.get("timezone") or "Asia/Shanghai")
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            return ZoneInfo("Asia/Shanghai")

    async def _llm_provider(self) -> Any | None:
        provider_id = str(self.config.get("llm_provider_id") or "").strip()
        if provider_id:
            return self.context.get_provider_by_id(provider_id)
        try:
            return await self.context.get_using_provider_async()
        except Exception:  # noqa: BLE001 - provider lookup must not break jobs
            return None

    # ---------- Plugin Page web API ----------

    def _register_routes(self) -> None:
        for route, methods, handler in ROUTE_TABLE:
            self.context.register_web_api(
                f"/{PLUGIN_NAME}{route}",
                self._wrap_handler(handler),
                list(methods),
                f"JX3 activity reminder {route}",
            )

    def _wrap_handler(self, handler: Any) -> Any:
        async def view(**path_params: Any):
            query = {
                key: web_request.query.get(key)
                for key in web_request.query.keys()
            }
            payload = await web_request.json(default={}) or {}
            try:
                result = await handler(self, query, payload, **path_params)
            except Exception as exc:  # noqa: BLE001 - reported to the dashboard
                logger.exception("Web 接口 %s 调用失败", handler.__name__)
                return error_response(f"内部错误：{exc}", status_code=500)
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], int):
                data, status = result
                if status >= 400:
                    data = data if isinstance(data, dict) else {}
                    return error_response(
                        str(data.get("error") or "request failed"),
                        status_code=status,
                        data={k: v for k, v in data.items() if k != "error"},
                    )
                return json_response(data, status_code=status)
            return json_response(result)

        return view

    # ---------- pipeline used by background jobs and web API ----------

    async def fetch_and_ingest(self, limit: int | None = None) -> dict[str, Any]:
        return await self.scheduler.fetch_and_ingest(self.client, limit)

    async def extract_and_schedule(self, announcement_ids: list[int] | None = None) -> int:
        created = await self.activity_service.extract_for_announcements(
            await self._llm_provider(), announcement_ids
        )
        if bool(self.config.get("reminder_enabled", True)):
            targets = ReminderTargets.from_config(self.config)
            self.scheduler.create_pending_reminders(targets, announcement_ids)
        return created

    async def fetch_job(self, limit: int | None = None) -> dict[str, Any]:
        result = await self.fetch_and_ingest(limit)
        self.scheduler.prune_logs()
        if result.get("success"):
            await self.extract_and_schedule()
        return result

    async def run_sync(self, limit: int | None = None) -> dict[str, Any]:
        """One guarded sync round, shared by the loops and the web API."""
        return await self._run_fetch_job_guarded(limit)

    def reminder_target_count(self) -> int:
        """Whitelist entries usable as reminder destinations (full addresses)."""
        return len(ReminderTargets.from_config(self.config).as_pairs())

    def valid_slot_keys(self) -> dict[tuple[int, str], str]:
        """(activity_id, scheduled_at) -> slot kind, inside current windows."""
        return self.scheduler.valid_slot_keys(
            ReminderTargets.from_config(self.config)
        )

    # ---------- reminders ----------

    async def _send_text(self, target_type: str, target_id: str, text: str) -> tuple[bool, str]:
        """target_id is a full session address ``platform:MessageType:id``."""
        try:
            sent = await self.context.send_message(target_id, MessageChain().message(text))
            return bool(sent), "" if sent else "未找到匹配的消息平台"
        except Exception as exc:  # noqa: BLE001 - logged into reminder history
            return False, str(exc)

    async def send_due_reminders(self) -> int:
        """Dispatch due reminders; several due for one target merge into one message."""
        if not bool(self.config.get("reminder_enabled", True)):
            return 0
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for reminder in self.scheduler.due_reminders(500):
            key = (reminder["target_type"], reminder["target_id"])
            groups.setdefault(key, []).append(reminder)

        sent = 0
        for (target_type, target_id), items in groups.items():
            text = merge_reminder_texts([item["message_text"] for item in items])
            ok, error = await self._send_text(target_type, target_id, text)
            # A failed send stays pending so the next scan retries it; the
            # rebuild turns it into a cancelled slot once the grace window ends.
            status = "sent" if ok else "pending"
            for item in items:
                self.scheduler.mark_reminder(int(item["id"]), status, error)
            if ok:
                sent += len(items)
            else:
                logger.warning("提醒发送失败（%s）：%s", target_id, error)
        return sent

    # ---------- background loops ----------

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self) -> None:
        # Fires only on a full AstrBot cold boot; reloads re-enter via
        # __init__ -> _try_start_background_tasks instead.
        self._start_background_tasks()

    def _try_start_background_tasks(self) -> None:
        """Start the loops when constructed inside a running event loop.

        AstrBot instantiates plugins asynchronously, so this covers plugin
        install/update reloads where on_astrbot_loaded never fires.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # constructed outside a loop (tests); the hook starts them
        self._start_background_tasks()

    def _start_background_tasks(self) -> None:
        if self._background_tasks:
            return
        for coro, name in (
            (self._sync_loop(), "jx3-announcement-sync"),
            (self._push_loop(), "jx3-announcement-push"),
            (self._reminder_loop(), "jx3-reminder-dispatch"),
        ):
            self._background_tasks.append(asyncio.create_task(coro, name=name))
        logger.info("剑网3活动提醒后台任务已启动")

    def background_alive(self) -> bool:
        return bool(self._background_tasks) and any(
            not task.done() for task in self._background_tasks
        )

    def _sync_interval_seconds(self) -> float:
        minutes = _as_int(self.config, "sync_interval_minutes", DEFAULT_SYNC_INTERVAL_MINUTES)
        return float(max(1, minutes) * 60)

    def next_sync(self) -> tuple[float, bool]:
        """Seconds until the next sync, and whether it is the daily full pull."""
        now = datetime.now(self.timezone)
        hour, minute = parse_hhmm(
            str(self.config.get("daily_full_pull_time") or DEFAULT_DAILY_PULL_TIME),
            (3, 0),
        )
        daily = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if daily <= now and self._daily_done == now.date():
            daily += timedelta(days=1)
        if daily <= now:
            # The pull time already passed today and has not run yet. Wait out
            # the previous failure instead of retrying in a tight loop.
            if self._daily_cooldown_until and now < self._daily_cooldown_until:
                return (self._daily_cooldown_until - now).total_seconds(), True
            return 0.0, True
        interval = self._sync_interval_seconds()
        if (daily - now).total_seconds() <= interval:
            return (daily - now).total_seconds(), True
        return interval, False

    async def _sync_loop(self) -> None:
        try:
            await self._startup_sync()
        except Exception:  # noqa: BLE001 - the loop must survive startup races
            logger.exception("启动同步/提醒调度检查失败")
        while True:
            wait, daily = self.next_sync()
            await asyncio.sleep(wait)
            limit = None
            run_day = datetime.now(self.timezone).date()
            if daily:
                limit = _as_limit(
                    self.config, "daily_full_pull_limit", DEFAULT_DAILY_PULL_LIMIT
                )
            result = await self._run_fetch_job_guarded(limit)
            # Only a successful pull counts for the day it was planned for: a
            # failure stays eligible for the catch-up branch of next_sync,
            # after a cooldown so a dead endpoint is not hammered.
            if daily:
                if result.get("success"):
                    self._daily_done = run_day
                    self._daily_cooldown_until = None
                else:
                    self._daily_cooldown_until = datetime.now(self.timezone) + (
                        timedelta(seconds=self._sync_interval_seconds())
                    )

    async def _startup_sync(self) -> None:
        # Fresh install: build the initial announcement base right away.
        if self.db.count("announcements") == 0:
            result = await self._run_fetch_job_guarded(None)
            if result.get("success"):
                self._daily_done = datetime.now(self.timezone).date()
        # Re-create future reminder slots on every start so a mid-cycle
        # update or restart does not miss today's remaining slots.
        if bool(self.config.get("reminder_enabled", True)):
            targets = ReminderTargets.from_config(self.config)
            if not targets.as_pairs():
                logger.warning(
                    "到期提醒已启用，但白名单中没有可用的完整会话地址"
                    "（平台ID:GroupMessage:群号 或 平台ID:FriendMessage:用户号），"
                    "提醒不会发送"
                )
            else:
                self.scheduler.create_pending_reminders(targets)

    def next_sync_at(self) -> str:
        """Next scheduled sync as wall-clock text, for the status page."""
        wait, _ = self.next_sync()
        return (datetime.now(self.timezone) + timedelta(seconds=wait)).isoformat(
            timespec="seconds"
        )

    async def _run_fetch_job_guarded(self, limit: int | None = None) -> dict[str, Any]:
        # Push arrivals, the timed sync and the daily full pull share one lock
        # so two of them never write the same announcement at once.
        async with self._sync_lock:
            try:
                result = await self.fetch_job(limit)
            except Exception as exc:  # noqa: BLE001 - the loop must survive failures
                logger.exception("同步任务异常终止")
                return {"success": False, "error": str(exc)}
        if not result.get("success"):
            logger.warning("同步失败：%s", result.get("error"))
        return result

    async def _on_push(self, detail: dict[str, Any]) -> None:
        """Write the pushed announcement at once; its body is filled in later."""
        outcome = self.ingest.ingest_push(detail)
        logger.info("推送入库：%s（%s）", detail.get("title") or "", outcome)
        self._request_backfill()

    def _request_backfill(self) -> None:
        # Coalesce a burst of pushes into one body fetch instead of one per
        # frame; the reader keeps draining the socket meanwhile.
        if self._backfill_task is not None and not self._backfill_task.done():
            self._backfill_pending = True
            return
        self._backfill_task = asyncio.create_task(
            self._drain_backfill(), name="jx3-announcement-backfill"
        )

    async def _drain_backfill(self) -> None:
        while True:
            self._backfill_pending = False
            await self._run_fetch_job_guarded(None)
            if not self._backfill_pending:
                return

    async def _push_loop(self) -> None:
        if not bool(self.config.get("socket_enabled", True)):
            logger.info("实时推送已关闭，仅按定时同步拉取公告")
            return
        await self.push.run(self._on_push)

    async def _reminder_loop(self) -> None:
        while True:
            try:
                await self.send_due_reminders()
            except Exception:  # noqa: BLE001 - the loop must survive failures
                logger.exception("提醒分发任务异常终止")
            await asyncio.sleep(REMINDER_LOOP_INTERVAL)

    async def terminate(self) -> None:
        tasks = list(self._background_tasks)
        self._background_tasks.clear()
        if self._backfill_task is not None:
            tasks.append(self._backfill_task)
            self._backfill_task = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
