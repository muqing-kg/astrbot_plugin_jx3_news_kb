"""AstrBot plugin entry: JX3 news knowledge base, QA and reminders."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star
from astrbot.api.web import error_response, json_response, request as web_request
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

try:
    # AstrBot loads this file as package ``data.plugins.<name>.main``.
    from .core.activities import ActivityService
    from .core.database import Database
    from .core.ingest import IngestService
    from .core.jx3api import NewsClient
    from .core.qa import QAService
    from .core.search import SearchService
    from .core.scheduler import (
        ReminderTargets,
        SchedulerService,
        extract_session_id,
        merge_reminder_texts,
    )
    from .web_api.routes import ROUTE_TABLE
except ImportError:  # imported as a top-level module (tests, direct run)
    from core.activities import ActivityService
    from core.database import Database
    from core.ingest import IngestService
    from core.jx3api import NewsClient
    from core.qa import QAService
    from core.search import SearchService
    from core.scheduler import (
        ReminderTargets,
        SchedulerService,
        extract_session_id,
        merge_reminder_texts,
    )
    from web_api.routes import ROUTE_TABLE

logger = logging.getLogger("astrbot.plugin.jx3_news_kb")

PLUGIN_NAME = "astrbot_plugin_jx3_news_kb"
REMINDER_LOOP_INTERVAL = 60


def _as_int(config: Any, key: str, default: int) -> int:
    try:
        return int(config.get(key, default))
    except (TypeError, ValueError):
        return default


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
        self._embedding_provider = self._resolve_embedding_provider()
        self._reranker_provider = self._resolve_reranker_provider()
        self.search_service = SearchService(
            self.db,
            embedding_provider=self._embedding_provider,
            reranker_provider=self._reranker_provider,
        )
        self.qa = QAService(
            self.search_service,
            context=context,
            llm_provider_id=str(self.config.get("llm_provider_id") or ""),
        )
        self.scheduler = SchedulerService(
            self.db,
            self.ingest,
            self.activity_service,
            timezone_name=self.timezone.key,
            initial_fetch_limit=_as_int(self.config, "initial_fetch_limit", 50),
            daily_fetch_limit=_as_int(self.config, "daily_fetch_limit", 10),
            catchup_fetch_limit=_as_int(self.config, "catchup_fetch_limit", 50),
        )
        self.client = NewsClient(
            base_url=str(self.config.get("api_base_url") or "https://www.jx3api.com"),
            records_path=str(self.config.get("news_records_path") or "/news/records"),
            token=str(self.config.get("api_token") or ""),
        )
        self._background_tasks: list[asyncio.Task] = []
        self._provider_probe: dict[str, Any] | None = None
        self._register_routes()
        self._try_start_background_tasks()

    # ---------- configuration helpers ----------

    def _resolve_timezone(self) -> ZoneInfo:
        name = str(self.config.get("timezone") or "Asia/Shanghai")
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            return ZoneInfo("Asia/Shanghai")

    def _resolve_embedding_provider(self) -> Any | None:
        provider_id = str(self.config.get("embedding_provider_id") or "").strip()
        if not provider_id:
            return None
        provider = self.context.get_provider_by_id(provider_id)
        if provider is None:
            logger.warning("embedding provider %s not found", provider_id)
            return None
        if not hasattr(provider, "get_embeddings_batch"):
            logger.warning(
                "provider %s does not support embeddings; vector recall disabled",
                provider_id,
            )
            return None
        return provider

    def _resolve_reranker_provider(self) -> Any | None:
        provider_id = str(self.config.get("reranker_provider_id") or "").strip()
        if not provider_id:
            return None
        provider = self.context.get_provider_by_id(provider_id)
        if provider is None or not hasattr(provider, "rerank"):
            logger.warning("reranker provider %s unavailable", provider_id)
            return None
        return provider

    @property
    def embedding_provider(self) -> Any | None:
        return self._embedding_provider

    @property
    def reranker_provider(self) -> Any | None:
        return self._reranker_provider

    async def probe_providers(self) -> dict[str, Any]:
        """Probe embedding dimension and reranker availability (cached per load)."""
        if self._provider_probe is None:
            self._provider_probe = await self._probe_providers_once()
        return self._provider_probe

    async def _probe_providers_once(self) -> dict[str, Any]:
        probe: dict[str, Any] = {
            "embedding": {
                "available": self._embedding_provider is not None,
                "provider_id": str(self.config.get("embedding_provider_id") or ""),
            },
            "reranker": {
                "available": self._reranker_provider is not None,
                "provider_id": str(self.config.get("reranker_provider_id") or ""),
            },
        }
        if self._embedding_provider is not None:
            try:
                vector = await self._embedding_provider.get_embedding("维度探测")
                probe["embedding"]["dim"] = len(vector)
            except Exception as exc:  # noqa: BLE001 - probe must not crash stats
                probe["embedding"]["available"] = False
                probe["embedding"]["error"] = str(exc)[:200]
        if self._reranker_provider is not None:
            try:
                await self._reranker_provider.rerank("能力探测", ["测试", "文档"], top_n=2)
                probe["reranker"]["ok"] = True
            except Exception as exc:  # noqa: BLE001
                probe["reranker"]["ok"] = False
                probe["reranker"]["error"] = str(exc)[:200]
        return probe

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
                f"JX3 news KB {route}",
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
                logger.exception("web api %s failed", handler.__name__)
                return error_response(f"internal error: {exc}", status_code=500)
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

    # ---------- pipeline used by daily job and web API ----------

    async def fetch_and_ingest(self, limit: int | None = None) -> dict[str, Any]:
        return await self.scheduler.fetch_and_ingest(
            self.client, self._embedding_provider, limit
        )

    async def extract_and_schedule(self, announcement_ids: list[int] | None = None) -> int:
        created = await self.activity_service.extract_for_announcements(
            await self._llm_provider(), announcement_ids
        )
        if bool(self.config.get("reminder_enabled", True)):
            targets = ReminderTargets.from_config(self.config)
            self.scheduler.create_pending_reminders(targets, announcement_ids)
        return created

    def reminder_target_count(self) -> int:
        """Whitelist entries usable as reminder destinations (full addresses)."""
        return len(ReminderTargets.from_config(self.config).as_pairs())

    async def daily_job(self) -> dict[str, Any]:
        result = await self.fetch_and_ingest()
        self.scheduler.prune_logs()
        if result.get("success"):
            await self.extract_and_schedule()
        return result

    # ---------- reminders ----------

    async def _send_text(self, target_type: str, target_id: str, text: str) -> tuple[bool, str]:
        """target_id is a full session address ``platform:MessageType:id``."""
        try:
            sent = await self.context.send_message(target_id, MessageChain().message(text))
            return bool(sent), "" if sent else "no matching platform"
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
            status = "sent" if ok else "failed"
            for item in items:
                self.scheduler.mark_reminder(int(item["id"]), status, error)
            if ok:
                sent += len(items)
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
        self._background_tasks.append(
            asyncio.create_task(self._daily_loop(), name="jx3-news-daily")
        )
        self._background_tasks.append(
            asyncio.create_task(self._reminder_loop(), name="jx3-news-reminders")
        )
        logger.info("JX3 news KB background tasks started")

    def background_alive(self) -> bool:
        return bool(self._background_tasks) and any(
            not task.done() for task in self._background_tasks
        )

    async def _daily_loop(self) -> None:
        # Fresh install: build the initial knowledge base right away.
        try:
            if self.db.count("announcements") == 0:
                await self._run_daily_job_guarded()
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
        except Exception:  # noqa: BLE001 - the loop must survive startup races
            logger.exception("startup fetch/schedule check failed")
        while True:
            delay = self._seconds_until(self._daily_fetch_time())
            logger.debug("next daily fetch in %.0f seconds", delay)
            await asyncio.sleep(delay)
            await self._run_daily_job_guarded()

    async def _run_daily_job_guarded(self) -> None:
        try:
            result = await self.daily_job()
            if not result.get("success"):
                logger.warning("daily fetch failed: %s", result.get("error"))
        except Exception:  # noqa: BLE001 - the loop must survive failures
            logger.exception("daily job crashed")

    def _daily_fetch_time(self) -> str:
        return str(self.config.get("daily_fetch_time") or "00:00")

    def _seconds_until(self, hhmm: str) -> float:
        try:
            hour_text, minute_text = hhmm.split(":", 1)
            hour, minute = int(hour_text), int(minute_text)
        except (ValueError, AttributeError):
            hour, minute = 0, 0
        now = datetime.now(self.timezone)
        moment = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if moment <= now:
            moment += timedelta(days=1)
        return max(1.0, (moment - now).total_seconds())

    async def _reminder_loop(self) -> None:
        while True:
            try:
                await self.send_due_reminders()
            except Exception:  # noqa: BLE001 - the loop must survive failures
                logger.exception("reminder dispatch crashed")
            await asyncio.sleep(REMINDER_LOOP_INTERVAL)

    async def terminate(self) -> None:
        tasks = list(self._background_tasks)
        self._background_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ---------- message handling ----------

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """Answer awakened messages that ask about JX3 announcements.

        ``is_wake`` is forced to True for every message matched by this plugin's
        event listener; ``is_at_or_wake_command`` only reflects a real wake
        (wake prefix, @bot, reply-to-bot or private chat). The answer is sent
        directly and the event stopped so AstrBot's default LLM stage does not
        answer the same question a second time.
        """
        if not getattr(event, "is_at_or_wake_command", False):
            return
        question = (event.message_str or "").strip()
        if not question or not self._session_allowed(event):
            return
        try:
            relevant, answer = await self.qa.answer(question, event.unified_msg_origin)
        except Exception:  # noqa: BLE001 - never break the bot's main flow
            logger.exception("qa pipeline failed")
            return
        if relevant and answer:
            await event.send(event.plain_result(answer))
            event.stop_event()

    def _session_allowed(self, event: AstrMessageEvent) -> bool:
        group_id = str(event.get_group_id() or "").strip()
        if group_id:
            if not bool(self.config.get("allow_group", True)):
                return False
            whitelist = [
                extract_session_id(str(item))
                for item in (self.config.get("whitelist_groups") or [])
            ]
            return not whitelist or group_id in whitelist
        if not bool(self.config.get("allow_private", True)):
            return False
        whitelist = [
            extract_session_id(str(item))
            for item in (self.config.get("whitelist_users") or [])
        ]
        sender = str(event.get_sender_id() or "").strip()
        return not whitelist or sender in whitelist
