"""AstrBot plugin entry: JX3 news knowledge base, QA and reminders."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star
from astrbot.api.web import error_response, json_response, request as web_request
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

try:
    # AstrBot loads this file as package ``data.plugins.<name>.main``.
    from astrbot.core.log import LogManager
    from .core.activities import ActivityService
    from .core.database import Database
    from .core.ingest import IngestService
    from .core.jx3api import NewsClient
    from .core.qa import QAService
    from .core.search import SearchService
    from .core.scheduler import (
        DEFAULT_FETCH_INTERVAL_MINUTES,
        ReminderTargets,
        SchedulerService,
        extract_session_id,
        merge_reminder_texts,
    )
    from .core.text import parse_date, strip_html
    from .web_api.routes import ROUTE_TABLE
except ImportError:  # imported as a top-level module (tests, direct run)
    from core.activities import ActivityService
    from core.database import Database
    from core.ingest import IngestService
    from core.jx3api import NewsClient
    from core.qa import QAService
    from core.search import SearchService
    from core.scheduler import (
        DEFAULT_FETCH_INTERVAL_MINUTES,
        ReminderTargets,
        SchedulerService,
        extract_session_id,
        merge_reminder_texts,
    )
    from core.text import parse_date, strip_html
    from web_api.routes import ROUTE_TABLE

# Official per-plugin logger: attaches the enricher filter (plugin_tag) and
# the WebUI log queue handler; a raw logging.getLogger child of "astrbot"
# would crash AstrBot's queue handler with KeyError: plugin_tag.
logger = LogManager.get_plugin_logger("jx3_news_kb")

PLUGIN_NAME = "astrbot_plugin_jx3_news_kb"
REMINDER_LOOP_INTERVAL = 60


def _as_int(config: Any, key: str, default: int) -> int:
    try:
        return int(config.get(key, default))
    except (TypeError, ValueError):
        return default


def _announcement_key(url: str) -> str:
    """Identify an announcement by its ``id`` query value, else by full URL."""
    text = str(url or "").strip()
    match = re.search(r"[?&]id=(\d+)", text)
    if match:
        return match.group(1)
    return text


def _announcement_context(
    title: Any,
    announcement_type: Any,
    url: Any,
    published_at: Any,
    content: str,
) -> dict[str, Any]:
    """Shape a loaded announcement like a search hit for the QA prompt."""
    return {
        "announcement_date": str(published_at or "")[:10],
        "type": str(announcement_type or "官方公告"),
        "title": str(title or ""),
        "url": str(url or ""),
        "content": content,
    }


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
        self.search_service = SearchService(
            self.db,
            embedding_provider=lambda: self.embedding_provider,
            reranker_provider=lambda: self.reranker_provider,
        )
        self.qa = QAService(
            self.search_service,
            context=context,
            llm_provider_id=str(self.config.get("llm_provider_id") or ""),
            link_resolver=self.resolve_announcement_link,
        )
        self.scheduler = SchedulerService(
            self.db,
            self.ingest,
            self.activity_service,
            timezone_name=self.timezone.key,
            initial_fetch_limit=_as_int(self.config, "initial_fetch_limit", 50),
            fetch_limit=_as_int(self.config, "fetch_limit", 10),
            catchup_fetch_limit=_as_int(self.config, "catchup_fetch_limit", 50),
        )
        self.client = NewsClient(
            base_url=str(self.config.get("api_base_url") or "https://www.jx3api.com"),
            records_path=str(self.config.get("news_records_path") or "/news/records"),
            token=str(self.config.get("api_token") or ""),
            proxy=str(self.config.get("proxy") or ""),
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
            logger.warning("未找到 Embedding 提供商 %s", provider_id)
            return None
        if not hasattr(provider, "get_embeddings_batch"):
            logger.warning(
                "提供商 %s 不支持向量接口，向量召回已禁用",
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
            logger.warning("Reranker 提供商 %s 不可用", provider_id)
            return None
        return provider

    @property
    def embedding_provider(self) -> Any | None:
        return self._resolve_embedding_provider()

    @property
    def reranker_provider(self) -> Any | None:
        return self._resolve_reranker_provider()

    async def probe_providers(self) -> dict[str, Any]:
        """Probe embedding dimension and reranker availability.

        Providers are initialized after plugins during AstrBot startup, so a
        failed probe is not cached until the embedding probe succeeds.
        """
        if self._provider_probe is not None and self._provider_probe["embedding"]["available"]:
            return self._provider_probe
        self._provider_probe = await self._probe_providers_once()
        return self._provider_probe

    async def _probe_providers_once(self) -> dict[str, Any]:
        probe: dict[str, Any] = {
            "embedding": {
                "available": self.embedding_provider is not None,
                "provider_id": str(self.config.get("embedding_provider_id") or ""),
            },
            "reranker": {
                "available": self.reranker_provider is not None,
                "provider_id": str(self.config.get("reranker_provider_id") or ""),
            },
        }
        if self.embedding_provider is not None:
            try:
                vector = await self.embedding_provider.get_embedding("维度探测")
                probe["embedding"]["dim"] = len(vector)
            except Exception as exc:  # noqa: BLE001 - probe must not crash stats
                probe["embedding"]["available"] = False
                probe["embedding"]["error"] = str(exc)[:200]
        if self.reranker_provider is not None:
            try:
                await self.reranker_provider.rerank("能力探测", ["测试", "文档"], top_n=2)
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

    # ---------- pipeline used by daily job and web API ----------

    async def fetch_and_ingest(self, limit: int | None = None) -> dict[str, Any]:
        return await self.scheduler.fetch_and_ingest(
            self.client, self.embedding_provider, limit
        )

    async def extract_and_schedule(self, announcement_ids: list[int] | None = None) -> int:
        created = await self.activity_service.extract_for_announcements(
            await self._llm_provider(), announcement_ids
        )
        if bool(self.config.get("reminder_enabled", True)):
            targets = ReminderTargets.from_config(self.config)
            self.scheduler.create_pending_reminders(targets, announcement_ids)
        return created

    async def resolve_announcement_link(self, url: str) -> dict[str, Any] | None:
        """Answer material for a pasted announcement link: local row, else JX3API."""
        target = _announcement_key(url)
        if not target:
            return None
        # LIKE 只做子串粗筛（id=1335902 也会命中 id=13359020），再按键精确比对。
        with self.db.connect() as conn:
            rows = conn.execute(
                """
                SELECT title, type, url, content_text, published_at
                FROM announcements
                WHERE url LIKE ?
                ORDER BY published_at DESC
                """,
                (f"%{target}%",),
            ).fetchall()
        row = next(
            (
                candidate
                for candidate in rows
                if _announcement_key(candidate["url"]) == target
                and str(candidate["content_text"] or "").strip()
            ),
            None,
        )
        if row is not None:
            return _announcement_context(
                row["title"],
                row["type"],
                row["url"],
                row["published_at"],
                row["content_text"],
            )

        try:
            items = await self.client.fetch(min(50, self.scheduler.catchup_fetch_limit))
        except Exception:  # noqa: BLE001 - upstream may be down; fall back to the index
            logger.warning("按链接抓取公告失败：%s", url, exc_info=True)
            return None
        for item in items:
            desc = item.get("desc") or {}
            item_url = str(item.get("url") or desc.get("url") or "").strip()
            if _announcement_key(item_url) != target:
                continue
            content = strip_html(desc.get("content")) or strip_html(desc.get("description"))
            if not content.strip():
                return None
            return _announcement_context(
                item.get("title") or desc.get("title"),
                item.get("type"),
                item_url,
                parse_date(str(item.get("date") or "")),
                content,
            )
        return None

    def reminder_target_count(self) -> int:
        """Whitelist entries usable as reminder destinations (full addresses)."""
        return len(ReminderTargets.from_config(self.config).as_pairs())

    def valid_slot_keys(self) -> dict[tuple[int, str], str]:
        """(activity_id, scheduled_at) -> slot kind, inside current windows."""
        return self.scheduler.valid_slot_keys(
            ReminderTargets.from_config(self.config)
        )

    async def fetch_job(self) -> dict[str, Any]:
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
            asyncio.create_task(self._fetch_loop(), name="jx3-news-fetch")
        )
        self._background_tasks.append(
            asyncio.create_task(self._reminder_loop(), name="jx3-news-reminders")
        )
        logger.info("剑网3新闻公告知识库后台任务已启动")

    def background_alive(self) -> bool:
        return bool(self._background_tasks) and any(
            not task.done() for task in self._background_tasks
        )

    async def _fetch_loop(self) -> None:
        # Fresh install: build the initial knowledge base right away.
        try:
            if self.db.count("announcements") == 0:
                await self._run_fetch_job_guarded()
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
            logger.exception("启动时抓取/提醒调度检查失败")
        while True:
            interval = self._fetch_interval_seconds()
            logger.debug("next fetch in %.0f seconds", interval)
            await asyncio.sleep(interval)
            await self._run_fetch_job_guarded()

    async def _run_fetch_job_guarded(self) -> None:
        try:
            result = await self.fetch_job()
            if not result.get("success"):
                logger.warning("抓取失败：%s", result.get("error"))
        except Exception:  # noqa: BLE001 - the loop must survive failures
            logger.exception("抓取任务异常终止")

    def _fetch_interval_seconds(self) -> float:
        minutes = _as_int(self.config, "fetch_interval_minutes", DEFAULT_FETCH_INTERVAL_MINUTES)
        return float(max(1, minutes) * 60)

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
            logger.exception("问答流程执行失败")
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
