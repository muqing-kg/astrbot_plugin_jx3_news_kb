"""Smoke tests for the plugin main class with a fake astrbot module."""

from __future__ import annotations

import asyncio
import importlib
import shutil
import sys
import types
from datetime import datetime, timedelta
from typing import Any

import pytest

ROUTE_COUNT = 9


def _install_fake_astrbot(monkeypatch, data_path) -> None:
    if "astrbot" in sys.modules:
        return

    astrbot = types.ModuleType("astrbot")
    api = types.ModuleType("astrbot.api")
    event = types.ModuleType("astrbot.api.event")
    star = types.ModuleType("astrbot.api.star")
    web = types.ModuleType("astrbot.api.web")
    core = types.ModuleType("astrbot.core")
    utils = types.ModuleType("astrbot.core.utils")
    log_mod = types.ModuleType("astrbot.core.log")

    class _FakeLogManager:
        @staticmethod
        def get_plugin_logger(plugin_name: str):
            import logging as _logging

            return _logging.getLogger(f"fake.plugin.{plugin_name}")

    log_mod.LogManager = _FakeLogManager
    astrbot_path = types.ModuleType("astrbot.core.utils.astrbot_path")

    event.EventMessageType = types.SimpleNamespace(
        ALL="ALL", GROUP_MESSAGE="GROUP", PRIVATE_MESSAGE="PRIVATE"
    )
    event.filter = types.SimpleNamespace(
        EventMessageType=event.EventMessageType,
        event_message_type=lambda *a, **k: (lambda fn: fn),
        on_astrbot_loaded=lambda *a, **k: (lambda fn: fn),
    )

    class _MessageChain:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.parts: list[Any] = []

        def message(self, text: str) -> "_MessageChain":
            self.parts.append(text)
            return self

    event.MessageChain = _MessageChain
    event.AstrMessageEvent = object

    class _Star:
        def __init__(self, context: Any) -> None:
            self.context = context

    star.Star = _Star
    star.Context = object

    def _json_response(data: Any = None, **kwargs: Any) -> dict[str, Any]:
        return {"__json__": data, **kwargs}

    def _error_response(message: str, *, status_code: int = 400, **kwargs: Any) -> dict[str, Any]:
        return {"__error__": message, "status_code": status_code}

    web.json_response = _json_response
    web.error_response = _error_response
    web.request = types.SimpleNamespace(
        query=types.SimpleNamespace(keys=lambda: iter([]), get=lambda k, d=None: d),
        path_params={},
        json=None,
    )
    astrbot_path.get_astrbot_plugin_data_path = lambda: str(data_path)

    astrbot.api = api
    astrbot.api.event = event
    astrbot.api.star = star
    astrbot.api.web = web
    astrbot.core = core
    astrbot.core.log = log_mod
    astrbot.core.utils = utils
    astrbot.core.utils.astrbot_path = astrbot_path
    for name, module in {
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.event": event,
        "astrbot.api.star": star,
        "astrbot.api.web": web,
        "astrbot.core": core,
        "astrbot.core.log": log_mod,
        "astrbot.core.utils": utils,
        "astrbot.core.utils.astrbot_path": astrbot_path,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


class FakeContext:
    def __init__(self):
        self.registered_apis: list[tuple[str, Any, list[str]]] = []
        self.sent: list[tuple[str, Any]] = []

    def register_web_api(self, route, handler, methods, desc):
        self.registered_apis.append((route, handler, methods))

    def get_provider_by_id(self, provider_id):
        return None

    async def get_using_provider_async(self, umo=None):
        return None

    async def send_message(self, session, chain):
        self.sent.append((session, chain))
        return True

    class _Platforms:
        platform_insts: list[Any] = [
            types.SimpleNamespace(meta=lambda: types.SimpleNamespace(id="fake"))
        ]

    platform_manager = _Platforms()


class _StubClient:
    def __init__(self, items: list[dict[str, Any]] | None = None) -> None:
        self.items = items or []
        self.calls: list[int | None] = []

    async def fetch(self, limit: int | None = None):
        self.calls.append(limit)
        return self.items


def _make_plugin(monkeypatch, tmp_path, config=None, stub_client=False):
    _install_fake_astrbot(monkeypatch, tmp_path)
    for name in list(sys.modules):
        if name == "main" or (name.startswith("main.") if isinstance(name, str) else False):
            del sys.modules[name]
    import main as main_module

    if stub_client:
        main_module.NewsClient = lambda **kwargs: _StubClient()

    context = FakeContext()
    plugin = main_module.JX3NewsKBPlugin(context, config or {})
    return main_module, plugin, context


def test_plugin_registers_all_routes(monkeypatch, tmp_path):
    main_module, plugin, context = _make_plugin(monkeypatch, tmp_path)
    routes = [route for route, _, _ in context.registered_apis]
    assert len(routes) == ROUTE_COUNT
    assert all(route.startswith("/astrbot_plugin_jx3_news_kb") for route in routes)
    assert "/astrbot_plugin_jx3_news_kb/announcements/<announcement_id>/delete" in routes
    assert "/astrbot_plugin_jx3_news_kb/activities/<activity_id>/delete" in routes
    assert not [route for route in routes if "rebuild" in route]


def test_news_client_uses_configured_proxy(monkeypatch, tmp_path):
    config = {"proxy": "http://127.0.0.1:7890"}

    _, plugin, _ = _make_plugin(monkeypatch, tmp_path, config=config)

    assert plugin.client.proxy == "http://127.0.0.1:7890"


def test_out_of_range_fetch_limits_are_clamped(monkeypatch, tmp_path):
    """A typo like 100 or 0 would make NewsClient reject every sync."""
    config = {
        "initial_fetch_limit": 500,
        "fetch_limit": 0,
        "catchup_fetch_limit": "abc",
    }

    _, plugin, _ = _make_plugin(monkeypatch, tmp_path, config=config)

    assert plugin.scheduler.initial_fetch_limit == 50
    assert plugin.scheduler.fetch_limit == 1
    assert plugin.scheduler.catchup_fetch_limit == 50


def test_push_writes_row_then_fills_body(monkeypatch, tmp_path):
    main_module, plugin, _ = _make_plugin(monkeypatch, tmp_path, stub_client=True)
    plugin.client = _StubClient(
        [
            {
                "catid": "24581",
                "date": "2026/09/11",
                "title": "中秋活动公告",
                "type": "官方公告",
                "url": "https://example.com/push",
                "desc": {
                    "id": "9",
                    "inputtime": "1789012800",
                    "content": "<p>活动到 9 月 20 日 7:00 结束。</p>",
                },
            }
        ]
    )

    async def scenario() -> None:
        await plugin._on_push(
            {
                "title": "中秋活动公告",
                "type": "官方公告",
                "url": "https://example.com/push",
                "date": "09/11",
            }
        )
        # The push returns before the body fetch; it runs as a background task.
        assert plugin._backfill_task is not None
        await plugin._backfill_task

    asyncio.run(scenario())

    assert len(plugin.client.calls) == 1
    with plugin.db.connect() as conn:
        row = conn.execute(
            "SELECT title, content_text FROM announcements WHERE url = ?",
            ("https://example.com/push",),
        ).fetchone()
    assert row["title"] == "中秋活动公告"
    assert "9 月 20 日" in row["content_text"]


def test_next_sync_uses_configured_interval(monkeypatch, tmp_path):
    _, plugin, _ = _make_plugin(
        monkeypatch,
        tmp_path,
        config={"sync_interval_minutes": 60, "daily_full_pull_time": "03:00"},
    )
    now = datetime.now(plugin.timezone)
    plugin.config["daily_full_pull_time"] = (now + timedelta(hours=10)).strftime("%H:%M")

    wait, daily = plugin.next_sync()
    if daily:  # 十小时后跨过午夜时，日更会比常规间隔更早
        assert wait <= 60 * 60
    else:
        assert wait == pytest.approx(3600)


def test_next_sync_switches_to_daily_full_pull(monkeypatch, tmp_path):
    _, plugin, _ = _make_plugin(
        monkeypatch,
        tmp_path,
        config={"sync_interval_minutes": 60, "daily_full_pull_time": "03:00"},
    )
    now = datetime.now(plugin.timezone)
    if now.hour >= 23:
        pytest.skip("临近午夜，日更与常规间隔会重叠")
    plugin.config["daily_full_pull_time"] = (now + timedelta(minutes=30)).strftime(
        "%H:%M"
    )

    wait, daily = plugin.next_sync()
    assert daily is True
    assert 0 < wait <= 30 * 60


def test_next_sync_catches_up_a_missed_daily_pull(monkeypatch, tmp_path):
    """A sync that straddles 03:00 must not silently skip the day's full pull."""
    _, plugin, _ = _make_plugin(
        monkeypatch,
        tmp_path,
        config={"sync_interval_minutes": 60, "daily_full_pull_time": "03:00"},
    )
    now = datetime.now(plugin.timezone)
    missed = now - timedelta(minutes=5)
    if missed.date() != now.date():
        pytest.skip("刚过午夜，日更时刻落在昨天")
    plugin.config["daily_full_pull_time"] = missed.strftime("%H:%M")

    wait, daily = plugin.next_sync()
    assert daily is True
    assert wait == 0.0

    plugin._daily_done = now.date()
    _, daily = plugin.next_sync()
    assert daily is False


def test_send_due_reminders_marks_sent(monkeypatch, tmp_path):
    main_module, plugin, context = _make_plugin(monkeypatch, tmp_path)
    with plugin.db.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO announcements(
                source_id, title, type, url, content_text, raw_json,
                content_hash, published_at, updated_at_source
            ) VALUES ('1', 't', 'g', 'u', 'c', '{}', 'h',
                      '2026-08-13T10:00:00+08:00', '2026-08-13T10:00:00+08:00')
            """
        )
        announcement_id = cursor.lastrowid
        cursor = conn.execute(
            """
            INSERT INTO activities(
                announcement_id, name, action, category, end_time
            ) VALUES (?, '签到领券', '使用', 'free_coupon',
                      '2099-09-17T07:00:00+08:00')
            """,
            (announcement_id,),
        )
        activity_id = cursor.lastrowid
        past = (
            plugin.scheduler.now() - timedelta(minutes=1)
        ).isoformat(timespec="seconds")
        conn.execute(
            """
            INSERT INTO reminders(
                activity_id, target_type, target_id, scheduled_at, message_text
            ) VALUES (?, 'group', 'fake:GroupMessage:10001', ?, '提醒内容')
            """,
            (activity_id, past),
        )

    sent = asyncio.run(plugin.send_due_reminders())
    assert sent == 1
    assert len(context.sent) == 1
    session, chain = context.sent[0]
    assert session == "fake:GroupMessage:10001"
    assert chain.parts == ["提醒内容"]

    with plugin.db.connect() as conn:
        status = conn.execute("SELECT status FROM reminders").fetchone()[0]
    assert status == "sent"


def test_send_due_reminders_merges_same_target(monkeypatch, tmp_path):
    main_module, plugin, context = _make_plugin(monkeypatch, tmp_path)
    with plugin.db.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO announcements(
                source_id, title, type, url, content_text, raw_json,
                content_hash, published_at, updated_at_source
            ) VALUES ('1', 't', 'g', 'u', 'c', '{}', 'h',
                      '2026-08-13T10:00:00+08:00', '2026-08-13T10:00:00+08:00')
            """
        )
        announcement_id = cursor.lastrowid
        for name in ("活动A", "活动B"):
            cursor = conn.execute(
                """
                INSERT INTO activities(
                    announcement_id, name, action, category, end_time
                ) VALUES (?, ?, '使用', 'free_coupon',
                          '2099-09-17T07:00:00+08:00')
                """,
                (announcement_id, name),
            )
            past = (
                plugin.scheduler.now() - timedelta(minutes=1)
            ).isoformat(timespec="seconds")
            conn.execute(
                """
                INSERT INTO reminders(
                    activity_id, target_type, target_id, scheduled_at, message_text
                ) VALUES (?, 'group', 'fake:GroupMessage:10001', ?, ?)
                """,
                (cursor.lastrowid, past, f"【{name} 到期提醒】\n待办：使用"),
            )

    sent = asyncio.run(plugin.send_due_reminders())
    assert sent == 2
    # One combined message per target instead of one message per reminder.
    assert len(context.sent) == 1
    _, chain = context.sent[0]
    assert "到期提醒 · 共 2 项" in chain.parts[0]
    assert "【活动A 到期提醒】" in chain.parts[0]
    assert "【活动B 到期提醒】" in chain.parts[0]

    with plugin.db.connect() as conn:
        statuses = [
            row[0] for row in conn.execute("SELECT status FROM reminders").fetchall()
        ]
    assert statuses == ["sent", "sent"]


def test_background_tasks_start_and_terminate(monkeypatch, tmp_path):
    main_module, plugin, _ = _make_plugin(monkeypatch, tmp_path)
    plugin.client = _StubClient()

    async def scenario():
        plugin._start_background_tasks()
        tasks = list(plugin._background_tasks)
        assert len(tasks) == 3
        await asyncio.sleep(0.1)
        alive = [not task.done() for task in tasks]
        await plugin.terminate()
        await asyncio.gather(*tasks, return_exceptions=True)
        finished = all(task.cancelled() or task.done() for task in tasks)
        return alive, finished

    alive, finished = asyncio.run(scenario())
    assert all(alive)
    assert finished
    assert plugin._background_tasks == []


async def test_tasks_start_on_init_inside_running_loop(monkeypatch, tmp_path):
    """AstrBot instantiates plugins in a running loop and never fires
    on_astrbot_loaded on reload; the loops must start from __init__."""
    main_module, plugin, _ = _make_plugin(monkeypatch, tmp_path, stub_client=True)
    # Tasks are created synchronously during construction.
    assert len(plugin._background_tasks) == 3
    assert plugin.background_alive() is True
    await asyncio.sleep(0.1)
    assert plugin.background_alive() is True
    await plugin.terminate()
    assert plugin.background_alive() is False


def test_plugin_loads_as_astrbot_package(monkeypatch, tmp_path):
    """AstrBot imports the plugin as ``data.plugins.<name>.main``; relative
    imports inside main.py must resolve in that package context."""
    _install_fake_astrbot(monkeypatch, tmp_path)
    plugin_dst = tmp_path / "data" / "plugins" / "astrbot_plugin_jx3_news_kb"
    plugin_dst.mkdir(parents=True)
    for item in ("main.py", "core", "web_api"):
        source = _plugin_source_dir() / item
        if source.is_dir():
            shutil.copytree(source, plugin_dst / item)
        else:
            shutil.copy2(source, plugin_dst / item)
    monkeypatch.syspath_prepend(str(tmp_path))

    module_name = "data.plugins.astrbot_plugin_jx3_news_kb.main"
    for name in [n for n in sys.modules if n.startswith("data.")]:
        del sys.modules[name]
    try:
        module = importlib.import_module(module_name)
    finally:
        for name in [n for n in sys.modules if n.startswith("data.")]:
            del sys.modules[name]

    assert module.__name__ == module_name
    context = FakeContext()
    plugin = module.JX3NewsKBPlugin(context, {})
    assert len(context.registered_apis) == ROUTE_COUNT


def _plugin_source_dir():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent


def test_send_failure_keeps_reminder_pending(monkeypatch, tmp_path):
    """A platform error must not burn the reminder: it stays pending."""
    _, plugin, context = _make_plugin(monkeypatch, tmp_path)

    async def _boom(session, chain):
        raise RuntimeError("platform offline")

    context.send_message = _boom

    with plugin.db.connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO announcements(
                source_id, title, type, url, content_text, raw_json,
                content_hash, published_at, updated_at_source
            ) VALUES ('1', 't', 'g', 'u', 'c', '{}', 'h',
                      '2026-08-13T10:00:00+08:00', '2026-08-13T10:00:00+08:00')
            """
        )
        announcement_id = cursor.lastrowid
        cursor = conn.execute(
            """
            INSERT INTO activities(
                announcement_id, name, action, category, end_time
            ) VALUES (?, '活动A', '使用', 'free_coupon',
                      '2099-09-17T07:00:00+08:00')
            """,
            (announcement_id,),
        )
        past = (plugin.scheduler.now() - timedelta(minutes=1)).isoformat(
            timespec="seconds"
        )
        conn.execute(
            """
            INSERT INTO reminders(
                activity_id, target_type, target_id, scheduled_at, message_text
            ) VALUES (?, 'group', 'fake:GroupMessage:10001', ?, ?)
            """,
            (cursor.lastrowid, past, "【活动A 到期提醒】\n待办：使用"),
        )

    assert asyncio.run(plugin.send_due_reminders()) == 0

    with plugin.db.connect() as conn:
        row = conn.execute("SELECT status, error FROM reminders").fetchone()
    assert row["status"] == "pending"
    assert "platform offline" in row["error"]


def test_next_sync_waits_out_the_daily_cooldown(monkeypatch, tmp_path):
    """A failed daily pull backs off instead of retrying in a tight loop."""
    from datetime import datetime, timedelta

    _, plugin, _ = _make_plugin(monkeypatch, tmp_path)
    plugin.config["daily_full_pull_time"] = "00:00"
    plugin._daily_done = None
    plugin._daily_cooldown_until = datetime.now(plugin.timezone) + timedelta(minutes=5)

    wait, daily = plugin.next_sync()

    assert daily is True
    assert 0 < wait <= 300

