"""Smoke tests for the plugin main class with a fake astrbot module."""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any


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
    astrbot.core.utils = utils
    astrbot.core.utils.astrbot_path = astrbot_path
    for name, module in {
        "astrbot": astrbot,
        "astrbot.api": api,
        "astrbot.api.event": event,
        "astrbot.api.star": star,
        "astrbot.api.web": web,
        "astrbot.core": core,
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


class FakeEvent:
    def __init__(self, is_wake=True, group_id="", sender_id="10086", message_str="新活动"):
        self.is_wake = is_wake
        self.message_str = message_str
        self.unified_msg_origin = "fake:GroupMessage:10001"
        self._group_id = group_id
        self._sender_id = sender_id

    def get_group_id(self):
        return self._group_id

    def get_sender_id(self):
        return self._sender_id

    def plain_result(self, text):
        return f"plain:{text}"


def _make_plugin(monkeypatch, tmp_path, config=None):
    _install_fake_astrbot(monkeypatch, tmp_path)
    for name in list(sys.modules):
        if name == "main" or (name.startswith("main.") if isinstance(name, str) else False):
            del sys.modules[name]
    import main as main_module

    context = FakeContext()
    plugin = main_module.JX3NewsKBPlugin(context, config or {})
    return main_module, plugin, context


def test_plugin_registers_all_routes(monkeypatch, tmp_path):
    main_module, plugin, context = _make_plugin(monkeypatch, tmp_path)
    routes = [route for route, _, _ in context.registered_apis]
    assert len(routes) == 9
    assert all(route.startswith("/astrbot_plugin_jx3_news_kb") for route in routes)
    assert "/astrbot_plugin_jx3_news_kb/announcements/<announcement_id>/delete" in routes


def test_message_requires_wake(monkeypatch, tmp_path):
    main_module, plugin, _ = _make_plugin(monkeypatch, tmp_path)
    called = []

    async def fake_answer(question, umo):
        called.append(question)
        return True, "answer"

    plugin.qa.answer = fake_answer

    async def consume(event):
        return [item async for item in plugin.on_message(event)]

    outputs = asyncio.run(consume(FakeEvent(is_wake=False)))
    assert outputs == []
    assert called == []

    outputs = asyncio.run(consume(FakeEvent(is_wake=True)))
    assert outputs == ["plain:answer"]
    assert called == ["新活动"]


def test_message_group_whitelist(monkeypatch, tmp_path):
    main_module, plugin, _ = _make_plugin(
        monkeypatch, tmp_path, config={"whitelist_groups": ["10001"]}
    )
    called = []

    async def fake_answer(question, umo):
        called.append(question)
        return True, "answer"

    plugin.qa.answer = fake_answer

    async def consume(event):
        return [item async for item in plugin.on_message(event)]

    outside = asyncio.run(consume(FakeEvent(group_id="99999")))
    assert outside == [] and called == []

    inside = asyncio.run(consume(FakeEvent(group_id="10001")))
    assert inside == ["plain:answer"]


def test_private_allow_switch(monkeypatch, tmp_path):
    main_module, plugin, _ = _make_plugin(
        monkeypatch, tmp_path, config={"allow_private": False}
    )
    called = []

    async def fake_answer(question, umo):
        called.append(question)
        return True, "answer"

    plugin.qa.answer = fake_answer

    async def consume(event):
        return [item async for item in plugin.on_message(event)]

    outputs = asyncio.run(consume(FakeEvent(group_id="")))
    assert outputs == [] and called == []


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
        conn.execute(
            """
            INSERT INTO reminders(
                activity_id, target_type, target_id, scheduled_at, message_text
            ) VALUES (?, 'group', '10001',
                      datetime('now', 'localtime', '-1 minute'), '提醒内容')
            """,
            (activity_id,),
        )

    sent = asyncio.run(plugin.send_due_reminders())
    assert sent == 1
    assert len(context.sent) == 1
    session, chain = context.sent[0]
    assert session.endswith(":GroupMessage:10001")
    assert chain.parts == ["提醒内容"]

    with plugin.db.connect() as conn:
        status = conn.execute("SELECT status FROM reminders").fetchone()[0]
    assert status == "sent"


class _StubClient:
    async def fetch(self, limit: int):
        return []


def test_background_tasks_start_and_terminate(monkeypatch, tmp_path):
    main_module, plugin, _ = _make_plugin(monkeypatch, tmp_path)
    plugin.client = _StubClient()

    async def scenario():
        await plugin.on_astrbot_loaded()
        tasks = list(plugin._background_tasks)
        assert len(tasks) == 2
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
