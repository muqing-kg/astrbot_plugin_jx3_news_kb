"""JX3API push channel: WebSocket action 2002 (官方资讯)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp

DEFAULT_SOCKET_URL = "wss://socket.nicemoe.cn/"
PUSH_ACTION = 2002
RECONNECT_MIN_SECONDS = 5
RECONNECT_MAX_SECONDS = 60
CONNECT_TIMEOUT = 20


def parse_frame(raw: str) -> tuple[int | None, dict[str, Any]] | None:
    """Unwrap the ``{action, detail}`` envelope; ``None`` when unusable."""
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        action = int(data.get("action"))
    except (TypeError, ValueError):
        action = None
    detail = data.get("detail")
    return action, detail if isinstance(detail, dict) else {}


class PushClient:
    """Reconnecting listener.

    The server pushes to every accepted connection without a subscribe
    handshake, so the client only connects and reads (verified against the
    live endpoint: an unsolicited frame never arrived, but sending
    ``{action: 2002}`` produced neither an error nor a frame either).
    """

    def __init__(
        self,
        url: str = DEFAULT_SOCKET_URL,
        proxy: str = "",
        ping_seconds: int = 30,
        logger: Any | None = None,
        on_connect: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.url = str(url or DEFAULT_SOCKET_URL).strip()
        self.proxy = str(proxy or "").strip()
        self.ping_seconds = max(1, int(ping_seconds))
        self.logger = logger
        self.on_connect = on_connect

    def _log(self, level: str, message: str, *args: Any) -> None:
        if self.logger is not None:
            getattr(self.logger, level)(message, *args)

    async def run(self, on_news: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        delay = RECONNECT_MIN_SECONDS
        while True:
            try:
                await self._listen(on_news)
                self._log("warning", "推送通道已关闭，%s 秒后重连", delay)
                # A clean close means the connection worked: retry promptly.
                delay = RECONNECT_MIN_SECONDS
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                self._log("warning", "推送通道异常：%s", str(exc)[:200])
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_SECONDS)

    async def _listen(
        self, on_news: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        # total=None keeps the long-lived socket alive; the connect timeout
        # stops a blackholed handshake from parking the loop forever.
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=CONNECT_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(
                self.url,
                heartbeat=self.ping_seconds,
                proxy=self.proxy or None,
            ) as websocket:
                self._log("info", "推送通道已连接：%s", self.url)
                if self.on_connect is not None:
                    try:
                        await self.on_connect()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - never drop the socket
                        self._log("warning", "连接后的补拉失败：%s", str(exc)[:200])
                async for message in websocket:
                    if message.type is aiohttp.WSMsgType.TEXT:
                        await self._dispatch(message.data, on_news)
                    elif message.type is aiohttp.WSMsgType.BINARY:
                        # Same envelope, packed as bytes: dropping these would
                        # silently lose every push if the server switches.
                        await self._dispatch(
                            message.data.decode("utf-8", errors="replace"), on_news
                        )
                    elif message.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        return

    async def _dispatch(
        self,
        raw: str,
        on_news: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        frame = parse_frame(raw)
        if frame is None:
            self._log("debug", "无法解析的推送帧：%s", raw[:200])
            return
        action, detail = frame
        if action != PUSH_ACTION:
            self._log("debug", "忽略推送 action=%s", action)
            return
        self._log(
            "info",
            "收到公告推送：%s %s",
            detail.get("title") or "",
            detail.get("url") or "",
        )
        try:
            await on_news(detail)
        except Exception as exc:  # noqa: BLE001 - one bad frame must not drop the channel
            self._log("warning", "处理推送帧失败：%s", str(exc)[:200])
