"""JX3API news/records client."""

from __future__ import annotations

from typing import Any

import aiohttp


class JX3APIError(RuntimeError):
    pass


class NewsClient:
    def __init__(
        self,
        base_url: str = "https://www.jx3api.com",
        records_path: str = "/news/records",
        token: str = "",
        proxy: str = "",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.records_path = "/" + records_path.lstrip("/")
        self.token = token
        self.proxy = str(proxy or "").strip()
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    def build_url(self, limit: int) -> str:
        return f"{self.base_url}{self.records_path}?limit={limit}"

    async def fetch(
        self,
        limit: int,
        session: aiohttp.ClientSession | None = None,
    ) -> list[dict[str, Any]]:
        limit = int(limit)
        if limit < 1 or limit > 50:
            raise JX3APIError("抓取条数必须在 1 到 50 之间")

        url = self.build_url(limit)
        headers = {"User-Agent": "AstrBot-JX3-Activity-Reminder/2.0"}
        params: dict[str, str] = {}
        if self.token:
            params["token"] = self.token
            headers["Authorization"] = f"Bearer {self.token}"

        owns_session = session is None
        if owns_session:
            session = aiohttp.ClientSession(timeout=self.timeout)
        assert session is not None
        try:
            async with session.get(
                url,
                headers=headers,
                params=params,
                proxy=self.proxy or None,
            ) as response:
                if response.status != 200:
                    detail = await response.text()
                    raise JX3APIError(f"HTTP {response.status}：{detail[:200]}")
                # Not every deployment labels the body as JSON, and an error
                # page or a bare list must not escape as a raw exception.
                payload = await response.json(content_type=None)
        except aiohttp.ClientError as exc:
            raise JX3APIError(f"请求失败：{exc}") from exc
        except ValueError as exc:
            raise JX3APIError(f"接口返回的不是合法 JSON：{exc}") from exc
        finally:
            if owns_session:
                await session.close()

        if not isinstance(payload, dict):
            raise JX3APIError("接口返回的 JSON 不是对象")

        # The live endpoint answers {"status": 200, "detail": [...]}; older
        # deployments used {"code": 200, "data": [...]}. Accept both.
        status = payload.get("status", payload.get("code"))
        if status != 200:
            message = payload.get("message") or payload.get("msg")
            raise JX3APIError(f"API error {status}: {message}")
        data = payload.get("detail", payload.get("data"))
        if not isinstance(data, list):
            raise JX3APIError("接口返回的 data 字段不是列表")
        return [item for item in data if isinstance(item, dict)]
