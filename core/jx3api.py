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
        headers = {"User-Agent": "AstrBot-JX3-News-KB/0.1"}
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
                payload: dict[str, Any] = await response.json()
        except aiohttp.ClientError as exc:
            raise JX3APIError(f"请求失败：{exc}") from exc
        finally:
            if owns_session:
                await session.close()

        if payload.get("code") != 200:
            raise JX3APIError(f"API error {payload.get('code')}: {payload.get('msg')}")
        data = payload.get("data")
        if not isinstance(data, list):
            raise JX3APIError("接口返回的 data 字段不是列表")
        return [item for item in data if isinstance(item, dict)]
