import pytest

from core.jx3api import JX3APIError, NewsClient


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status = status
        self._payload = payload
        self._text = text

    async def json(self):
        return self._payload

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def get(self, url, headers=None, params=None, proxy=None):
        self.requests.append((url, headers, params, proxy))
        return self.response


async def test_fetch_parses_success_payload():
    session = FakeSession(FakeResponse(payload={"code": 200, "data": [{"url": "a"}]}))
    client = NewsClient()
    data = await client.fetch(10, session=session)
    assert data == [{"url": "a"}]
    assert session.requests[0][0].endswith("/news/records?limit=10")


async def test_fetch_passes_proxy_to_request():
    session = FakeSession(FakeResponse(payload={"code": 200, "data": []}))
    client = NewsClient(proxy="http://127.0.0.1:7890")

    await client.fetch(10, session=session)

    assert session.requests[0][3] == "http://127.0.0.1:7890"


async def test_fetch_uses_direct_connection_without_proxy():
    session = FakeSession(FakeResponse(payload={"code": 200, "data": []}))
    client = NewsClient(proxy="")

    await client.fetch(10, session=session)

    assert session.requests[0][3] is None


async def test_fetch_rejects_limit_above_maximum():
    client = NewsClient()
    with pytest.raises(JX3APIError):
        await client.fetch(51, session=FakeSession(FakeResponse()))


async def test_fetch_raises_for_api_error():
    session = FakeSession(FakeResponse(payload={"code": 400, "msg": "bad"}))
    with pytest.raises(JX3APIError, match="bad"):
        await NewsClient().fetch(1, session=session)
