import asyncio

from core.socket import PUSH_ACTION, PushClient, parse_frame


def test_parse_frame_unwraps_envelope():
    action, detail = parse_frame(
        '{"action":2002,"detail":{"title":"公告","url":"https://a","date":"09/11"}}'
    )
    assert action == PUSH_ACTION
    assert detail["url"] == "https://a"


def test_parse_frame_tolerates_missing_detail_and_string_action():
    action, detail = parse_frame('{"action":"2002","detail":null}')
    assert action == PUSH_ACTION
    assert detail == {}


def test_parse_frame_rejects_garbage():
    assert parse_frame("not json") is None
    assert parse_frame("[1, 2]") is None
    assert parse_frame('"text"') is None


def test_dispatch_only_forwards_push_action():
    client = PushClient()
    seen: list[dict] = []

    async def on_news(detail):
        seen.append(detail)

    asyncio.run(client._dispatch('{"action":2001,"detail":{"x":1}}', on_news))
    asyncio.run(client._dispatch('{"action":2002,"detail":{"title":"t"}}', on_news))
    asyncio.run(client._dispatch("broken", on_news))

    assert seen == [{"title": "t"}]


def test_client_settings_fall_back_to_defaults():
    client = PushClient(url="", proxy=None, ping_seconds=0)
    assert client.url.startswith("wss://")
    assert client.proxy == ""
    assert client.ping_seconds == 1


def test_dispatch_survives_a_failing_handler():
    """A broken handler must not kill the read loop (which would reconnect)."""
    client = PushClient()

    async def boom(detail):
        raise RuntimeError("ingest failed")

    asyncio.run(client._dispatch('{"action":2002,"detail":{"title":"t"}}', boom))
