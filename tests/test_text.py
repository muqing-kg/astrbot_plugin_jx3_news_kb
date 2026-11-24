from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from core.text import (
    content_fingerprint,
    parse_date,
    parse_push_date,
    strip_html,
    unix_to_iso,
)

TZ = ZoneInfo("Asia/Shanghai")


def test_strip_html_keeps_text_and_line_breaks():
    raw = "<p>第一段</p><style>bad{}</style><div>第二段 &amp; 内容</div>"
    assert strip_html(raw) == "第一段\n第二段 & 内容"


def test_fingerprint_is_stable_and_whitespace_insensitive():
    left = content_fingerprint("https://a", "标题", "内容\n\n 内容")
    right = content_fingerprint("https://a", "标题", "内容 内容")
    assert left == right


def test_unix_to_iso_uses_timezone():
    assert unix_to_iso(1789012800).startswith("2026-09-10T12:00:00+08:00")


def test_unix_to_iso_rejects_garbage_instead_of_passing_it_through():
    assert unix_to_iso("明天") == ""
    assert unix_to_iso(10**18) == ""


def test_parse_date_supports_api_format():
    assert parse_date("2026/09/11") == "2026-09-11"


def test_parse_push_date_fills_current_year():
    today = datetime.now(TZ).date()
    value = f"{today.month}/{today.day}"
    assert parse_push_date(value) == today.isoformat()


def test_parse_push_date_rolls_back_a_year_for_future_month_day():
    today = datetime.now(TZ).date()
    future = today + timedelta(days=10)
    if future.year != today.year:
        pytest.skip("十天后的日期已跨年，不再是同一年内的未来日期")
    try:
        expected = future.replace(year=today.year - 1)
    except ValueError:  # 2 月 29 日，去年的同一天不存在
        pytest.skip("去年的同一天不存在")
    assert parse_push_date(f"{future.month}/{future.day}") == expected.isoformat()


def test_parse_push_date_rolls_forward_across_the_new_year():
    """A 1 月 date pushed in late December belongs to next year."""
    today = datetime.now(TZ).date()
    old = today - timedelta(days=200)
    if (old.month, old.day) == (2, 29):
        pytest.skip("闰日兜底另有分支")
    try:
        expected = old.replace(year=today.year + 1)
    except ValueError:
        pytest.skip("明年的同一天不存在")
    assert parse_push_date(f"{old.month}/{old.day}") == expected.isoformat()


def test_strip_html_does_not_double_unescape_entities():
    assert strip_html("<p>&amp;lt; 不是 &lt;</p>") == "&lt; 不是 <"


def test_parse_push_date_falls_back_to_full_date():
    assert parse_push_date("2026-09-10") == "2026-09-10"
    assert parse_push_date("") == ""
    # MM/DD/YYYY is not a push format; unreadable input stays empty so the
    # caller keeps whatever date it already had.
    assert parse_push_date("03/31/2026") == ""
