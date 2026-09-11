from core.text import chunk_text, content_fingerprint, parse_date, strip_html, unix_to_iso


def test_strip_html_keeps_text_and_line_breaks():
    raw = "<p>第一段</p><style>bad{}</style><div>第二段 &amp; 内容</div>"
    assert strip_html(raw) == "第一段\n第二段 & 内容"


def test_fingerprint_is_stable_and_whitespace_insensitive():
    left = content_fingerprint("https://a", "标题", "内容\n\n 内容")
    right = content_fingerprint("https://a", "标题", "内容 内容")
    assert left == right


def test_unix_to_iso_uses_timezone():
    assert unix_to_iso(1789012800).startswith("2026-09-10T12:00:00+08:00")


def test_parse_date_supports_api_format():
    assert parse_date("2026/09/11") == "2026-09-11"


def test_chunk_text_limits_long_paragraphs():
    chunks = chunk_text("字" * 2100, max_chars=900, overlap_chars=100)
    assert len(chunks) == 3
    assert all(len(chunk) <= 900 for chunk in chunks)
