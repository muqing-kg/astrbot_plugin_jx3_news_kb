"""Text cleaning, hashing and date helpers."""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from zoneinfo import ZoneInfo


_TAG_BLOCKS = {
    "p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "tr", "br", "table", "blockquote",
}
_SKIP_TAGS = {"script", "style", "noscript"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag in _TAG_BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        # convert_charrefs already decoded entities; unescaping again would
        # turn a literal "&lt;" in the text into "<".
        text = data.strip()
        if text:
            self.parts.append(text)
            self.parts.append(" ")


def strip_html(raw: str | None) -> str:
    """Convert announcement HTML into stable plain text."""
    parser = _TextExtractor()
    parser.feed(raw or "")
    text = "".join(parser.parts)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t\r\f]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def content_fingerprint(url: str, title: str, content_text: str) -> str:
    normalized = "\n".join((url, normalize_text(title), normalize_text(content_text)))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def unix_to_iso(value: str | int | None, timezone_name: str = "Asia/Shanghai") -> str:
    if value in (None, "", 0, "0"):
        return ""
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return ""
    try:
        return datetime.fromtimestamp(seconds, timezone.utc).astimezone(
            ZoneInfo(timezone_name)
        ).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        # A nonsense epoch must not abort a whole batch of announcements.
        return ""


def parse_date(value: str) -> str:
    """Source date as YYYY-MM-DD, or "" when it cannot be read.

    An unreadable date must not reach the database as raw text: it is
    concatenated into ``published_at`` and SQLite's ``date()`` would return
    NULL for the whole row.
    """
    value = (value or "").strip()
    for pattern in ("%Y/%m/%d", "%Y-%m-%d", "%Y年%m月%d日"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            continue
    match = re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", value)
    if not match:
        return ""
    try:
        return date(*(int(part) for part in match.groups())).isoformat()
    except ValueError:
        return ""
