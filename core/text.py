"""Text cleaning, hashing and chunking helpers."""

from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime, timezone
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
        text = html.unescape(data).strip()
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
        return str(value)
    return datetime.fromtimestamp(seconds, timezone.utc).astimezone(
        ZoneInfo(timezone_name)
    ).isoformat(timespec="seconds")


def parse_date(value: str) -> str:
    value = (value or "").strip()
    for pattern in ("%Y/%m/%d", "%Y-%m-%d", "%Y年%m月%d日"):
        try:
            return datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            continue
    return value


def chunk_text(
    text: str,
    max_chars: int = 900,
    overlap_chars: int = 120,
) -> list[str]:
    """Split long text by paragraphs, then by a sliding character window."""
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue
        step = max(1, max_chars - overlap_chars)
        for start in range(0, len(paragraph), step):
            piece = paragraph[start : start + max_chars].strip()
            if piece:
                pieces.append(piece)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        addition = len(piece) + (1 if current else 0)
        if current and current_len + addition > max_chars:
            chunks.append("\n".join(current))
            available = max(0, max_chars - len(piece) - 1)
            tail = "\n".join(current)[-min(overlap_chars, available):] if available else ""
            current = [tail, piece] if overlap_chars and tail else [piece]
            current_len = len("\n".join(current))
        else:
            current.append(piece)
            current_len += addition
    if current:
        chunks.append("\n".join(current))
    return chunks
