"""Tests for grounded answer verification (citation + number checking)."""

from __future__ import annotations

import asyncio
from typing import Any

from core.qa import (
    QAService,
    _numeric_tokens,
    clean_markdown,
    format_sources,
    verify_answer,
)

CONTEXT = [
    {
        "chunk_id": 1,
        "announcement_id": 1,
        "revision_id": 1,
        "title": "9月10日1.5.0.9951版本更新公告",
        "announcement_date": "2026-09-10",
        "type": "官方公告",
        "url": "https://example.com/1",
        "content": "联动签到第一期9月10日7:00至10月12日7:00，签到第7天送后缀称号。300%积分回馈期间充值时间与消耗通宝享3倍积分，9月14日7:00结束。",
    },
    {
        "chunk_id": 2,
        "announcement_id": 2,
        "revision_id": 2,
        "title": "9月5日活动公告",
        "announcement_date": "2026-09-05",
        "type": "官方公告",
        "url": "https://example.com/2",
        "content": "武林争霸赛第七届季后赛开启，大攻防9月12日晚开始。",
    },
]


def test_numeric_tokens_equivalence():
    tokens = _numeric_tokens("07:00 68,800通宝 300%")
    assert "7" in tokens and "07" in tokens
    assert "68800" in tokens
    assert "300" in tokens


def test_verify_keeps_supported_sentences_and_strips_markers():
    answer = "签到第一期在10月12日7:00结束[1]。300%积分回馈9月14日结束[1]。"
    kept, sources = verify_answer(answer, CONTEXT)
    assert kept == [
        "签到第一期在10月12日7:00结束。",
        "300%积分回馈9月14日结束。",
    ]
    assert sources == {1}
    assert all("[1]" not in line for line in kept)


def test_verify_drops_sentences_with_out_of_range_citation():
    answer = "签到第一期在10月12日结束[5]。"
    kept, sources = verify_answer(answer, CONTEXT)
    assert kept == []
    assert sources == set()


def test_verify_drops_sentences_with_unsupported_numbers():
    # 9月30日 never appears in the cited announcement text.
    answer = "签到第一期在9月30日结束[1]。"
    kept, _ = verify_answer(answer, CONTEXT)
    assert kept == []


def test_verify_drops_uncited_factual_sentences_but_keeps_transitions():
    answer = "以下是公告要点：\n签到第一期在10月12日结束[1]。\n新坐骑售价68800通宝。"
    kept, _ = verify_answer(answer, CONTEXT)
    assert kept == ["以下是公告要点：", "签到第一期在10月12日结束。"]


def test_verify_multi_citation_requires_all_support():
    answer = "签到第一期10月12日结束，大攻防9月12日开始[1][2]。"
    kept, sources = verify_answer(answer, CONTEXT)
    assert len(kept) == 1
    assert sources == {1, 2}


def test_format_sources_dedupes_same_announcement():
    note = format_sources({1, 2}, CONTEXT)
    assert note.count("（来源：") == 2
    assert "2026-09-10《9月10日1.5.0.9951版本更新公告》" in note


def test_clean_markdown_strips_headings_bold_and_code():
    raw = (
        "### 一、联动活动\n"
        "1. **渡厄问心**：参与活动\n"
        "2. 联动签到`第一期`\n"
        "**来源：2026-09-10 版本更新公告**"
    )
    cleaned = clean_markdown(raw)
    assert "###" not in cleaned
    assert "**" not in cleaned
    assert "`" not in cleaned
    assert "一、联动活动" in cleaned
    assert "渡厄问心：参与活动" in cleaned


class _FakeSearch:
    async def search(self, query: str, **kwargs: Any):
        return CONTEXT


class _FakeProvider:
    def __init__(self, answer: str):
        self.answer = answer

    async def text_chat(self, prompt: str, system_prompt: str = "", **kwargs: Any):
        class _Response:
            completion_text = self.answer

        return _Response()


def _service_with_provider(answer: str) -> QAService:
    service = QAService(_FakeSearch(), context=None)
    provider = _FakeProvider(answer)

    async def fake_provider(umo: str | None = None):
        return provider

    service._provider = fake_provider
    return service


async def test_answer_ends_with_deduped_source_note_and_no_markers():
    service = _service_with_provider(
        "签到第一期在10月12日7:00结束[1]。\n300%积分回馈9月14日结束[1]。"
    )
    relevant, answer = await service.answer("签到活动什么时候结束")
    assert relevant is True
    assert "[1]" not in answer
    assert answer.endswith("（来源：2026-09-10《9月10日1.5.0.9951版本更新公告》）")
    assert "10月12日" in answer


async def test_answer_all_sentences_unsupported_returns_no_finding():
    service = _service_with_provider("双人名片即将于9月30日上线[1]。")
    relevant, answer = await service.answer("双人名片什么时候上线")
    assert relevant is True
    assert answer == "现有公告资料中没有找到相关内容。"
