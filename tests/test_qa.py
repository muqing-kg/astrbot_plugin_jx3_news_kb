"""Tests for grounded answer verification (citation + number checking)."""

from __future__ import annotations

from typing import Any

from core.qa import (
    QAService,
    _numeric_tokens,
    clean_markdown,
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
        "content": "联动签到第一期9月10日7:00至10月12日7:00，签到第7天送后缀称号【不凡】。300%积分回馈期间充值时间与消耗通宝享3倍积分，9月14日7:00结束。",
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
    kept = verify_answer(answer, CONTEXT)
    assert kept == [
        "签到第一期在10月12日7:00结束。",
        "300%积分回馈9月14日结束。",
    ]
    assert all("[1]" not in line for line in kept)


def test_verify_drops_sentences_with_out_of_range_citation():
    answer = "签到第一期在10月12日结束[5]。"
    assert verify_answer(answer, CONTEXT) == []


def test_verify_drops_sentences_with_unsupported_numbers():
    # 9月30日 never appears in the cited announcement text.
    answer = "签到第一期在9月30日结束[1]。"
    assert verify_answer(answer, CONTEXT) == []


def test_verify_drops_fabricated_item_names():
    # 【双人名片】 does not appear anywhere in the cited announcements.
    answer = "双人名片【双人名片】将于9月30日上线[1]。"
    assert verify_answer(answer, CONTEXT) == []


def test_verify_keeps_real_item_names():
    answer = "签到第7天送后缀称号【不凡】[1]。"
    kept = verify_answer(answer, CONTEXT)
    assert kept == ["签到第7天送后缀称号【不凡】。"]


def test_verify_drops_uncited_factual_sentences_but_keeps_transitions():
    answer = "以下是公告要点：\n签到第一期在10月12日结束[1]。\n新坐骑售价68800通宝。"
    kept = verify_answer(answer, CONTEXT)
    assert kept == ["以下是公告要点：", "签到第一期在10月12日结束。"]


def test_verify_multi_citation_requires_all_support():
    answer = "签到第一期10月12日结束，大攻防9月12日开始[1][2]。"
    kept = verify_answer(answer, CONTEXT)
    assert len(kept) == 1


def test_verify_exempts_group_headings_with_source_tag():
    answer = (
        "唐门技改分两类。\n"
        "一、已生效调整（9月10日版本更新）\n"
        "1. 奇穴增伤叠加异常已修复[1]；\n"
        "二、计划调整（9月8日资料片预告，尚未上线）\n"
        "1. 神机值体系联动整合[1]。"
    )
    kept = verify_answer(answer, CONTEXT)
    assert kept[0] == "唐门技改分两类。"
    # One blank line before every group heading for visual separation.
    assert kept == [
        "唐门技改分两类。",
        "",
        "一、已生效调整（9月10日版本更新）",
        "1. 奇穴增伤叠加异常已修复；",
        "",
        "二、计划调整（9月8日资料片预告，尚未上线）",
        "1. 神机值体系联动整合。",
    ]


def test_verify_ignores_item_index_numbers():
    answer = "1. 签到第7天送后缀称号[1]；"
    kept = verify_answer(answer, CONTEXT)
    assert kept == ["1. 签到第7天送后缀称号；"]


def test_verify_passes_through_when_format_ignored():
    answer = "签到第一期10月12日7:00结束。300%积分回馈9月14日结束。"
    kept = verify_answer(answer, CONTEXT)
    assert kept == ["签到第一期10月12日7:00结束。300%积分回馈9月14日结束。"]


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
        assert "当前时间：" in prompt
        return type("R", (), {"completion_text": self.answer})()


def _service_with_provider(answer: str) -> QAService:
    service = QAService(_FakeSearch(), context=None)
    provider = _FakeProvider(answer)

    async def fake_provider(umo: str | None = None):
        return provider

    service._provider = fake_provider
    return service


async def test_answer_grouped_layout():
    service = _service_with_provider(
        "唐门技改分两类。\n"
        "一、已生效调整（9月10日版本更新）\n"
        "1. 奇穴增伤叠加异常已修复[1]；\n"
        "二、计划调整（9月8日资料片预告，尚未上线）\n"
        "1. 神机值体系联动整合[1]。"
    )
    relevant, answer = await service.answer("唐门技改")
    assert relevant is True
    assert "[1]" not in answer
    assert "（9月10日版本更新）" in answer
    assert "（9月8日资料片预告，尚未上线）" in answer
    assert "来源：" not in answer


async def test_answer_simple_with_inline_source():
    service = _service_with_provider(
        "签到第一期在10月12日7:00结束[1]（9月10日版本更新）。"
    )
    relevant, answer = await service.answer("签到什么时候结束")
    assert relevant is True
    assert "[1]" not in answer
    assert "（9月10日版本更新）" in answer
    assert "10月12日" in answer


async def test_answer_all_sentences_unsupported_returns_no_finding():
    service = _service_with_provider("双人名片即将于9月30日上线[1]。")
    relevant, answer = await service.answer("双人名片什么时候上线")
    assert relevant is True
    assert answer == "现有公告资料中没有找到相关内容。"


async def test_answer_untouched_when_model_ignores_citations():
    service = _service_with_provider("签到第一期在10月12日7:00结束。")
    relevant, answer = await service.answer("签到什么时候结束")
    assert relevant is True
    assert answer == "签到第一期在10月12日7:00结束。"
