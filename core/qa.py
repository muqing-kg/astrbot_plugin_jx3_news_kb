"""LLM relevance routing, query rewriting and grounded answers."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from .search import SearchService


ROUTE_SYSTEM_PROMPT = """你负责判断消息是否需要剑网3公告知识库处理。
只输出一个 JSON 对象，不要输出 Markdown 或解释。
JSON 字段：
- relevant: 布尔值，只有剑网3/JX3官方公告、新闻、活动、版本、奖励、兑换、领取、调整相关才为 true
- rewrite: 适合检索的短句
- keywords: 2到6个关键词
- date_from: YYYY-MM-DD 或 null
- date_to: YYYY-MM-DD 或 null
- announcement_type: "官方公告"、"官方新闻" 或 null
"""

ANSWER_SYSTEM_PROMPT = """你是剑网3官方公告知识库助手。
严格依据用户消息后面的“公告资料”回答，禁止编造。
默认以最新公告为准；如果资料中存在冲突，明确说明最新公告日期。
只有用户询问“是否改过”“以前怎样”“历史变化”时，才比较新旧公告。
回答使用简体中文纯文本，禁止 Markdown 符号（#、*、`、表格等）。
直接回答，不要开场白和客套语。

回答格式：
- 信息简单时直接回答，一两句说完，句末标注资料编号如[1]，句后用括号注明来源，如“（9月10日版本更新）”。
- 内容较多时，第一行用一句话概括；随后按主题用“一、二、三”分组，组标题末尾用括号注明该组来源，如“（9月10日版本更新）”；组内条目用“1. 2. 3.”编号，每条末尾标注资料编号如[2]。
- 凡包含具体数字、日期、时间的句子，都必须标注资料编号；一句依据多条资料时写作[1][3]。
- 数字、日期、数值保持公告原文的写法，不要换算、改写或省略。
- 完整覆盖资料中与问题相关的信息，包括：活动与系统名称，招式与奇穴，物品、道具与外观奖励，日期、时间与次数，数值、档位与货币金额，获取条件与参与资格，设置与操作路径，修复的问题，调整前后的变化，以及适用的门派、体型、区服或账号范围。
- 招式、物品等游戏名称保持公告原文的写法，如【神风宝箱】【神机千变·悟】，不要改写或自创简称。
- 来源注明使用简短描述（如“9月10日版本更新”），不要照抄完整标题；同一来源只在首次出现处注明一次。
- 编号用于内容核对，格式为[n]；除括号内注明的来源外，不要写其他来源说明。

如果用户说法与公告原文冲突，先指出公告原文的内容再回答。
如果资料不足，明确说“现有公告资料中没有找到”。
"""


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；;])\s*|\n+")
_GROUP_HEADING_RE = re.compile(r"^[一二三四五六七八九十]+、")
_ITEM_INDEX_RE = re.compile(r"^\d+\s*[\.、]\s*")
_BRACKET_NAME_RE = re.compile(r"【([^【】]{1,24})】")
_NO_FINDING_REPLY = "现有公告资料中没有找到相关内容。"


def clean_markdown(text: str) -> str:
    """Strip raw Markdown markers that would show up verbatim in chat messages."""
    text = _MD_HEADING_RE.sub("", text or "")
    text = text.replace("**", "").replace("`", "")
    return text.strip()


def _numeric_tokens(text: str) -> set[str]:
    """Numeric tokens in several equivalent spellings (07 == 7, 68,800 == 68800)."""
    tokens: set[str] = set()
    for match in _NUMBER_RE.finditer(text or ""):
        raw = match.group(0).replace(",", "")
        tokens.add(raw)
        stripped = raw.lstrip("0") or "0"
        tokens.add(stripped)
        try:
            tokens.add(str(float(stripped)))
        except ValueError:
            pass
    return tokens


def verify_answer(
    answer: str, context_items: list[dict[str, Any]]
) -> list[str]:
    """Fact-check a cited answer against the retrieved announcements.

    Sentences are dropped when they cite an out-of-range item, name a
    【game item】 that does not appear in the cited text, or contain numbers
    absent from the cited text. Group headings ("一、...") carry the short
    source tag and are exempt from number/name checks; list-item indices
    ("1. ") are not treated as claim numbers. If the model ignored the
    citation format entirely, the answer passes through unpruned rather than
    losing every factual sentence.
    """
    text = (answer or "").strip()
    if not _CITATION_RE.search(text):
        # Model ignored the citation format; keep the answer as-is instead of
        # deleting every factual sentence.
        return [text] if text else []

    kept: list[str] = []
    max_index = len(context_items)
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        citations = {int(m.group(1)) for m in _CITATION_RE.finditer(sentence)}
        bare = _CITATION_RE.sub("", sentence).strip()
        if citations and not all(1 <= index <= max_index for index in citations):
            continue

        if _GROUP_HEADING_RE.match(bare):
            kept.append(bare)
            continue

        item_match = _ITEM_INDEX_RE.match(bare)
        body = bare[item_match.end():] if item_match else bare

        if not citations:
            if not _numeric_tokens(body):
                kept.append(bare)
            continue
        cited_text = " ".join(
            str(context_items[index - 1].get("content") or "")
            for index in citations
        )
        numbers = _numeric_tokens(body)
        if numbers and not numbers <= _numeric_tokens(cited_text):
            continue
        names = _BRACKET_NAME_RE.findall(body)
        if names and not all(name in cited_text for name in names):
            continue
        kept.append(bare)

    # Visual separation: one blank line before every group heading.
    spaced: list[str] = []
    for line in kept:
        if _GROUP_HEADING_RE.match(line) and spaced:
            spaced.append("")
        spaced.append(line)
    return spaced


class QAService:
    def __init__(
        self,
        search_service: SearchService,
        context: Any | None = None,
        llm_provider_id: str = "",
        max_context_items: int = 16,
        max_answer_length: int = 1200,
    ) -> None:
        self.search = search_service
        self.context = context
        self.llm_provider_id = llm_provider_id
        self.max_context_items = max(1, int(max_context_items))
        self.max_answer_length = max(200, int(max_answer_length))

    async def _provider(self, umo: str | None = None) -> Any | None:
        if self.context is None:
            return None
        if self.llm_provider_id:
            return self.context.get_provider_by_id(self.llm_provider_id)
        return await self.context.get_using_provider_async(umo)

    async def route_query(
        self,
        question: str,
        provider: Any | None,
        umo: str | None = None,
    ) -> dict[str, Any]:
        fallback = {
            "relevant": True,
            "rewrite": question,
            "keywords": [question],
            "date_from": None,
            "date_to": None,
            "announcement_type": None,
        }
        if provider is None:
            return fallback
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        response = await provider.text_chat(
            prompt=f"当前时间：{now}\n用户消息：{question}",
            system_prompt=ROUTE_SYSTEM_PROMPT,
        )
        text = str(getattr(response, "completion_text", "") or "")
        routed = extract_json_object(text)
        if not routed or not isinstance(routed.get("relevant"), bool):
            return fallback
        routed.setdefault("rewrite", question)
        routed.setdefault("keywords", [question])
        routed["keywords"] = [str(item) for item in routed["keywords"] if str(item).strip()]
        return routed

    async def answer(
        self,
        question: str,
        umo: str | None = None,
    ) -> tuple[bool, str]:
        provider = await self._provider(umo)
        routed = await self.route_query(question, provider, umo)
        if not routed.get("relevant", False):
            return False, ""

        results = await self.search.search(
            str(routed.get("rewrite") or question),
            keywords=routed.get("keywords"),
            date_from=routed.get("date_from"),
            date_to=routed.get("date_to"),
            announcement_type=routed.get("announcement_type"),
            top_k=self.max_context_items,
        )
        if not results:
            if provider is None:
                return True, _NO_FINDING_REPLY
            response = await provider.text_chat(
                prompt=(
                    f"当前时间：{datetime.now().astimezone().isoformat(timespec='seconds')}\n"
                    f"用户问题：{question}\n\n公告资料：无\n"
                    f"请按系统规则回答。最大长度：{self.max_answer_length} 字。"
                ),
                system_prompt=ANSWER_SYSTEM_PROMPT
            )
            return True, clean_markdown(str(getattr(response, "completion_text", "") or ""))

        context_parts: list[str] = []
        for index, item in enumerate(results, start=1):
            context_parts.append(
                "\n".join(
                    [
                        f"[{index}] 日期：{item['announcement_date']}",
                        f"类型：{item['type']}",
                        f"标题：{item['title']}",
                        f"链接：{item['url']}",
                        "内容：",
                        item["content"],
                    ]
                )
            )
        context_text = "\n\n".join(context_parts)
        if provider is None:
            return True, self._fallback_answer(results)

        response = await provider.text_chat(
            prompt=(
                f"当前时间：{datetime.now().astimezone().isoformat(timespec='seconds')}\n"
                f"用户问题：{question}\n\n公告资料：\n{context_text}\n\n"
                f"最大回答长度：{self.max_answer_length} 字。"
            ),
            system_prompt=ANSWER_SYSTEM_PROMPT,
        )
        answer_text = clean_markdown(
            str(getattr(response, "completion_text", "") or "")
        )
        kept = verify_answer(answer_text, results)
        if not kept:
            return True, _NO_FINDING_REPLY
        return True, "\n".join(kept)

    def _fallback_answer(self, results: list[dict[str, Any]]) -> str:
        lines = ["根据现有公告资料，相关内容如下："]
        for item in results[:3]:
            snippet = item["content"].replace("\n", " ")
            lines.append(
                f"- {item['announcement_date']}《{item['title']}》：{snippet[:240]}（{item['url']}）"
            )
        return "\n".join(lines)
