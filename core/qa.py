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
回答使用简体中文纯文本，直接、准确、可操作，不超过指定字数。
禁止使用任何 Markdown 符号（#、*、`、表格等），聊天窗口按纯文本显示。
关键结论后用括号标注来源：公告日期 + 标题；多条结论来自同一篇公告时，只在回答末尾统一标注一次，不要每条重复。
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


def clean_markdown(text: str) -> str:
    """Strip raw Markdown markers that would show up verbatim in chat messages."""
    text = _MD_HEADING_RE.sub("", text or "")
    text = text.replace("**", "").replace("`", "")
    return text.strip()


class QAService:
    def __init__(
        self,
        search_service: SearchService,
        context: Any | None = None,
        llm_provider_id: str = "",
        max_context_items: int = 8,
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
                return True, "知识库中没有找到相关公告。"
            response = await provider.text_chat(
                prompt=(
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
                f"用户问题：{question}\n\n公告资料：\n{context_text}\n\n"
                f"最大回答长度：{self.max_answer_length} 字。"
            ),
            system_prompt=ANSWER_SYSTEM_PROMPT,
        )
        return True, clean_markdown(str(getattr(response, "completion_text", "") or ""))

    def _fallback_answer(self, results: list[dict[str, Any]]) -> str:
        lines = ["根据现有公告资料，相关内容如下："]
        for item in results[:3]:
            snippet = item["content"].replace("\n", " ")
            lines.append(
                f"- {item['announcement_date']}《{item['title']}》：{snippet[:240]}（{item['url']}）"
            )
        return "\n".join(lines)
