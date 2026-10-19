"""
Causal-enhanced generation module.

Assembles a structured prompt from retrieval context, then calls the LLM.

Prompt structure:
  1. System: role + reasoning instructions
  2. Retrieved context (causal graph, path summary, vector chunks)
  3. Causal chain exposition request
  4. Counterfactual reasoning request (optional)
  5. Self-reflection step (optional)
  6. User question
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from loguru import logger

from config.settings import settings
from utils.llm_client import llm_client
from retrieval.hybrid_retriever import RetrievalContext


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
你是一个擅长因果推理的知识问答助手。在回答问题时，你需要：
0. 先判断检索到的知识库内容是否真的与用户问题相关；不要被表面相似但实际无关的片段误导
1. 优先利用提供的因果链信息进行推理
2. 识别并说明问题中涉及的因果关系
3. 使推理过程显式、可追溯
4. 诚实地区分"有直接证据支持的因果关系"和"推断"
5. 如果知识库内容明显无关，但问题属于常识、通用知识或你可以稳妥回答的内容，可以直接基于通用知识作答
6. 只有当问题必须依赖当前知识库中的专门事实、而知识库又没有提供时，才说明无法根据知识库作答
"""

_CONTEXT_TMPL = """\
以下是与问题相关的知识库内容：

{context}
"""

_CHAT_HISTORY_TMPL = """\
以下是当前对话的最近历史（按时间顺序）：
{history}
"""

_RAG_QUERY_NOTE_TMPL = """\
本轮检索使用的改写查询（仅用于检索）：
{rag_query}
"""

_RETRIEVAL_STATUS_TMPL = """\
本次检索状态：
- L1因果图命中：{l1}
- L2文本图命中：{l2}
- 向量片段数：{vector_count}

解释：
- 如果 L1/L2 都未命中，只返回了一些向量片段，那么这些片段只是兜底召回，不代表它们一定真的和问题相关
- 你必须自行判断这些片段是否可用；如果明显不相关，应忽略
"""

_CAUSAL_CHAIN_INSTRUCTION = """\
在回答前，请先梳理与本问题相关的因果链：
- 列出主要的"原因 → 结果"关系
- 指出哪些因果关系有直接证据支持
"""

_COUNTERFACTUAL_INSTRUCTION = """\
请进行反事实思考：
- 如果上述关键原因未发生，结果会如何变化？
- 是否存在其他可能导致相同结果的原因路径？
"""

_SELF_REFLECTION_INSTRUCTION = """\
在给出最终答案前，请进行自我反思：
- 你的推理是否有遗漏的中间步骤？
- 提供的因果信息是否足以支撑你的结论？
- 如果信息不足，请明确说明不确定之处。
"""

_RELEVANCE_FALLBACK_INSTRUCTION = """\
请先判断“检索到的知识库内容”和“用户问题”是否属于同一主题：
- 如果高度相关，请优先基于知识库内容回答
- 如果只有弱相关或明显不相关，请忽略这些无关片段，不要因为知识库没写到就拒绝回答
- 对于常识性、通用性问题，可以直接给出基于通用知识的回答，并明确标注“以下回答基于通用知识，不依赖当前知识库”
- 只有在问题要求的是当前知识库中的专有事实、规则、名单、时间、数值等内容，而检索内容又不足时，才说明无法依据知识库回答
"""

_ANSWER_INSTRUCTION = """\
请基于以上分析，给出完整、准确的最终答案。
如果最终采用的是通用知识回答，请在答案开头明确写出：以下回答基于通用知识，不依赖当前知识库。
"""


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    answer: str
    used_causal_context: bool
    l1_used: bool
    l2_used: bool
    vector_chunks_count: int


class CausalGenerator:
    def __init__(self):
        self._cfg = settings.generation

    def generate(
        self,
        query: str,
        ctx: RetrievalContext,
        conversation_history: list[dict[str, str]] | None = None,
        retrieval_query: str = "",
    ) -> GenerationResult:
        context_str = ctx.build_context_string()
        if len(context_str) > self._cfg.max_context_length:
            context_str = context_str[: self._cfg.max_context_length] + "\n...[截断]"

        messages = self._build_messages(
            query=query,
            context_str=context_str,
            ctx=ctx,
            conversation_history=conversation_history or [],
            retrieval_query=retrieval_query,
        )
        logger.debug(
            f"Generation: {len(messages)} message(s), "
            f"context_len={len(context_str)}, causal={ctx.has_causal_context}"
        )

        answer = llm_client.chat(messages)

        return GenerationResult(
            answer=answer,
            used_causal_context=ctx.has_causal_context,
            l1_used=ctx.l1_success,
            l2_used=ctx.l2_success,
            vector_chunks_count=len(ctx.vector_chunks),
        )

    def _build_messages(
        self,
        query: str,
        context_str: str,
        ctx: RetrievalContext,
        conversation_history: list[dict[str, str]],
        retrieval_query: str,
    ) -> list[dict[str, str]]:
        # Accumulate user turn content
        user_parts: list[str] = []
        has_causal = ctx.has_causal_context

        history_text = self._format_history(conversation_history)
        if history_text:
            user_parts.append(_CHAT_HISTORY_TMPL.format(history=history_text))

        if retrieval_query.strip():
            user_parts.append(_RAG_QUERY_NOTE_TMPL.format(rag_query=retrieval_query.strip()))

        if context_str:
            user_parts.append(_CONTEXT_TMPL.format(context=context_str))

        user_parts.append(
            _RETRIEVAL_STATUS_TMPL.format(
                l1="是" if ctx.l1_success else "否",
                l2="是" if ctx.l2_success else "否",
                vector_count=len(ctx.vector_chunks),
            )
        )

        user_parts.append(_RELEVANCE_FALLBACK_INSTRUCTION)

        if has_causal:
            user_parts.append(_CAUSAL_CHAIN_INSTRUCTION)

        if has_causal and self._cfg.use_counterfactual:
            user_parts.append(_COUNTERFACTUAL_INSTRUCTION)

        if self._cfg.use_self_reflection:
            user_parts.append(_SELF_REFLECTION_INSTRUCTION)

        user_parts.append(_ANSWER_INSTRUCTION)
        user_parts.append(f"\n问题：{query}")

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    def _format_history(self, turns: list[dict[str, str]]) -> str:
        if not turns:
            return ""
        max_turns = max(1, int(self._cfg.max_chat_history_turns))
        clipped = turns[-max_turns:]
        lines: list[str] = []
        for turn in clipped:
            role = str(turn.get("role", "")).strip().lower()
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            role_name = "Assistant" if role == "assistant" else "User"
            lines.append(f"{role_name}: {content}")
        return "\n".join(lines)


generator = CausalGenerator()
