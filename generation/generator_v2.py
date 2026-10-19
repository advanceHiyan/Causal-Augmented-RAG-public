"""
Causal-enhanced generation module (v2).

Implements adaptive fallback:
1) causal mode if L1/L2 causal evidence exists
2) retrieval mode if vector snippets are relevant
3) general mode if retrieval is weak/irrelevant
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from loguru import logger

from config.settings import settings
from retrieval.hybrid_retriever import RetrievalContext
from utils.llm_client import llm_client


@dataclass
class GenerationResult:
    answer: str
    causal_analysis: str
    counterfactual: str
    self_reflection: str
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
        profile_context: str = "",
        query_preanalysis_context: str = "",
        answer_instruction: str = "",
    ) -> GenerationResult:
        context_str = ctx.build_context_string()
        if len(context_str) > self._cfg.max_context_length:
            context_str = context_str[: self._cfg.max_context_length] + "\n...[truncated]"
        mode = self._decide_generation_mode(query=query, ctx=ctx)
        history_text = self._format_history(conversation_history or [])
        logger.debug(
            f"Generation(v2 staged): mode={mode}, context_len={len(context_str)}, "
            f"causal={ctx.has_causal_context}"
        )

        try:
            stage1 = self._run_stage1_reasoning(
                query=query,
                mode=mode,
                context_str=context_str,
                history_text=history_text,
                retrieval_query=retrieval_query,
                profile_context=profile_context,
                query_preanalysis_context=query_preanalysis_context,
                ctx=ctx,
            )
            final_answer = self._run_stage2_answer(
                query=query,
                mode=mode,
                context_str=context_str,
                history_text=history_text,
                retrieval_query=retrieval_query,
                profile_context=profile_context,
                query_preanalysis_context=query_preanalysis_context,
                answer_instruction=answer_instruction,
                stage1=stage1,
                ctx=ctx,
            )
            parsed = {
                "answer": (final_answer or "").strip(),
                "causal_analysis": self._ensure_process_text(stage1.get("causal_analysis", "")),
                "counterfactual": self._ensure_process_text(stage1.get("counterfactual", "")),
                "self_reflection": self._ensure_process_text(stage1.get("self_reflection", "")),
            }
        except Exception as e:
            logger.warning(f"Generation(v2 staged) failed, fallback single-pass: {e}")
            messages = self._build_messages(
                query=query,
                context_str=context_str,
                ctx=ctx,
                conversation_history=conversation_history or [],
                retrieval_query=retrieval_query,
                profile_context=profile_context,
                query_preanalysis_context=query_preanalysis_context,
                answer_instruction=answer_instruction,
            )
            raw_output = llm_client.chat(messages)
            parsed = self._parse_structured_output(raw_output)
        return GenerationResult(
            answer=parsed["answer"],
            causal_analysis=parsed["causal_analysis"],
            counterfactual=parsed["counterfactual"],
            self_reflection=parsed["self_reflection"],
            used_causal_context=ctx.has_causal_context,
            l1_used=ctx.l1_success,
            l2_used=ctx.l2_success,
            vector_chunks_count=len(ctx.vector_chunks),
        )

    def _run_stage1_reasoning(
        self,
        query: str,
        mode: str,
        context_str: str,
        history_text: str,
        retrieval_query: str,
        profile_context: str,
        query_preanalysis_context: str,
        ctx: RetrievalContext,
    ) -> dict[str, str]:
        """
        Stage 1: short, structured causal reasoning + reflection checks.
        Keep this prompt concise to avoid overthinking/overfitting.
        """
        system = (
            "You are a concise causal QA reasoner. Output JSON only. "
            "Do not produce long explanations."
        )
        user_parts = []
        if history_text:
            user_parts.append("Conversation history:\n" + history_text)
        if retrieval_query.strip():
            user_parts.append("Retrieval query:\n" + retrieval_query.strip())
        if (profile_context or "").strip():
            user_parts.append("Student status context:\n" + profile_context.strip())
        if (query_preanalysis_context or "").strip():
            user_parts.append("Query pre-analysis (hypothesis only):\n" + query_preanalysis_context.strip())
        user_parts.append(
            "Retrieval status:\n"
            f"- mode: {mode}\n"
            f"- L1 causal hit: {ctx.l1_success}\n"
            f"- L2 graph-causal hit: {ctx.l2_success}\n"
            f"- vector chunks: {len(ctx.vector_chunks)}"
        )
        if context_str:
            user_parts.append("Evidence:\n" + context_str)
        user_parts.append(
            "Task: produce three short fields for downstream answer generation.\n"
            "Treat structured relations (necessary/sufficient/precondition/trigger/disqualify/exception/temporal_before)\n"
            "as causal evidence forms. Combine them with user state/constraints to build causal chains.\n"
            "Checks to include in self_reflection (briefly):\n"
            "1) multi-cause one-effect (missing alternative causes)\n"
            "2) common-cause confounding (A-B correlation both from C)\n"
            "3) reverse causality possibility\n"
            "4) mediator/path gap\n"
            "5) temporal order conflict (cause after effect)\n"
            "6) selection/sample bias (single-source evidence)\n"
            "If non-causal or evidence weak, keep cautious concise statements, do not hallucinate.\n"
            "Output JSON format:\n"
            "{\"causal_analysis\":\"...\",\"counterfactual\":\"...\",\"self_reflection\":\"...\"}\n"
            "Each field <= 2 short sentences and must be non-empty."
        )
        user_parts.append("Question:\n" + query)
        data = llm_client.chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "\n\n".join(user_parts)},
            ],
            temperature=0.0,
            max_tokens=max(4096, int(getattr(settings.llm, "max_tokens", 8192))),
            parse_retry=1,
        )
        if not isinstance(data, dict):
            data = {}
        return {
            "causal_analysis": self._ensure_process_text(str(data.get("causal_analysis", "")).strip()),
            "counterfactual": self._ensure_process_text(str(data.get("counterfactual", "")).strip()),
            "self_reflection": self._ensure_process_text(str(data.get("self_reflection", "")).strip()),
        }

    def _run_stage2_answer(
        self,
        query: str,
        mode: str,
        context_str: str,
        history_text: str,
        retrieval_query: str,
        profile_context: str,
        query_preanalysis_context: str,
        answer_instruction: str,
        stage1: dict[str, str],
        ctx: RetrievalContext,
    ) -> str:
        """
        Stage 2: generate final answer only, using Stage-1 structured outputs.
        """
        system = (
            "You are a retrieval-grounded QA assistant. "
            "Produce final answer only; no extra section headers."
        )
        parts = []
        if history_text:
            parts.append("Conversation history:\n" + history_text)
        if retrieval_query.strip():
            parts.append("Retrieval query:\n" + retrieval_query.strip())
        if (profile_context or "").strip():
            parts.append("Student status context:\n" + profile_context.strip())
        if (query_preanalysis_context or "").strip():
            parts.append("Query pre-analysis (hypothesis only):\n" + query_preanalysis_context.strip())
        if (answer_instruction or "").strip():
            parts.append("Evaluation-specific answer instruction:\n" + answer_instruction.strip())
        parts.append(
            "Retrieval status:\n"
            f"- mode: {mode}\n"
            f"- L1 causal hit: {ctx.l1_success}\n"
            f"- L2 graph-causal hit: {ctx.l2_success}\n"
            f"- vector chunks: {len(ctx.vector_chunks)}"
        )
        if context_str:
            parts.append("Evidence:\n" + context_str)
        parts.append(
            "Stage-1 reasoning (already computed):\n"
            f"- Causal analysis: {stage1.get('causal_analysis','')}\n"
            f"- Counterfactual: {stage1.get('counterfactual','')}\n"
            f"- Self reflection: {stage1.get('self_reflection','')}\n"
        )
        parts.append(
            "Write a clear final answer. "
            "Do not over-claim causality. "
            "If uncertainty exists, state it briefly and give safest actionable guidance."
        )
        parts.append("Question:\n" + query)
        out = llm_client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": "\n\n".join(parts)},
            ],
            temperature=0.0,
            max_tokens=max(4096, int(getattr(settings.llm, "max_tokens", 8192))),
        )
        return (out or "").strip()

    def _ensure_process_text(self, text: str) -> str:
        t = (text or "").strip()
        if not t or t.upper() == "MISSING_PROCESS":
            return "N/A: evidence is insufficient for a reliable causal-process statement."
        return t

    def _build_messages(
        self,
        query: str,
        context_str: str,
        ctx: RetrievalContext,
        conversation_history: list[dict[str, str]],
        retrieval_query: str,
        profile_context: str = "",
        query_preanalysis_context: str = "",
        answer_instruction: str = "",
    ) -> list[dict[str, str]]:
        language = self._detect_language(query)
        mode = self._decide_generation_mode(query=query, ctx=ctx)

        user_parts: list[str] = []
        history_text = self._format_history(conversation_history)
        if history_text:
            user_parts.append("Conversation history:\n" + history_text)
        if retrieval_query.strip():
            user_parts.append("Retrieval query:\n" + retrieval_query.strip())
        if (profile_context or "").strip():
            user_parts.append("Student status context:\n" + profile_context.strip())
        if (query_preanalysis_context or "").strip():
            user_parts.append("Query pre-analysis (hypothesis only):\n" + query_preanalysis_context.strip())
        if (answer_instruction or "").strip():
            user_parts.append("Evaluation-specific answer instruction:\n" + answer_instruction.strip())

        user_parts.append(
            "Retrieval status:\n"
            f"- L1 causal hit: {ctx.l1_success}\n"
            f"- L2 graph-causal hit: {ctx.l2_success}\n"
            f"- vector chunks: {len(ctx.vector_chunks)}"
        )

        if mode == "causal":
            if context_str:
                user_parts.append("Retrieved evidence:\n" + context_str)
            user_parts.append(
                "Mode=causal. Prioritize causal evidence from L1/L2. "
                "Structured relation labels are causal storage forms; use them to compose condition-to-outcome chains. "
                "If adding causal explanation, keep it directly tied to evidence. "
                "Provide detailed, actionable steps when the question is how-to."
            )
        elif mode == "retrieval":
            vec_ctx = self._build_vector_context(query=query, ctx=ctx)
            if vec_ctx:
                user_parts.append("Retrieved snippets:\n" + vec_ctx)
            user_parts.append(
                "Mode=retrieval-only fallback. No reliable causal graph evidence. "
                "Answer from relevant vector snippets and preserve specific terms/menu labels from snippets. "
                "Do not output generic causal reasoning templates. "
                "If snippets contain similar but different features, disambiguate using the user's exact wording; "
                "if still ambiguous, provide the most likely path first and then a short alternative. "
                "Do NOT infer missing facts. If a requested item is not explicitly stated in snippets, say it is not specified."
            )
        else:
            user_parts.append(
                "Mode=general-knowledge fallback. Retrieved evidence is weak/irrelevant. "
                "Answer with cautious general knowledge. "
                "Do not claim inability solely because retrieval is weak."
            )

        user_parts.append(
            f"Answer language: {'Chinese' if language == 'zh' else 'English'} (match user query language)."
        )
        user_parts.append(
            "Output policy: detailed answers are allowed and encouraged when helpful. "
            "Reasoning can be shown, but avoid unsupported claims. "
            "For procedural questions, prefer numbered steps."
        )
        user_parts.append(
            "Output format (strict, all sections required and non-empty):\n"
            "[Final Answer]\n...\n\n"
            "[Causal Analysis]\n...\n\n"
            "[Counterfactual]\n...\n\n"
            "[Self Reflection]\n..."
        )
        user_parts.append("Question:\n" + query)

        system = (
            "You are a retrieval-grounded QA assistant with causal reasoning capability. "
            "Use this hierarchy: causal evidence > relevant retrieval snippets > cautious general knowledge. "
            "Never force causal analysis when evidence is absent."
        )
        logger.debug(
            f"Generator(v2) mode={mode} lang={language} "
            f"l1={ctx.l1_success} l2={ctx.l2_success} chunks={len(ctx.vector_chunks)}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    def _parse_structured_output(self, raw: str) -> dict[str, str]:
        text = (raw or "").strip()
        if not text:
            logger.warning("Generator(v2) returned empty output; process sections missing.")
            return {
                "answer": "",
                "causal_analysis": "MISSING_PROCESS",
                "counterfactual": "MISSING_PROCESS",
                "self_reflection": "MISSING_PROCESS",
            }

        pattern = re.compile(
            r"\[Final Answer\]\s*(.*?)\s*\[Causal Analysis\]\s*(.*?)\s*\[Counterfactual\]\s*(.*?)\s*\[Self Reflection\]\s*(.*)$",
            re.DOTALL | re.IGNORECASE,
        )
        m = pattern.search(text)
        if not m:
            logger.warning("Generator(v2) output missing required process sections; fallback parser used.")
            return {
                "answer": text,
                "causal_analysis": "MISSING_PROCESS",
                "counterfactual": "MISSING_PROCESS",
                "self_reflection": "MISSING_PROCESS",
            }

        answer = m.group(1).strip()
        causal = m.group(2).strip()
        counterfactual = m.group(3).strip()
        self_reflection = m.group(4).strip()
        return {
            "answer": answer or text,
            "causal_analysis": causal or "MISSING_PROCESS",
            "counterfactual": counterfactual or "MISSING_PROCESS",
            "self_reflection": self_reflection or "MISSING_PROCESS",
        }

    def _detect_language(self, text: str) -> str:
        zh = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
        en = len(re.findall(r"[A-Za-z]", text or ""))
        return "zh" if zh >= max(2, en // 3) else "en"

    def _build_vector_context(self, query: str, ctx: RetrievalContext, top_k: int = 4) -> str:
        if not ctx.vector_chunks:
            return ""
        rows: list[str] = []
        q = (query or "").strip()
        q_tokens = self._query_tokens(q)
        scored: list[tuple[float, dict]] = []
        for c in ctx.vector_chunks:
            text = str(c.get("text", "")).strip()
            if not text:
                continue
            text_l = text.lower()
            overlap = 0.0
            if q_tokens:
                overlap = sum(1 for t in q_tokens if t in text_l) / max(len(q_tokens), 1)
            rerank = float(c.get("rerank_score", 0.0) or 0.0)
            score = float(c.get("score", 0.0) or 0.0)
            final = overlap * 1.5 + rerank + score * 0.2
            scored.append((final, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [c for _, c in scored[:top_k]]
        for i, c in enumerate(picked, start=1):
            score = float(c.get("rerank_score", c.get("score", 0.0)) or 0.0)
            text = str(c.get("text", "")).strip()
            rows.append(f"[snippet {i} | score={score:.3f}]\n{text}")
        return "\n\n".join(rows)

    def _decide_generation_mode(self, query: str, ctx: RetrievalContext) -> str:
        if (
            self._is_causal_question(query)
            and ctx.has_causal_context
            and self._is_causal_relevant(query=query, ctx=ctx)
        ):
            return "causal"
        if self._is_vector_relevant(query=query, chunks=ctx.vector_chunks):
            return "retrieval"
        return "general"

    def _is_causal_question(self, query: str) -> bool:
        q = (query or "").lower()
        causal_en = (
            "why", "cause", "causes", "caused", "effect", "affect", "impact",
            "lead to", "result in", "due to", "because", "reason",
        )
        causal_zh = ("为什么", "原因", "导致", "影响", "结果", "因果", "由于", "所以")
        return any(k in q for k in causal_en) or any(k in query for k in causal_zh)

    def _is_causal_relevant(self, query: str, ctx: RetrievalContext) -> bool:
        q = (query or "").strip()
        if not q:
            return False
        causal_text = f"{ctx.causal_graph_text or ''}\n{ctx.causal_path_summary or ''}".lower()
        if not causal_text.strip():
            return False
        if self._detect_language(q) == "zh":
            q_chars = [ch for ch in q if "\u4e00" <= ch <= "\u9fff"]
            if not q_chars:
                return False
            uniq = set(q_chars)
            overlap = sum(1 for ch in uniq if ch in causal_text) / max(len(uniq), 1)
            return overlap >= 0.18

        stop = {
            "the", "a", "an", "to", "of", "in", "on", "for", "and", "or",
            "is", "are", "was", "were", "be", "how", "what", "why", "when",
            "where", "i", "you", "my", "your", "it", "this", "that",
        }
        toks = [t for t in re.findall(r"[a-z0-9]+", q.lower()) if len(t) >= 3 and t not in stop]
        if not toks:
            return False
        overlap = sum(1 for t in set(toks) if t in causal_text) / max(len(set(toks)), 1)
        return overlap >= 0.20

    def _is_vector_relevant(self, query: str, chunks: list[dict]) -> bool:
        if not chunks:
            return False
        q = (query or "").strip()
        if not q:
            return False

        top_chunks = chunks[:3]
        joined = " ".join(str(c.get("text", "")) for c in top_chunks).lower()
        top_rerank = max(float(c.get("rerank_score", 0.0) or 0.0) for c in top_chunks)
        top_score = max(float(c.get("score", 0.0) or 0.0) for c in top_chunks)

        if self._detect_language(q) == "zh":
            q_chars = [ch for ch in q if "\u4e00" <= ch <= "\u9fff"]
            if not q_chars:
                overlap = 0.0
            else:
                uniq = set(q_chars)
                overlap = sum(1 for ch in uniq if ch in joined) / max(len(uniq), 1)
        else:
            stop = {
                "the", "a", "an", "to", "of", "in", "on", "for", "and", "or",
                "is", "are", "was", "were", "be", "how", "what", "why", "when",
                "where", "i", "you", "my", "your", "it", "this", "that",
            }
            toks = [t for t in re.findall(r"[a-z0-9]+", q.lower()) if len(t) >= 3 and t not in stop]
            if not toks:
                overlap = 0.0
            else:
                overlap = sum(1 for t in set(toks) if t in joined) / max(len(set(toks)), 1)

        return bool(
            overlap >= 0.20
            or top_rerank >= 0.35
            or (top_score >= 0.82 and overlap >= 0.08)
        )

    def _query_tokens(self, query: str) -> list[str]:
        if self._detect_language(query) == "zh":
            chars = [ch for ch in query if "\u4e00" <= ch <= "\u9fff"]
            return list(dict.fromkeys(chars))[:30]
        stop = {
            "the", "a", "an", "to", "of", "in", "on", "for", "and", "or",
            "is", "are", "was", "were", "be", "how", "what", "why", "when",
            "where", "i", "you", "my", "your", "it", "this", "that",
        }
        toks = [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(t) >= 3 and t not in stop]
        return list(dict.fromkeys(toks))[:30]

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
