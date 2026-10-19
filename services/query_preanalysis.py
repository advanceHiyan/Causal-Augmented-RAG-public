from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any

from loguru import logger

from utils.llm_client import llm_client


@dataclass
class QueryPreanalysisResult:
    enabled: bool = True
    llm_used: bool = False
    question: str = ""
    causal_question: bool = False
    analysis: dict[str, Any] | None = None
    note: str = ""

    def to_meta(self) -> dict[str, Any]:
        return asdict(self)

    def to_context_text(self) -> str:
        data = self.analysis or {}
        return json.dumps(data, ensure_ascii=False, indent=2)


def _is_causal_question(query: str) -> bool:
    q = (query or "").strip().lower()
    if not q:
        return False
    causal_en = ("cause", "effect", "if", "then", "impact", "influence", "would", "will")
    causal_zh = ("因果", "原因", "导致", "影响", "如果", "那么", "会不会", "是否")
    return any(t in q for t in causal_en) or any(t in (query or "") for t in causal_zh)


def _fallback_analysis(query: str) -> dict[str, Any]:
    # Lightweight fallback: extract two likely variables from "A ... B ..." style clauses.
    text = (query or "").strip()
    vars_guess: list[str] = []
    for seg in re.split(r"[，,。？?；;、\n]+", text):
        seg = seg.strip()
        if not seg:
            continue
        if len(vars_guess) >= 3:
            break
        vars_guess.append(seg[:24])
    if not vars_guess:
        vars_guess = ["输入变量A", "结果变量B"]
    return {
        "analysis_type": "query_only_hypothesis",
        "assumptions": [
            "该分析仅基于问题语义，不代表个体真实数据结论。",
            "未观测混杂变量可能改变方向与强度。"
        ],
        "candidate_variables": vars_guess,
        "candidate_causal_chain": [
            {
                "from": vars_guess[0],
                "to": vars_guess[-1],
                "direction": "possible",
                "polarity": "unknown",
                "confidence": "low",
                "reason": "query_semantic_hypothesis_only",
            }
        ],
        "counter_hypothesis": [
            "存在反向因果或共同原因导致的伪相关。"
        ],
        "what_data_needed": [
            "时间序列观测（干预前后）",
            "关键混杂变量（如压力、课程难度、健康状态）"
        ],
        "safety_note": "不可据此直接下个体诊断性结论。",
    }


def build_query_preanalysis(query: str) -> QueryPreanalysisResult:
    q = (query or "").strip()
    result = QueryPreanalysisResult(question=q, causal_question=_is_causal_question(q))
    if not q:
        result.enabled = False
        result.note = "empty_query"
        return result
    if not result.causal_question:
        result.analysis = {
            "analysis_type": "non_causal_question",
            "note": "问题不显著包含因果推理意图，跳过预分析。",
        }
        result.note = "non_causal_skip"
        return result

    system = (
        "You are a cautious causal pre-analysis planner. "
        "Use query-only semantics (no retrieval evidence). "
        "Output JSON only."
    )
    user = (
        "Task: produce a hypothesis-only causal pre-analysis for downstream evidence selection.\n"
        "Hard constraints:\n"
        "1) Do NOT claim facts about the specific user.\n"
        "2) Mark all links as hypothesis-level.\n"
        "3) Include confounding/reverse-causality risks.\n"
        "4) Keep concise.\n"
        "JSON schema:\n"
        "{\n"
        "  \"analysis_type\":\"query_only_hypothesis\",\n"
        "  \"assumptions\":[\"...\"],\n"
        "  \"candidate_variables\":[\"...\"],\n"
        "  \"candidate_causal_chain\":[{\"from\":\"...\",\"to\":\"...\",\"direction\":\"possible\",\"polarity\":\"positive|negative|unknown\",\"confidence\":\"low|medium\",\"reason\":\"...\"}],\n"
        "  \"counter_hypothesis\":[\"...\"],\n"
        "  \"what_data_needed\":[\"...\"],\n"
        "  \"safety_note\":\"...\"\n"
        "}\n"
        f"Question:\n{q}"
    )
    try:
        data = llm_client.chat_json(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=1200,
            parse_retry=1,
        )
        if not isinstance(data, dict):
            raise ValueError("query preanalysis json is not dict")
        result.llm_used = True
        result.analysis = data
        result.note = "ok"
        return result
    except Exception as e:
        logger.debug(f"query preanalysis fallback: {e}")
        result.llm_used = False
        result.analysis = _fallback_analysis(q)
        result.note = "fallback"
        return result

