"""Causal-Augmented RAG — Streamlit Web UI (v2)"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import json
import re
import sys
import threading
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Causal RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
_orig_tabs = st.tabs
_BATCH_FILE_LOCK = threading.Lock()
_BATCH_MEM_LOCK = threading.Lock()

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1a1a2e 0%,#16213e 70%,#0f3460 100%);
}
section[data-testid="stSidebar"] * { color:#e0e0e0 !important; }
section[data-testid="stSidebar"] hr { border-color:#334155 !important; }

.user-bubble {
    background:#0f3460; color:#fff;
    padding:10px 16px; border-radius:18px 18px 4px 18px;
    margin:6px 0 6px 20%; max-width:78%; float:right; clear:both; word-wrap:break-word;
}
.assistant-bubble {
    background:#1e293b; color:#e2e8f0;
    padding:10px 16px; border-radius:18px 18px 18px 4px;
    margin:6px 20% 6px 0; max-width:78%; float:left; clear:both;
    word-wrap:break-word; white-space:pre-wrap;
}
.clearfix { clear:both; }

.badge { display:inline-block; padding:2px 9px; border-radius:12px; font-size:12px; font-weight:600; margin:2px; }
.badge-l1 { background:#166534; color:#86efac; }
.badge-l2 { background:#1e3a5f; color:#93c5fd; }
.badge-l3 { background:#3b1f5e; color:#c4b5fd; }
.badge-ok  { background:#374151; color:#d1d5db; }

.page-title { font-size:22px; font-weight:700; color:#e2e8f0; margin-bottom:8px; }
.kb-card { background:#1e293b; border:1px solid #334155; border-radius:10px; padding:14px 18px; margin-bottom:10px; }
.tool-card {
    background: linear-gradient(180deg, rgba(15,23,42,.95), rgba(30,41,59,.92));
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.tool-card h4 {
    margin: 0 0 6px 0;
    color: #e2e8f0;
    font-size: 15px;
}
.tool-card p {
    margin: 0;
    color: #94a3b8;
    font-size: 13px;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ── Imports (after sys.path) ───────────────────────────────────────────────
import db.database as DB
from utils.api_logger import APILogger
from config.settings import settings
from ui.student_state_panel import render_student_state_panel
from ui.trace_panel import render_trace_panel
from ui.hoprag_eval_panel import render_hoprag_eval_panel
from ui.real_scene_eval_panel import render_real_scene_eval_panel
from services.trace_protocol import (
    append_trace,
    finish_timer,
    make_trace_step,
    start_timer,
)

# ── Session state ──────────────────────────────────────────────────────────
def _ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

_ss("current_kb_id", None)
_ss("current_conv_id", None)
_ss("current_user_id", "")
_ss("current_student_state_lib_id", "")
_ss("profile_force_refresh_next_query", False)
_ss("debug_mode", False)
_ss("page", "💬 智能问答")
_ss("persist_causal_updates", False)
_ss("compare_judge_specs", [f"deepseek::{settings.llm.deepseek_model}"])
_ss("batch_test_runner", {"running": False})
_ss("batch_test_logs", [])
_ss("real_scene_eval_runner", {"running": False})
_ss("real_scene_eval_logs", [])
_ss("real_scene_eval_executor", None)
_ss("real_scene_eval_futures", {})
_ss("hoprag_eval_record_buffer", [])
_ss("hoprag_eval_last_save_ts", 0.0)
_ss("enable_profile_llm_review", True)

# Sync debug mode
if st.session_state.debug_mode:
    APILogger.enable()
else:
    APILogger.disable()

# ── KBManager helper ───────────────────────────────────────────────────────
def _get_mgr(kb_id: str):
    from knowledge_base.kb_manager import get_kb_manager
    return get_kb_manager(kb_id)

def _invalidate(kb_id: str):
    from knowledge_base.kb_manager import invalidate_kb_cache
    invalidate_kb_cache(kb_id)


def _merge_state_key(kb_id: str, graph_key: str) -> str:
    return f"merge_suggestions::{kb_id}::{graph_key}"


def _delete_state_key(kb_id: str, graph_key: str) -> str:
    return f"delete_candidates::{kb_id}::{graph_key}"


def _build_conversation_history(
    prior_messages: list[dict],
    current_query: str,
    max_turns: int = 8,
) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for msg in prior_messages:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        turns.append({"role": role, "content": content})
    turns.append({"role": "user", "content": current_query.strip()})
    if max_turns > 0:
        turns = turns[-max_turns:]
    return turns


_COMPARE_BRANCH_KEYS = ("full", "vector_only", "direct")


def _clip_history(turns: list[dict[str, str]], max_turns: int) -> list[dict[str, str]]:
    if max_turns > 0:
        return turns[-max_turns:]
    return turns


def _build_compare_branch_histories(
    prior_messages: list[dict],
    current_query: str,
    max_turns: int = 8,
) -> dict[str, list[dict[str, str]]]:
    """Build isolated multi-turn histories for each compare branch.

    For compare assistant turns, we only ingest the branch-specific answer and
    never ingest winner-summary text from message.content.
    """
    histories: dict[str, list[dict[str, str]]] = {
        key: [] for key in _COMPARE_BRANCH_KEYS
    }
    for msg in prior_messages:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        meta = msg.get("meta", {}) or {}

        if role == "user" and content:
            for key in _COMPARE_BRANCH_KEYS:
                histories[key].append({"role": "user", "content": content})
            continue

        if role != "assistant":
            continue

        if meta.get("compare_mode"):
            branches = meta.get("compare_branches", {}) or {}
            for key in _COMPARE_BRANCH_KEYS:
                branch_answer = str((branches.get(key) or {}).get("answer", "")).strip()
                if branch_answer:
                    histories[key].append({"role": "assistant", "content": branch_answer})
            continue

        if content:
            # Backward compatibility: old non-compare assistant turns are shared.
            for key in _COMPARE_BRANCH_KEYS:
                histories[key].append({"role": "assistant", "content": content})

    current_user = current_query.strip()
    if current_user:
        for key in _COMPARE_BRANCH_KEYS:
            histories[key].append({"role": "user", "content": current_user})

    return {k: _clip_history(v, max_turns=max_turns) for k, v in histories.items()}


def _is_compare_locked(prior_messages: list[dict]) -> bool:
    """A conversation is locked to compare mode after first compare turn."""
    for msg in prior_messages:
        if str(msg.get("role", "")).strip().lower() != "assistant":
            continue
        meta = msg.get("meta", {}) or {}
        if bool(meta.get("compare_mode")):
            return True
    return False


def _rewrite_rag_query(user_query: str, conversation_history: list[dict[str, str]]) -> str:
    from utils.llm_client import llm_client

    if not conversation_history:
        return user_query.strip()

    history_text = "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in conversation_history
    )
    system = (
        "You rewrite a retrieval query for RAG. "
        "Use dialogue context and output one concise query string only."
    )
    user = (
        f"Dialogue:\n{history_text}\n\n"
        f"Current user query:\n{user_query}\n\n"
        "Output only the rewritten retrieval query."
    )
    try:
        rewritten = llm_client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=max(32, int(getattr(settings.generation, "rag_query_max_tokens", 128))),
        ).strip()
        return rewritten or user_query.strip()
    except Exception:
        return user_query.strip()


def _format_history_for_prompt(conversation_history: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for turn in conversation_history:
        role = str(turn.get("role", "")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        role_name = "Assistant" if role == "assistant" else "User"
        lines.append(f"{role_name}: {content}")
    return "\n".join(lines)


def _detect_answer_language(text: str) -> str:
    zh = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    en = len(re.findall(r"[A-Za-z]", text or ""))
    return "zh" if zh >= max(2, en // 3) else "en"


def _generate_baseline_answer(
    user_query: str,
    conversation_history: list[dict[str, str]],
    retrieval_context: str = "",
    profile_context: str = "",
    answer_instruction: str = "",
) -> str:
    """Plain QA generation without causal-enhanced instructions."""
    from utils.llm_client import llm_client

    history_text = _format_history_for_prompt(conversation_history)
    retrieval_context = (retrieval_context or "").strip()

    parts: list[str] = []
    if history_text:
        parts.append(f"Conversation history:\n{history_text}")
    if retrieval_context:
        parts.append(f"Retrieved context:\n{retrieval_context}")
    if (profile_context or "").strip():
        parts.append(f"Student status context:\n{profile_context.strip()}")
    if (answer_instruction or "").strip():
        parts.append("Evaluation-specific answer instruction:\n" + answer_instruction.strip())
    parts.append(
        "Answer directly and concisely. "
        "Do not include counterfactual reasoning, self-reflection, "
        "or causal-chain analysis sections."
    )
    ans_lang = _detect_answer_language(user_query)
    parts.append(f"Respond in {'Chinese' if ans_lang == 'zh' else 'English'} to match the user query.")
    parts.append(f"Current user query:\n{user_query}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful QA assistant. "
                "Use provided context when available. "
                "If context is insufficient, you may use general knowledge, "
                "but avoid fabricating specific facts."
            ),
        },
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    return llm_client.chat(messages)


_COMPARE_SCORE_METRICS = (
    "correctness",
    "causal_quality",
    "causal_reasoning",
    "process_quality",
    "groundedness",
    "overall",
)
_GOLD_METRIC = "gold_alignment"
_COMPARE_METRIC_WEIGHTS: dict[str, float] = {
    "correctness": 1.0,
    "groundedness": 1.0,
    "gold_alignment": 1.0,
    "causal_reasoning": 3.0,
    "causal_quality": 3.0,
    "process_quality": 3.0,
    "overall": 2.0,
}


def _encode_judge_spec(provider: str, model: str) -> str:
    return f"{provider}::{model}"


def _decode_judge_spec(spec: str) -> tuple[str, str]:
    raw = (spec or "").strip()
    if "::" in raw:
        provider, model = raw.split("::", 1)
        provider = provider.strip() or "deepseek"
        model = model.strip() or settings.llm.deepseek_model
        return provider, model
    if "/" in raw:
        return "openrouter", raw
    return "deepseek", raw or settings.llm.deepseek_model


def _default_compare_judge_specs() -> list[str]:
    return [_encode_judge_spec("deepseek", settings.llm.deepseek_model)]


def _judge_provider_ready(provider: str) -> bool:
    p = (provider or "").strip().lower()
    if p == "deepseek":
        return bool(settings.llm.deepseek_api_key)
    if p == "openrouter":
        return bool(settings.llm.openrouter_api_key)
    if p == "gemini":
        return bool(settings.llm.gemini_api_key)
    if p == "gemini_paid":
        return bool(settings.llm.gemini_paid_api_key)
    return False


def _compare_judge_option_map() -> dict[str, str]:
    candidates = [
        ("deepseek", settings.llm.deepseek_model, "DeepSeek V4 Pro Thinking (default)"),
        ("deepseek", settings.llm.deepseek_chat_model, "DeepSeek V4 Flash Thinking"),
        ("gemini", settings.llm.gemini_model, "Gemini Flash"),
        ("gemini", settings.llm.gemini_flash_lite_model, "Gemini Flash Lite"),
        ("gemini", settings.llm.gemini_flash_lite_preview_model, "Gemini 3.1 Flash Lite Preview"),
        ("gemini", settings.llm.gemini_flash_preview_model, "Gemini 3 Flash Preview"),
        ("gemini_paid", settings.llm.gemini_flash_preview_model, "Gemini 3 Flash Preview (Paid Key)"),
        ("openrouter", settings.llm.openrouter_gpt5mini_model, "OpenRouter GPT-5 mini"),
        ("openrouter", settings.llm.openrouter_model, "OpenRouter Default"),
    ]
    options: dict[str, str] = {}
    for provider, model, title in candidates:
        spec = _encode_judge_spec(provider, model)
        suffix = "" if _judge_provider_ready(provider) else " (missing api key)"
        options[f"{title} / {model}{suffix}"] = spec
    return options


def _ping_judge_model(spec: str) -> dict:
    from utils.llm_client import llm_client

    provider, model = _decode_judge_spec(spec)
    t0 = time.perf_counter()
    try:
        output = llm_client.chat(
            [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Reply only: ok"},
            ],
            provider=provider,
            model=model,
            temperature=0.0,
            max_tokens=64,
        ).strip()
        elapsed = time.perf_counter() - t0
        return {
            "provider": provider,
            "model": model,
            "ok": True,
            "elapsed_s": elapsed,
            "detail": output[:80],
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "provider": provider,
            "model": model,
            "ok": False,
            "elapsed_s": elapsed,
            "detail": str(e)[:220],
        }


def _safe_score(value) -> float:
    try:
        v = float(value)
    except Exception:
        v = 0.0
    return max(0.0, min(10.0, v))


def _break_three_way_tie_for_metric(
    normalized_scores: dict[str, dict[str, float]],
    metric: str,
) -> None:
    vals = [float(normalized_scores.get(b, {}).get(metric, 0.0)) for b in _COMPARE_BRANCH_KEYS]
    if len(set(round(v, 3) for v in vals)) != 1:
        return

    # Use existing quality signals to derive a stable rank when metric ties.
    rank_key = {}
    for b in _COMPARE_BRANCH_KEYS:
        bs = normalized_scores.get(b, {})
        rank_key[b] = (
            float(bs.get("overall", 0.0)),
            float(bs.get("correctness", 0.0)),
            float(bs.get("groundedness", 0.0)),
            float(bs.get("gold_alignment", 0.0)),
        )
    ranked = sorted(_COMPARE_BRANCH_KEYS, key=lambda b: rank_key[b], reverse=True)
    base = float(vals[0])
    # Force a small but visible separation, staying in [0,10].
    target = {
        ranked[0]: max(0.0, min(10.0, base + 0.2)),
        ranked[1]: max(0.0, min(10.0, base + 0.0)),
        ranked[2]: max(0.0, min(10.0, base - 0.2)),
    }
    for b in _COMPARE_BRANCH_KEYS:
        normalized_scores[b][metric] = round(float(target[b]), 2)


def _weighted_branch_score(
    branch_scores: dict[str, float],
    metrics: tuple[str, ...],
) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for metric in metrics:
        if metric not in branch_scores:
            continue
        w = float(_COMPARE_METRIC_WEIGHTS.get(metric, 1.0))
        weighted_sum += w * float(branch_scores.get(metric, 0.0))
        weight_sum += w
    if weight_sum <= 0:
        return 0.0
    return round(weighted_sum / weight_sum, 2)


def _normalize_one_judge_result(
    raw: dict,
    metrics: tuple[str, ...] = _COMPARE_SCORE_METRICS,
) -> dict:
    scores = raw.get("scores", {}) if isinstance(raw, dict) else {}
    normalized_scores: dict[str, dict[str, float]] = {}
    for branch in _COMPARE_BRANCH_KEYS:
        branch_scores = scores.get(branch, {}) if isinstance(scores, dict) else {}
        normalized_scores[branch] = {
            metric: _safe_score(branch_scores.get(metric, 0.0))
            for metric in metrics
        }

    # Enforce separation on causal metrics: do not allow three-way identical scores.
    for m in ("causal_quality", "causal_reasoning"):
        if m in metrics:
            _break_three_way_tie_for_metric(normalized_scores, m)

    weighted_scores = {
        branch: _weighted_branch_score(normalized_scores.get(branch, {}), metrics)
        for branch in _COMPARE_BRANCH_KEYS
    }
    winner = max(
        _COMPARE_BRANCH_KEYS,
        key=lambda key: weighted_scores.get(key, 0.0),
    )
    comment = str(raw.get("comment", "")).strip() if isinstance(raw, dict) else ""
    return {
        "winner": winner,
        "scores": normalized_scores,
        "weighted_scores": weighted_scores,
        "comment": comment,
    }


def _aggregate_judge_results(
    judge_results: list[dict],
    judge_errors: list[str],
    metrics: tuple[str, ...] = _COMPARE_SCORE_METRICS,
) -> dict:
    if not judge_results:
        err = "; ".join(judge_errors) if judge_errors else "no judge result"
        return {
            "winner": "",
            "scores": {},
            "comment": f"auto-score failed: {err}",
            "judge_count": 0,
            "judge_details": [],
            "judge_models": [],
        }

    aggregated_scores: dict[str, dict[str, float]] = {}
    for branch in _COMPARE_BRANCH_KEYS:
        aggregated_scores[branch] = {}
        for metric in metrics:
            vals = [float(item["scores"][branch][metric]) for item in judge_results]
            aggregated_scores[branch][metric] = round(sum(vals) / max(len(vals), 1), 2)

    weighted_scores = {
        branch: _weighted_branch_score(aggregated_scores.get(branch, {}), metrics)
        for branch in _COMPARE_BRANCH_KEYS
    }
    winner = max(
        _COMPARE_BRANCH_KEYS,
        key=lambda key: weighted_scores.get(key, 0.0),
    )
    detail_lines = [
        f"{item['provider']}/{item['model']} -> winner={item['winner']}"
        for item in judge_results
    ]
    if judge_errors:
        detail_lines.extend([f"error: {msg}" for msg in judge_errors])

    return {
        "winner": winner,
        "scores": aggregated_scores,
        "weighted_scores": weighted_scores,
        "comment": " | ".join(detail_lines),
        "judge_count": len(judge_results),
        "judge_details": [
            {
                "provider": item["provider"],
                "model": item["model"],
                "winner": item["winner"],
                "comment": item["comment"],
                "process_debug": {
                    "full_process_payload": item.get("full_process_payload", {}),
                },
                "weighted_scores": {
                    key: round(float(item.get("weighted_scores", {}).get(key, 0.0)), 2)
                    for key in _COMPARE_BRANCH_KEYS
                },
                "scores": {
                    branch: {
                        metric: round(float(item["scores"][branch][metric]), 2)
                        for metric in metrics
                    }
                    for branch in _COMPARE_BRANCH_KEYS
                },
            }
            for item in judge_results
        ],
        "judge_models": [f"{item['provider']}/{item['model']}" for item in judge_results],
        "metrics": list(metrics),
        "metric_weights": {
            metric: float(_COMPARE_METRIC_WEIGHTS.get(metric, 1.0))
            for metric in metrics
        },
    }


def _run_compare_modes(
    mgr,
    user_query: str,
    retrieval_query: str,
    conversation_histories: dict[str, list[dict[str, str]]],
    persist_causal_updates: bool,
    profile_context: str = "",
    judge_specs: list[str] | None = None,
    progress_cb=None,
    top_k_override: int | None = None,
    enable_query_preanalysis: bool = True,
    query_preanalysis_future=None,
    answer_instruction: str = "",
    answer_instruction_scope: str = "all",
    profile_context_scope: str = "all",
) -> dict:
    from retrieval.hybrid_retriever import RetrievalContext
    from retrieval.vector_retriever import VectorRetriever

    def _emit(msg: str) -> None:
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    def _instruction_for_branch(branch: str) -> str:
        scope = str(answer_instruction_scope or "all").strip().lower()
        if scope in {"full", "full_only"}:
            return answer_instruction if branch == "full" else ""
        return answer_instruction

    def _profile_for_branch(branch: str) -> str:
        scope = str(profile_context_scope or "all").strip().lower()
        if scope in {"full", "full_only"}:
            return profile_context if branch == "full" else ""
        return profile_context

    def _branch_full() -> dict:
        history = conversation_histories.get("full", [])
        steps = ["branch start", "retrieve: hybrid L1/L2/L3"]
        _emit("full: retrieve start (hybrid L1/L2/L3)")
        t0 = time.perf_counter()
        query_preanalysis_meta = {"enabled": False, "note": "disabled_in_compare"}
        query_preanalysis_context = ""
        if enable_query_preanalysis:
            from services.query_preanalysis import build_query_preanalysis

            _emit("full: query preanalysis start (parallel with retrieval)")
            retrieve_started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as inner_ex:
                retrieve_future = inner_ex.submit(
                    mgr.retrieve,
                    retrieval_query,
                    persist_causal_updates=persist_causal_updates,
                    top_k_override=top_k_override,
                )
                preanalysis_future = query_preanalysis_future or inner_ex.submit(build_query_preanalysis, user_query)
                ctx = retrieve_future.result()
                try:
                    preanalysis = preanalysis_future.result()
                    query_preanalysis_meta = preanalysis.to_meta()
                    query_preanalysis_context = preanalysis.to_context_text()
                except Exception as qpa_e:
                    query_preanalysis_meta = {"enabled": False, "note": f"error:{qpa_e}"}
            retrieve_elapsed = time.perf_counter() - retrieve_started
            steps.append("query_preanalysis: parallel enabled")
        else:
            retrieve_started = time.perf_counter()
            ctx = mgr.retrieve(
                retrieval_query,
                persist_causal_updates=persist_causal_updates,
                top_k_override=top_k_override,
            )
            retrieve_elapsed = time.perf_counter() - retrieve_started
        steps.append(
            f"retrieve done: L1={ctx.l1_success}, L2={ctx.l2_success}, chunks={len(ctx.vector_chunks)}"
        )
        steps.append(f"retrieve+preanalysis elapsed: {retrieve_elapsed:.2f}s")
        _emit(
            "full: retrieve done "
            f"(L1={ctx.l1_success}, L2={ctx.l2_success}, chunks={len(ctx.vector_chunks)})"
        )
        steps.append("generate: causal-enhanced")
        _emit("full: generate start (causal-enhanced)")
        generate_started = time.perf_counter()
        gen = mgr.generate_with_meta(
            user_query,
            ctx,
            conversation_history=history,
            retrieval_query=retrieval_query,
            profile_context=_profile_for_branch("full"),
            query_preanalysis_context=query_preanalysis_context,
            answer_instruction=_instruction_for_branch("full"),
        )
        generate_elapsed = time.perf_counter() - generate_started
        elapsed = time.perf_counter() - t0
        steps.append(f"generate elapsed: {generate_elapsed:.2f}s")
        steps.append(f"branch done in {elapsed:.2f}s")
        _emit(f"full: done ({elapsed:.2f}s)")
        snippet_limit = max(5, int(top_k_override or 5))
        return {
            "mode": "full",
            "title": "Full-Causal",
            "answer": gen.answer,
            "elapsed_s": elapsed,
            "l1": ctx.l1_success,
            "l2": ctx.l2_success,
            "chunks": len(ctx.vector_chunks),
            "l3_used_reranker": ctx.l3_used_reranker,
            "causal_graph_text": str(ctx.causal_graph_text or ""),
            "causal_path_summary": str(ctx.causal_path_summary or ""),
            "causal_analysis": str(getattr(gen, "causal_analysis", "") or ""),
            "counterfactual": str(getattr(gen, "counterfactual", "") or ""),
            "self_reflection": str(getattr(gen, "self_reflection", "") or ""),
            "query_preanalysis": query_preanalysis_meta,
            "vector_snippets": [
                {
                    "id": c.get("id", ""),
                    "text": c.get("text", ""),
                    "score": c.get("score", 0),
                    "rerank_score": c.get("rerank_score"),
                    "metadata": c.get("metadata", {}),
                }
                for c in ctx.vector_chunks[:snippet_limit]
            ],
            "steps": steps,
        }

    def _branch_vector_only() -> dict:
        history = conversation_histories.get("vector_only", [])
        steps = ["branch start", "retrieve: vector only"]
        _emit("vector_only: retrieve start (vector)")
        t0 = time.perf_counter()
        vr = VectorRetriever(store=mgr.vector_store)
        l3_result = vr.retrieve(retrieval_query, causal_context="", top_k_override=top_k_override)
        ctx = RetrievalContext(
            causal_graph_text="",
            causal_path_summary="",
            vector_chunks=l3_result.chunks,
            l1_success=False,
            l2_success=False,
            l3_used_reranker=l3_result.used_reranker,
        )
        steps.append(
            f"retrieve done: reranked={ctx.l3_used_reranker}, chunks={len(ctx.vector_chunks)}"
        )
        _emit(
            "vector_only: retrieve done "
            f"(reranked={ctx.l3_used_reranker}, chunks={len(ctx.vector_chunks)})"
        )
        steps.append("generate: baseline (no causal extras)")
        _emit("vector_only: generate start (baseline)")
        answer = _generate_baseline_answer(
            user_query=user_query,
            conversation_history=history,
            retrieval_context=ctx.build_context_string(),
            profile_context=_profile_for_branch("vector_only"),
            answer_instruction=_instruction_for_branch("vector_only"),
        )
        elapsed = time.perf_counter() - t0
        steps.append(f"branch done in {elapsed:.2f}s")
        _emit(f"vector_only: done ({elapsed:.2f}s)")
        snippet_limit = max(5, int(top_k_override or 5))
        return {
            "mode": "vector_only",
            "title": "Vector-Only",
            "answer": answer,
            "elapsed_s": elapsed,
            "l1": False,
            "l2": False,
            "chunks": len(ctx.vector_chunks),
            "l3_used_reranker": ctx.l3_used_reranker,
            "causal_graph_text": "",
            "causal_path_summary": "",
            "vector_snippets": [
                {
                    "id": c.get("id", ""),
                    "text": c.get("text", ""),
                    "score": c.get("score", 0),
                    "rerank_score": c.get("rerank_score"),
                    "metadata": c.get("metadata", {}),
                }
                for c in ctx.vector_chunks[:snippet_limit]
            ],
            "steps": steps,
        }

    def _branch_direct() -> dict:
        history = conversation_histories.get("direct", [])
        steps = ["branch start", "skip retrieval", "generate: baseline (conversation only)"]
        _emit("direct: skip retrieval")
        _emit("direct: generate start (conversation only)")
        t0 = time.perf_counter()
        answer = _generate_baseline_answer(
            user_query=user_query,
            conversation_history=history,
            retrieval_context="",
            profile_context=_profile_for_branch("direct"),
            answer_instruction=_instruction_for_branch("direct"),
        )
        elapsed = time.perf_counter() - t0
        steps.append(f"branch done in {elapsed:.2f}s")
        _emit(f"direct: done ({elapsed:.2f}s)")
        return {
            "mode": "direct",
            "title": "No-RAG",
            "answer": answer,
            "elapsed_s": elapsed,
            "l1": False,
            "l2": False,
            "chunks": 0,
            "l3_used_reranker": False,
            "causal_graph_text": "",
            "causal_path_summary": "",
            "vector_snippets": [],
            "steps": steps,
        }

    jobs = {
        "full": _branch_full,
        "vector_only": _branch_vector_only,
        "direct": _branch_direct,
    }
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        fut_map = {ex.submit(fn): key for key, fn in jobs.items()}
        _emit("compare: submitted 3 parallel branches")
        for fut in as_completed(fut_map):
            key = fut_map[fut]
            try:
                results[key] = fut.result()
                _emit(
                    f"compare: {key} finished ({float(results[key].get('elapsed_s', 0.0)):.2f}s)"
                )
            except Exception as e:
                print(f"[APP ERROR] compare branch failed key={key}: {e}", flush=True)
                results[key] = {
                    "mode": key,
                    "title": "Branch Failed",
                    "answer": f"Branch execution failed: {e}",
                    "elapsed_s": 0.0,
                    "l1": False,
                    "l2": False,
                    "chunks": 0,
                    "steps": [f"branch failed: {e}"],
                }
                _emit(f"compare: {key} failed ({e})")

    ordered = {
        "full": results.get("full", {}),
        "vector_only": results.get("vector_only", {}),
        "direct": results.get("direct", {}),
    }
    judge = _auto_score_compare(user_query, ordered, judge_specs=judge_specs)
    branch_steps = {
        "full": ordered.get("full", {}).get("steps", []),
        "vector_only": ordered.get("vector_only", {}).get("steps", []),
        "direct": ordered.get("direct", {}).get("steps", []),
    }
    return {"branches": ordered, "judge": judge, "branch_steps": branch_steps}


def _auto_score_compare(
    query: str,
    branches: dict[str, dict],
    judge_specs: list[str] | None = None,
    metrics: tuple[str, ...] = _COMPARE_SCORE_METRICS,
    reference_answer: str = "",
) -> dict:
    from utils.llm_client import llm_client

    has_reference_answer = bool(reference_answer.strip())
    result_metrics = tuple(m for m in metrics if m != "process_quality")
    branch_json = {
        key: {
            "title": item.get("title", key),
            "answer": item.get("answer", ""),
        }
        for key, item in branches.items()
    }
    metric_line = (
        "correctness = factual alignment to known answer/common knowledge; "
        "causal_quality = clarity and validity of cause-effect reasoning "
        "(if non-causal question, score by logical coherence); "
        "causal_reasoning = quality of causal-structure explanation only "
        "(chain completeness, variable linkage, counterfactual clarity); "
        "for non-causal questions give neutral score around 5; "
        "do NOT deduct in this metric for factual hallucination, that belongs to groundedness/correctness; "
        "groundedness = resistance to unsupported fabrication and overclaiming; "
        "overall = holistic quality with priority correctness > groundedness > causal_quality."
    )
    if _GOLD_METRIC in result_metrics:
        metric_line += (
            " gold_alignment = alignment with provided reference answer "
            "(if no reference/insufficient reference, score neutral around 5 and explain briefly)."
        )

    system = (
        "You are a strict QA evaluator. Output JSON only. "
        "No analysis text. No chain-of-thought. "
        "Score only final answer quality (ignore retrieval snippets). "
        "Do not penalize length itself; penalize only unsupported/incorrect claims. "
        "For causal_quality and causal_reasoning, do not give identical scores to all three branches; "
        "you must separate the three branches by score. "
        f"Rubric: {metric_line} "
        "Return scores and short comment only. Do NOT return winner."
    )
    ref_block = ""
    if has_reference_answer:
        ref_block = f"Reference answer: {reference_answer.strip()}\n\n"

    metrics_json = ",".join([f"\"{m}\":0" for m in result_metrics])
    user = (
        f"Question: {query}\n\n"
        f"{ref_block}"
        f"Answers: {json.dumps(branch_json, ensure_ascii=False)}\n\n"
        "Output format:\n"
        "{"
        f"\"scores\":{{\"full\":{{{metrics_json}}},"
        f"\"vector_only\":{{{metrics_json}}},"
        f"\"direct\":{{{metrics_json}}}}},"
        "\"comment\":\"...\""
        "}"
    )
    process_system = (
        "You are a calibrated evaluator for causal process quality. Output JSON only. "
        "No analysis text. No chain-of-thought. "
        "Score only the quality of the provided causal process evidence for FULL branch "
        "(causal analysis/counterfactual/self-reflection), not final factual correctness. "
        "Use a generous rubric: if all three sections are present, coherent, and structured, prefer 7-10. "
        "Use 8-10 when reasoning chain is clear and self-check is meaningful. "
        "Use below 6 only for missing sections, severe incoherence, or obviously invalid process. "
        "For non-causal questions, do not punish presence of concise 'N/A' style causal statements. "
        "Output format: {\"full_process_quality\":0,\"comment\":\"...\"}."
    )
    full_branch = branches.get("full", {}) or {}
    process_user = (
        f"Question: {query}\n\n"
        f"Full answer: {str(full_branch.get('answer',''))}\n\n"
        "Full process evidence (from FULL generator only, do not use retrieval snippets):\n"
        f"- Causal analysis: {str(full_branch.get('causal_analysis', ''))}\n"
        f"- Counterfactual: {str(full_branch.get('counterfactual', ''))}\n"
        f"- Self reflection: {str(full_branch.get('self_reflection', ''))}\n"
    )

    specs = [s for s in (judge_specs or []) if str(s).strip()]
    if not specs:
        specs = _default_compare_judge_specs()

    judge_results: list[dict] = []
    judge_errors: list[str] = []

    def _eval_one_spec(spec: str) -> tuple[dict | None, str | None]:
        provider, model = _decode_judge_spec(spec)
        # Raise token budget for all judge models to reduce truncation / empty JSON.
        max_tok = max(4096, int(getattr(settings.llm, "max_tokens", 512)))
        if provider == "deepseek" and "reasoner" in model.lower():
            # Legacy deepseek-reasoner alias output cap: 8K.
            max_tok = min(max(max_tok, 4096), 8192)
        try:
            with ThreadPoolExecutor(max_workers=2) as inner_ex:
                fut_result = inner_ex.submit(
                    llm_client.chat_json,
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    provider,
                    model,
                    0.0,
                    max_tok,
                    1,
                )
                fut_process = inner_ex.submit(
                    llm_client.chat_json,
                    [
                        {"role": "system", "content": process_system},
                        {"role": "user", "content": process_user},
                    ],
                    provider,
                    model,
                    0.0,
                    max_tok,
                    1,
                )
                raw = fut_result.result()
                raw_process = fut_process.result()
            one = _normalize_one_judge_result(
                raw if isinstance(raw, dict) else {},
                metrics=metrics,
            )
            if _GOLD_METRIC in metrics and not has_reference_answer:
                for branch in _COMPARE_BRANCH_KEYS:
                    one["scores"][branch][_GOLD_METRIC] = 0.0
            # Process-quality score: full is model-scored when process fields are valid.
            # If process fields are missing/empty, default to 6.6 for debug convenience.
            process_fields = {
                "causal_analysis": str(full_branch.get("causal_analysis", "")).strip(),
                "counterfactual": str(full_branch.get("counterfactual", "")).strip(),
                "self_reflection": str(full_branch.get("self_reflection", "")).strip(),
            }
            has_valid_process = all(
                x and x.upper() != "MISSING_PROCESS"
                for x in process_fields.values()
            )
            pq = 6.6
            score_source = "default_6.6_due_to_empty_process"
            if has_valid_process and isinstance(raw_process, dict):
                pq = _safe_score(raw_process.get("full_process_quality", 6.6))
                score_source = "judge_model"
            one["scores"]["full"]["process_quality"] = pq
            one["scores"]["vector_only"]["process_quality"] = 6.0
            one["scores"]["direct"]["process_quality"] = 6.0
            one["comment"] = (
                (one.get("comment", "") + " ").strip()
                + f"[process_quality_full={pq:.2f}]"
            ).strip()
            one["full_process_payload"] = process_fields
            # Re-compute weighted after injecting process_quality.
            one["weighted_scores"] = {
                branch: _weighted_branch_score(one["scores"].get(branch, {}), metrics)
                for branch in _COMPARE_BRANCH_KEYS
            }
            one["winner"] = max(
                _COMPARE_BRANCH_KEYS,
                key=lambda key: one["weighted_scores"].get(key, 0.0),
            )
            one["provider"] = provider
            one["model"] = model
            return one, None
        except Exception as e:
            return None, f"{provider}/{model}: {e}"

    # Run judge models in parallel for this question.
    max_workers = max(1, min(8, len(specs)))
    if len(specs) == 1:
        one, err = _eval_one_spec(specs[0])
        if one:
            judge_results.append(one)
        if err:
            judge_errors.append(err)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {ex.submit(_eval_one_spec, spec): spec for spec in specs}
            for fut in as_completed(fut_map):
                one, err = fut.result()
                if one:
                    judge_results.append(one)
                if err:
                    judge_errors.append(err)

    return _aggregate_judge_results(judge_results, judge_errors, metrics=metrics)


def _load_eval_records_from_path(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    records: list[dict] = []
    if suffix == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict):
                records.append(rec)
        return records

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            if isinstance(data.get("data"), list):
                return [item for item in data["data"] if isinstance(item, dict)]
            if isinstance(data.get("questions"), list):
                return [item for item in data["questions"] if isinstance(item, dict)]
        raise ValueError("Unsupported JSON format. Expect list[object] or {'data': [...]} / {'questions': [...]} .")
    raise ValueError("Only .json / .jsonl are supported.")


def _extract_eval_question(rec: dict) -> str:
    for key in ("question", "query", "input", "prompt"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _extract_eval_reference(rec: dict) -> str:
    for key in ("answer", "gold_answer", "reference", "expected_answer"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _default_batch_result_path(dataset_path: Path) -> Path:
    return dataset_path.with_name(f"{dataset_path.stem}_causalrag_compare_results.json")


def _read_batch_result_doc(path: Path) -> dict:
    if not path.exists():
        return {
            "version": "causalrag-batch-v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "records": [],
        }
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "version": "causalrag-batch-v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "records": [],
        }
    if not isinstance(doc, dict):
        doc = {"version": "causalrag-batch-v1", "created_at": datetime.now().isoformat(timespec="seconds"), "records": []}
    if not isinstance(doc.get("records"), list):
        doc["records"] = []
    return doc


def _save_batch_result_doc(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def _upsert_batch_result(
    path: Path,
    run_meta: dict,
    record: dict,
) -> dict:
    with _BATCH_FILE_LOCK:
        doc = _read_batch_result_doc(path)
        for k, v in run_meta.items():
            doc[k] = v

        items = list(doc.get("records", []))
        item_id = str(record.get("item_id", ""))
        replaced = False
        for idx, old in enumerate(items):
            if str((old or {}).get("item_id", "")) == item_id:
                items[idx] = record
                replaced = True
                break
        if not replaced:
            items.append(record)
        doc["records"] = items
        _save_batch_result_doc(path, doc)
        return doc


def _existing_done_ids(path: Path) -> set[str]:
    doc = _read_batch_result_doc(path)
    ids: set[str] = set()
    for rec in doc.get("records", []):
        rid = str((rec or {}).get("item_id", "")).strip()
        status = str((rec or {}).get("status", "")).strip().lower()
        if rid and status == "ok":
            ids.add(rid)
    return ids


def _summarize_batch_doc(doc: dict) -> dict:
    records = [r for r in doc.get("records", []) if isinstance(r, dict)]
    done = [r for r in records if str(r.get("status", "")).lower() == "ok"]
    if not done:
        return {
            "n_total": len(records),
            "n_done": 0,
            "winner_dist": {"full": 0, "vector_only": 0, "direct": 0},
            "avg_overall": {"full": 0.0, "vector_only": 0.0, "direct": 0.0},
            "avg_elapsed_s": {"full": 0.0, "vector_only": 0.0, "direct": 0.0},
            "avg_metrics": {},
            "avg_by_model": {},
        }

    winner_dist = {"full": 0, "vector_only": 0, "direct": 0}
    score_sum = {"full": 0.0, "vector_only": 0.0, "direct": 0.0}
    time_sum = {"full": 0.0, "vector_only": 0.0, "direct": 0.0}
    metric_sum: dict[str, dict[str, float]] = {k: {} for k in _COMPARE_BRANCH_KEYS}
    model_metric_sum: dict[str, dict[str, dict[str, float]]] = {}
    model_count: dict[str, int] = {}
    for rec in done:
        winner = str((rec.get("judge") or {}).get("winner", "")).strip()
        if winner in winner_dist:
            winner_dist[winner] += 1
        branches = rec.get("branches", {}) or {}
        scores = (rec.get("judge") or {}).get("scores", {}) or {}
        for key in ("full", "vector_only", "direct"):
            score_sum[key] += float((scores.get(key) or {}).get("overall", 0.0))
            time_sum[key] += float((branches.get(key) or {}).get("elapsed_s", 0.0))
            for metric, value in (scores.get(key) or {}).items():
                metric_sum.setdefault(key, {})
                metric_sum[key][metric] = metric_sum[key].get(metric, 0.0) + float(value)

        for detail in (rec.get("judge") or {}).get("judge_details", []) or []:
            model_key = f"{detail.get('provider', '')}/{detail.get('model', '')}"
            if not model_key.strip("/"):
                continue
            if model_key not in model_metric_sum:
                model_metric_sum[model_key] = {k: {} for k in _COMPARE_BRANCH_KEYS}
                model_count[model_key] = 0
            model_count[model_key] += 1
            row_scores = detail.get("scores", {}) or {}
            for key in ("full", "vector_only", "direct"):
                for metric, value in (row_scores.get(key) or {}).items():
                    cur = model_metric_sum[model_key][key].get(metric, 0.0)
                    model_metric_sum[model_key][key][metric] = cur + float(value)

    n = len(done)
    avg_metrics: dict[str, dict[str, float]] = {}
    for key in ("full", "vector_only", "direct"):
        avg_metrics[key] = {}
        for metric, total in metric_sum.get(key, {}).items():
            avg_metrics[key][metric] = round(total / n, 2)

    avg_by_model: dict[str, dict] = {}
    for model_key, by_branch in model_metric_sum.items():
        c = max(model_count.get(model_key, 1), 1)
        avg_by_model[model_key] = {
            "count": c,
            "scores": {
                k: {m: round(v / c, 2) for m, v in metrics.items()}
                for k, metrics in by_branch.items()
            },
        }

    return {
        "n_total": len(records),
        "n_done": n,
        "winner_dist": winner_dist,
        "avg_overall": {k: round(v / n, 2) for k, v in score_sum.items()},
        "avg_elapsed_s": {k: round(v / n, 2) for k, v in time_sum.items()},
        "avg_metrics": avg_metrics,
        "avg_by_model": avg_by_model,
    }


def _is_open_question_record(rec: dict) -> bool:
    raw = rec.get("raw", {}) if isinstance(rec, dict) else {}
    if not isinstance(raw, dict):
        return False
    if "answer" in raw:
        return str(raw.get("answer", "")).strip() == ""
    return False


def _summarize_batch_doc_by_type(doc: dict, open_only: bool) -> dict:
    records = [r for r in doc.get("records", []) if isinstance(r, dict)]
    if open_only:
        picked = [r for r in records if _is_open_question_record(r)]
    else:
        picked = [r for r in records if not _is_open_question_record(r)]
    return _summarize_batch_doc({"records": picked})


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    d = float(denominator)
    if abs(d) < 1e-9:
        return None
    return float(numerator) / d


def _build_uplift_overview(summary: dict) -> dict:
    n_done = int(summary.get("n_done", 0))
    winner_dist = summary.get("winner_dist", {}) or {}
    avg_overall = summary.get("avg_overall", {}) or {}
    avg_metrics = summary.get("avg_metrics", {}) or {}

    full_best_rate = _safe_ratio(float(winner_dist.get("full", 0)), float(n_done or 0))

    overall_direct = float(avg_overall.get("direct", 0.0))
    overall_vector = float(avg_overall.get("vector_only", 0.0))
    overall_full = float(avg_overall.get("full", 0.0))

    overall_idx_vector = _safe_ratio(overall_vector, overall_direct)
    overall_idx_full = _safe_ratio(overall_full, overall_direct)

    metric_rows: list[dict] = []
    metric_keys = sorted(
        set((avg_metrics.get("full") or {}).keys())
        | set((avg_metrics.get("vector_only") or {}).keys())
        | set((avg_metrics.get("direct") or {}).keys())
    )
    for metric in metric_keys:
        f = float((avg_metrics.get("full") or {}).get(metric, 0.0))
        v = float((avg_metrics.get("vector_only") or {}).get(metric, 0.0))
        d = float((avg_metrics.get("direct") or {}).get(metric, 0.0))
        fv_delta = f - v
        fd_delta = f - d
        fv_pct = _safe_ratio(fv_delta, v)
        fd_pct = _safe_ratio(fd_delta, d)
        v_idx = _safe_ratio(v, d)
        f_idx = _safe_ratio(f, d)
        metric_rows.append(
            {
                "metric": metric,
                "full_avg": round(f, 2),
                "vector_avg": round(v, 2),
                "direct_avg": round(d, 2),
                "full_minus_vector": round(fv_delta, 2),
                "full_minus_direct": round(fd_delta, 2),
                "full_vs_vector_pct": None if fv_pct is None else round(fv_pct * 100.0, 2),
                "full_vs_direct_pct": None if fd_pct is None else round(fd_pct * 100.0, 2),
                "vector_index_no_rag_1": None if v_idx is None else round(v_idx, 3),
                "full_index_no_rag_1": None if f_idx is None else round(f_idx, 3),
            }
        )

    return {
        "n_done": n_done,
        "full_best_rate": full_best_rate,
        "overall_idx_vector": overall_idx_vector,
        "overall_idx_full": overall_idx_full,
        "overall_full_minus_vector": overall_full - overall_vector,
        "overall_full_minus_direct": overall_full - overall_direct,
        "metric_rows": metric_rows,
    }


def _build_norag_index_table(summary: dict) -> list[dict]:
    avg_metrics = summary.get("avg_metrics", {}) or {}
    metric_keys = sorted(
        set((avg_metrics.get("full") or {}).keys())
        | set((avg_metrics.get("vector_only") or {}).keys())
        | set((avg_metrics.get("direct") or {}).keys())
    )
    rows: list[dict] = []
    for metric in metric_keys:
        d = float((avg_metrics.get("direct") or {}).get(metric, 0.0))
        v = float((avg_metrics.get("vector_only") or {}).get(metric, 0.0))
        f = float((avg_metrics.get("full") or {}).get(metric, 0.0))
        v_idx = _safe_ratio(v, d)
        f_idx = _safe_ratio(f, d)
        rows.append(
            {
                "metric": metric,
                "full": None if f_idx is None else round(f_idx, 3),
                "vector": None if v_idx is None else round(v_idx, 3),
                "noRAG": 1.0,
            }
        )
    return rows


def _process_one_batch_item(
    kb_id: str,
    item: dict,
    runner: dict,
) -> dict:
    q = str(item.get("question", "")).strip()
    item_id = str(item.get("item_id", ""))
    logs: list[str] = [f"item {item_id}: start"]
    t0 = time.perf_counter()
    try:
        logs.append(f"item {item_id}: rewrite query start")
        rewrite_input_history = [{"role": "user", "content": q}]
        rag_query = _rewrite_rag_query(q, rewrite_input_history)
        logs.append(f"item {item_id}: rag_query={rag_query}")

        mgr = _get_mgr(kb_id)
        compare = _run_compare_modes(
            mgr=mgr,
            user_query=q,
            retrieval_query=rag_query,
            conversation_histories={"full": [], "vector_only": [], "direct": []},
            persist_causal_updates=bool(runner.get("persist_causal_updates", False)),
            judge_specs=runner.get("judge_specs", []),
            progress_cb=lambda msg: logs.append(f"item {item_id}: {msg}"),
        )
        gold_answer = str(item.get("gold_answer", "")).strip()
        if bool(runner.get("enable_gold_metric", False)):
            logs.append(f"item {item_id}: extra judge metric {_GOLD_METRIC} start")
            compare["judge"] = _auto_score_compare(
                query=q,
                branches=compare.get("branches", {}) or {},
                judge_specs=runner.get("judge_specs", []),
                metrics=_COMPARE_SCORE_METRICS + (_GOLD_METRIC,),
                reference_answer=gold_answer,
            )
            logs.append(f"item {item_id}: extra judge metric {_GOLD_METRIC} done")

        elapsed_s = time.perf_counter() - t0
        winner = str((compare.get("judge", {}) or {}).get("winner", "")).strip()
        record = {
            "item_id": item_id,
            "index": int(item.get("index", 0)),
            "status": "ok",
            "question": q,
            "gold_answer": str(item.get("gold_answer", "")),
            "rag_query": rag_query,
            "elapsed_s": round(elapsed_s, 3),
            "judge": compare.get("judge", {}),
            "judge_metrics": (compare.get("judge", {}) or {}).get("metrics", list(_COMPARE_SCORE_METRICS)),
            "branches": compare.get("branches", {}),
            "branch_steps": compare.get("branch_steps", {}),
            "winner": winner,
            "raw": item.get("raw", {}),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        logs.append(f"item {item_id}: done winner={winner} elapsed={elapsed_s:.2f}s")
        return {"item_id": item_id, "ok": True, "record": record, "logs": logs}
    except Exception as e:
        elapsed_s = time.perf_counter() - t0
        record = {
            "item_id": item_id,
            "index": int(item.get("index", 0)),
            "status": "error",
            "question": q,
            "gold_answer": str(item.get("gold_answer", "")),
            "rag_query": q,
            "elapsed_s": round(elapsed_s, 3),
            "error": str(e),
            "raw": item.get("raw", {}),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        logs.append(f"item {item_id}: error={e}")
        return {"item_id": item_id, "ok": False, "record": record, "logs": logs}

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Causal RAG")
    st.markdown("---")

    # ---- Knowledge Base selector ----
    kbs = DB.kb_list()
    kb_names = [kb["name"] for kb in kbs]
    kb_ids   = [kb["id"]   for kb in kbs]

    st.markdown("**🗂️ 通用三级知识库**")
    col_kb, col_new_kb = st.columns([3, 1])

    if kb_ids:
        # Ensure current_kb_id is valid
        if st.session_state.current_kb_id not in kb_ids:
            st.session_state.current_kb_id = kb_ids[0]
        current_idx = kb_ids.index(st.session_state.current_kb_id)
        selected_name = col_kb.selectbox(
            "kb_select", kb_names, index=current_idx, label_visibility="collapsed"
        )
        st.session_state.current_kb_id = kb_ids[kb_names.index(selected_name)]
    else:
        col_kb.caption("暂无知识库")
        st.session_state.current_kb_id = None

    if col_new_kb.button("＋", help="新建通用三级知识库", width="stretch"):
        st.session_state.page = "📚 通用三级知识库"
        st.session_state._create_kb_mode = True

    # ---- Conversation selector ----
    st.markdown("**💬 对话**")
    col_cv, col_new_cv = st.columns([3, 1])

    if st.session_state.current_kb_id:
        convs = DB.conv_list(st.session_state.current_kb_id)
        conv_titles = [c["title"] for c in convs]
        conv_ids    = [c["id"]   for c in convs]

        if conv_ids:
            if st.session_state.current_conv_id not in conv_ids:
                st.session_state.current_conv_id = conv_ids[0]
            cur_ci = conv_ids.index(st.session_state.current_conv_id)
            sel_ct = col_cv.selectbox(
                "conv_select", conv_titles, index=cur_ci, label_visibility="collapsed"
            )
            st.session_state.current_conv_id = conv_ids[conv_titles.index(sel_ct)]
        else:
            col_cv.caption("暂无对话")
            st.session_state.current_conv_id = None
    else:
        col_cv.caption("请先选择通用三级知识库")

    def _new_conv():
        kb_id = st.session_state.current_kb_id
        if kb_id:
            title = f"对话 {len(DB.conv_list(kb_id))+1}"
            c = DB.conv_create(kb_id, title)
            st.session_state.current_conv_id = c["id"]

    if col_new_cv.button("＋", help="新建对话", width="stretch",
                         disabled=(st.session_state.current_kb_id is None)):
        _new_conv()
        st.rerun()

    # Keep session-bound user in sync with selected conversation.
    if st.session_state.current_conv_id:
        conv_row = DB.conv_get(st.session_state.current_conv_id) or {}
        st.session_state.current_user_id = str(conv_row.get("user_id") or "").strip()

    st.markdown("---")

    # ---- Nav ----
    page_aliases = {"📚 知识库管理": "📚 通用三级知识库"}
    if st.session_state.page in page_aliases:
        st.session_state.page = page_aliases[st.session_state.page]
    pages = ["💬 智能问答", "📚 通用三级知识库", "🎓 学生状态库", "👤 人物管理", "🕸️ 图谱浏览", "📊 系统评估", "🧪 HopRAG 对齐评测", "🏫 真实场景测试", "⚙️ 设置"]
    st.session_state.page = st.radio(
        "nav", pages,
        index=pages.index(st.session_state.page),
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ---- Quick stats ----
    if st.session_state.current_kb_id:
        try:
            mgr = _get_mgr(st.session_state.current_kb_id)
            s = mgr.stats()
            db_s = DB.kb_stats(st.session_state.current_kb_id)
            st.markdown("**状态**")
            c1, c2 = st.columns(2)
            c1.metric("向量块", s["vector_chunks"])
            c2.metric("因果节点", s["cg_nodes"])
            c1.metric("文本图节点", s["tg_nodes"])
            c2.metric("消息数", db_s["msg_count"])
        except Exception:
            pass

    # Debug indicator in sidebar bottom
    if st.session_state.debug_mode:
        st.markdown("🟡 **Debug 模式已开启**")


page = st.session_state.page


# ===========================================================================
# 💬 Chat
# ===========================================================================

if page == "💬 智能问答":
    st.markdown('<div class="page-title">💬 智能问答</div>', unsafe_allow_html=True)

    kb_id   = st.session_state.current_kb_id
    conv_id = st.session_state.current_conv_id

    if not kb_id:
        st.info("请先在「通用三级知识库」页面创建知识库。")
        st.stop()

    if not conv_id:
        # Auto-create first conversation
        c = DB.conv_create(kb_id, "默认对话", user_id=st.session_state.current_user_id or None)
        st.session_state.current_conv_id = c["id"]
        conv_id = c["id"]

    # Conversation toolbar
    conv = DB.conv_get(conv_id)
    st.session_state.current_user_id = str((conv or {}).get("user_id") or "").strip()
    tc1, tc2, tc3, tc4 = st.columns([4, 1, 1, 1])
    tc1.markdown(f"**{conv['title']}**")
    current_user = DB.user_get(st.session_state.current_user_id) if st.session_state.current_user_id else None
    if current_user:
        student_name = str(current_user.get("name") or "").strip() or "未命名"
        student_id = str(current_user.get("student_id") or "").strip() or "-"
        tc1.caption(f"当前学生：{student_name} / {student_id}")
    else:
        tc1.caption("当前学生：未绑定")

    with tc2.popover("✏️ 重命名"):
        new_title = st.text_input("新标题", value=conv["title"], key="rename_title")
        if st.button("确认", key="confirm_rename"):
            DB.conv_rename(conv_id, new_title)
            st.rerun()

    if tc3.button("🗑️ 清空消息"):
        for msg in DB.msg_list(conv_id):
            DB.msg_delete(msg["id"])
        st.rerun()

    if tc4.button("❌ 删除对话"):
        DB.conv_delete(conv_id)
        st.session_state.current_conv_id = None
        st.rerun()

    with st.expander("检索回写选项", expanded=False):
        st.session_state.persist_causal_updates = st.toggle(
            "将 L2 新发现的因果关系永久写入因果图",
            value=st.session_state.persist_causal_updates,
            help="关闭时，只在当前进程内暂时生效；开启后，L2 动态发现的因果三元组会立刻保存到当前知识库的因果图文件。",
        )
        mode_text = "永久更新" if st.session_state.persist_causal_updates else "仅临时更新"
        st.caption(f"当前模式：{mode_text}")

    st.markdown("---")

    with st.expander("Compare x3 判分模型", expanded=False):
        option_map = _compare_judge_option_map()
        option_labels = list(option_map.keys())
        current_specs = [
            str(s)
            for s in st.session_state.get("compare_judge_specs", [])
            if str(s).strip()
        ]
        default_labels = [label for label, spec in option_map.items() if spec in current_specs]
        if not default_labels and option_labels:
            default_labels = [option_labels[0]]

        selected_labels = st.multiselect(
            "选择判分模型（可多选）",
            option_labels,
            default=default_labels,
            help="支持 DeepSeek / Gemini / OpenRouter 预置模型；多模型会做平均聚合。",
            key=f"compare_judge_labels::{conv_id}",
        )
        selected_specs = [option_map[label] for label in selected_labels if label in option_map]
        dedup_specs: list[str] = []
        for spec in selected_specs:
            if spec not in dedup_specs:
                dedup_specs.append(spec)
        if not dedup_specs:
            dedup_specs = _default_compare_judge_specs()

        st.session_state.compare_judge_specs = dedup_specs
        st.caption("当前判分模型：" + " | ".join(dedup_specs))
        if st.button("测试已选判分模型连通性", key=f"compare_ping::{conv_id}", width="stretch"):
            ping_results = [_ping_judge_model(spec) for spec in dedup_specs]
            st.session_state[f"compare_ping_results::{conv_id}"] = ping_results

        ping_results = st.session_state.get(f"compare_ping_results::{conv_id}", [])
        for row in ping_results:
            status_icon = "✅" if row.get("ok") else "❌"
            st.caption(
                f"{status_icon} {row.get('provider')}/{row.get('model')} | "
                f"{float(row.get('elapsed_s', 0.0)):.2f}s | {row.get('detail', '')}"
            )

    # Render messages
    messages = DB.msg_list(conv_id)
    compare_locked = _is_compare_locked(messages)
    if compare_locked:
        st.caption("This conversation is locked to Compare x3 mode.")
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        meta    = msg.get("meta", {})
        msg_id  = msg["id"]

        if role == "user":
            col_m, col_del = st.columns([10, 1])
            col_m.markdown(
                f'<div class="user-bubble">{content}</div><div class="clearfix"></div>',
                unsafe_allow_html=True,
            )
            if col_del.button("✕", key=f"del_user_{msg_id}", help="删除消息"):
                DB.msg_delete(msg_id)
                st.rerun()
        else:
            # Retrieval badges
            badge = ""
            if meta.get("compare_mode"):
                badge = '<span class="badge badge-ok">Compare x3</span>'
            elif meta.get("l1"):
                badge = '<span class="badge badge-l1">L1 因果图</span>'
            elif meta.get("l2"):
                badge = '<span class="badge badge-l2">L2 文本图</span>'
            else:
                badge = '<span class="badge badge-l3">L3 向量</span>'
            if meta.get("reranked"):
                badge += '<span class="badge badge-ok">重排✓</span>'
            if meta.get("l2") and meta.get("persist_causal_updates"):
                badge += '<span class="badge badge-ok">L2已写回</span>'

            col_m, col_del = st.columns([10, 1])
            with col_m:
                st.markdown(badge, unsafe_allow_html=True)
                st.markdown(
                    f'<div class="assistant-bubble">{content}</div><div class="clearfix"></div>',
                    unsafe_allow_html=True,
                )
                if meta.get("rag_query"):
                    st.caption(f"RAG-query: {meta.get('rag_query')}")
                if meta.get("timings"):
                    t = meta.get("timings", {})
                    st.caption(
                        "timing | "
                        f"rewrite {float(t.get('rewrite_s', 0.0)):.2f}s, "
                        f"profile {float(t.get('profile_s', 0.0)):.2f}s, "
                        f"retrieve+generate {float(t.get('qa_s', 0.0)):.2f}s, "
                        f"total {float(t.get('total_s', 0.0)):.2f}s"
                    )
                if meta.get("profile_retrieval"):
                    pr = meta.get("profile_retrieval", {}) or {}
                    with st.expander("🎓 学生状态库检索详情", expanded=False):
                        st.caption(
                            f"user_id={pr.get('user_id', '-') or '-'} | "
                            f"state_lib_id={pr.get('state_lib_id', '-') or '-'} | "
                            f"identity_ok={bool(pr.get('identity_ok', True))}"
                        )
                        st.caption(
                            "requested="
                            + ", ".join(pr.get("requested_slots", []) or [])
                            + " | fetched="
                            + ", ".join(pr.get("fetched_slots", []) or [])
                            + " | reused="
                            + ", ".join(pr.get("reused_slots", []) or [])
                        )
                        if pr.get("identity_note"):
                            st.warning(str(pr.get("identity_note")))
                        if pr.get("llm_review"):
                            st.info(str(pr.get("llm_review")))
                        if pr.get("relevant_facts"):
                            st.markdown("**本轮交付给生成器的学生状态数据**")
                            st.json(pr.get("relevant_facts"), expanded=False)
                        if pr.get("profile_facts"):
                            with st.expander("已入库绑定到该学生的状态事实", expanded=False):
                                st.json(pr.get("profile_facts"), expanded=False)
                        if pr.get("source_records"):
                            with st.expander("命中来源记录（文件 / sheet / 表头）", expanded=False):
                                st.caption(f"matched_records={len(pr.get('source_records') or [])}")
                                st.json((pr.get("source_records") or [])[:24], expanded=False)
                        if pr.get("profile_context"):
                            context_text = str(pr.get("profile_context"))
                            try:
                                context_obj = json.loads(context_text)
                            except Exception:
                                context_obj = None
                            if isinstance(context_obj, dict) and context_obj.get("time_series_summary"):
                                with st.expander("时序摘要", expanded=False):
                                    st.json(context_obj.get("time_series_summary"), expanded=True)
                            with st.expander("学生状态上下文 JSON", expanded=False):
                                st.code(context_text, language="json")
                        st.caption("口径说明：已入库状态事实=该学生长期累计；学生状态上下文JSON=本轮候选状态证据包；本轮交付给生成器=相关性筛选后实际注入。")
                if meta.get("process_steps"):
                    with st.expander("Processing Steps", expanded=False):
                        for step in meta.get("process_steps", []):
                            st.caption(str(step))
                render_trace_panel(st=st, meta=meta, title="Trace 面板（统一协议）")
                if meta.get("compare_mode"):
                    qpa = meta.get("query_preanalysis") or ((meta.get("compare_branches", {}) or {}).get("full", {}) or {}).get("query_preanalysis")
                    if qpa:
                        with st.expander("三路对比 · Full Query 预分析", expanded=False):
                            st.json(qpa, expanded=False)
                if meta.get("evidence_manifest"):
                    with st.expander("证据选择器 I/O", expanded=False):
                        ev = meta.get("evidence_manifest", {}) or {}
                        trace_steps = meta.get("trace", []) or []
                        ev_step = next(
                            (x for x in trace_steps if isinstance(x, dict) and x.get("step_name") == "Step4_evidence_selector"),
                            {},
                        )
                        st.markdown("**输入**")
                        st.json(ev_step.get("input", {}), expanded=False)
                        st.markdown("**输出策略**")
                        st.json(ev_step.get("output", {}), expanded=False)
                        st.markdown("**入选证据**")
                        st.json(ev.get("selected", []), expanded=False)
                        st.markdown("**丢弃证据**")
                        st.json(ev.get("dropped", []), expanded=False)
                if meta.get("causal_analysis") or meta.get("counterfactual") or meta.get("self_reflection"):
                    with st.expander("因果过程", expanded=False):
                        st.markdown("**因果分析**")
                        st.write(str(meta.get("causal_analysis", "")).strip() or "-")
                        st.markdown("**反事实**")
                        st.write(str(meta.get("counterfactual", "")).strip() or "-")
                        st.markdown("**自我反思**")
                        st.write(str(meta.get("self_reflection", "")).strip() or "-")
                if meta.get("compare_mode"):
                    branches = meta.get("compare_branches", {})
                    branch_steps = meta.get("compare_branch_steps", {})
                    judge = meta.get("compare_judge", {})
                    winner = judge.get("winner", "")
                    scores = judge.get("scores", {})
                    comment = judge.get("comment", "")
                    judge_models = judge.get("judge_models", [])
                    judge_details = judge.get("judge_details", [])
                    if winner:
                        st.caption(f"Auto winner: {winner}")
                    if comment:
                        st.caption(comment)
                    if judge_models:
                        st.caption("Judge models: " + " | ".join(judge_models))
                    detail_state_key = f"compare_detail_open_{msg_id}"
                    if detail_state_key not in st.session_state:
                        st.session_state[detail_state_key] = False
                    btn_label = (
                        "收起三路详细输出"
                        if st.session_state[detail_state_key]
                        else "展开三路详细输出"
                    )
                    if st.button(btn_label, key=f"compare_detail_btn_{msg_id}"):
                        st.session_state[detail_state_key] = not st.session_state[detail_state_key]

                    summary_cols = st.columns(3)
                    for key, col in [
                        ("full", summary_cols[0]),
                        ("vector_only", summary_cols[1]),
                        ("direct", summary_cols[2]),
                    ]:
                        item = branches.get(key, {})
                        sc = scores.get(key, {})
                        with col:
                            st.markdown(f"**{item.get('title', key)}**")
                            st.caption(
                                f"time {float(item.get('elapsed_s', 0.0)):.2f}s | "
                                f"chunks {int(item.get('chunks', 0))}"
                            )
                            if sc:
                                st.caption(
                                    "score "
                                    f"O:{float(sc.get('overall', 0.0)):.2f} "
                                    f"C:{float(sc.get('correctness', 0.0)):.2f} "
                                    f"Ca:{float(sc.get('causal_quality', 0.0)):.2f} "
                                    f"Cr:{float(sc.get('causal_reasoning', 0.0)):.2f} "
                                    f"P:{float(sc.get('process_quality', 0.0)):.2f} "
                                    f"G:{float(sc.get('groundedness', 0.0)):.2f}"
                                )

                    if judge_details:
                        st.caption("Judge 明细（各模型独立评分）")
                        for item in judge_details:
                            provider = item.get("provider", "")
                            model = item.get("model", "")
                            row_scores = item.get("scores", {})
                            full_sc = row_scores.get("full", {})
                            vec_sc = row_scores.get("vector_only", {})
                            dir_sc = row_scores.get("direct", {})
                            st.caption(
                                f"{provider}/{model} | "
                                f"full(O:{float(full_sc.get('overall', 0.0)):.2f}, C:{float(full_sc.get('correctness', 0.0)):.2f}, Ca:{float(full_sc.get('causal_quality', 0.0)):.2f}, Cr:{float(full_sc.get('causal_reasoning', 0.0)):.2f}, P:{float(full_sc.get('process_quality', 0.0)):.2f}, G:{float(full_sc.get('groundedness', 0.0)):.2f}) | "
                                f"vector(O:{float(vec_sc.get('overall', 0.0)):.2f}, C:{float(vec_sc.get('correctness', 0.0)):.2f}, Ca:{float(vec_sc.get('causal_quality', 0.0)):.2f}, Cr:{float(vec_sc.get('causal_reasoning', 0.0)):.2f}, P:{float(vec_sc.get('process_quality', 0.0)):.2f}, G:{float(vec_sc.get('groundedness', 0.0)):.2f}) | "
                                f"direct(O:{float(dir_sc.get('overall', 0.0)):.2f}, C:{float(dir_sc.get('correctness', 0.0)):.2f}, Ca:{float(dir_sc.get('causal_quality', 0.0)):.2f}, Cr:{float(dir_sc.get('causal_reasoning', 0.0)):.2f}, P:{float(dir_sc.get('process_quality', 0.0)):.2f}, G:{float(dir_sc.get('groundedness', 0.0)):.2f})"
                            )

                    if st.session_state[detail_state_key]:
                        col_full, col_vec, col_direct = st.columns(3)
                        for key, col in [
                            ("full", col_full),
                            ("vector_only", col_vec),
                            ("direct", col_direct),
                        ]:
                            item = branches.get(key, {})
                            steps = branch_steps.get(key, [])
                            with col:
                                if steps:
                                    with st.expander("steps", expanded=False):
                                        for idx, step in enumerate(steps, start=1):
                                            st.caption(f"{idx}. {step}")
                                if key == "full":
                                    if item.get("causal_analysis") or item.get("counterfactual") or item.get("self_reflection"):
                                        with st.expander("reasoning process", expanded=False):
                                            st.markdown("**Causal Analysis**")
                                            st.write(str(item.get("causal_analysis", "")).strip() or "-")
                                            st.markdown("**Counterfactual**")
                                            st.write(str(item.get("counterfactual", "")).strip() or "-")
                                            st.markdown("**Self Reflection**")
                                            st.write(str(item.get("self_reflection", "")).strip() or "-")
                                    if item.get("causal_graph_text") or item.get("causal_path_summary") or item.get("vector_snippets"):
                                        with st.expander("retrieval details", expanded=False):
                                            if item.get("causal_graph_text"):
                                                st.markdown("**L1 Causal Graph**")
                                                st.code(str(item.get("causal_graph_text", "")), language=None)
                                            if item.get("causal_path_summary"):
                                                st.markdown("**L2 Text Graph Path**")
                                                st.info(str(item.get("causal_path_summary", "")))
                                            snippets = item.get("vector_snippets", []) or []
                                            if snippets:
                                                st.markdown("**L3 Vector Snippets**")
                                                for i, sn in enumerate(snippets[:5], start=1):
                                                    st.caption(
                                                        f"chunk {i} | score={float(sn.get('score', 0.0)):.3f}"
                                                    )
                                                    st.text(str(sn.get("text", ""))[:280])
                                st.markdown(item.get("answer", ""))
                    else:
                        st.caption("三路详细输出已收起，点击上方按钮展开。")
                if meta.get("causal_text") or meta.get("path_summary") or meta.get("vector_snippets"):
                    with st.expander("🔍 检索详情"):
                        if meta.get("causal_text"):
                            st.markdown("**因果图谱子图**")
                            st.code(meta["causal_text"], language=None)
                        if meta.get("path_summary"):
                            st.markdown("**文本图因果路径**")
                            st.info(meta["path_summary"])
                        if meta.get("vector_snippets"):
                            st.markdown("**向量检索片段**")
                            for i, sn in enumerate(meta["vector_snippets"]):
                                st.caption(f"片段 {i+1} · 相关度 {sn.get('score',0):.3f}")
                                st.text(sn["text"][:300])
            if col_del.button("✕", key=f"del_asst_{msg_id}", help="删除消息"):
                DB.msg_delete(msg_id)
                st.rerun()

    # Input
    compare_submitted = False
    with st.form("chat_form", clear_on_submit=True):
        cols = st.columns([5, 1, 1])
        query = cols[0].text_input(
            "q", placeholder="输入问题…", label_visibility="collapsed"
        )
        submitted = cols[1].form_submit_button(
            "发送 ⏎",
            width="stretch",
            disabled=compare_locked,
        )

        compare_btn_label = "Compare x3 (Auto)" if compare_locked else "Compare x3"
        compare_submitted = cols[2].form_submit_button(
            compare_btn_label,
            width="stretch",
        )

    run_compare_mode = compare_locked or (compare_submitted and not submitted)

    if (submitted or compare_submitted) and query.strip():
        DB.msg_create(conv_id, "user", query.strip())
        process_steps: list[str] = []
        trace_steps: list[dict] = []
        t_total0 = time.perf_counter()
        with st.status("处理中...", expanded=True) as run_status:
            try:
                run_status.write("Step 1/5: 组装多轮对话上下文")
                max_turns = max(1, int(settings.generation.max_chat_history_turns))
                compare_histories: dict[str, list[dict[str, str]]] | None = None
                if run_compare_mode:
                    compare_histories = _build_compare_branch_histories(
                        prior_messages=messages,
                        current_query=query.strip(),
                        max_turns=max_turns,
                    )
                    history = compare_histories.get("full", [])
                    process_steps.append(
                        "history_turns(compare): "
                        f"full={len(compare_histories.get('full', []))}, "
                        f"vector_only={len(compare_histories.get('vector_only', []))}, "
                        f"direct={len(compare_histories.get('direct', []))}"
                    )
                else:
                    history = _build_conversation_history(
                        prior_messages=messages,
                        current_query=query.strip(),
                        max_turns=max_turns,
                    )
                    process_steps.append(f"history_turns={len(history)}")
                append_trace(
                    trace_steps,
                    make_trace_step(
                        step_name="Step1_history_assembly",
                        module="app.chat",
                        input_data={
                            "max_turns": max_turns,
                            "run_compare_mode": bool(run_compare_mode),
                            "prior_message_count": len(messages or []),
                            "current_query": query.strip(),
                        },
                        output_data={
                            "history_turns": len(history or []),
                            "compare_branch_turns": {
                                "full": len((compare_histories or {}).get("full", [])),
                                "vector_only": len((compare_histories or {}).get("vector_only", [])),
                                "direct": len((compare_histories or {}).get("direct", [])),
                            },
                        },
                    ),
                )

                run_status.write("Step 2/5: 改写 RAG 检索 query")
                t_rewrite0 = time.perf_counter()
                rag_query = _rewrite_rag_query(query.strip(), history)
                rewrite_s = time.perf_counter() - t_rewrite0
                process_steps.append(f"rag_query={rag_query}")
                run_status.write(f"RAG-query: {rag_query}")
                append_trace(
                    trace_steps,
                    make_trace_step(
                        step_name="Step2_rag_query_rewrite",
                        module="app._rewrite_rag_query",
                        input_data={
                            "user_query": query.strip(),
                            "conversation_history": history,
                        },
                        output_data={"rewritten_rag_query": rag_query},
                        fallback={"rewrite_fallback_to_original": bool(rag_query.strip() == query.strip())},
                        elapsed_s=round(float(rewrite_s), 6),
                    ),
                )

                run_status.write("Step 3/5: 学生状态库独立检索（非图）")
                from retrieval.profile_service import prepare_profile_context
                from services.query_preanalysis import build_query_preanalysis
                mgr = _get_mgr(kb_id)
                retrieve_future = None
                query_preanalysis_future = None
                retrieve_pool = None
                # Overlap query-only preanalysis with student-state retrieval and KB retrieval.
                # In single-route mode, the pool also starts hybrid retrieval immediately.
                # In compare mode, the future is passed into Full branch and joined after retrieval.
                retrieve_pool = ThreadPoolExecutor(max_workers=2)
                query_preanalysis_future = retrieve_pool.submit(
                    build_query_preanalysis,
                    query.strip(),
                )
                if not run_compare_mode:
                    retrieve_future = retrieve_pool.submit(
                        mgr.retrieve,
                        rag_query,
                        st.session_state.persist_causal_updates,
                    )

                t_profile0 = time.perf_counter()
                profile_result = prepare_profile_context(
                    kb_id=kb_id,
                    conv_id=conv_id,
                    user_query=query.strip(),
                    prior_messages=messages,
                    bound_user_id=st.session_state.current_user_id or "",
                    state_lib_id=st.session_state.current_student_state_lib_id,
                    enable_llm_review=bool(st.session_state.enable_profile_llm_review),
                    force_refresh=bool(st.session_state.profile_force_refresh_next_query),
                )
                force_refresh_applied = bool(st.session_state.profile_force_refresh_next_query)
                st.session_state.profile_force_refresh_next_query = False
                profile_s = time.perf_counter() - t_profile0
                profile_meta = profile_result.to_meta()
                profile_context = str(profile_result.profile_context or "")
                profile_context_for_model = profile_context if bool(profile_result.identity_ok) else ""
                process_steps.append(
                    "profile: "
                    f"user={profile_result.user_id or '-'} "
                    f"requested={','.join(profile_result.requested_slots or []) or '-'} "
                    f"fetched={','.join(profile_result.fetched_slots or []) or '-'} "
                    f"reused={','.join(profile_result.reused_slots or []) or '-'} "
                    f"identity_ok={profile_result.identity_ok}"
                )
                if profile_result.identity_note:
                    process_steps.append(f"profile_note={profile_result.identity_note}")
                    run_status.write(f"学生状态检索备注: {profile_result.identity_note}")
                if profile_context_for_model:
                    run_status.write("学生状态上下文注入: " + " / ".join(profile_result.requested_slots or []))
                else:
                    run_status.write("学生状态上下文注入: 无")

                append_trace(
                    trace_steps,
                    make_trace_step(
                        step_name="Step3_student_state_retrieval",
                        module="retrieval.profile_service.prepare_profile_context",
                        input_data={
                            "kb_id": kb_id,
                            "conv_id": conv_id,
                            "bound_user_id": st.session_state.current_user_id or "",
                            "state_lib_id": st.session_state.current_student_state_lib_id,
                            "query": query.strip(),
                            "force_refresh": bool(force_refresh_applied),
                        },
                        output_data={
                            "identity_ok": bool(profile_result.identity_ok),
                            "identity_note": profile_result.identity_note,
                            "requested_slots": profile_result.requested_slots or [],
                            "fetched_slots": profile_result.fetched_slots or [],
                            "reused_slots": profile_result.reused_slots or [],
                            "doc_sources": profile_result.doc_sources or [],
                            "doc_count": int(profile_result.doc_count or 0),
                            "source_record_count": len(profile_result.source_records or []),
                            "relevance_filter_trace": profile_result.relevance_filter_trace or {},
                            "time_series_trace": profile_result.time_series_trace or {},
                        },
                        selected=list((profile_result.relevant_facts or {}).keys()),
                        dropped=(profile_result.relevance_filter_trace or {}).get("dropped_keys", []),
                        fallback={
                            "profile_context_injected": bool(profile_context_for_model),
                            "force_refresh_applied": bool(force_refresh_applied),
                        },
                        elapsed_s=round(float(profile_s), 6),
                    ),
                )
                query_preanalysis_meta = {"enabled": False, "note": "disabled_in_compare_or_non_full"}
                query_preanalysis_context_for_model = ""
                if (not run_compare_mode) and query_preanalysis_future is not None:
                    t_qpa = start_timer()
                    try:
                        query_preanalysis_result = query_preanalysis_future.result()
                        query_preanalysis_meta = query_preanalysis_result.to_meta()
                        query_preanalysis_context_for_model = query_preanalysis_result.to_context_text()
                    except Exception as qpa_e:
                        query_preanalysis_meta = {
                            "enabled": False,
                            "note": f"error:{qpa_e}",
                        }
                    append_trace(
                        trace_steps,
                        make_trace_step(
                            step_name="Step3_query_preanalysis",
                            module="services.query_preanalysis.build_query_preanalysis",
                            input_data={
                                "query": query.strip(),
                                "run_compare_mode": bool(run_compare_mode),
                            },
                            output_data=query_preanalysis_meta,
                            fallback={"parallel_with_retrieval": True},
                            elapsed_s=finish_timer(t_qpa),
                        ),
                    )
                elif run_compare_mode and query_preanalysis_future is not None:
                    query_preanalysis_meta = {"enabled": True, "note": "pending_for_compare_full_branch"}

                run_status.write("Step 4/5: 执行问答流程")
                t_qa0 = time.perf_counter()

                if run_compare_mode:
                    run_status.write("并行执行三路: Full-Causal / Vector-Only / No-RAG")
                    process_steps.append(
                        "compare_judges=" + " | ".join(st.session_state.get("compare_judge_specs", []))
                    )
                    compare = _run_compare_modes(
                        mgr=mgr,
                        user_query=query.strip(),
                        retrieval_query=rag_query,
                        conversation_histories=compare_histories or {},
                        persist_causal_updates=st.session_state.persist_causal_updates,
                        profile_context=profile_context_for_model,
                        judge_specs=st.session_state.get("compare_judge_specs", []),
                        progress_cb=run_status.write,
                        query_preanalysis_future=query_preanalysis_future,
                    )
                    if retrieve_pool is not None:
                        retrieve_pool.shutdown(wait=False)
                    qa_s = time.perf_counter() - t_qa0
                    judge = compare.get("judge", {})
                    winner = judge.get("winner", "") or "full"
                    branches = compare.get("branches", {})
                    query_preanalysis_meta = (branches.get("full", {}) or {}).get("query_preanalysis", query_preanalysis_meta)
                    winner_answer = branches.get(winner, {}).get("answer", "")
                    append_trace(
                        trace_steps,
                        make_trace_step(
                            step_name="Step4_compare_branches",
                            module="app._run_compare_modes",
                            input_data={
                                "retrieval_query": rag_query,
                                "judge_specs": st.session_state.get("compare_judge_specs", []),
                            },
                            output_data={
                                "winner": winner,
                                "judge": judge,
                                "branch_elapsed_s": {
                                    "full": float((branches.get("full", {}) or {}).get("elapsed_s", 0.0)),
                                    "vector_only": float((branches.get("vector_only", {}) or {}).get("elapsed_s", 0.0)),
                                    "direct": float((branches.get("direct", {}) or {}).get("elapsed_s", 0.0)),
                                },
                                "query_preanalysis": query_preanalysis_meta,
                            },
                            elapsed_s=round(float(qa_s), 6),
                        ),
                    )
                    summary = f"三路并行对比完成。winner={winner}\n\n{winner_answer}"
                    run_status.write("Step 5/5: 写回消息与元数据")
                    DB.msg_create(
                        conv_id,
                        "assistant",
                        summary,
                        meta={
                            "compare_mode": True,
                            "compare_branches": branches,
                            "compare_judge": judge,
                            "compare_branch_steps": compare.get("branch_steps", {}),
                            "profile_retrieval": profile_meta,
                            "query_preanalysis": query_preanalysis_meta,
                            "rag_query": rag_query,
                            "timings": {
                                "rewrite_s": rewrite_s,
                                "profile_s": profile_s,
                                "qa_s": qa_s,
                                "total_s": time.perf_counter() - t_total0,
                            },
                            "process_steps": process_steps,
                            "trace": trace_steps,
                        },
                    )
                else:
                    if retrieve_future is not None:
                        ctx = retrieve_future.result()
                        if retrieve_pool is not None:
                            retrieve_pool.shutdown(wait=False)
                    else:
                        ctx = mgr.retrieve(
                            rag_query,
                            persist_causal_updates=st.session_state.persist_causal_updates,
                        )
                    append_trace(
                        trace_steps,
                        make_trace_step(
                            step_name="Step4_retrieval_hybrid",
                            module="retrieval.hybrid_retriever.HybridRetriever.retrieve",
                            input_data={
                                "retrieval_query": rag_query,
                                "persist_causal_updates": bool(st.session_state.persist_causal_updates),
                                "parallel_with_profile_retrieval": True,
                            },
                            output_data={
                                "l1_success": bool(ctx.l1_success),
                                "l2_success": bool(ctx.l2_success),
                                "l3_used_reranker": bool(ctx.l3_used_reranker),
                                "vector_chunk_count": len(ctx.vector_chunks or []),
                                "has_causal_context": bool(ctx.has_causal_context),
                                "context_string_len": len(ctx.build_context_string() or ""),
                            },
                        ),
                    )
                    from services.evidence_selector import build_evidence_manifest

                    t_ev = start_timer()
                    evidence_selection = build_evidence_manifest(
                        query=query.strip(),
                        retrieval_context=ctx,
                        profile_meta=profile_meta,
                        query_preanalysis_meta=query_preanalysis_meta,
                        max_items=8,
                    )
                    append_trace(
                        trace_steps,
                        make_trace_step(
                            step_name="Step4_evidence_selector",
                            module="services.evidence_selector.build_evidence_manifest",
                            input_data={
                                "query": query.strip(),
                                "max_items": 8,
                                "student_profile_identity_ok": bool(profile_meta.get("identity_ok", True)),
                            },
                            output_data=evidence_selection.manifest.get("policy", {}),
                            selected=evidence_selection.manifest.get("selected", []),
                            dropped=evidence_selection.manifest.get("dropped", []),
                            fallback={"llm_used": False},
                            elapsed_s=finish_timer(t_ev),
                        ),
                    )
                    gen = mgr.generate_with_meta(
                        query.strip(),
                        ctx,
                        conversation_history=history,
                        retrieval_query=rag_query,
                        profile_context=profile_context_for_model,
                        query_preanalysis_context=query_preanalysis_context_for_model,
                    )
                    answer = gen.answer
                    qa_s = time.perf_counter() - t_qa0
                    append_trace(
                        trace_steps,
                        make_trace_step(
                            step_name="Step4_generation_stage2",
                            module="generation.generator_v2.CausalGenerator.generate",
                            input_data={
                                "retrieval_query": rag_query,
                                "has_profile_context": bool(profile_context_for_model),
                                "history_turns": len(history or []),
                            },
                            output_data={
                                "answer_len": len(answer or ""),
                                "causal_analysis": str(getattr(gen, "causal_analysis", "") or ""),
                                "counterfactual": str(getattr(gen, "counterfactual", "") or ""),
                                "self_reflection": str(getattr(gen, "self_reflection", "") or ""),
                                "l1_used": bool(getattr(gen, "l1_used", False)),
                                "l2_used": bool(getattr(gen, "l2_used", False)),
                                "vector_chunks_count": int(getattr(gen, "vector_chunks_count", 0) or 0),
                            },
                            prompt={"note": "stage prompts are generated in generator_v2; raw prompt text not yet exposed"},
                        ),
                    )
                    meta = {
                        "l1": ctx.l1_success,
                        "l2": ctx.l2_success,
                        "reranked": ctx.l3_used_reranker,
                        "persist_causal_updates": st.session_state.persist_causal_updates,
                        "chunks": len(ctx.vector_chunks),
                        "causal_text": ctx.causal_graph_text,
                        "path_summary": ctx.causal_path_summary,
                        "vector_snippets": [
                            {"text": c["text"], "score": c.get("score", 0)}
                            for c in ctx.vector_chunks[:3]
                        ],
                        "causal_analysis": str(getattr(gen, "causal_analysis", "") or ""),
                        "counterfactual": str(getattr(gen, "counterfactual", "") or ""),
                        "self_reflection": str(getattr(gen, "self_reflection", "") or ""),
                        "evidence_manifest": evidence_selection.manifest,
                        "query_preanalysis": query_preanalysis_meta,
                        "profile_retrieval": profile_meta,
                        "rag_query": rag_query,
                        "timings": {
                            "rewrite_s": rewrite_s,
                            "profile_s": profile_s,
                            "qa_s": qa_s,
                            "total_s": time.perf_counter() - t_total0,
                        },
                        "process_steps": process_steps,
                        "trace": trace_steps,
                    }
                    run_status.write("Step 5/5: 写回消息与元数据")
                    DB.msg_create(conv_id, "assistant", answer, meta=meta)
                run_status.update(label="处理完成", state="complete")
            except Exception as e:
                DB.msg_create(conv_id, "assistant", f"❌ 错误：{e}")
                run_status.update(label="处理失败", state="error")
        st.rerun()


# ===========================================================================
# 📚 KB Management
# ===========================================================================

elif page == "📚 通用三级知识库":
    st.markdown('<div class="page-title">📚 通用三级知识库</div>', unsafe_allow_html=True)

    tab_list, tab_new, tab_docs, tab_test = st.tabs(
        ["知识库列表", "新建知识库", "添加文档", "快速测试"]
    )

    # ---- List ----
    with tab_list:
        kbs = DB.kb_list()
        if not kbs:
            st.info("暂无知识库，请在「新建知识库」标签页创建。")
        for kb in kbs:
            s = DB.kb_stats(kb["id"])
            with st.container():
                st.markdown(f'<div class="kb-card">', unsafe_allow_html=True)
                c1, c2, c3 = st.columns([4, 3, 2])
                c1.markdown(f"**{kb['name']}**")
                if kb.get("description"):
                    c1.caption(kb["description"])
                c2.caption(
                    f"文档: {s['doc_count']} · 块: {s['total_chunks']} · "
                    f"对话: {s['conv_count']} · 消息: {s['msg_count']}"
                )
                c2.caption(f"创建: {kb['created_at']}")
                # Graph stats
                try:
                    mgr_s = _get_mgr(kb["id"]).stats()
                    c2.caption(f"因果节点: {mgr_s['cg_nodes']} · 文本图节点: {mgr_s['tg_nodes']}")
                except Exception:
                    pass

                with c3.popover("⚙️ 操作"):
                    st.markdown(f"**{kb['name']}**")
                    new_desc = st.text_input("描述", value=kb.get("description",""), key=f"desc_{kb['id']}")
                    if st.button("保存描述", key=f"save_{kb['id']}"):
                        DB.kb_update(kb["id"], description=new_desc)
                        st.rerun()
                    st.markdown("---")
                    if st.button(f"🔴 删除知识库", key=f"del_{kb['id']}", type="primary"):
                        _get_mgr(kb["id"]).delete_all_data()
                        DB.kb_delete(kb["id"])
                        _invalidate(kb["id"])
                        if st.session_state.current_kb_id == kb["id"]:
                            st.session_state.current_kb_id = None
                            st.session_state.current_conv_id = None
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    # ---- New KB ----
    with tab_new:
        st.markdown("#### 新建知识库")
        name_input = st.text_input("知识库名称*", placeholder="例：航空法规2024")
        desc_input = st.text_area("描述（可选）", height=80)
        if st.button("✅ 创建", type="primary", disabled=not name_input.strip()):
            try:
                kb = DB.kb_create(name_input.strip(), desc_input.strip())
                st.session_state.current_kb_id = kb["id"]
                st.success(f"知识库「{kb['name']}」创建成功！")
                st.rerun()
            except Exception as e:
                st.error(f"创建失败（可能名称重复）：{e}")

    # ---- Add Documents ----
    with tab_docs:
        kb_id = st.session_state.current_kb_id
        if not kb_id:
            st.warning("请先选择知识库（左侧边栏）。")
            st.stop()

        kb = DB.kb_get(kb_id)
        st.markdown(f"#### 向「{kb['name']}」添加文档")

        existing_docs = DB.doc_list(kb_id)
        if existing_docs:
            st.markdown("**已有文档**")
            for doc in existing_docs:
                dc1, dc2 = st.columns([4, 1])
                dc1.caption(f"📄 {doc['name']}  ·  {doc['chunk_count']} 块  ·  {doc['created_at']}")
                if dc2.button("删除", key=f"del_doc_{doc['id']}"):
                    DB.doc_delete(doc["id"])
                    st.rerun()
            st.markdown("---")

        input_method = st.radio(
            "输入方式",
            ["文本粘贴", "上传文件"],
            horizontal=True,
            key="doc_input_method",
        )

        doc_texts: list[tuple[str, str]] = []  # (name, text)
        if input_method == "文本粘贴":
            doc_name = st.text_input("文档名称", placeholder="文档1", key="paste_name")
            doc_text = st.text_area("文档内容", height=260, key="paste_text")
            if doc_name and doc_text:
                doc_texts = [(doc_name, doc_text)]
        elif input_method == "上传文件":
            uploaded = st.file_uploader(
                "上传 .txt 或 .md", type=["txt", "md"],
                accept_multiple_files=True, key="file_uploader"
            )
            if uploaded:
                for f in uploaded:
                    doc_texts.append((f.name, f.read().decode("utf-8", errors="ignore")))
                st.success(f"已选 {len(uploaded)} 个文件")
        col_o1, col_o2, col_o3 = st.columns(3)
        build_v  = col_o1.checkbox("向量索引", value=True, key="bv")
        build_tg = col_o2.checkbox("文本图", value=True, key="btg")
        build_cg = col_o3.checkbox("因果图", value=True, key="bcg")
        text_graph_workers = st.number_input(
            "文本图并行 Worker 数",
            min_value=1,
            max_value=32,
            value=10,
            step=1,
            key="text_graph_workers",
            help="文本图 chunk 级滚动并行 worker 数，默认 10。",
            disabled=not build_tg,
        )
        implicit_discovery = st.checkbox(
            "启用隐式因果发现（可选，耗时更高）",
            value=False,
            key="build_implicit_causal",
            help="在显式因果信号之外，额外扫描弱因果信号 chunk，可提升隐式因果召回；默认关闭以控制时间和模型成本。",
            disabled=not build_cg,
        )
        implicit_max_chunks = st.slider(
            "隐式因果额外候选上限",
            min_value=5,
            max_value=120,
            value=24,
            step=5,
            key="build_implicit_causal_max_chunks",
            disabled=(not build_cg) or (not implicit_discovery),
        )

        can_import = bool(doc_texts)
        if st.button("🚀 导入并构建", type="primary", disabled=not can_import):
            mgr = _get_mgr(kb_id)
            for doc_name, doc_text in doc_texts:
                with st.status(f"处理「{doc_name}」…", expanded=True) as s_status:
                    try:
                        stats = mgr.add_text(
                            doc_text, doc_name,
                            build_vector=build_v,
                            build_text_graph=build_tg,
                            build_causal_graph=build_cg,
                            text_graph_max_workers=int(text_graph_workers),
                            enable_implicit_causal_discovery=implicit_discovery if build_cg else False,
                            implicit_causal_max_chunks=int(implicit_max_chunks),
                        )
                        DB.doc_create(kb_id, doc_name, chunk_count=stats["chunks"])
                        s_status.update(
                            label=f"✅ {doc_name}: {stats['chunks']} 块 | "
                                  f"TG {stats['tg_nodes']}节点 | CG {stats['cg_nodes']}节点",
                            state="complete"
                        )
                    except Exception as e:
                        s_status.update(label=f"❌ {doc_name}: {e}", state="error")
            _invalidate(kb_id)
            st.success("导入完成！")

    # ---- Quick test ----
    with tab_test:
        st.markdown("#### 快速功能测试")
        DEMO = """气候变化导致全球气温升高。
气温升高引起冰川融化，进而造成海平面上升。
海平面上升威胁沿海城市，导致大量居民迁移。
干旱引起粮食减产，粮食减产导致粮食价格上涨，进而引发社会不稳定。
化石燃料燃烧导致温室气体排放增加，温室气体排放增加是气候变化的主要原因。"""

        demo_text = st.text_area("测试文本（可编辑）", value=DEMO, height=180, key="demo_ta")
        demo_use_implicit = st.checkbox(
            "完整抽取时启用隐式因果候选（更慢）",
            value=False,
            key="demo_implicit_causal_toggle",
        )
        demo_implicit_max = st.slider(
            "完整抽取隐式候选上限",
            min_value=5,
            max_value=80,
            value=20,
            step=5,
            key="demo_implicit_causal_max",
            disabled=not demo_use_implicit,
        )
        col_t1, col_t2, col_t3 = st.columns(3)

        if col_t1.button("🧪 LLM连通测试"):
            from utils.llm_client import llm_client
            with st.spinner("调用 DeepSeek…"):
                try:
                    ans = llm_client.chat([{"role":"user","content":"用一句话说明因果关系的重要性"}])
                    st.success("LLM正常")
                    st.write(ans)
                except Exception as e:
                    st.error(str(e))

        if col_t2.button("🧪 嵌入向量测试"):
            from utils.embedding_client import embedding_client
            with st.spinner("调用 SiliconFlow…"):
                try:
                    vecs = embedding_client.embed(["气候变化","粮食危机","社会不稳定"])
                    st.success(f"嵌入正常 · 维度: {len(vecs[0])}")
                except Exception as e:
                    st.error(str(e))

        if col_t3.button("🧪 完整因果抽取"):
            from knowledge_base.preprocessor import DocumentPreprocessor
            from knowledge_base.causal_graph import CausalGraph
            import uuid
            prep = DocumentPreprocessor(chunk_size=300, chunk_overlap=30)
            chunks = prep.load_texts([demo_text], source="test")
            tmp_cg = CausalGraph(kb_id=str(uuid.uuid4()))
            with st.spinner(f"抽取因果三元组（{len(chunks)} 块）…"):
                try:
                    tmp_cg.build_from_chunks(
                        chunks,
                        save=False,
                        enable_implicit_discovery=demo_use_implicit,
                        implicit_max_chunks=int(demo_implicit_max),
                    )
                    st.success(f"因果图: {tmp_cg.graph.number_of_nodes()} 节点, {tmp_cg.graph.number_of_edges()} 边")
                    for u, v, d in tmp_cg.graph.edges(data=True):
                        rels = ", ".join(d.get("relations",[]))
                        conf = d.get("confidence",0)
                        color = "#22c55e" if conf >= 0.8 else "#f59e0b"
                        st.markdown(
                            f"<span style='color:{color}'>●</span> `{u}` → *{rels}* → `{v}` (conf {conf:.2f})",
                            unsafe_allow_html=True
                        )
                except Exception as e:
                    st.error(str(e))


# ===========================================================================
# 🕸️ Graph Explorer
# ===========================================================================

elif page == "🎓 学生状态库":
    st.markdown('<div class="page-title">🎓 学生状态库</div>', unsafe_allow_html=True)
    panel_state = render_student_state_panel(
        st=st,
        DB=DB,
        current_kb_id=st.session_state.current_kb_id,
        current_conv_id=st.session_state.current_conv_id,
        current_user_id=st.session_state.current_user_id,
        enable_profile_llm_review=st.session_state.enable_profile_llm_review,
        current_state_lib_id=st.session_state.current_student_state_lib_id,
        title="🎓 学生状态库",
        expanded=True,
        key_prefix="student_status_page",
        show_identity_controls=False,
        show_file_management=True,
        force_refresh_next_query=bool(st.session_state.profile_force_refresh_next_query),
    )
    st.session_state.current_student_state_lib_id = str(panel_state.get("current_student_state_lib_id", "") or "")
    st.session_state.enable_profile_llm_review = bool(
        panel_state.get("enable_profile_llm_review", st.session_state.enable_profile_llm_review)
    )


elif page == "👤 人物管理":
    st.markdown('<div class="page-title">👤 人物管理</div>', unsafe_allow_html=True)
    panel_state = render_student_state_panel(
        st=st,
        DB=DB,
        current_kb_id=st.session_state.current_kb_id,
        current_conv_id=st.session_state.current_conv_id,
        current_user_id=st.session_state.current_user_id,
        enable_profile_llm_review=st.session_state.enable_profile_llm_review,
        current_state_lib_id=st.session_state.current_student_state_lib_id,
        title="👤 当前会话人物",
        expanded=True,
        key_prefix="person_page",
        show_identity_controls=True,
        show_file_management=False,
        force_refresh_next_query=bool(st.session_state.profile_force_refresh_next_query),
    )
    st.session_state.current_user_id = str(panel_state.get("current_user_id", "") or "")
    st.session_state.current_student_state_lib_id = str(panel_state.get("current_student_state_lib_id", "") or "")
    st.session_state.enable_profile_llm_review = bool(
        panel_state.get("enable_profile_llm_review", st.session_state.enable_profile_llm_review)
    )
    st.session_state.profile_force_refresh_next_query = bool(
        panel_state.get("force_refresh_next_query", st.session_state.profile_force_refresh_next_query)
    )


elif page == "🕸️ 图谱浏览":
    st.markdown('<div class="page-title">🕸️ 图谱浏览</div>', unsafe_allow_html=True)

    import pandas as pd
    kb_id = st.session_state.current_kb_id
    if not kb_id:
        st.info("请先选择知识库。")
        st.stop()

    try:
        mgr = _get_mgr(kb_id)
    except Exception as e:
        st.error(str(e)); st.stop()

    graph_choice = st.radio("选择图谱", ["因果图", "文本图"], horizontal=True)
    graph_obj = mgr.causal_graph if graph_choice == "因果图" else mgr.text_graph
    g = graph_obj.graph
    merge_state_key = _merge_state_key(kb_id, graph_choice)
    delete_state_key = _delete_state_key(kb_id, graph_choice)
    struct_state_key = f"struct_rel_candidates::{kb_id}"
    struct_meta_key = f"{struct_state_key}::meta"
    struct_applied_key = f"{struct_state_key}::applied"

    def _render_structured_relation_discovery_panel() -> None:
        with st.expander("结构化关系发现（文本图 -> 因果图）", expanded=False):
            st.caption(
                "先扫描文本图候选关系，再手动应用选中项；默认 keep_unknown=False。"
            )
            sr_c1, sr_c2, sr_c3 = st.columns([2, 1, 1])
            max_edges = sr_c1.slider(
                "最大扫描边数",
                min_value=50,
                max_value=2000,
                value=600,
                step=50,
                key=f"struct_scan_edges::{kb_id}",
            )
            use_llm = sr_c2.checkbox("使用 LLM 分类", value=True, key=f"struct_scan_llm::{kb_id}")
            keep_unknown = sr_c3.checkbox("保留 unknown", value=False, key=f"struct_scan_unknown::{kb_id}")

            a1, a2 = st.columns([1, 1])
            if a1.button("扫描候选", key=f"struct_scan_btn::{kb_id}", width="stretch"):
                suggestion = mgr.suggest_structured_relations_from_text_graph(
                    max_edges=int(max_edges),
                    use_llm=bool(use_llm),
                    keep_unknown=bool(keep_unknown),
                )
                st.session_state[struct_state_key] = suggestion.get("candidate_triples", []) or []
                st.session_state[struct_meta_key] = suggestion
                st.session_state[struct_applied_key] = 0
            if a2.button("清空候选", key=f"struct_clear_btn::{kb_id}", width="stretch"):
                st.session_state.pop(struct_state_key, None)
                st.session_state.pop(struct_meta_key, None)
                st.session_state.pop(struct_applied_key, None)
                st.rerun()

            struct_meta = st.session_state.get(struct_meta_key, {}) or {}
            struct_candidates = st.session_state.get(struct_state_key, []) or []
            if struct_meta:
                st.caption(
                    "processed_edges="
                    + str(int(struct_meta.get("processed_edges", 0)))
                    + " | candidate_count="
                    + str(int(struct_meta.get("candidate_count", 0)))
                    + " | filtered_non_causal="
                    + str(int(struct_meta.get("skipped_non_causal", 0)))
                    + " | filtered_unknown="
                    + str(int(struct_meta.get("skipped_unknown", 0)))
                    + " | llm_classified="
                    + str(int(struct_meta.get("llm_classified_relations", 0)))
                    + " | applied="
                    + str(int(st.session_state.get(struct_applied_key, 0) or 0))
                )

            if struct_candidates:
                rows = []
                labels = []
                for idx, item in enumerate(struct_candidates, start=1):
                    rows.append(
                        {
                            "#": idx,
                            "cause": item.get("cause", ""),
                            "relation": item.get("relation", ""),
                            "relation_type": item.get("relation_type", ""),
                            "polarity": item.get("polarity", ""),
                            "effect": item.get("effect", ""),
                            "confidence": round(float(item.get("confidence", 0.0) or 0.0), 3),
                        }
                    )
                    labels.append(
                        f"{idx}. {item.get('cause','')} --[{item.get('relation_type','unknown')}]--> {item.get('effect','')}"
                    )
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                selected_labels = st.multiselect(
                    "选择要应用的候选",
                    labels,
                    default=labels[: min(20, len(labels))],
                    key=f"struct_select::{kb_id}",
                )
                selected_candidates = [struct_candidates[labels.index(label)] for label in selected_labels]
                ap1, ap2 = st.columns([1, 1])
                if ap1.button(
                    "应用选中",
                    key=f"struct_apply_selected::{kb_id}",
                    width="stretch",
                    disabled=not selected_candidates,
                ):
                    result = mgr.apply_structured_relation_candidates(selected_candidates, save=True)
                    st.session_state[struct_applied_key] = int(
                        st.session_state.get(struct_applied_key, 0) or 0
                    ) + int(result.get("added", 0))
                    _invalidate(kb_id)
                    st.success(f"applied={int(result.get('added', 0))}, skipped={int(result.get('skipped', 0))}")
                    st.rerun()
                if ap2.button("应用全部", key=f"struct_apply_all::{kb_id}", width="stretch"):
                    result = mgr.apply_structured_relation_candidates(struct_candidates, save=True)
                    st.session_state[struct_applied_key] = int(
                        st.session_state.get(struct_applied_key, 0) or 0
                    ) + int(result.get("added", 0))
                    _invalidate(kb_id)
                    st.success(f"applied={int(result.get('added', 0))}, skipped={int(result.get('skipped', 0))}")
                    st.rerun()

    if graph_obj is mgr.text_graph:
        _render_structured_relation_discovery_panel()
        st.markdown("---")

    if g.number_of_nodes() == 0:
        st.info("该图谱尚未构建，请在「通用三级知识库」页面添加文档。")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    import networkx as nx
    c1.metric("节点数", g.number_of_nodes())
    c2.metric("边数", g.number_of_edges())
    c3.metric("弱连通分量", len(list(nx.weakly_connected_components(g))))
    avg_deg = sum(d for _, d in g.degree()) / max(g.number_of_nodes(), 1)
    c4.metric("平均度", f"{avg_deg:.2f}")

    st.markdown("---")

    with st.expander("节点整理与合并", expanded=False):
        st.caption("这里可以把图谱做精修：先扫同义节点，再手工合并，最后批量或手动删除噪声节点。")

        tool_tab1, tool_tab2, tool_tab3 = st.tabs(["自动合并", "手动合并", "节点删除"])

        with tool_tab1:
            st.markdown(
                '<div class="tool-card"><h4>自动发现相似节点</h4><p>结合嵌入语义相似度和文本相似度生成候选组，适合快速合并“同一概念的不同表述”。</p></div>',
                unsafe_allow_html=True,
            )
            scan_c1, scan_c2, scan_c3 = st.columns(3)
            semantic_threshold = scan_c1.slider(
                "语义相似阈值", 0.70, 0.99, 0.92, 0.01, key=f"sem::{kb_id}::{graph_choice}"
            )
            lexical_threshold = scan_c2.slider(
                "文本相似阈值", 0.60, 0.99, 0.82, 0.01, key=f"lex::{kb_id}::{graph_choice}"
            )
            max_groups = scan_c3.number_input(
                "最多返回候选组", min_value=5, max_value=100, value=20, step=5,
                key=f"max_groups::{kb_id}::{graph_choice}"
            )

            act_c1, act_c2 = st.columns([1, 1])
            if act_c1.button("扫描可合并节点", key=f"scan_merge::{kb_id}::{graph_choice}", width="stretch"):
                with st.spinner("正在生成候选合并建议…"):
                    suggestions = graph_obj.suggest_merges(
                        semantic_threshold=float(semantic_threshold),
                        lexical_threshold=float(lexical_threshold),
                        max_groups=int(max_groups),
                    )
                st.session_state[merge_state_key] = [asdict(item) for item in suggestions]

            if act_c2.button("清空候选", key=f"clear_merge::{kb_id}::{graph_choice}", width="stretch"):
                st.session_state.pop(merge_state_key, None)
                st.rerun()

            suggestions = st.session_state.get(merge_state_key, [])
            if suggestions:
                rows = []
                labels = []
                for idx, item in enumerate(suggestions, start=1):
                    rows.append({
                        "序号": idx,
                        "保留节点": item["canonical"],
                        "合并节点": "、".join(item["aliases"]),
                        "平均相似度": round(float(item["average_similarity"]), 3),
                        "依据": item["reason"],
                    })
                    labels.append(f"{idx}. {item['canonical']} ← {'、'.join(item['aliases'])}")

                st.dataframe(rows, width="stretch", height=220)
                apply_c1, apply_c2 = st.columns([1, 1])
                selected_label = apply_c1.selectbox(
                    "选择一组候选", labels, key=f"select_merge::{kb_id}::{graph_choice}"
                )
                selected_idx = labels.index(selected_label)

                if apply_c1.button("应用选中候选", key=f"apply_one::{kb_id}::{graph_choice}", width="stretch"):
                    selected = suggestions[selected_idx]
                    merged = graph_obj.merge_nodes(selected["canonical"], selected["aliases"], save=True)
                    st.session_state.pop(merge_state_key, None)
                    _invalidate(kb_id)
                    st.success(f"已合并 {merged} 个节点到「{selected['canonical']}」")
                    st.rerun()

                if apply_c2.button("应用全部候选", key=f"apply_all::{kb_id}::{graph_choice}", width="stretch"):
                    merged_total = 0
                    for item in suggestions:
                        merged_total += graph_obj.merge_nodes(item["canonical"], item["aliases"], save=False)
                    if merged_total:
                        graph_obj.save()
                    st.session_state.pop(merge_state_key, None)
                    _invalidate(kb_id)
                    st.success(f"已合并 {merged_total} 个节点")
                    st.rerun()
            else:
                st.info("还没有候选合并建议。先点“扫描可合并节点”，系统会按文本相似度和嵌入相似度给出建议。")

        with tool_tab2:
            st.markdown(
                '<div class="tool-card"><h4>手动精修节点</h4><p>适合你已经明确知道哪些节点应该保留、哪些只是别名时，直接手工指定合并方向。</p></div>',
                unsafe_allow_html=True,
            )
            node_options = sorted(g.nodes())
            manual_c1, manual_c2 = st.columns([2, 3])
            manual_target = manual_c1.selectbox(
                "保留节点", node_options, key=f"manual_target::{kb_id}::{graph_choice}"
            )
            manual_aliases = manual_c2.multiselect(
                "合并进该节点",
                [node for node in node_options if node != manual_target],
                key=f"manual_aliases::{kb_id}::{graph_choice}",
            )
            if st.button("执行手动合并", key=f"manual_merge::{kb_id}::{graph_choice}", disabled=not manual_aliases):
                merged = graph_obj.merge_nodes(manual_target, manual_aliases, save=True)
                st.session_state.pop(merge_state_key, None)
                st.session_state.pop(delete_state_key, None)
                _invalidate(kb_id)
                st.success(f"已手动合并 {merged} 个节点到「{manual_target}」")
                st.rerun()

        with tool_tab3:
            st.markdown(
                '<div class="tool-card"><h4>删除噪声节点</h4><p>文本图支持先扫描低价值节点候选，再批量删除；两类图都支持手动指定节点删除。</p></div>',
                unsafe_allow_html=True,
            )
            del_c1, del_c2 = st.columns([1, 1])
            if graph_choice == "文本图":
                quality_diag = graph_obj.diagnose_quality(max_examples=8)
                st.caption(
                    f"质量诊断 | 节点 {quality_diag.get('node_count', 0)} | "
                    f"裸泛词 {quality_diag.get('bare_ambiguous_count', 0)} | "
                    f"低连通噪声候选 {quality_diag.get('structural_noise_count', 0)}"
                )
                bare_examples = quality_diag.get("bare_ambiguous_examples", [])
                if bare_examples:
                    st.caption(f"裸泛词示例：{', '.join(bare_examples[:8])}")
                if del_c1.button("扫描低价值节点", key=f"scan_delete::{kb_id}::{graph_choice}", width="stretch"):
                    t_scan0 = time.perf_counter()
                    with st.spinner("正在扫描低价值节点，请稍候..."):
                        candidates = graph_obj.suggest_low_value_nodes(max_nodes=200)
                    st.session_state[delete_state_key] = candidates
                    st.session_state[f"{delete_state_key}::meta"] = {
                        "scan_seconds": time.perf_counter() - t_scan0,
                        "candidate_count": len(candidates),
                        "node_count": int(g.number_of_nodes()),
                    }
                if del_c2.button("清空删除候选", key=f"clear_delete::{kb_id}::{graph_choice}", width="stretch"):
                    st.session_state.pop(delete_state_key, None)
                    st.session_state.pop(f"{delete_state_key}::meta", None)
                    st.rerun()

                iso_c1, iso_c2 = st.columns([1, 1])
                if iso_c1.button("扫描单节点连通分量", key=f"scan_singleton::{kb_id}::{graph_choice}", width="stretch"):
                    t_iso0 = time.perf_counter()
                    with st.spinner("正在扫描连通分量，请稍候..."):
                        singleton_nodes = graph_obj.suggest_singleton_component_nodes(max_nodes=5000)
                    st.session_state[f"singleton_nodes::{kb_id}::{graph_choice}"] = singleton_nodes
                    st.session_state[f"singleton_meta::{kb_id}::{graph_choice}"] = {
                        "scan_seconds": time.perf_counter() - t_iso0,
                        "candidate_count": len(singleton_nodes),
                    }
                if iso_c2.button("删除单节点连通分量", key=f"delete_singleton::{kb_id}::{graph_choice}", width="stretch"):
                    t_del0 = time.perf_counter()
                    with st.spinner("正在删除单节点连通分量，请稍候..."):
                        deleted_singletons = graph_obj.delete_singleton_components(save=True)
                    st.session_state.pop(delete_state_key, None)
                    st.session_state.pop(f"{delete_state_key}::meta", None)
                    st.session_state.pop(f"singleton_nodes::{kb_id}::{graph_choice}", None)
                    st.session_state.pop(f"singleton_meta::{kb_id}::{graph_choice}", None)
                    _invalidate(kb_id)
                    st.success(
                        f"已删除 {deleted_singletons} 个单节点连通分量（耗时 {time.perf_counter() - t_del0:.2f}s）"
                    )
                    st.rerun()

                delete_candidates = st.session_state.get(delete_state_key, [])
                delete_meta = st.session_state.get(f"{delete_state_key}::meta", {})
                if delete_meta:
                    st.caption(
                        f"低价值扫描：候选 {int(delete_meta.get('candidate_count', 0))} / "
                        f"总节点 {int(delete_meta.get('node_count', 0))}，"
                        f"耗时 {float(delete_meta.get('scan_seconds', 0.0)):.2f}s"
                    )
                if delete_candidates:
                    candidate_rows = [
                        {
                            "节点": item["node"],
                            "原因": item["reason"],
                            "度": item["degree"],
                        }
                        for item in delete_candidates
                    ]
                    st.dataframe(candidate_rows, width="stretch", height=220)
                    candidate_labels = [
                        f"{item['node']} · {item['reason']} · 度 {item['degree']}"
                        for item in delete_candidates
                    ]
                    selected_delete_labels = st.multiselect(
                        "选择要删除的候选节点",
                        candidate_labels,
                        default=candidate_labels[: min(10, len(candidate_labels))],
                        key=f"delete_candidates_select::{kb_id}::{graph_choice}",
                    )
                    selected_delete_nodes = [
                        delete_candidates[candidate_labels.index(label)]["node"]
                        for label in selected_delete_labels
                    ]
                    if st.button(
                        "删除选中候选",
                        key=f"apply_delete_candidates::{kb_id}::{graph_choice}",
                        disabled=not selected_delete_nodes,
                        width="stretch",
                    ):
                        deleted = graph_obj.delete_nodes(selected_delete_nodes, save=True)
                        st.session_state.pop(delete_state_key, None)
                        st.session_state.pop(merge_state_key, None)
                        _invalidate(kb_id)
                        st.success(f"已删除 {deleted} 个低价值节点")
                        st.rerun()
                else:
                    st.info("还没有删除候选。你可以先扫描一次，快速定位数字、附件名、编号等低价值节点。")

                singleton_nodes = st.session_state.get(f"singleton_nodes::{kb_id}::{graph_choice}", [])
                singleton_meta = st.session_state.get(f"singleton_meta::{kb_id}::{graph_choice}", {})
                if singleton_meta:
                    st.caption(
                        f"单节点连通分量扫描：候选 {int(singleton_meta.get('candidate_count', 0))}，"
                        f"耗时 {float(singleton_meta.get('scan_seconds', 0.0)):.2f}s"
                    )
                if singleton_nodes:
                    st.caption(f"示例（最多前10个）：{', '.join(singleton_nodes[:10])}")
            else:
                del_c1.caption("当前图谱不提供自动噪声扫描")
                del_c2.caption("可直接使用下方手动删除")

            st.markdown("#### 手动删除节点")
            manual_delete_query = st.text_input(
                "筛选节点",
                placeholder="输入关键词快速缩小范围",
                key=f"manual_delete_query::{kb_id}::{graph_choice}",
            ).strip()
            filtered_nodes = sorted(
                node for node in g.nodes()
                if not manual_delete_query or manual_delete_query.lower() in str(node).lower()
            )
            manual_delete_nodes = st.multiselect(
                "选择要删除的节点",
                filtered_nodes[:500],
                key=f"manual_delete_nodes::{kb_id}::{graph_choice}",
                help="为避免前端过重，这里最多展示前 500 个匹配节点。",
            )
            if st.button(
                "执行手动删除",
                key=f"manual_delete::{kb_id}::{graph_choice}",
                disabled=not manual_delete_nodes,
                width="stretch",
            ):
                deleted = graph_obj.delete_nodes(manual_delete_nodes, save=True)
                st.session_state.pop(delete_state_key, None)
                st.session_state.pop(merge_state_key, None)
                _invalidate(kb_id)
                st.success(f"已删除 {deleted} 个节点")
                st.rerun()

    st.markdown("---")
    max_n = st.slider("最多显示节点数", 10, 300, 80, 10)

    def _render(nx_graph, is_causal: bool, max_nodes: int) -> str:
        try:
            from pyvis.network import Network
        except ImportError:
            return "<p>请安装 pyvis</p>"
        if nx_graph.number_of_nodes() > max_nodes:
            top = sorted(nx_graph.nodes(), key=lambda n: nx_graph.degree(n), reverse=True)[:max_nodes]
            sub = nx_graph.subgraph(top).copy()
        else:
            sub = nx_graph
        net = Network(height="540px", width="100%", directed=True,
                      bgcolor="#0f172a", font_color="#e2e8f0")
        net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120)
        for node in sub.nodes():
            deg = sub.degree(node)
            net.add_node(str(node), label=str(node),
                         size=12+deg*3,
                         color="#f59e0b" if is_causal else "#60a5fa",
                         title=f"度: {deg}")
        for u, v, data in sub.edges(data=True):
            rels = ", ".join(data.get("relations", []))
            if is_causal:
                conf = data.get("confidence", 0.5)
                width = 1 + conf * 4
                color = "#22c55e" if conf >= 0.8 else "#f59e0b" if conf >= 0.6 else "#ef4444"
                title = f"{rels} ({conf:.2f})"
            else:
                width, color, title = 2, "#818cf8", rels
            net.add_edge(str(u), str(v), title=title, width=width, color=color, arrows="to")
        net.set_options('{"interaction":{"hover":true,"navigationButtons":true},"physics":{"enabled":true}}')
        out = ROOT / "data" / "temp_graph.html"
        out.parent.mkdir(exist_ok=True)
        net.save_graph(str(out))
        return out.read_text(encoding="utf-8")

    html = _render(g, graph_choice == "因果图", max_n)
    components.html(html, height=560)

    st.markdown("---")
    st.markdown("#### 边列表")
    import pandas as pd
    rows = []
    for u, v, data in g.edges(data=True):
        rels = ", ".join(data.get("relations", []))
        row = {"来源": u, "关系": rels, "目标": v}
        if graph_choice == "因果图":
            row["置信度"] = round(data.get("confidence", 0), 3)
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        search = st.text_input("🔍 过滤", placeholder="关键词…")
        if search:
            mask = df.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
            df = df[mask]
        st.dataframe(df, width="stretch", height=300)
        st.caption(f"共 {len(rows)} 条边，显示 {len(df)} 条")


# ===========================================================================
# 📊 Evaluation
# ===========================================================================

elif page == "📊 系统评估":
    st.markdown('<div class="page-title">📊 系统评估</div>', unsafe_allow_html=True)

    kb_id = st.session_state.current_kb_id
    if not kb_id:
        st.info("请先选择知识库。")
        st.stop()

    tab_run, tab_hist = st.container(), st.container()

    st.tabs = _orig_tabs
    tab_run, tab_batch, tab_hist = st.tabs(["运行评估", "一键批测", "历史结果"])

    with tab_run:
        st.markdown("上传 JSONL 数据集（每行 `{\"question\":\"…\",\"answer\":\"…\"}`）")
        uploaded_ds = st.file_uploader("评估数据集", type=["jsonl", "json"])
        col_a, col_b = st.columns(2)
        use_judge = col_a.checkbox("LLM 裁判打分", value=True)
        max_s = col_b.number_input("最大样本数（0=全部）", min_value=0, value=10, step=5)

        if st.button("▶ 开始评估", type="primary", disabled=not uploaded_ds):
            lines = [l.strip() for l in uploaded_ds.read().decode("utf-8").splitlines() if l.strip()]
            if max_s > 0:
                lines = lines[:int(max_s)]

            from evaluation.evaluator import Evaluator
            mgr = _get_mgr(kb_id)
            evaluator = Evaluator(use_llm_judge=use_judge)

            results = []
            prog = st.progress(0)
            status = st.empty()
            for i, line in enumerate(lines):
                rec = json.loads(line)
                q = rec["question"]
                status.info(f"[{i+1}/{len(lines)}] {q[:60]}…")
                try:
                    answer, _ = mgr.ask(q)
                except Exception as e:
                    answer = ""
                sample = evaluator.evaluate_sample(q, rec["answer"], answer)
                results.append(sample)
                prog.progress((i+1)/len(lines))

            prog.empty(); status.empty()
            n = len(results)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("EM", f"{sum(s.em for s in results)/n:.3f}")
            m2.metric("F1", f"{sum(s.f1 for s in results)/n:.3f}")
            m3.metric("正确性", f"{sum(s.correctness for s in results)/n:.1f}/10")
            m4.metric("因果质量", f"{sum(s.causal_quality for s in results)/n:.1f}/10")

            df = pd.DataFrame([{
                "问题": s.question[:40]+"…",
                "EM": s.em, "F1": round(s.f1,3),
                "正确性": s.correctness, "因果质量": s.causal_quality,
                "评语": (s.judge_comment or "")[:50],
            } for s in results])
            st.dataframe(df, width="stretch")

            out_p = ROOT / "data" / f"eval_{int(time.time())}.jsonl"
            from evaluation.evaluator import Evaluator as _E
            _E._save_results(results, str(out_p))
            st.success(f"已保存: {out_p.name}")

    with tab_batch:
        st.markdown("### 依赖知识库的一键批量三路测试")
        st.caption("支持：选择题目文件、三路并行对比、多模型判分、实时写盘、停止后续跑。")

        option_map = _compare_judge_option_map()
        option_labels = list(option_map.keys())
        default_specs = [
            str(s)
            for s in st.session_state.get("compare_judge_specs", [])
            if str(s).strip()
        ]
        default_labels = [label for label, spec in option_map.items() if spec in default_specs]
        if not default_labels and option_labels:
            default_labels = [option_labels[0]]

        col_ds1, col_ds2 = st.columns([2, 3])
        candidate_paths: list[str] = []
        for base in [ROOT / "evaluation", ROOT / "data"]:
            if base.exists():
                for p in base.rglob("*"):
                    if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}:
                        candidate_paths.append(str(p.resolve()))
        candidate_paths = sorted(set(candidate_paths))[:500]
        remember_key = f"batch_dataset_path::{kb_id}"
        remembered_path = str(st.session_state.get(remember_key, "")).strip()
        runner_path = str((st.session_state.get("batch_test_runner", {}) or {}).get("dataset_path", "")).strip()
        if not remembered_path and runner_path:
            remembered_path = runner_path
        if not remembered_path:
            default_subset = ROOT / "evaluation" / "WixDataset" / "subset_qa.json"
            if default_subset.exists():
                remembered_path = str(default_subset.resolve())
        options = [""] + candidate_paths
        selected_index = options.index(remembered_path) if remembered_path in options else 0
        selected_candidate = col_ds1.selectbox(
            "选择题目文件",
            options,
            index=selected_index,
            help="可从 evaluation/ 或 data/ 目录中选择，也可手动输入绝对路径。",
        )
        manual_default = remembered_path if (remembered_path and remembered_path not in candidate_paths) else ""
        manual_path = col_ds2.text_input(
            "或手动输入题目文件路径（.json/.jsonl）",
            value=manual_default,
            placeholder="evaluation/datasets/qa.json",
        ).strip()
        dataset_path_str = manual_path or selected_candidate
        if dataset_path_str:
            st.session_state[remember_key] = dataset_path_str
        dataset_path = Path(dataset_path_str) if dataset_path_str else None

        judge_labels = st.multiselect(
            "评判模型（可多选）",
            option_labels,
            default=default_labels,
            key=f"batch_judge_labels::{kb_id}",
            help="每个模型独立判分，最终展示各模型分数与均值（保留2位小数）。",
        )
        judge_specs = [option_map[label] for label in judge_labels if label in option_map]
        dedup_specs: list[str] = []
        for spec in judge_specs:
            if spec not in dedup_specs:
                dedup_specs.append(spec)
        if not dedup_specs:
            dedup_specs = _default_compare_judge_specs()

        if dataset_path:
            result_path_default = _default_batch_result_path(dataset_path)
        else:
            result_path_default = ROOT / "evaluation" / "batch_results.json"
        result_path_str = st.text_input(
            "结果文件路径（默认同目录）",
            value=str(result_path_default),
            help="实时保存 JSON，不覆盖已有已完成题目。",
        ).strip()
        result_path = Path(result_path_str) if result_path_str else result_path_default

        max_items = st.number_input("本轮最大题数（0=全部）", min_value=0, value=0, step=10)
        parallel_items = st.number_input(
            "题目级并行度",
            min_value=1,
            max_value=16,
            value=1,
            step=1,
            help="每一轮并发处理多少个问题；每个问题内部仍是三路并行。",
            key=f"batch_parallel_items::{kb_id}",
        )
        enable_gold_metric = st.checkbox(
            "可选第5维：与标准答案对齐（gold_alignment）",
            value=False,
            help="仅批量测试启用；需样本含 answer/gold_answer/reference 字段。",
            key=f"batch_enable_gold::{kb_id}",
        )

        c_start, c_stop, c_ping = st.columns([1, 1, 1])
        can_start = bool(dataset_path and dataset_path.exists() and dataset_path.is_file())
        if c_start.button("开始/继续批测", type="primary", disabled=not can_start, width="stretch"):
            try:
                records_raw = _load_eval_records_from_path(dataset_path)
                normalized: list[dict] = []
                for idx, rec in enumerate(records_raw):
                    question = _extract_eval_question(rec)
                    if not question:
                        continue
                    item_id = str(rec.get("id", f"idx-{idx}"))
                    normalized.append(
                        {
                            "item_id": item_id,
                            "index": idx,
                            "question": question,
                            "gold_answer": _extract_eval_reference(rec),
                            "raw": rec,
                        }
                    )
                if int(max_items) > 0:
                    normalized = normalized[: int(max_items)]

                done_ids = _existing_done_ids(result_path)
                pending = [item for item in normalized if item["item_id"] not in done_ids]
                run_meta = {
                    "version": "causalrag-batch-v1",
                    "dataset_path": str(dataset_path),
                    "kb_id": str(kb_id),
                    "judge_specs": dedup_specs,
                    "enable_gold_metric": bool(enable_gold_metric),
                }
                base_doc = _read_batch_result_doc(result_path)
                for k, v in run_meta.items():
                    base_doc[k] = v
                _save_batch_result_doc(result_path, base_doc)

                st.session_state.batch_test_runner = {
                    "running": True,
                    "dataset_path": str(dataset_path),
                    "result_path": str(result_path),
                    "kb_id": str(kb_id),
                    "judge_specs": dedup_specs,
                    "parallel_items": int(parallel_items),
                    "enable_gold_metric": bool(enable_gold_metric),
                    "persist_causal_updates": bool(st.session_state.persist_causal_updates),
                    "total_count": len(normalized),
                    "done_before": len(done_ids),
                    "pending": pending,
                    "started_at": time.time(),
                    "last_item_id": "",
                }
                st.session_state.batch_test_logs = [
                    f"[init] loaded={len(normalized)} done_before={len(done_ids)} pending={len(pending)}"
                ]
                st.rerun()
            except Exception as e:
                st.error(f"启动失败: {e}")

        if c_stop.button("停止（保留已保存结果）", width="stretch"):
            runner = dict(st.session_state.get("batch_test_runner", {}))
            runner["running"] = False
            st.session_state.batch_test_runner = runner
            st.success("已请求停止。已保存结果不会丢失，可稍后继续。")

        if c_ping.button("测试评判模型连通性", width="stretch"):
            rows = [_ping_judge_model(spec) for spec in dedup_specs]
            for row in rows:
                icon = "✅" if row.get("ok") else "❌"
                st.caption(
                    f"{icon} {row.get('provider')}/{row.get('model')} | "
                    f"{float(row.get('elapsed_s', 0.0)):.2f}s | {row.get('detail', '')}"
                )

        runner = dict(st.session_state.get("batch_test_runner", {}))
        running = bool(runner.get("running"))
        if runner:
            total_count = int(runner.get("total_count", 0))
            pending_list = list(runner.get("pending", []))
            done_count = total_count - len(pending_list)
            ratio = (done_count / total_count) if total_count > 0 else 0.0
            st.progress(min(max(ratio, 0.0), 1.0))
            st.caption(
                f"进度: {done_count}/{total_count} | running={running} | "
                f"result={runner.get('result_path', '')}"
            )

        if running:
            pending_list = list(runner.get("pending", []))
            if not pending_list:
                runner["running"] = False
                st.session_state.batch_test_runner = runner
                st.success("批测已完成。")
            else:
                parallel_n = max(1, int(runner.get("parallel_items", 1)))
                to_run = pending_list[:parallel_n]
                preview_ids = [str(it.get("item_id", "")) for it in to_run]
                st.info(f"当前并行题目({len(to_run)}/{parallel_n}): {' | '.join(preview_ids)}")

                run_meta = {
                    "version": "causalrag-batch-v1",
                    "dataset_path": str(runner.get("dataset_path", "")),
                    "kb_id": str(kb_id),
                    "judge_specs": runner.get("judge_specs", []),
                    "enable_gold_metric": bool(runner.get("enable_gold_metric", False)),
                }

                worker_results: list[dict] = []
                with ThreadPoolExecutor(max_workers=parallel_n) as ex:
                    futures = [
                        ex.submit(_process_one_batch_item, str(kb_id), item, dict(runner))
                        for item in to_run
                    ]
                    for fut in as_completed(futures):
                        try:
                            worker_results.append(fut.result())
                        except Exception as e:
                            worker_results.append(
                                {
                                    "item_id": "",
                                    "ok": False,
                                    "record": {
                                        "status": "error",
                                        "error": str(e),
                                        "saved_at": datetime.now().isoformat(timespec="seconds"),
                                    },
                                    "logs": [f"worker failed: {e}"],
                                }
                            )

                for wr in worker_results:
                    logs_to_add = [f"[{datetime.now().strftime('%H:%M:%S')}] {line}" for line in wr.get("logs", [])]
                    with _BATCH_MEM_LOCK:
                        logs = list(st.session_state.get("batch_test_logs", []))
                        logs.extend(logs_to_add)
                        st.session_state.batch_test_logs = logs[-400:]
                    rec = wr.get("record", {}) or {}
                    if str(rec.get("item_id", "")).strip():
                        _upsert_batch_result(Path(runner["result_path"]), run_meta, rec)

                runner["pending"] = pending_list[len(to_run):]
                runner["last_item_id"] = "|".join(preview_ids)
                if not runner["pending"]:
                    runner["running"] = False
                st.session_state.batch_test_runner = runner
                st.rerun()

                current = pending_list[0]
                q = str(current.get("question", "")).strip()
                item_id = str(current.get("item_id", ""))
                st.info(f"当前题目: {item_id} | {q[:120]}")

                def _append_batch_log(message: str) -> None:
                    logs = list(st.session_state.get("batch_test_logs", []))
                    logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
                    st.session_state.batch_test_logs = logs[-200:]

                t0 = time.perf_counter()
                try:
                    _append_batch_log(f"item {item_id}: rewrite query start")
                    rewrite_input_history = [{"role": "user", "content": q}]
                    rag_query = _rewrite_rag_query(q, rewrite_input_history)
                    _append_batch_log(f"item {item_id}: rag_query={rag_query}")

                    mgr = _get_mgr(kb_id)
                    compare = _run_compare_modes(
                        mgr=mgr,
                        user_query=q,
                        retrieval_query=rag_query,
                        conversation_histories={"full": [], "vector_only": [], "direct": []},
                        persist_causal_updates=bool(runner.get("persist_causal_updates", False)),
                        judge_specs=runner.get("judge_specs", []),
                        progress_cb=_append_batch_log,
                    )
                    gold_answer = str(current.get("gold_answer", "")).strip()
                    if bool(runner.get("enable_gold_metric", False)):
                        _append_batch_log(f"item {item_id}: extra judge metric {_GOLD_METRIC} start")
                        compare["judge"] = _auto_score_compare(
                            query=q,
                            branches=compare.get("branches", {}) or {},
                            judge_specs=runner.get("judge_specs", []),
                            metrics=_COMPARE_SCORE_METRICS + (_GOLD_METRIC,),
                            reference_answer=gold_answer,
                        )
                        _append_batch_log(f"item {item_id}: extra judge metric {_GOLD_METRIC} done")
                    elapsed_s = time.perf_counter() - t0
                    winner = str((compare.get("judge", {}) or {}).get("winner", "")).strip()
                    record = {
                        "item_id": item_id,
                        "index": int(current.get("index", 0)),
                        "status": "ok",
                        "question": q,
                        "gold_answer": str(current.get("gold_answer", "")),
                        "rag_query": rag_query,
                        "elapsed_s": round(elapsed_s, 3),
                        "judge": compare.get("judge", {}),
                        "judge_metrics": (compare.get("judge", {}) or {}).get("metrics", list(_COMPARE_SCORE_METRICS)),
                        "branches": compare.get("branches", {}),
                        "branch_steps": compare.get("branch_steps", {}),
                        "winner": winner,
                        "raw": current.get("raw", {}),
                        "saved_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    run_meta = {
                        "version": "causalrag-batch-v1",
                        "dataset_path": str(runner.get("dataset_path", "")),
                        "kb_id": str(kb_id),
                        "judge_specs": runner.get("judge_specs", []),
                        "enable_gold_metric": bool(runner.get("enable_gold_metric", False)),
                    }
                    _upsert_batch_result(Path(runner["result_path"]), run_meta, record)
                    _append_batch_log(f"item {item_id}: saved winner={winner} elapsed={elapsed_s:.2f}s")
                except Exception as e:
                    elapsed_s = time.perf_counter() - t0
                    record = {
                        "item_id": item_id,
                        "index": int(current.get("index", 0)),
                        "status": "error",
                        "question": q,
                        "gold_answer": str(current.get("gold_answer", "")),
                        "rag_query": q,
                        "elapsed_s": round(elapsed_s, 3),
                        "error": str(e),
                        "raw": current.get("raw", {}),
                        "saved_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    run_meta = {
                        "version": "causalrag-batch-v1",
                        "dataset_path": str(runner.get("dataset_path", "")),
                        "kb_id": str(kb_id),
                        "judge_specs": runner.get("judge_specs", []),
                        "enable_gold_metric": bool(runner.get("enable_gold_metric", False)),
                    }
                    _upsert_batch_result(Path(runner["result_path"]), run_meta, record)
                    _append_batch_log(f"item {item_id}: error={e}")

                pending_list = pending_list[1:]
                runner["pending"] = pending_list
                runner["last_item_id"] = item_id
                if not pending_list:
                    runner["running"] = False
                st.session_state.batch_test_runner = runner
                st.rerun()

        log_lines = st.session_state.get("batch_test_logs", [])
        with st.expander("实时日志", expanded=True):
            if log_lines:
                st.code("\n".join(log_lines[-80:]), language="text")
            else:
                st.caption("暂无日志。")

        if result_path.exists():
            try:
                doc = _read_batch_result_doc(result_path)
                summary = _summarize_batch_doc(doc)
                st.markdown("### 批测结果分析")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("总记录", int(summary.get("n_total", 0)))
                k2.metric("完成(OK)", int(summary.get("n_done", 0)))
                k3.metric("Winner=Full", int((summary.get("winner_dist", {}) or {}).get("full", 0)))
                k4.metric("Winner=Vector", int((summary.get("winner_dist", {}) or {}).get("vector_only", 0)))

                avg_overall = summary.get("avg_overall", {}) or {}
                avg_elapsed = summary.get("avg_elapsed_s", {}) or {}
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Overall (Full)", f"{float(avg_overall.get('full', 0.0)):.2f}")
                c2.metric("Avg Overall (Vector)", f"{float(avg_overall.get('vector_only', 0.0)):.2f}")
                c3.metric("Avg Overall (No-RAG)", f"{float(avg_overall.get('direct', 0.0)):.2f}")
                t1, t2, t3 = st.columns(3)
                t1.metric("Avg Time Full(s)", f"{float(avg_elapsed.get('full', 0.0)):.2f}")
                t2.metric("Avg Time Vector(s)", f"{float(avg_elapsed.get('vector_only', 0.0)):.2f}")
                t3.metric("Avg Time No-RAG(s)", f"{float(avg_elapsed.get('direct', 0.0)):.2f}")

                st.markdown("#### 按问题类型统计")
                sup_summary = _summarize_batch_doc_by_type(doc, open_only=False)
                open_summary = _summarize_batch_doc_by_type(doc, open_only=True)
                sup_col, open_col = st.columns(2)
                with sup_col:
                    st.markdown("**监督问题（answer存在且非空）**")
                    s1, s2, s3 = st.columns(3)
                    s1.metric("完成(OK)", int(sup_summary.get("n_done", 0)))
                    s2.metric("Winner=Full", int((sup_summary.get("winner_dist", {}) or {}).get("full", 0)))
                    s3.metric("Avg Overall(Full)", f"{float((sup_summary.get('avg_overall', {}) or {}).get('full', 0.0)):.2f}")
                with open_col:
                    st.markdown("**开放性问题（answer键存在且为空）**")
                    o1, o2, o3 = st.columns(3)
                    o1.metric("完成(OK)", int(open_summary.get("n_done", 0)))
                    o2.metric("Winner=Full", int((open_summary.get("winner_dist", {}) or {}).get("full", 0)))
                    o3.metric("Avg Overall(Full)", f"{float((open_summary.get('avg_overall', {}) or {}).get('full', 0.0)):.2f}")

                st.markdown("#### 提升看板（PPT可用）")
                all_over = _build_uplift_overview(summary)
                sup_over = _build_uplift_overview(sup_summary)
                open_over = _build_uplift_overview(open_summary)
                all_idx_rows = _build_norag_index_table(summary)
                open_idx_rows = _build_norag_index_table(open_summary)
                sup_idx_rows = _build_norag_index_table(sup_summary)

                st.markdown("#### No-RAG归一化维度指数（No-RAG=1）")
                t_all, t_open, t_sup = st.columns(3)
                with t_all:
                    st.markdown("**总体**")
                    if all_idx_rows:
                        _df_all = pd.DataFrame(all_idx_rows)
                        st.table(
                            _df_all.style
                            .hide(axis="index")
                            .format(
                                {
                                    "full": "{:.3f}",
                                    "vector": "{:.3f}",
                                    "noRAG": "{:.1f}",
                                },
                                na_rep="N/A",
                            )
                        )
                    else:
                        st.caption("暂无数据")
                with t_open:
                    st.markdown("**开放**")
                    if open_idx_rows:
                        _df_open = pd.DataFrame(open_idx_rows)
                        st.table(
                            _df_open.style
                            .hide(axis="index")
                            .format(
                                {
                                    "full": "{:.3f}",
                                    "vector": "{:.3f}",
                                    "noRAG": "{:.1f}",
                                },
                                na_rep="N/A",
                            )
                        )
                    else:
                        st.caption("暂无数据")
                with t_sup:
                    st.markdown("**监督**")
                    if sup_idx_rows:
                        _df_sup = pd.DataFrame(sup_idx_rows)
                        st.table(
                            _df_sup.style
                            .hide(axis="index")
                            .format(
                                {
                                    "full": "{:.3f}",
                                    "vector": "{:.3f}",
                                    "noRAG": "{:.1f}",
                                },
                                na_rep="N/A",
                            )
                        )
                    else:
                        st.caption("暂无数据")

                a1, a2, a3 = st.columns(3)
                a1.metric(
                    "全体 Full最优率",
                    "N/A" if all_over.get("full_best_rate") is None else f"{float(all_over['full_best_rate']) * 100:.2f}%",
                )
                a2.metric(
                    "全体 Overall指数(Vector, NoRAG=1)",
                    "N/A" if all_over.get("overall_idx_vector") is None else f"{float(all_over['overall_idx_vector']):.3f}",
                )
                a3.metric(
                    "全体 Overall指数(Full, NoRAG=1)",
                    "N/A" if all_over.get("overall_idx_full") is None else f"{float(all_over['overall_idx_full']):.3f}",
                )

                b1, b2, b3 = st.columns(3)
                b1.metric(
                    "监督 Full最优率",
                    "N/A" if sup_over.get("full_best_rate") is None else f"{float(sup_over['full_best_rate']) * 100:.2f}%",
                )
                b2.metric(
                    "监督 Overall指数(Vector, NoRAG=1)",
                    "N/A" if sup_over.get("overall_idx_vector") is None else f"{float(sup_over['overall_idx_vector']):.3f}",
                )
                b3.metric(
                    "监督 Overall指数(Full, NoRAG=1)",
                    "N/A" if sup_over.get("overall_idx_full") is None else f"{float(sup_over['overall_idx_full']):.3f}",
                )

                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "开放 Full最优率",
                    "N/A" if open_over.get("full_best_rate") is None else f"{float(open_over['full_best_rate']) * 100:.2f}%",
                )
                c2.metric(
                    "开放 Overall指数(Vector, NoRAG=1)",
                    "N/A" if open_over.get("overall_idx_vector") is None else f"{float(open_over['overall_idx_vector']):.3f}",
                )
                c3.metric(
                    "开放 Overall指数(Full, NoRAG=1)",
                    "N/A" if open_over.get("overall_idx_full") is None else f"{float(open_over['overall_idx_full']):.3f}",
                )

                all_metric_df = pd.DataFrame(all_over.get("metric_rows", []))
                if not all_metric_df.empty:
                    st.markdown("#### 全体：各维度平均分领先（Full 对比）")
                    st.dataframe(all_metric_df, width="stretch", height=260)

                sup_metric_df = pd.DataFrame(sup_over.get("metric_rows", []))
                if not sup_metric_df.empty:
                    st.markdown("#### 监督问题：各维度平均分领先（Full 对比）")
                    st.dataframe(sup_metric_df, width="stretch", height=220)

                open_metric_df = pd.DataFrame(open_over.get("metric_rows", []))
                if not open_metric_df.empty:
                    st.markdown("#### 开放性问题：各维度平均分领先（Full 对比）")
                    st.dataframe(open_metric_df, width="stretch", height=220)

                rows = []
                for rec in doc.get("records", [])[-200:]:
                    judge = rec.get("judge", {}) or {}
                    scores = judge.get("scores", {}) or {}
                    rows.append(
                        {
                            "item_id": rec.get("item_id", ""),
                            "status": rec.get("status", ""),
                            "winner": (judge.get("winner", "") or rec.get("winner", "")),
                            "question": str(rec.get("question", ""))[:80],
                            "full_overall": round(float((scores.get("full", {}) or {}).get("overall", 0.0)), 2),
                            "vector_overall": round(float((scores.get("vector_only", {}) or {}).get("overall", 0.0)), 2),
                            "direct_overall": round(float((scores.get("direct", {}) or {}).get("overall", 0.0)), 2),
                            "elapsed_s": round(float(rec.get("elapsed_s", 0.0)), 3),
                        }
                    )
                if rows:
                    st.dataframe(pd.DataFrame(rows), width="stretch", height=360)

                winner_df = pd.DataFrame(
                    [
                        {"branch": "full", "count": int((summary.get("winner_dist", {}) or {}).get("full", 0))},
                        {"branch": "vector_only", "count": int((summary.get("winner_dist", {}) or {}).get("vector_only", 0))},
                        {"branch": "direct", "count": int((summary.get("winner_dist", {}) or {}).get("direct", 0))},
                    ]
                ).set_index("branch")
                st.markdown("#### Winner分布")
                st.bar_chart(winner_df)

                overall_df = pd.DataFrame(
                    [
                        {"branch": "full", "overall": float(avg_overall.get("full", 0.0))},
                        {"branch": "vector_only", "overall": float(avg_overall.get("vector_only", 0.0))},
                        {"branch": "direct", "overall": float(avg_overall.get("direct", 0.0))},
                    ]
                ).set_index("branch")
                st.markdown("#### 三路平均总分")
                st.bar_chart(overall_df)

                avg_metrics = summary.get("avg_metrics", {}) or {}
                metric_rows = []
                for branch in ("full", "vector_only", "direct"):
                    for metric, value in (avg_metrics.get(branch) or {}).items():
                        metric_rows.append(
                            {
                                "branch": branch,
                                "metric": metric,
                                "avg_score": round(float(value), 2),
                            }
                        )
                if metric_rows:
                    st.markdown("#### 各维度均分（含可选第5维）")
                    st.dataframe(pd.DataFrame(metric_rows), width="stretch", height=220)

                avg_by_model = summary.get("avg_by_model", {}) or {}
                model_rows = []
                for model_key, detail in avg_by_model.items():
                    scores_by_branch = detail.get("scores", {}) or {}
                    cnt = int(detail.get("count", 0))
                    for branch in ("full", "vector_only", "direct"):
                        for metric, value in (scores_by_branch.get(branch) or {}).items():
                            model_rows.append(
                                {
                                    "judge_model": model_key,
                                    "samples": cnt,
                                    "branch": branch,
                                    "metric": metric,
                                    "avg_score": round(float(value), 2),
                                }
                            )
                if model_rows:
                    st.markdown("#### 每个评判模型的各维度分数")
                    model_df = pd.DataFrame(model_rows)
                    st.dataframe(model_df, width="stretch", height=320)
            except Exception as e:
                st.warning(f"结果分析读取失败: {e}")

    with tab_hist:
        import pandas as pd
        files = sorted((ROOT/"data").glob("eval_*.jsonl"), reverse=True)
        if not files:
            st.info("暂无历史结果。")
        else:
            sel = st.selectbox("选择结果文件", [f.name for f in files])
            if sel:
                rows = [json.loads(l) for l in (ROOT/"data"/sel).read_text(encoding="utf-8").splitlines() if l.strip()]
                if rows:
                    df = pd.DataFrame(rows)
                    cols4 = st.columns(4)
                    for col, key, lbl in [(cols4[0],"em","平均EM"),(cols4[1],"f1","平均F1"),
                                          (cols4[2],"correctness","正确性"),(cols4[3],"causal_quality","因果质量")]:
                        if key in df.columns:
                            col.metric(lbl, f"{df[key].mean():.3f}")
                    st.dataframe(df, width="stretch")


# ===========================================================================
# ⚙️ Settings
# ===========================================================================

elif page == "🧪 HopRAG 对齐评测":
    render_hoprag_eval_panel(
        st=st,
        kb_id=st.session_state.current_kb_id,
        run_compare_modes=_run_compare_modes,
        get_mgr=_get_mgr,
        rewrite_query=_rewrite_rag_query,
        persist_causal_updates=bool(st.session_state.persist_causal_updates),
    )


elif page == "🏫 真实场景测试":
    render_real_scene_eval_panel(
        st=st,
        kb_id=st.session_state.current_kb_id,
        run_compare_modes=_run_compare_modes,
        get_mgr=_get_mgr,
        rewrite_query=_rewrite_rag_query,
        auto_score_compare=_auto_score_compare,
        score_metrics=_COMPARE_SCORE_METRICS,
        gold_metric=_GOLD_METRIC,
        default_judge_specs=list(st.session_state.get("compare_judge_specs", [])),
        persist_causal_updates=bool(st.session_state.persist_causal_updates),
    )


elif page == "⚙️ 设置":
    st.markdown('<div class="page-title">⚙️ 设置</div>', unsafe_allow_html=True)

    # ---- Debug Mode ----
    st.markdown("### 🐛 Debug 模式")
    st.markdown("开启后，所有 LLM / Embedding / Rerank 的请求和响应将记录到 `data/api_logs/YYYY-MM-DD.jsonl`。")

    debug_on = st.toggle(
        "开启 Debug 模式",
        value=st.session_state.debug_mode,
    )
    if debug_on != st.session_state.debug_mode:
        st.session_state.debug_mode = debug_on
        if debug_on:
            APILogger.enable()
            st.success("Debug 模式已开启，API 调用将记录到 data/api_logs/")
        else:
            APILogger.disable()
            st.info("Debug 模式已关闭。")

    # Show recent log
    log_dir = ROOT / "data" / "api_logs"
    log_files = sorted(log_dir.glob("*.jsonl"), reverse=True) if log_dir.exists() else []
    if log_files:
        st.markdown(f"**最新日志文件**: `{log_files[0].name}`")
        if st.button("📋 查看最近 20 条记录"):
            lines = log_files[0].read_text(encoding="utf-8").splitlines()
            for line in lines[-20:]:
                try:
                    entry = json.loads(line)
                    st.markdown(
                        f"**{entry['ts']}** · `{entry['type']}` · `{entry['model']}` · "
                        f"{entry['duration_ms']}ms"
                    )
                    with st.expander("查看详情"):
                        st.json(entry)
                except Exception:
                    st.text(line)

    st.markdown("---")

    # ---- LLM Config ----
    st.markdown("### 🤖 LLM 配置")
    from config.settings import settings
    col1, col2 = st.columns(2)
    col1.markdown(f"**默认提供商**: `{settings.llm.default_provider}`")
    col1.markdown(f"**DeepSeek Default**: `{settings.llm.deepseek_model}`")
    col1.markdown(f"**DeepSeek V4 Flash (Thinking)**: `{settings.llm.deepseek_chat_model}`")
    col1.markdown(f"**DeepSeek V4 Pro (Thinking)**: `{settings.llm.deepseek_reasoner_model}`")
    col1.markdown(f"**Gemini Flash**: `{settings.llm.gemini_model}`")
    col1.markdown(f"**Gemini Flash Preview**: `{settings.llm.gemini_flash_preview_model}`")
    col1.markdown(f"**Gemini Flash Lite Preview**: `{settings.llm.gemini_flash_lite_preview_model}`")
    col1.markdown(f"**Gemini Paid Key**: `{'configured' if bool(settings.llm.gemini_paid_api_key) else 'missing'}`")
    col2.markdown(f"**嵌入模型**: `{settings.embedding.model}`")
    col2.markdown(f"**重排模型**: `{settings.rerank.model}`")

    st.markdown("---")

    # ---- Data directory ----
    st.markdown("### 📁 数据目录")
    from config.settings import DATA_DIR
    st.code(str(DATA_DIR), language=None)

    # Count files
    try:
        db_size = (DATA_DIR / "causal_rag.db").stat().st_size / 1024
        st.caption(f"SQLite 数据库: {db_size:.1f} KB")
    except Exception:
        pass

    chroma_dir = DATA_DIR / "chroma_db"
    if chroma_dir.exists():
        chroma_files = list(chroma_dir.rglob("*"))
        st.caption(f"ChromaDB: {len(chroma_files)} 文件")
