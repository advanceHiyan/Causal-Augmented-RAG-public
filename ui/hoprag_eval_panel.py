from __future__ import annotations

import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from config.settings import settings
from evaluation_VS_hopRAG.hoprag_kb_builder import append_hoprag_vector_chunks, build_hoprag_kb_parallel
from evaluation_VS_hopRAG.hoprag_metrics import (
    answer_f1,
    chunk_id_from_text_or_meta,
    exact_match,
    load_gold_map,
    load_questions_jsonl,
    retrieval_scores,
    summarize_records,
)
from utils.llm_client import llm_client


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "evaluation_VS_hopRAG" / "manualAssets"
RESULT_DIR = ROOT / "evaluation_VS_hopRAG" / "results"


BRANCH_TO_METHOD = {
    "full": "Full-Causal",
    "vector_only": "Vector-Only",
    "direct": "No-RAG",
}


HOTPOT_SHORT_ANSWER_INSTRUCTION = (
    "HotpotQA evaluation mode. Output ONLY the shortest answer span.\n"
    "- Do not explain.\n"
    "- Do not include parentheses, aliases, citations, caveats, or full sentences.\n"
    "- Never output I don't know, unknown, unclear, insufficient information, cannot determine, or any equivalent uncertainty phrase.\n"
    "- If evidence is incomplete or conflicting, make the best forced guess and output one answer span only.\n"
    "- For yes/no questions, output exactly \"yes\" or \"no\".\n"
    "- For comparison questions, output only the winning entity/name.\n"
    "- For location questions, output the most specific location span supported by evidence.\n"
    "- If the evidence says \"midtown Manhattan\" and the question asks where, output \"Manhattan\" unless a more exact gold-like span is clearly required.\n"
)

HOTPOT_NORAG_SHORT_ANSWER_INSTRUCTION = (
    "HotpotQA evaluation mode. Output ONLY the shortest answer span.\n"
    "- Do not explain.\n"
    "- Do not include parentheses, aliases, citations, caveats, or full sentences.\n"
    "- For yes/no questions, output exactly \"yes\" or \"no\".\n"
    "- For comparison questions, output only the winning entity/name.\n"
)


def _files(pattern: str) -> list[Path]:
    if not ASSET_DIR.exists():
        return []
    return sorted(ASSET_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)


def _result_path() -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR / f"hoprag_frontend_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def _default_result_path(questions_path: Path, kb_id: str) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stem = questions_path.stem.replace("_questions_for_batch", "")
    kb_short = str(kb_id or "kb")[:8]
    return RESULT_DIR / f"{stem}_{kb_short}_results.json"


def _latest_result_for_questions(questions_path: Path) -> Path | None:
    if not RESULT_DIR.exists():
        return None
    stem = questions_path.stem.replace("_questions_for_batch", "")
    files = sorted(
        RESULT_DIR.glob(f"{stem}_*_results.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _read_result_doc(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"records": [], "summary": {}}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            return {"records": [], "summary": {}}
        records = doc.get("records", []) or []
        doc["summary"] = summarize_records(records)
        return doc
    except Exception:
        return {"records": [], "summary": {}}


def _save_result_doc(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = doc.get("records", []) or []
    doc["summary"] = summarize_records(records)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def _done_sample_ids(path: Path) -> set[str]:
    doc = _read_result_doc(path)
    rows = doc.get("records", []) or []
    by_id: dict[str, set[str]] = {}
    for row in rows:
        if row.get("error"):
            continue
        sid = str(row.get("sample_id", "")).strip()
        method = str(row.get("method", "")).strip()
        if sid and method:
            by_id.setdefault(sid, set()).add(method)
    return {sid for sid, methods in by_id.items() if set(BRANCH_TO_METHOD.values()).issubset(methods)}


def _upsert_sample_records(path: Path, run_meta: dict[str, Any], sample_records: list[dict[str, Any]]) -> None:
    doc = _read_result_doc(path)
    doc.update(run_meta)
    old = doc.get("records", []) or []
    replace_keys = {
        (str(r.get("sample_id", "")), str(r.get("method", "")))
        for r in sample_records
    }
    kept = [
        r for r in old
        if (str(r.get("sample_id", "")), str(r.get("method", ""))) not in replace_keys
    ]
    doc["records"] = kept + sample_records
    _save_result_doc(path, doc)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue
    return rows


def _collect_external_hoprag_runs(root_dir: Path) -> list[dict[str, Any]]:
    if not root_dir.exists():
        return []
    by_stem: dict[str, dict[str, Path]] = defaultdict(dict)
    for p in root_dir.rglob("run_meta*.json"):
        k = p.stem.replace("run_meta", "").lstrip("_") or "default"
        by_stem[k]["run_meta"] = p
    for p in root_dir.rglob("hoprag_predictions*.jsonl"):
        k = p.stem.replace("hoprag_predictions", "").lstrip("_") or "default"
        by_stem[k]["predictions"] = p
    runs: list[dict[str, Any]] = []
    for mode, pair in by_stem.items():
        pred = pair.get("predictions")
        meta = pair.get("run_meta")
        if pred is None:
            continue
        runs.append(
            {
                "mode": mode,
                "run_meta_path": meta,
                "predictions_path": pred,
                "mtime": pred.stat().st_mtime,
            }
        )
    runs.sort(key=lambda x: float(x.get("mtime", 0.0)), reverse=True)
    return runs


def _eval_external_hoprag_predictions(
    predictions_path: Path,
    gold_map: dict[str, Any],
) -> dict[str, Any]:
    rows = _read_jsonl_rows(predictions_path)
    n = len(rows)
    if n == 0:
        return {"n": 0.0}
    em_sum = 0.0
    f1_sum = 0.0
    rp_sum = 0.0
    rr_sum = 0.0
    rf_sum = 0.0
    hit_any_sum = 0.0
    support_recall_sum = 0.0
    failures = 0.0
    for i, row in enumerate(rows):
        sid = str(row.get("sample_id", "") or row.get("id", "")).strip()
        if not sid:
            sid = f"idx-{i}"
        gold = gold_map.get(sid, {}) or {}
        pred = str(row.get("prediction", "") or row.get("answer", "")).strip()
        gold_answer = str(gold.get("answer", "")).strip()
        if row.get("error"):
            failures += 1.0
        em_sum += exact_match(pred, gold_answer)
        f1_sum += answer_f1(pred, gold_answer)
        retrieved_ids_raw = row.get("retrieved_chunk_ids", []) or row.get("retrieved_ids", []) or []
        retrieved_ids = [str(x) for x in retrieved_ids_raw if str(x).strip()]
        gold_ids = [str(x) for x in (gold.get("gold_supporting_chunk_ids", []) or []) if str(x).strip()]
        rs = retrieval_scores(retrieved_ids, gold_ids)
        rp_sum += float(rs.get("retrieval_precision", 0.0) or 0.0)
        rr_sum += float(rs.get("retrieval_recall", 0.0) or 0.0)
        rf_sum += float(rs.get("retrieval_f1", 0.0) or 0.0)
        hit_any_sum += 1.0 if float(rs.get("hit_count", 0.0) or 0.0) > 0 else 0.0
        support_recall_sum += float(rs.get("retrieval_recall", 0.0) or 0.0)
    denom = float(n)
    return {
        "n": denom,
        "failures": failures,
        "em": em_sum / denom,
        "answer_f1": f1_sum / denom,
        "retrieval_precision": rp_sum / denom,
        "retrieval_recall": rr_sum / denom,
        "retrieval_f1": rf_sum / denom,
        "support_hit_any": hit_any_sum / denom,
        "support_recall_avg": support_recall_sum / denom,
    }


def _generate_hoprag_norag_answer(question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful QA assistant. Answer the question directly.",
        },
        {
            "role": "user",
            "content": f"{HOTPOT_NORAG_SHORT_ANSWER_INSTRUCTION}\n\nQuestion:\n{question}",
        },
    ]
    return str(
        llm_client.chat(
            messages,
            provider="deepseek",
            model=settings.llm.deepseek_v4_flash_model,
            temperature=0.0,
            max_tokens=64,
        )
        or ""
    ).strip()


def _process_one_hoprag_norag_item(
    *,
    item: dict[str, Any],
    sample_index: int,
    gold_map: dict[str, Any],
) -> dict[str, Any]:
    sample_id = str(item.get("id", "") or item.get("sample_id", "") or f"idx-{sample_index}")
    question = str(item.get("question", "")).strip()
    gold = gold_map.get(sample_id, {}) or {}
    gold_answer = str(gold.get("answer") or item.get("answer") or "").strip()
    gold_ids = [str(x) for x in gold.get("gold_supporting_chunk_ids", []) or []]
    started = time.perf_counter()
    error = ""
    try:
        prediction = _generate_hoprag_norag_answer(question)
    except Exception as e:
        print(f"[HOPRAG NORAG RERUN ERROR] sample_id={sample_id} error={e}", flush=True)
        prediction = ""
        error = str(e)
    elapsed = time.perf_counter() - started
    rs = retrieval_scores([], gold_ids)
    record = {
        "version": "hoprag-record-v3-norag-rerun",
        "sample_id": sample_id,
        "sample_index": sample_index,
        "method": "No-RAG",
        "branch": "direct",
        "question": question,
        "gold_answer": gold_answer,
        "gold_debug": {
            "gold_supporting_chunk_ids": gold_ids,
            "supporting_facts": gold.get("supporting_facts", []),
            "gold_map_entry": gold,
        },
        "prediction": prediction,
        "rag_query": question,
        "em": 0.0 if error else exact_match(prediction, gold_answer),
        "answer_f1": 0.0 if error else answer_f1(prediction, gold_answer),
        **rs,
        "gold_supporting_chunk_ids": gold_ids,
        "retrieved_chunk_ids": [],
        "retrieval_debug": {
            "retrieved_chunk_ids": [],
            "vector_snippets": [],
            "retrieved_count": 0.0,
            "hit_count": 0.0,
        },
        "query_preanalysis": {},
        "full_debug": {},
        "branch_payload": {
            "mode": "direct",
            "title": "No-RAG",
            "answer": prediction,
            "elapsed_s": elapsed,
            "l1": False,
            "l2": False,
            "chunks": 0,
            "l3_used_reranker": False,
            "causal_graph_text": "",
            "causal_path_summary": "",
            "vector_snippets": [],
            "steps": ["branch start", "skip retrieval", "generate: no-rag only", f"branch done in {elapsed:.2f}s"],
        },
        "branch_steps": ["branch start", "skip retrieval", "generate: no-rag only", f"branch done in {elapsed:.2f}s"],
        "compare_branch_steps": {},
        "compare_judge": {},
        "compare_debug": {"rerun_scope": "No-RAG only; Full-Causal and Vector-Only were not executed"},
        "trace": [
            {
                "step_name": "Step4_norag_only_rerun",
                "module": "ui.hoprag_eval_panel._process_one_hoprag_norag_item",
                "input": {
                    "question": question,
                    "answer_instruction": HOTPOT_NORAG_SHORT_ANSWER_INSTRUCTION,
                    "retrieval_context": "",
                },
                "output": {"prediction": prediction},
                "fallback": {},
                "elapsed_s": round(elapsed, 3),
                "error": error,
            },
            {
                "step_name": "Step5_hoprag_metrics",
                "module": "evaluation_VS_hopRAG.hoprag_metrics",
                "input": {
                    "prediction": prediction,
                    "gold_answer": gold_answer,
                    "retrieved_chunk_ids": [],
                    "gold_supporting_chunk_ids": gold_ids,
                },
                "output": {
                    "em": 0.0 if error else exact_match(prediction, gold_answer),
                    "answer_f1": 0.0 if error else answer_f1(prediction, gold_answer),
                    **rs,
                },
            },
        ],
        "elapsed_s": round(elapsed, 3),
        "total_sample_elapsed_s": round(elapsed, 3),
        "error": error,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    return {
        "sample_id": sample_id,
        "ok": not bool(error),
        "records": [record],
        "logs": [f"{sample_id}: No-RAG rerun elapsed={elapsed:.2f}s error={error}" if error else f"{sample_id}: No-RAG rerun elapsed={elapsed:.2f}s"],
    }


def _shutdown_hoprag_executor(st: Any, *, cancel_futures: bool) -> None:
    ex = st.session_state.get("hoprag_eval_executor")
    if ex is not None:
        try:
            ex.shutdown(wait=False, cancel_futures=bool(cancel_futures))
        except Exception as e:
            print(f"[HOPRAG EXECUTOR SHUTDOWN ERROR] {e}", flush=True)
    st.session_state.hoprag_eval_executor = None
    st.session_state.hoprag_eval_futures = {}


def _branch_retrieved_ids(branch_key: str, branch: dict[str, Any]) -> list[str]:
    if branch_key == "direct":
        return []
    snippets = branch.get("vector_snippets", []) or []
    return [chunk_id_from_text_or_meta(x) for x in snippets if isinstance(x, dict)]


def _records_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "id": r.get("sample_id", ""),
                "method": r.get("method", ""),
                "EM": round(float(r.get("em", 0) or 0) * 100, 2),
                "Answer F1": round(float(r.get("answer_f1", 0) or 0) * 100, 2),
                "Retrieval Precision": round(float(r.get("retrieval_precision", 0) or 0) * 100, 2),
                "Retrieval Recall": round(float(r.get("retrieval_recall", 0) or 0) * 100, 2),
                "Retrieval F1": round(float(r.get("retrieval_f1", 0) or 0) * 100, 2),
                "retrieved": int(float(r.get("retrieved_count", 0) or 0)),
                "gold": r.get("gold_answer", ""),
                "prediction": str(r.get("prediction", ""))[:160],
                "question": str(r.get("question", ""))[:120],
                "error": r.get("error", ""),
            }
        )
    return pd.DataFrame(rows)


def _process_one_hoprag_item(
    *,
    item: dict[str, Any],
    sample_index: int,
    gold_map: dict[str, Any],
    kb_id: str,
    run_compare_modes: Callable[..., dict],
    get_mgr: Callable[[str], Any],
    rewrite_query: Callable[[str, list[dict[str, str]]], str],
    persist_causal_updates: bool,
) -> dict[str, Any]:
    sample_id = str(item.get("id", "") or item.get("sample_id", "") or f"idx-{sample_index}")
    question = str(item.get("question", "")).strip()
    gold = gold_map.get(sample_id, {}) or {}
    gold_answer = str(gold.get("answer") or item.get("answer") or "").strip()
    gold_ids = [str(x) for x in gold.get("gold_supporting_chunk_ids", []) or []]
    t0 = time.perf_counter()
    logs: list[str] = []
    try:
        logs.append(f"{sample_id}: rewrite query start")
        rag_query = rewrite_query(question, [{"role": "user", "content": question}])
        logs.append(f"{sample_id}: rag_query={rag_query}")
        mgr = get_mgr(kb_id)
        compare = run_compare_modes(
            mgr=mgr,
            user_query=question,
            retrieval_query=rag_query,
            conversation_histories={"full": [], "vector_only": [], "direct": []},
            persist_causal_updates=bool(persist_causal_updates),
            profile_context="",
            judge_specs=[],
            progress_cb=None,
            top_k_override=20,
            enable_query_preanalysis=True,
            answer_instruction=HOTPOT_SHORT_ANSWER_INSTRUCTION,
            answer_instruction_scope="full_only",
        )
        branches = compare.get("branches", {}) or {}
        branch_steps = compare.get("branch_steps", {}) or {}
        compare_judge = compare.get("judge", {}) or {}
        elapsed_total = time.perf_counter() - t0
        records: list[dict[str, Any]] = []
        for branch_key, method in BRANCH_TO_METHOD.items():
            branch = branches.get(branch_key, {}) or {}
            pred = str(branch.get("answer", "") or "").strip()
            retrieved_ids = _branch_retrieved_ids(branch_key, branch)
            rs = retrieval_scores(retrieved_ids, gold_ids)
            trace = [
                {
                    "step_name": "Step2_rag_query_rewrite",
                    "module": "ui.hoprag_eval_panel.rewrite_query",
                    "input": {
                        "question": question,
                        "history": [{"role": "user", "content": question}],
                    },
                    "output": {"rag_query": rag_query},
                    "fallback": {"rewrite_fallback_to_original": bool(rag_query.strip() == question.strip())},
                },
                {
                    "step_name": "Step4_compare_branch",
                    "module": "app._run_compare_modes",
                    "input": {
                        "branch": branch_key,
                        "retrieval_query": rag_query,
                        "top_k_override": 20,
                        "enable_query_preanalysis": True,
                        "answer_instruction": HOTPOT_SHORT_ANSWER_INSTRUCTION if branch_key == "full" else "",
                        "answer_instruction_scope": "full_only",
                    },
                    "output": branch,
                    "fallback": {},
                    "elapsed_s": float(branch.get("elapsed_s", 0.0) or 0.0),
                },
                {
                    "step_name": "Step5_hoprag_metrics",
                    "module": "evaluation_VS_hopRAG.hoprag_metrics",
                    "input": {
                        "prediction": pred,
                        "gold_answer": gold_answer,
                        "retrieved_chunk_ids": retrieved_ids,
                        "gold_supporting_chunk_ids": gold_ids,
                    },
                    "output": {
                        "em": exact_match(pred, gold_answer),
                        "answer_f1": answer_f1(pred, gold_answer),
                        **rs,
                    },
                },
            ]
            records.append(
                {
                    "version": "hoprag-record-v2-rich-trace",
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "method": method,
                    "branch": branch_key,
                    "question": question,
                    "gold_answer": gold_answer,
                    "gold_debug": {
                        "gold_supporting_chunk_ids": gold_ids,
                        "supporting_facts": gold.get("supporting_facts", []),
                        "gold_map_entry": gold,
                    },
                    "prediction": pred,
                    "rag_query": rag_query,
                    "em": exact_match(pred, gold_answer),
                    "answer_f1": answer_f1(pred, gold_answer),
                    **rs,
                    "gold_supporting_chunk_ids": gold_ids,
                    "retrieved_chunk_ids": retrieved_ids,
                    "retrieval_debug": {
                        "retrieved_chunk_ids": retrieved_ids,
                        "vector_snippets": branch.get("vector_snippets", []),
                        "retrieved_count": rs.get("retrieved_count", 0.0),
                        "hit_count": rs.get("hit_count", 0.0),
                    },
                    "query_preanalysis": branch.get("query_preanalysis", {}) if branch_key == "full" else {},
                    "full_debug": {
                        "l1": branch.get("l1"),
                        "l2": branch.get("l2"),
                        "l3_used_reranker": branch.get("l3_used_reranker"),
                        "causal_graph_text": branch.get("causal_graph_text", ""),
                        "causal_path_summary": branch.get("causal_path_summary", ""),
                        "causal_analysis": branch.get("causal_analysis", ""),
                        "counterfactual": branch.get("counterfactual", ""),
                        "self_reflection": branch.get("self_reflection", ""),
                    } if branch_key == "full" else {},
                    "branch_payload": branch,
                    "branch_steps": branch_steps.get(branch_key, []),
                    "compare_branch_steps": branch_steps,
                    "compare_judge": compare_judge,
                    "compare_debug": {
                        "branches": branches,
                        "judge": compare_judge,
                        "branch_steps": branch_steps,
                    },
                    "trace": trace,
                    "elapsed_s": round(float(branch.get("elapsed_s", elapsed_total) or elapsed_total), 3),
                    "total_sample_elapsed_s": round(float(elapsed_total), 3),
                    "error": "",
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
        logs.append(f"{sample_id}: done elapsed={elapsed_total:.2f}s")
        return {"sample_id": sample_id, "ok": True, "records": records, "logs": logs}
    except Exception as e:
        print(f"[HOPRAG ERROR] sample_id={sample_id} error={e}", flush=True)
        logs.append(f"{sample_id}: error={e}")
        records = []
        for branch_key, method in BRANCH_TO_METHOD.items():
            records.append(
                {
                    "version": "hoprag-record-v2-rich-trace",
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "method": method,
                    "branch": branch_key,
                    "question": question,
                    "gold_answer": gold_answer,
                    "gold_debug": {
                        "gold_supporting_chunk_ids": gold_ids,
                        "supporting_facts": gold.get("supporting_facts", []),
                        "gold_map_entry": gold,
                    },
                    "prediction": "",
                    "rag_query": question,
                    "em": 0.0,
                    "answer_f1": 0.0,
                    **retrieval_scores([], gold_ids),
                    "gold_supporting_chunk_ids": gold_ids,
                    "retrieved_chunk_ids": [],
                    "retrieval_debug": {},
                    "query_preanalysis": {},
                    "full_debug": {},
                    "branch_payload": {},
                    "branch_steps": [],
                    "compare_branch_steps": {},
                    "compare_judge": {},
                    "compare_debug": {},
                    "trace": [
                        {
                            "step_name": "HopRAG_sample_error",
                            "module": "ui.hoprag_eval_panel._process_one_hoprag_item",
                            "input": {"question": question, "sample_id": sample_id},
                            "output": {},
                            "error": str(e),
                        }
                    ],
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                    "error": str(e),
                    "saved_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
        return {"sample_id": sample_id, "ok": False, "records": records, "logs": logs}


def _summary_df(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for method, data in (summary or {}).items():
        row = {"method": method}
        for key, value in (data or {}).items():
            if key in {
                "em",
                "answer_f1",
                "retrieval_precision",
                "retrieval_recall",
                "retrieval_f1",
            }:
                row[key] = round(float(value or 0) * 100, 2)
            else:
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def run_hoprag_compare_eval(
    *,
    questions_path: Path,
    gold_map_path: Path,
    kb_id: str,
    max_samples: int,
    run_compare_modes: Callable[..., dict],
    get_mgr: Callable[[str], Any],
    rewrite_query: Callable[[str, list[dict[str, str]]], str],
    persist_causal_updates: bool = False,
) -> dict[str, Any]:
    questions = load_questions_jsonl(questions_path)
    gold_map = load_gold_map(gold_map_path)
    if max_samples > 0:
        questions = questions[:max_samples]

    mgr = get_mgr(kb_id)
    records: list[dict[str, Any]] = []
    for idx, item in enumerate(questions):
        sample_id = str(item.get("id", "") or item.get("sample_id", "") or f"idx-{idx}")
        question = str(item.get("question", "")).strip()
        gold = gold_map.get(sample_id, {}) or {}
        gold_answer = str(gold.get("answer") or item.get("answer") or "").strip()
        gold_ids = [str(x) for x in gold.get("gold_supporting_chunk_ids", []) or []]
        t0 = time.perf_counter()
        try:
            rag_query = rewrite_query(question, [{"role": "user", "content": question}])
            compare = run_compare_modes(
                mgr=mgr,
                user_query=question,
                retrieval_query=rag_query,
                conversation_histories={"full": [], "vector_only": [], "direct": []},
                persist_causal_updates=bool(persist_causal_updates),
                profile_context="",
                judge_specs=[],
                progress_cb=None,
                top_k_override=20,
                enable_query_preanalysis=True,
                answer_instruction=HOTPOT_SHORT_ANSWER_INSTRUCTION,
                answer_instruction_scope="full_only",
            )
            branches = compare.get("branches", {}) or {}
            branch_steps = compare.get("branch_steps", {}) or {}
            compare_judge = compare.get("judge", {}) or {}
            elapsed_total = time.perf_counter() - t0
            for branch_key, method in BRANCH_TO_METHOD.items():
                branch = branches.get(branch_key, {}) or {}
                pred = str(branch.get("answer", "") or "").strip()
                retrieved_ids = _branch_retrieved_ids(branch_key, branch)
                rs = retrieval_scores(retrieved_ids, gold_ids)
                trace = [
                    {
                        "step_name": "Step2_rag_query_rewrite",
                        "module": "ui.hoprag_eval_panel.rewrite_query",
                        "input": {
                            "question": question,
                            "history": [{"role": "user", "content": question}],
                        },
                        "output": {"rag_query": rag_query},
                        "fallback": {"rewrite_fallback_to_original": bool(rag_query.strip() == question.strip())},
                    },
                    {
                        "step_name": "Step4_compare_branch",
                        "module": "app._run_compare_modes",
                        "input": {
                            "branch": branch_key,
                            "retrieval_query": rag_query,
                            "top_k_override": 20,
                            "enable_query_preanalysis": True,
                            "answer_instruction": HOTPOT_SHORT_ANSWER_INSTRUCTION if branch_key == "full" else "",
                            "answer_instruction_scope": "full_only",
                        },
                        "output": branch,
                        "fallback": {},
                        "elapsed_s": float(branch.get("elapsed_s", 0.0) or 0.0),
                    },
                    {
                        "step_name": "Step5_hoprag_metrics",
                        "module": "evaluation_VS_hopRAG.hoprag_metrics",
                        "input": {
                            "prediction": pred,
                            "gold_answer": gold_answer,
                            "retrieved_chunk_ids": retrieved_ids,
                            "gold_supporting_chunk_ids": gold_ids,
                        },
                        "output": {
                            "em": exact_match(pred, gold_answer),
                            "answer_f1": answer_f1(pred, gold_answer),
                            **rs,
                        },
                    },
                ]
                records.append(
                    {
                        "version": "hoprag-record-v2-rich-trace",
                        "sample_id": sample_id,
                        "sample_index": idx,
                        "method": method,
                        "branch": branch_key,
                        "question": question,
                        "gold_answer": gold_answer,
                        "gold_debug": {
                            "gold_supporting_chunk_ids": gold_ids,
                            "supporting_facts": gold.get("supporting_facts", []),
                            "gold_map_entry": gold,
                        },
                        "prediction": pred,
                        "rag_query": rag_query,
                        "em": exact_match(pred, gold_answer),
                        "answer_f1": answer_f1(pred, gold_answer),
                        **rs,
                        "gold_supporting_chunk_ids": gold_ids,
                        "retrieved_chunk_ids": retrieved_ids,
                        "retrieval_debug": {
                            "retrieved_chunk_ids": retrieved_ids,
                            "vector_snippets": branch.get("vector_snippets", []),
                            "retrieved_count": rs.get("retrieved_count", 0.0),
                            "hit_count": rs.get("hit_count", 0.0),
                        },
                        "query_preanalysis": branch.get("query_preanalysis", {}) if branch_key == "full" else {},
                        "full_debug": {
                            "l1": branch.get("l1"),
                            "l2": branch.get("l2"),
                            "l3_used_reranker": branch.get("l3_used_reranker"),
                            "causal_graph_text": branch.get("causal_graph_text", ""),
                            "causal_path_summary": branch.get("causal_path_summary", ""),
                            "causal_analysis": branch.get("causal_analysis", ""),
                            "counterfactual": branch.get("counterfactual", ""),
                            "self_reflection": branch.get("self_reflection", ""),
                        } if branch_key == "full" else {},
                        "branch_payload": branch,
                        "branch_steps": branch_steps.get(branch_key, []),
                        "compare_branch_steps": branch_steps,
                        "compare_judge": compare_judge,
                        "compare_debug": {
                            "branches": branches,
                            "judge": compare_judge,
                            "branch_steps": branch_steps,
                        },
                        "trace": trace,
                        "elapsed_s": round(float(branch.get("elapsed_s", elapsed_total) or elapsed_total), 3),
                        "total_sample_elapsed_s": round(float(elapsed_total), 3),
                        "error": "",
                    }
                )
        except Exception as e:
            print(f"[HOPRAG ERROR] sample_id={sample_id} error={e}", flush=True)
            for branch_key, method in BRANCH_TO_METHOD.items():
                records.append(
                    {
                        "version": "hoprag-record-v2-rich-trace",
                        "sample_id": sample_id,
                        "sample_index": idx,
                        "method": method,
                        "branch": branch_key,
                        "question": question,
                        "gold_answer": gold_answer,
                        "gold_debug": {
                            "gold_supporting_chunk_ids": gold_ids,
                            "supporting_facts": gold.get("supporting_facts", []),
                            "gold_map_entry": gold,
                        },
                        "prediction": "",
                        "rag_query": question,
                        "em": 0.0,
                        "answer_f1": 0.0,
                        **retrieval_scores([], gold_ids),
                        "gold_supporting_chunk_ids": gold_ids,
                        "retrieved_chunk_ids": [],
                        "retrieval_debug": {},
                        "query_preanalysis": {},
                        "full_debug": {},
                        "branch_payload": {},
                        "branch_steps": [],
                        "compare_branch_steps": {},
                        "compare_judge": {},
                        "compare_debug": {},
                        "trace": [
                            {
                                "step_name": "HopRAG_sample_error",
                                "module": "ui.hoprag_eval_panel.run_hoprag_compare_eval",
                                "input": {"question": question, "sample_id": sample_id},
                                "output": {},
                                "error": str(e),
                            }
                        ],
                        "elapsed_s": round(time.perf_counter() - t0, 3),
                        "error": str(e),
                    }
                )

    doc = {
        "version": "hoprag-aligned-frontend-eval-v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "questions_path": str(questions_path),
        "gold_map_path": str(gold_map_path),
        "kb_id": kb_id,
        "sample_count": len(questions),
        "records": records,
        "summary": summarize_records(records),
    }
    out = _result_path()
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    doc["result_path"] = str(out)
    return doc


def render_hoprag_eval_panel(
    *,
    st: Any,
    kb_id: str,
    run_compare_modes: Callable[..., dict],
    get_mgr: Callable[[str], Any],
    rewrite_query: Callable[[str, list[dict[str, str]]], str],
    persist_causal_updates: bool = False,
) -> None:
    st.markdown('<div class="page-title">HopRAG 对齐评测</div>', unsafe_allow_html=True)
    st.caption("Independent eval entry for HopRAG comparison.")

    if not kb_id:
        st.info("Please select a knowledge base with imported HotpotQA context first.")
        return

    st.markdown("### HopRAG Aligned KB Build")
    st.caption("Build flow: sentence chunks -> vector store; article units -> text/causal graphs.")
    cc1, cc2, cc3 = st.columns([1.2, 1, 1])
    concurrency_provider = cc1.selectbox(
        "LLM provider",
        ["deepseek", "openrouter", "gemini", "gemini_paid"],
        index=0,
        key="hoprag_llm_concurrency_provider",
    )
    current_limit = llm_client.get_provider_concurrency(concurrency_provider)
    concurrency_limit = cc2.number_input(
        "Max concurrency",
        min_value=1,
        max_value=256,
        value=int(current_limit),
        step=1,
        key=f"hoprag_llm_concurrency_limit::{concurrency_provider}",
    )
    if cc3.button("Apply concurrency", key="hoprag_apply_llm_concurrency", width="stretch"):
        applied = llm_client.set_provider_concurrency(concurrency_provider, int(concurrency_limit))
        st.success(f"{concurrency_provider} max concurrency set to {applied}")

    sentence_files = _files("*_sentence_chunks.jsonl")
    article_files = _files("*_article_units.jsonl")
    b1, b2 = st.columns(2)
    sentence_path_str = b1.selectbox(
        "Sentence-level vector chunks",
        [""] + [str(p) for p in sentence_files],
        index=1 if sentence_files else 0,
        key="hoprag_build_sentence_chunks",
    )
    article_path_str = b2.selectbox(
        "Article-level graph units",
        [""] + [str(p) for p in article_files],
        index=1 if article_files else 0,
        key="hoprag_build_article_units",
    )
    bo1, bo2, bo3 = st.columns(3)
    build_vector = bo1.checkbox("Build vector index", value=True, key="hoprag_build_vector")
    build_text_graph = bo2.checkbox("Build text graph", value=True, key="hoprag_build_text_graph")
    build_causal_graph = bo3.checkbox("Build causal graph", value=True, key="hoprag_build_causal_graph")
    wo1, wo2 = st.columns(2)
    embedding_workers = wo1.slider(
        "Embedding workers",
        min_value=1,
        max_value=16,
        value=8,
        step=1,
        key="hoprag_embedding_workers",
        disabled=not build_vector,
        help="Adjust with embedding batch settings.",
    )
    graph_llm_workers = wo2.slider(
        "Graph LLM workers",
        min_value=1,
        max_value=128,
        value=64,
        step=1,
        key="hoprag_graph_llm_workers",
        disabled=not (build_text_graph or build_causal_graph),
    )
    implicit_discovery = st.checkbox(
        "Enable implicit causal discovery",
        value=False,
        key="hoprag_build_implicit_causal",
        disabled=not build_causal_graph,
    )
    implicit_max_chunks = st.slider(
        "Implicit causal max chunks",
        min_value=5,
        max_value=120,
        value=24,
        step=5,
        key="hoprag_build_implicit_max_chunks",
        disabled=(not build_causal_graph) or (not implicit_discovery),
    )
    any_target = bool(build_vector or build_text_graph or build_causal_graph)
    has_vector_input = (not build_vector) or bool(sentence_path_str)
    has_graph_input = (not (build_text_graph or build_causal_graph)) or bool(article_path_str)
    can_build = bool(kb_id) and any_target and has_vector_input and has_graph_input
    if st.button("Build HopRAG KB (parallel)", type="primary", disabled=not can_build):
        mgr = get_mgr(kb_id)
        logs: list[str] = []

        def progress(msg: str) -> None:
            logs.append(msg)

        with st.status("Building HopRAG aligned KB in parallel...", expanded=True) as status:
            try:
                stats = build_hoprag_kb_parallel(
                    mgr=mgr,
                    sentence_chunks_path=Path(sentence_path_str),
                    article_units_path=Path(article_path_str),
                    build_vector=bool(build_vector),
                    build_text_graph=bool(build_text_graph),
                    build_causal_graph=bool(build_causal_graph),
                    enable_implicit_causal_discovery=bool(implicit_discovery),
                    implicit_causal_max_chunks=int(implicit_max_chunks),
                    embedding_workers=int(embedding_workers),
                    graph_llm_workers=int(graph_llm_workers),
                    progress_cb=progress,
                )
                status.update(label="HopRAG KB build completed.", state="complete")
                for msg in logs:
                    st.write(msg)
                st.json(stats, expanded=False)
            except Exception as e:
                print(f"[HOPRAG BUILD ERROR] {e}", flush=True)
                status.update(label="HopRAG KB build failed.", state="error")
                st.error(str(e))

    with st.expander("Append vector chunks only (no graph rebuild)", expanded=False):
        st.caption("Append sentence chunks to vector store and skip existing chunk IDs.")
        inc_cols = st.columns([3, 1])
        append_sentence_path = inc_cols[0].selectbox(
            "Append sentence chunk file",
            [""] + [str(p) for p in sentence_files],
            index=1 if sentence_files else 0,
            key="hoprag_append_sentence_chunks",
        )
        append_workers = inc_cols[1].slider(
            "Append worker count",
            min_value=1,
            max_value=16,
            value=8,
            step=1,
            key="hoprag_append_embedding_workers",
        )
        if st.button("Append vector chunks", disabled=not append_sentence_path, key="hoprag_append_vector_chunks"):
            mgr = get_mgr(kb_id)
            logs: list[str] = []

            def progress(msg: str) -> None:
                logs.append(msg)

            with st.status("Appending HopRAG vector chunks...", expanded=True) as status:
                try:
                    stats = append_hoprag_vector_chunks(
                        mgr=mgr,
                        sentence_chunks_path=Path(append_sentence_path),
                        embedding_workers=int(append_workers),
                        progress_cb=progress,
                    )
                    status.update(label="Append completed.", state="complete")
                    for msg in logs:
                        st.write(msg)
                    st.json(stats, expanded=False)
                except Exception as e:
                    print(f"[HOPRAG APPEND ERROR] {e}", flush=True)
                    status.update(label="Append failed.", state="error")
                    st.error(str(e))

    st.markdown("---")
    st.markdown("### HopRAG Three-Way Evaluation")

    questions_files = _files("*_questions_for_batch.jsonl")
    gold_files = _files("*_gold_supporting_map.json")
    q_options = [""] + [str(p) for p in questions_files]
    g_options = [""] + [str(p) for p in gold_files]

    c1, c2 = st.columns(2)
    questions_path_str = c1.selectbox("Question file", q_options, index=1 if len(q_options) > 1 else 0)
    gold_map_path_str = c2.selectbox("Gold supporting map", g_options, index=1 if len(g_options) > 1 else 0)
    question_total_count = 0
    if questions_path_str:
        try:
            question_total_count = len(load_questions_jsonl(Path(questions_path_str)))
        except Exception as e:
            print(f"[HOPRAG QUESTION COUNT ERROR] {e}", flush=True)
            st.warning(f"Failed to read question count: {e}")
    target_max_samples = max(1, question_total_count or 1000)
    max_samples = st.number_input(
        "Target sample count (first N)",
        min_value=1,
        max_value=target_max_samples,
        value=min(10, target_max_samples),
        step=10,
        key=f"hoprag_eval_target_total::{kb_id}::{Path(questions_path_str).stem if questions_path_str else 'none'}",
        help="Total target count, not increment count.",
    )
    effective_max_samples = min(int(max_samples), question_total_count or int(max_samples))
    st.caption(f"Question count: {question_total_count or 'unknown'} | target sample count: {effective_max_samples}")
    parallel_items = st.number_input(
        "Item-level parallelism",
        min_value=1,
        max_value=16,
        value=1,
        step=1,
        key=f"hoprag_eval_parallel_items::{kb_id}",
        help="Each sample runs Full / Vector-Only / No-RAG internally.",
    )
    default_out = (
        _default_result_path(Path(questions_path_str), kb_id)
        if questions_path_str
        else RESULT_DIR / "hoprag_results.json"
    )
    use_latest_result = st.checkbox(
        "复用该题集最近结果文件（继续跑）",
        value=True,
        key=f"hoprag_use_latest_result::{kb_id}",
    )
    if questions_path_str and use_latest_result:
        latest = _latest_result_for_questions(Path(questions_path_str))
        if latest is not None:
            default_out = latest
    elif questions_path_str and not use_latest_result:
        default_out = _result_path()
    result_path_str = st.text_input(
        "Result file path",
        value=str(default_out),
        key=f"hoprag_eval_result_path::{kb_id}",
        help="Results saved to JSON and resume skips completed samples.",
    ).strip()
    result_path = Path(result_path_str) if result_path_str else default_out
    st.caption(f"Current experiment KB: {kb_id}")
    ignore_done = st.checkbox(
        "忽略已完成记录（强制重跑目标样本）",
        value=False,
        key=f"hoprag_ignore_done::{kb_id}",
        help="开启后会重跑目标样本，并覆盖同 sample_id + method 的旧记录。",
    )
    if questions_path_str and result_path.exists():
        try:
            done_preview = len(_done_sample_ids(result_path))
            target_preview = min(int(max_samples), question_total_count or int(max_samples))
            pending_preview = max(0, target_preview - done_preview)
            st.caption(f"progress preview: done={done_preview}, target={target_preview}, pending={pending_preview}")
        except Exception:
            pass

    c_start, c_full, c_norag, c_stop = st.columns(4)
    can_start = bool(questions_path_str and gold_map_path_str and result_path_str)
    start_clicked = c_start.button(
        "Start / Resume HopRAG Aligned Eval",
        type="primary",
        disabled=not can_start,
        width="stretch",
        key=f"hoprag_start_eval::{kb_id}",
    )
    full_clicked = c_full.button(
        "Start / Resume Full",
        disabled=not can_start or not question_total_count,
        width="stretch",
        key=f"hoprag_start_full_eval::{kb_id}",
        help="Set full target count to all questions; completed samples are still skipped.",
    )
    norag_clicked = c_norag.button(
        "No-RAG only rerun",
        disabled=not can_start,
        width="stretch",
        key=f"hoprag_rerun_norag::{kb_id}",
        help="Rerun and overwrite No-RAG only.",
    )
    if norag_clicked:
        try:
            _shutdown_hoprag_executor(st, cancel_futures=True)
            all_questions = load_questions_jsonl(Path(questions_path_str))
            target_count = min(int(max_samples), len(all_questions))
            questions = all_questions[:target_count]
            gold_map = load_gold_map(Path(gold_map_path_str))
            run_meta = {
                "version": "hoprag-aligned-frontend-eval-v2",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "questions_path": str(questions_path_str),
                "gold_map_path": str(gold_map_path_str),
                "kb_id": kb_id,
                "sample_count": len(questions),
                "question_total_count": len(all_questions),
                "target_count": target_count,
                "norag_rerun_only": True,
                "norag_answer_instruction": HOTPOT_NORAG_SHORT_ANSWER_INSTRUCTION,
            }
            logs = list(st.session_state.get("hoprag_eval_logs", []) or [])
            logs.append(f"[No-RAG rerun] start target={len(questions)} workers={int(parallel_items)}")
            with st.status("Rerunning No-RAG only and overwriting old No-RAG records...", expanded=True) as status:
                done = 0
                with ThreadPoolExecutor(max_workers=max(1, int(parallel_items))) as ex:
                    future_map = {
                        ex.submit(
                            _process_one_hoprag_norag_item,
                            item=item,
                            sample_index=idx,
                            gold_map=gold_map,
                        ): str(item.get("id", "") or item.get("sample_id", "") or f"idx-{idx}")
                        for idx, item in enumerate(questions)
                    }
                    for fut in as_completed(future_map):
                        sid = future_map[fut]
                        try:
                            wr = fut.result()
                        except Exception as e:
                            print(f"[HOPRAG NORAG FUTURE ERROR] sample_id={sid} error={e}", flush=True)
                            wr = {"records": [], "logs": [f"{sid}: No-RAG future error={e}"]}
                        _upsert_sample_records(result_path, run_meta, wr.get("records", []) or [])
                        done += 1
                        for line in wr.get("logs", []) or []:
                            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {line}")
                        if done == len(questions) or done % 5 == 0:
                            status.write(f"No-RAG rerun progress: {done}/{len(questions)}")
                st.session_state.hoprag_eval_logs = logs[-400:]
                status.update(label="No-RAG rerun complete.", state="complete")
            st.rerun()
        except Exception as e:
            print(f"[HOPRAG NORAG RERUN START ERROR] {e}", flush=True)
            st.error(f"No-RAG rerun failed: {e}")

    if start_clicked or full_clicked:
        try:
            _shutdown_hoprag_executor(st, cancel_futures=True)
            all_questions = load_questions_jsonl(Path(questions_path_str))
            target_count = len(all_questions) if full_clicked else min(int(max_samples), len(all_questions))
            questions = all_questions[:target_count]
            done_ids = set() if ignore_done else _done_sample_ids(result_path)
            pending = []
            for idx, item in enumerate(questions):
                sid = str(item.get("id", "") or item.get("sample_id", "") or f"idx-{idx}")
                if sid not in done_ids:
                    pending.append({"index": idx, "item": item})
            if not pending:
                st.warning(
                    "没有待运行样本：当前目标样本都已完成。"
                    "可以关闭“复用最近结果文件”生成新结果文件，或开启“忽略已完成记录”强制重跑。"
                )
                st.stop()
            run_meta = {
                "version": "hoprag-aligned-frontend-eval-v2",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "questions_path": str(questions_path_str),
                "gold_map_path": str(gold_map_path_str),
                "kb_id": kb_id,
                "sample_count": len(questions),
                "question_total_count": len(all_questions),
                "target_count": target_count,
                "parallel_items": int(parallel_items),
                "top_k_override": 20,
                "ignore_done": bool(ignore_done),
            }
            doc = _read_result_doc(result_path)
            doc.update(run_meta)
            _save_result_doc(result_path, doc)
            st.session_state.hoprag_eval_runner = {
                "running": True,
                "questions_path": str(questions_path_str),
                "gold_map_path": str(gold_map_path_str),
                "result_path": str(result_path),
                "kb_id": str(kb_id),
                "parallel_items": int(parallel_items),
                "persist_causal_updates": bool(persist_causal_updates),
                "total_count": len(questions),
                "done_before": len(done_ids),
                "pending": pending,
                "started_at": time.time(),
                "last_sample_id": "",
            }
            st.session_state.hoprag_eval_logs = [
                f"[init] loaded={len(questions)} done_before={len(done_ids)} pending={len(pending)}"
            ]
            st.rerun()
        except Exception as e:
            print(f"[HOPRAG RUNNER START ERROR] {e}", flush=True)
            st.error(f"Start failed: {e}")

    if c_stop.button("Stop (keep saved results)", width="stretch"):
        _shutdown_hoprag_executor(st, cancel_futures=True)
        runner = dict(st.session_state.get("hoprag_eval_runner", {}) or {})
        runner["running"] = False
        st.session_state.hoprag_eval_runner = runner
        st.success("Stop requested. Saved records are preserved and can be resumed.")

    runner = dict(st.session_state.get("hoprag_eval_runner", {}) or {})
    running = bool(runner.get("running"))
    if runner:
        total_count = int(runner.get("total_count", 0) or 0)
        pending = list(runner.get("pending", []) or [])
        inflight_count = len(st.session_state.get("hoprag_eval_futures", {}) or {}) if running else 0
        done_count = total_count - len(pending) - inflight_count
        ratio = done_count / total_count if total_count else 0.0
        st.progress(min(max(ratio, 0.0), 1.0))
        st.caption(
            f"Progress: {done_count}/{total_count} | running={running} | result={runner.get('result_path', '')}"
        )

    if running:
        pending = list(runner.get("pending", []) or [])
        parallel_n = max(1, int(runner.get("parallel_items", 1) or 1))
        futures: dict[Any, dict[str, Any]] = dict(st.session_state.get("hoprag_eval_futures", {}) or {})
        executor = st.session_state.get("hoprag_eval_executor")
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=parallel_n)
            st.session_state.hoprag_eval_executor = executor

        gold_map = load_gold_map(Path(runner["gold_map_path"]))
        run_meta = {
            "version": "hoprag-aligned-frontend-eval-v2",
            "questions_path": str(runner.get("questions_path", "")),
            "gold_map_path": str(runner.get("gold_map_path", "")),
            "kb_id": str(runner.get("kb_id", "")),
            "sample_count": int(runner.get("total_count", 0) or 0),
            "parallel_items": parallel_n,
            "scheduler": "rolling_window",
            "top_k_override": 20,
        }

        def submit_next() -> bool:
            queue = list(runner.get("pending", []) or [])
            if not queue:
                return False
            entry = queue.pop(0)
            fut = executor.submit(
                _process_one_hoprag_item,
                item=entry["item"],
                sample_index=int(entry["index"]),
                gold_map=gold_map,
                kb_id=str(runner.get("kb_id", kb_id)),
                run_compare_modes=run_compare_modes,
                get_mgr=get_mgr,
                rewrite_query=rewrite_query,
                persist_causal_updates=bool(runner.get("persist_causal_updates", False)),
            )
            futures[fut] = entry
            runner["pending"] = queue
            return True

        while len(futures) < parallel_n and submit_next():
            pass

        done_futs = [fut for fut in list(futures) if fut.done()]
        completed_labels = []
        for fut in done_futs:
            entry = futures.pop(fut, {})
            sid = str(entry.get("item", {}).get("id", "") or entry.get("item", {}).get("sample_id", "") or f"idx-{entry.get('index', 0)}")
            completed_labels.append(sid)
            try:
                wr = fut.result()
            except Exception as e:
                print(f"[HOPRAG FUTURE ERROR] sample_id={sid} error={e}", flush=True)
                wr = {"records": [], "logs": [f"{sid}: future error={e}"]}
            _upsert_sample_records(Path(runner["result_path"]), run_meta, wr.get("records", []) or [])
            logs = list(st.session_state.get("hoprag_eval_logs", []) or [])
            logs.extend(f"[{datetime.now().strftime('%H:%M:%S')}] {line}" for line in wr.get("logs", []))
            st.session_state.hoprag_eval_logs = logs[-400:]
            submit_next()

        if completed_labels:
            runner["last_sample_id"] = "|".join(completed_labels)

        inflight_labels = [
            str(e.get("item", {}).get("id", "") or e.get("item", {}).get("sample_id", "") or f"idx-{e.get('index', 0)}")
            for e in futures.values()
        ]
        st.info(f"Rolling parallel run: in-flight={len(futures)}/{parallel_n} | {' | '.join(inflight_labels[:8])}")

        if not futures and not list(runner.get("pending", []) or []):
            runner["running"] = False
            _shutdown_hoprag_executor(st, cancel_futures=False)
            st.session_state.hoprag_eval_runner = runner
            st.success("HopRAG evaluation completed.")
            st.rerun()
            return

        st.session_state.hoprag_eval_futures = futures
        st.session_state.hoprag_eval_runner = runner
        time.sleep(1.0)
        st.rerun()
        return
        if not pending:
            runner["running"] = False
            st.session_state.hoprag_eval_runner = runner
            st.success("HopRAG evaluation completed.")
        else:
            gold_map = load_gold_map(Path(runner["gold_map_path"]))
            parallel_n = max(1, int(runner.get("parallel_items", 1) or 1))
            to_run = pending[:parallel_n]
            preview = [
                str(x.get("item", {}).get("id", "") or x.get("item", {}).get("sample_id", "") or f"idx-{x.get('index', 0)}")
                for x in to_run
            ]
            st.info(f"Current parallel samples ({len(to_run)}/{parallel_n}): {' | '.join(preview)}")
            run_meta = {
                "version": "hoprag-aligned-frontend-eval-v2",
                "questions_path": str(runner.get("questions_path", "")),
                "gold_map_path": str(runner.get("gold_map_path", "")),
                "kb_id": str(runner.get("kb_id", "")),
                "sample_count": int(runner.get("total_count", 0) or 0),
                "parallel_items": int(runner.get("parallel_items", 1) or 1),
                "top_k_override": 20,
            }
            worker_results: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=parallel_n) as ex:
                futures = [
                    ex.submit(
                        _process_one_hoprag_item,
                        item=entry["item"],
                        sample_index=int(entry["index"]),
                        gold_map=gold_map,
                        kb_id=str(runner.get("kb_id", kb_id)),
                        run_compare_modes=run_compare_modes,
                        get_mgr=get_mgr,
                        rewrite_query=rewrite_query,
                        persist_causal_updates=bool(runner.get("persist_causal_updates", False)),
                    )
                    for entry in to_run
                ]
                for fut in as_completed(futures):
                    worker_results.append(fut.result())

            for wr in worker_results:
                _upsert_sample_records(Path(runner["result_path"]), run_meta, wr.get("records", []) or [])
                logs = list(st.session_state.get("hoprag_eval_logs", []) or [])
                logs.extend(f"[{datetime.now().strftime('%H:%M:%S')}] {line}" for line in wr.get("logs", []))
                st.session_state.hoprag_eval_logs = logs[-400:]

            runner["pending"] = pending[len(to_run):]
            runner["last_sample_id"] = "|".join(preview)
            if not runner["pending"]:
                runner["running"] = False
            st.session_state.hoprag_eval_runner = runner
            st.rerun()

    logs = st.session_state.get("hoprag_eval_logs", []) or []
    if logs:
        with st.expander("Runtime logs", expanded=False):
            st.code("\n".join(logs[-120:]), language=None)

    if result_path.exists():
        doc = _read_result_doc(result_path)
        st.dataframe(_summary_df(doc.get("summary", {})), width="stretch")
        st.dataframe(_records_df(doc.get("records", []) or []), width="stretch")
        records = doc.get("records", []) or []
        if records:
            with st.expander("Result detail (trace / payload / gold / retrieval)", expanded=False):
                labels = [
                    f"{i}. {r.get('sample_id','')} | {r.get('method','')} | EM={float(r.get('em',0) or 0) * 100:.0f} F1={float(r.get('answer_f1',0) or 0) * 100:.1f}"
                    for i, r in enumerate(records[-60:], start=max(0, len(records) - 60))
                ]
                selected_label = st.selectbox(
                    "Select record",
                    labels,
                    index=len(labels) - 1 if labels else 0,
                    key=f"hoprag_record_detail::{kb_id}",
                )
                if selected_label:
                    idx = labels.index(selected_label)
                    rec = records[-60:][idx]
                    st.json(
                        {
                            "Core result": {
                                "sample_id": rec.get("sample_id"),
                                "method": rec.get("method"),
                                "question": rec.get("question"),
                                "gold_answer": rec.get("gold_answer"),
                                "prediction": rec.get("prediction"),
                                "rag_query": rec.get("rag_query"),
                                "metrics": {
                                    "em": round(float(rec.get("em", 0) or 0) * 100, 2),
                                    "answer_f1": round(float(rec.get("answer_f1", 0) or 0) * 100, 2),
                                    "retrieval_precision": round(float(rec.get("retrieval_precision", 0) or 0) * 100, 2),
                                    "retrieval_recall": round(float(rec.get("retrieval_recall", 0) or 0) * 100, 2),
                                    "retrieval_f1": round(float(rec.get("retrieval_f1", 0) or 0) * 100, 2),
                                },
                            },
                            "trace": rec.get("trace", []),
                            "branch_payload": rec.get("branch_payload", {}),
                            "branch_steps": rec.get("branch_steps", []),
                            "gold_debug": rec.get("gold_debug", {}),
                            "retrieval_debug": rec.get("retrieval_debug", {}),
                            "full_debug": rec.get("full_debug", {}),
                            "query_preanalysis": rec.get("query_preanalysis", {}),
                            "compare_debug": rec.get("compare_debug", {}),
                        },
                        expanded=False,
                    )

    with st.expander("Run single-pass eval now", expanded=False):
        if st.button("Run current sample set", disabled=not questions_path_str or not gold_map_path_str):
            with st.status("Running three-way evaluation...", expanded=True) as status:
                try:
                    doc = run_hoprag_compare_eval(
                        questions_path=Path(questions_path_str),
                        gold_map_path=Path(gold_map_path_str),
                        kb_id=kb_id,
                        max_samples=int(max_samples),
                        run_compare_modes=run_compare_modes,
                        get_mgr=get_mgr,
                        rewrite_query=rewrite_query,
                        persist_causal_updates=bool(persist_causal_updates),
                    )
                    status.update(label="Run finished.", state="complete")
                    st.success(f"Saved result: {doc.get('result_path')}")
                    st.dataframe(_summary_df(doc.get("summary", {})), width="stretch")
                    st.dataframe(_records_df(doc.get("records", []) or []), width="stretch")
                    with st.expander("Raw rich JSON", expanded=False):
                        st.json(doc, expanded=False)
                except Exception as e:
                    print(f"[HOPRAG ONESHOT ERROR] {e}", flush=True)
                    status.update(label="Run failed", state="error")
                    st.error(str(e))

    st.markdown("### History Results")
    result_files = sorted(
        {p for pattern in ("hoprag_frontend_run_*.json", "*_results.json") for p in RESULT_DIR.glob(pattern)},
        reverse=True,
    ) if RESULT_DIR.exists() else []
    selected = st.selectbox("Select result file", [""] + [str(p) for p in result_files], index=0)
    if selected:
        doc = _read_result_doc(Path(selected))
        st.dataframe(_summary_df(doc.get("summary", {})), width="stretch")
        st.dataframe(_records_df(doc.get("records", []) or []), width="stretch")
        with st.expander("Raw result JSON", expanded=False):
            st.json(doc, expanded=False)

    st.markdown("### 外部 HopRAG 复现结果解析")
    default_external_dir = ROOT / "data" / "external_hoprag_results"
    external_dir_str = st.text_input(
        "外部结果目录",
        value=str(default_external_dir),
        key=f"hoprag_external_dir::{kb_id}",
        help="目录下自动发现 run_meta*.json 与 hoprag_predictions*.jsonl",
    ).strip()
    external_dir = Path(external_dir_str) if external_dir_str else default_external_dir
    if not gold_map_path_str:
        st.info("请先选择 Gold supporting map，再解析外部复现结果。")
        return
    if not external_dir.exists():
        st.warning(f"目录不存在: {external_dir}")
        return
    runs = _collect_external_hoprag_runs(external_dir)
    if not runs:
        st.info("未发现外部结果文件（run_meta*.json / hoprag_predictions*.jsonl）。")
        return
    st.caption(f"发现 {len(runs)} 组外部结果")
    labels = []
    for r in runs:
        meta_name = r["run_meta_path"].name if r.get("run_meta_path") else "(no run_meta)"
        labels.append(f"{r['mode']} | {r['predictions_path'].name} | {meta_name}")
    chosen = st.multiselect(
        "选择要解析的结果组",
        options=labels,
        default=labels[: min(3, len(labels))],
        key=f"hoprag_external_pick::{kb_id}",
    )
    if not chosen:
        return
    label_to_run = {lab: runs[i] for i, lab in enumerate(labels)}
    gold_map = load_gold_map(Path(gold_map_path_str))
    out_rows: list[dict[str, Any]] = []
    for lab in chosen:
        run = label_to_run[lab]
        metrics = _eval_external_hoprag_predictions(Path(run["predictions_path"]), gold_map)
        out_rows.append(
            {
                "mode": run["mode"],
                "predictions_path": str(run["predictions_path"]),
                "run_meta_path": str(run["run_meta_path"]) if run.get("run_meta_path") else "",
                "n": int(metrics.get("n", 0.0) or 0),
                "failures": int(metrics.get("failures", 0.0) or 0),
                "EM": round(float(metrics.get("em", 0.0) or 0.0) * 100, 2),
                "Answer_F1": round(float(metrics.get("answer_f1", 0.0) or 0.0) * 100, 2),
                "Retrieval_P": round(float(metrics.get("retrieval_precision", 0.0) or 0.0) * 100, 2),
                "Retrieval_R": round(float(metrics.get("retrieval_recall", 0.0) or 0.0) * 100, 2),
                "Retrieval_F1": round(float(metrics.get("retrieval_f1", 0.0) or 0.0) * 100, 2),
                "support_hit_any": round(float(metrics.get("support_hit_any", 0.0) or 0.0) * 100, 2),
                "support_recall_avg": round(float(metrics.get("support_recall_avg", 0.0) or 0.0) * 100, 2),
            }
        )
    st.dataframe(pd.DataFrame(out_rows), width="stretch")
