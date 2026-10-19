from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import altair as alt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "evaluation" / "BUAAEduPolicyQA"
RESULT_DIR = DATASET_DIR / "real_scene_results"

ROUTE_ORDER: list[tuple[str, str]] = [
    ("direct", "No-RAG"),
    ("bm25", "BM25"),
    ("cosine_shuffle", "Cosine"),
    ("vector_only", "Vector-Only"),
    ("vector_analysis", "Vector+Analysis"),
    ("full", "Full-Causal"),
]

QTYPE_LABELS: dict[str, str] = {
    "counterfactual_policy": "政策反事实",
    "eligibility_causality": "资格因果",
    "process_causality": "流程因果",
    "boundary_condition": "边界条件",
    "evidence_requirement": "材料要求",
    "counterfactual": "一般反事实",
    "policy_explanation": "政策解释",
    "unknown": "未知",
}

QTYPE_ORDER: list[str] = [
    "政策反事实",
    "资格因果",
    "流程因果",
    "边界条件",
    "材料要求",
    "一般反事实",
    "政策解释",
    "未知",
]

SOFT_WEIGHTS: dict[str, float] = {
    "correctness": 1.0,
    "groundedness": 1.0,
    "gold_alignment": 1.0,
    "causal_reasoning": 3.0,
    "causal_quality": 3.0,
    "process_quality": 3.0,
    "overall": 2.0,
    "answer_f1": 1.0,
}


def _dataset_files() -> list[Path]:
    if not DATASET_DIR.exists():
        return []
    files: list[Path] = []
    for pat in ("buaa_causal_qa_unified_*.json", "buaa_policy_qa_*.json", "buaa_policy_qa_*.jsonl"):
        files.extend(DATASET_DIR.glob(pat))
    files = [p for p in files if "preview" not in p.name and "rejected" not in p.name]
    return sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)


def _default_result_path(dataset_path: Path, kb_id: str) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return RESULT_DIR / f"{dataset_path.stem}_{str(kb_id or 'kb')[:8]}_real_scene_results.json"


def _read_result_doc(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": "buaa-real-scene-eval-v1", "records": []}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            rec = obj.get("records")
            if not isinstance(rec, list):
                obj["records"] = []
            return obj
    except Exception:
        pass
    return {"version": "buaa-real-scene-eval-v1", "records": []}


def _safe_score(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    if x > 1.0:
        x = x / 10.0
    return max(0.0, min(1.0, x))


def _tokens(text: str) -> list[str]:
    s = str(text or "").lower()
    zh = re.findall(r"[\u4e00-\u9fff]", s)
    en = re.findall(r"[a-z0-9_]+", s)
    return zh + en


def _prf(prediction: str, gold: str) -> tuple[float, float, float]:
    p = _tokens(prediction)
    g = _tokens(gold)
    if not p and not g:
        return 1.0, 1.0, 1.0
    if not p or not g:
        return 0.0, 0.0, 0.0
    same = sum((Counter(p) & Counter(g)).values())
    if same <= 0:
        return 0.0, 0.0, 0.0
    precision = same / len(p)
    recall = same / len(g)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    b2 = beta * beta
    den = b2 * precision + recall
    if den <= 0:
        return 0.0
    return (1.0 + b2) * precision * recall / den


def _judge_score(rec: dict[str, Any], route_key: str) -> float | None:
    judge = rec.get("judge", {}) or {}
    weighted = judge.get("weighted_scores", {}) or {}
    v = weighted.get(route_key)
    if isinstance(v, (int, float)):
        return float(v)
    sc = (judge.get("scores", {}) or {}).get(route_key, {}) or {}
    if isinstance(sc.get("soft_f1"), (int, float)):
        return float(sc.get("soft_f1"))
    if isinstance(sc.get("overall"), (int, float)):
        return float(sc.get("overall"))
    return None


def _branch_scores_with_fallback(rec: dict[str, Any], route_key: str) -> dict[str, Any]:
    judge = rec.get("judge", {}) or {}
    scores = (judge.get("scores", {}) or {}).get(route_key, {}) or {}
    out = dict(scores)
    branches = rec.get("branches", {}) or {}
    pred = str((branches.get(route_key, {}) or {}).get("answer", "") or "")
    gold = str(rec.get("gold_answer", "") or "")
    p, r, f1 = _prf(pred, gold)
    out.setdefault("answer_precision", _safe_score(p))
    out.setdefault("answer_recall", _safe_score(r))
    out.setdefault("answer_f1", _safe_score(f1))
    out.setdefault("hard_f1_weighted", _safe_score(_f_beta(p, r, beta=2.0)))
    if "soft_f1" not in out or not isinstance(out.get("soft_f1"), (int, float)):
        ws = 0.0
        wt = 0.0
        for k, w in SOFT_WEIGHTS.items():
            vv = out.get(k)
            if isinstance(vv, (int, float)):
                ws += float(w) * _safe_score(vv)
                wt += float(w)
        out["soft_f1"] = round(ws / wt, 4) if wt > 0 else None
    return out


def _summary_rows(records: list[dict[str, Any]]) -> pd.DataFrame:
    ok = [r for r in records if isinstance(r, dict) and str(r.get("status", "")).lower() == "ok"]
    metrics = [
        "overall",
        "correctness",
        "groundedness",
        "gold_alignment",
        "causal_quality",
        "causal_reasoning",
        "process_quality",
        "answer_precision",
        "answer_recall",
        "answer_f1",
        "hard_f1_weighted",
        "soft_f1",
        "judge_score",
    ]
    rows: list[dict[str, Any]] = []
    for key, label in ROUTE_ORDER:
        vals: dict[str, list[float]] = {m: [] for m in metrics}
        for rec in ok:
            sc = _branch_scores_with_fallback(rec, key)
            for m in metrics:
                v = _judge_score(rec, key) if m == "judge_score" else sc.get(m)
                if isinstance(v, (int, float)):
                    vals[m].append(float(v))
        row = {"route": label, "route_key": key, "n_ok": len(ok)}
        for m in metrics:
            row[m] = round(sum(vals[m]) / len(vals[m]), 4) if vals[m] else None
        rows.append(row)
    return pd.DataFrame(rows)


def _detail_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        qtype_raw = str(rec.get("question_type", "") or "unknown")
        qtype = QTYPE_LABELS.get(qtype_raw, qtype_raw)
        one = {
            "id": rec.get("item_id", ""),
            "status": rec.get("status", ""),
            "type": qtype,
            "winner": rec.get("winner", ""),
            "elapsed_s": rec.get("elapsed_s", ""),
            "question": str(rec.get("question", ""))[:160],
        }
        for key, _label in ROUTE_ORDER:
            sc = _branch_scores_with_fallback(rec, key)
            one[f"{key}.judge_score"] = _judge_score(rec, key)
            one[f"{key}.soft_f1"] = sc.get("soft_f1")
            one[f"{key}.hard_f1_weighted"] = sc.get("hard_f1_weighted")
        rows.append(one)
    return pd.DataFrame(rows)


def _type_pct_chart_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    route_index = {label: i for i, (_k, label) in enumerate(ROUTE_ORDER)}
    rows: list[dict[str, Any]] = []
    for rec in records:
        if str(rec.get("status", "")).lower() != "ok":
            continue
        qtype_raw = str(rec.get("question_type", "") or "unknown")
        qtype = QTYPE_LABELS.get(qtype_raw, qtype_raw)
        for key, label in ROUTE_ORDER:
            js = _judge_score(rec, key)
            if isinstance(js, (int, float)):
                rows.append({"qtype": qtype, "route": label, "route_index": route_index[label], "judge_score": float(js)})
    if not rows:
        return pd.DataFrame(columns=["qtype", "route", "route_index", "judge_pct"])
    df = pd.DataFrame(rows)
    agg = df.groupby(["qtype", "route", "route_index"], as_index=False)["judge_score"].sum()
    total = agg.groupby("qtype")["judge_score"].transform("sum")
    agg["judge_pct"] = (agg["judge_score"] / total.where(total > 0, 1.0) * 100.0).round(2)
    agg["qtype"] = pd.Categorical(agg["qtype"], categories=QTYPE_ORDER, ordered=True)
    agg["route"] = pd.Categorical(agg["route"], categories=[x[1] for x in ROUTE_ORDER], ordered=True)
    return agg.sort_values(["qtype", "route_index"])


def render_real_scene_eval_panel(
    *,
    st: Any,
    kb_id: str,
    run_compare_modes: Callable[..., dict],
    get_mgr: Callable[[str], Any],
    rewrite_query: Callable[[str, list[dict[str, str]]], str],
    auto_score_compare: Callable[..., dict],
    score_metrics: tuple[str, ...],
    gold_metric: str,
    default_judge_specs: list[str],
    persist_causal_updates: bool = False,
) -> None:
    st.markdown('<div class="page-title">北航真实场景测试</div>', unsafe_allow_html=True)
    st.caption("结果查看与对比分析面板。")

    files = _dataset_files()
    selected = st.selectbox("测试集", [str(p) for p in files], index=0 if files else None) if files else ""
    ds_path = Path(st.text_input("测试集路径", value=str(selected or DATASET_DIR / "buaa_causal_qa_unified_230_shuffled.json")).strip())
    res_default = _default_result_path(ds_path, kb_id or "kb")
    res_path = Path(st.text_input("结果路径", value=str(res_default)).strip())

    if not res_path.exists():
        st.warning("结果文件不存在。")
        return

    doc = _read_result_doc(res_path)
    records = [r for r in (doc.get("records", []) or []) if isinstance(r, dict)]

    st.markdown("### 各路指标均值（自动统计）")
    st.dataframe(_summary_rows(records), width="stretch")

    st.markdown("### 结果明细")
    st.dataframe(_detail_df(records), width="stretch")

    st.markdown("### 按题型对比（Judge Score 占比）")
    cdf = _type_pct_chart_df(records)
    if cdf.empty:
        st.info("暂无可绘图数据。")
    else:
        pivot = cdf.pivot(index="qtype", columns="route", values="judge_pct").fillna(0.0).reset_index()
        cols = ["qtype"] + [x[1] for x in ROUTE_ORDER]
        for c in cols[1:]:
            if c not in pivot.columns:
                pivot[c] = 0.0
        st.dataframe(pivot[cols].rename(columns={"qtype": "问题类型"}), width="stretch")

        chart = (
            alt.Chart(cdf)
            .mark_bar()
            .encode(
                x=alt.X("qtype:N", title="问题类型", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("judge_pct:Q", title="判分占比(%)", stack="zero"),
                color=alt.Color("route:N", scale=alt.Scale(domain=[x[1] for x in ROUTE_ORDER]), legend=alt.Legend(title="路线")),
                order=alt.Order("route_index:Q", sort="descending"),
                tooltip=[
                    alt.Tooltip("qtype:N", title="题型"),
                    alt.Tooltip("route:N", title="路线"),
                    alt.Tooltip("judge_pct:Q", title="占比(%)", format=".2f"),
                ],
            )
            .properties(height=380)
        )
        st.altair_chart(chart, use_container_width=True)

    ok_n = sum(1 for r in records if str(r.get("status", "")).lower() == "ok")
    st.markdown("### 汇总")
    st.dataframe(
        pd.DataFrame(
            [
                {"metric": "records", "value": len(records)},
                {"metric": "ok", "value": ok_n},
                {"metric": "error", "value": len(records) - ok_n},
            ]
        ),
        width="stretch",
    )
