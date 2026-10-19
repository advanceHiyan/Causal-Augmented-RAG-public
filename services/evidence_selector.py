from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any


SOURCE_GENERAL = "general_kb"
SOURCE_STUDENT_STATUS = "student_status"
SOURCE_QUERY_PREANALYSIS = "query_preanalysis"


@dataclass
class EvidenceItem:
    source_type: str
    source_name: str = ""
    level: str = ""
    content: str = ""
    summary: str = ""
    relevance_score: float = 0.0
    reliability_score: float = 0.0
    specificity_score: float = 0.0
    noise_risk: float = 0.0
    selected_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_score(self) -> float:
        return (
            self.relevance_score * 0.45
            + self.reliability_score * 0.25
            + self.specificity_score * 0.20
            - self.noise_risk * 0.10
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["final_score"] = round(self.final_score, 4)
        return data


@dataclass
class EvidenceSelection:
    selected: list[EvidenceItem]
    dropped: list[EvidenceItem]
    policy: dict[str, Any]
    manifest: dict[str, Any]

    def to_context_json(self) -> str:
        return json.dumps(self.manifest, ensure_ascii=False, indent=2)


def _clip_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tokens(text: str) -> set[str]:
    raw = (text or "").lower()
    words = set(re.findall(r"[a-z0-9_]{2,}|\d+|[\u4e00-\u9fff]", raw))
    return {w for w in words if w.strip()}


def _relevance(query: str, text: str) -> float:
    qt = _tokens(query)
    if not qt:
        return 0.0
    tt = _tokens(text)
    if not tt:
        return 0.0
    overlap = len(qt & tt)
    return _clip_score(overlap / max(3, min(len(qt), 12)))


def _dedupe_key(item: EvidenceItem) -> str:
    compact = re.sub(r"\s+", " ", (item.summary or item.content or "")).strip().lower()
    return f"{item.source_type}:{item.level}:{compact[:160]}"


def detect_question_intent(query: str) -> dict[str, bool]:
    q = (query or "").lower()
    causal_terms = ("因果", "原因", "导致", "影响", "反事实", "干预", "why", "cause", "effect")
    status_terms = (
        "我",
        "本人",
        "学生",
        "成绩",
        "绩点",
        "消费",
        "一卡通",
        "考勤",
        "奖学金",
        "预警",
        "score",
        "gpa",
        "attendance",
    )
    policy_terms = ("政策", "规则", "要求", "定义", "标准", "办法", "policy", "rule", "definition")
    return {
        "causal": any(t in q for t in causal_terms),
        "student_status": any(t in q for t in status_terms),
        "policy": any(t in q for t in policy_terms),
    }


def normalize_general_retrieval(query: str, retrieval_context: Any) -> list[EvidenceItem]:
    if not retrieval_context:
        return []

    def _get(name: str, default: Any = "") -> Any:
        if isinstance(retrieval_context, dict):
            return retrieval_context.get(name, default)
        return getattr(retrieval_context, name, default)

    items: list[EvidenceItem] = []
    l1_text = str(_get("causal_graph_text", "") or "").strip()
    if bool(_get("l1_success", False)) and l1_text:
        items.append(
            EvidenceItem(
                source_type=SOURCE_GENERAL,
                source_name="causal_graph",
                level="L1",
                content=l1_text,
                summary=l1_text[:800],
                relevance_score=max(0.55, _relevance(query, l1_text)),
                reliability_score=0.9,
                specificity_score=0.85,
                noise_risk=0.1,
                selected_reason="L1 causal graph hit",
            )
        )

    l2_text = str(_get("causal_path_summary", "") or _get("path_summary", "") or "").strip()
    if bool(_get("l2_success", False) or _get("l2", False)) and l2_text:
        items.append(
            EvidenceItem(
                source_type=SOURCE_GENERAL,
                source_name="text_graph",
                level="L2",
                content=l2_text,
                summary=l2_text[:800],
                relevance_score=max(0.45, _relevance(query, l2_text)),
                reliability_score=0.75,
                specificity_score=0.75,
                noise_risk=0.2,
                selected_reason="L2 text graph causal path hit",
            )
        )

    chunks = _get("vector_chunks", []) or _get("vector_snippets", []) or _get("chunks", []) or []
    if isinstance(chunks, list):
        for idx, chunk in enumerate(chunks[:12]):
            text = ""
            source = "vector_store"
            score = 0.0
            if isinstance(chunk, dict):
                text = str(chunk.get("text") or chunk.get("content") or "")
                source = str(chunk.get("source") or chunk.get("filename") or source)
                score = float(chunk.get("score") or chunk.get("similarity") or 0.0)
            else:
                text = str(chunk)
            text = text.strip()
            if not text:
                continue
            rel = _clip_score(score) if score else _relevance(query, text)
            items.append(
                EvidenceItem(
                    source_type=SOURCE_GENERAL,
                    source_name=source,
                    level="L3",
                    content=text,
                    summary=text[:700],
                    relevance_score=rel,
                    reliability_score=0.6,
                    specificity_score=0.55,
                    noise_risk=0.35,
                    selected_reason=f"L3 vector chunk #{idx + 1}",
                )
            )
    return items


def normalize_student_status(query: str, profile_meta: dict[str, Any] | None) -> list[EvidenceItem]:
    meta = profile_meta or {}
    if not isinstance(meta, dict) or not meta.get("identity_ok", True):
        return []

    items: list[EvidenceItem] = []
    context_raw = str(meta.get("profile_context") or "").strip()
    context: dict[str, Any] = {}
    if context_raw:
        try:
            parsed = json.loads(context_raw)
            if isinstance(parsed, dict):
                context = parsed
        except Exception:
            context = {}

    relevant_facts = meta.get("relevant_facts") or context.get("relevant_fields") or {}
    if isinstance(relevant_facts, dict) and relevant_facts:
        content = json.dumps(relevant_facts, ensure_ascii=False)
        items.append(
            EvidenceItem(
                source_type=SOURCE_STUDENT_STATUS,
                source_name="profile_facts",
                level="facts",
                content=content,
                summary=content[:800],
                relevance_score=max(0.55, _relevance(query, content)),
                reliability_score=0.85,
                specificity_score=0.95,
                noise_risk=0.15,
                selected_reason="student-bound relevant facts",
                metadata={"state_lib_id": meta.get("state_lib_id", "")},
            )
        )

    ts = context.get("time_series_summary") or {}
    if isinstance(ts, dict) and ts:
        content = json.dumps(ts, ensure_ascii=False)
        items.append(
            EvidenceItem(
                source_type=SOURCE_STUDENT_STATUS,
                source_name="time_series_summary",
                level="time_series",
                content=content,
                summary=content[:1000],
                relevance_score=max(0.65, _relevance(query, content)),
                reliability_score=0.9,
                specificity_score=0.95,
                noise_risk=0.1,
                selected_reason="aggregated student time-series evidence",
                metadata={"state_lib_id": meta.get("state_lib_id", "")},
            )
        )

    records = meta.get("source_records") or context.get("matched_records") or []
    if isinstance(records, list):
        for idx, record in enumerate(records[:8]):
            if not isinstance(record, dict):
                continue
            content = json.dumps(record, ensure_ascii=False)
            items.append(
                EvidenceItem(
                    source_type=SOURCE_STUDENT_STATUS,
                    source_name=str(record.get("source_file") or "student_status_record"),
                    level="record_sample",
                    content=content,
                    summary=content[:700],
                    relevance_score=max(0.35, _relevance(query, content)),
                    reliability_score=0.8,
                    specificity_score=0.9,
                    noise_risk=0.3,
                    selected_reason=f"student status source record sample #{idx + 1}",
                    metadata={
                        "sheet": record.get("sheet", ""),
                        "orientation": record.get("orientation", ""),
                    },
                )
            )
    return items


def normalize_query_preanalysis(query: str, query_preanalysis_meta: dict[str, Any] | None) -> list[EvidenceItem]:
    meta = query_preanalysis_meta or {}
    if not isinstance(meta, dict):
        return []
    analysis = meta.get("analysis")
    if not isinstance(analysis, dict) or not analysis:
        return []
    content = json.dumps(analysis, ensure_ascii=False)
    return [
        EvidenceItem(
            source_type=SOURCE_QUERY_PREANALYSIS,
            source_name="query_preanalysis",
            level="Q0",
            content=content,
            summary=content[:1000],
            relevance_score=max(0.55, _relevance(query, content)),
            reliability_score=0.35,
            specificity_score=0.55,
            noise_risk=0.35,
            selected_reason="query-only causal hypothesis scaffold",
            metadata={
                "causal_question": bool(meta.get("causal_question", False)),
                "llm_used": bool(meta.get("llm_used", False)),
                "note": str(meta.get("note", "")),
            },
        )
    ]


def select_evidence(
    *,
    query: str,
    general_items: list[EvidenceItem] | None = None,
    student_items: list[EvidenceItem] | None = None,
    query_items: list[EvidenceItem] | None = None,
    max_items: int = 8,
) -> EvidenceSelection:
    intent = detect_question_intent(query)
    candidates = list(general_items or []) + list(student_items or []) + list(query_items or [])
    seen: set[str] = set()
    unique: list[EvidenceItem] = []
    dropped: list[EvidenceItem] = []
    for item in candidates:
        key = _dedupe_key(item)
        if key in seen:
            item.selected_reason = "dropped: duplicate evidence"
            dropped.append(item)
            continue
        seen.add(key)
        unique.append(item)

    selected: list[EvidenceItem] = []

    def _add_best(predicate, reason: str) -> None:
        pool = [it for it in unique if predicate(it) and it not in selected]
        if not pool:
            return
        best = max(pool, key=lambda it: it.final_score)
        best.selected_reason = reason
        selected.append(best)

    if intent["causal"]:
        _add_best(lambda it: it.source_type == SOURCE_GENERAL and it.level in {"L1", "L2"}, "required causal mechanism evidence")
        _add_best(lambda it: it.source_type == SOURCE_QUERY_PREANALYSIS, "query-only causal scaffold")
    if intent["policy"]:
        _add_best(lambda it: it.source_type == SOURCE_GENERAL, "required general knowledge evidence")
    if intent["student_status"]:
        _add_best(lambda it: it.source_type == SOURCE_STUDENT_STATUS and it.level in {"time_series", "facts"}, "required student-specific status evidence")

    remaining = [it for it in unique if it not in selected]
    remaining.sort(key=lambda it: it.final_score, reverse=True)
    for item in remaining:
        if len(selected) >= max_items:
            item.selected_reason = "dropped: context budget"
            dropped.append(item)
            continue
        if item.final_score < 0.12 and selected:
            item.selected_reason = "dropped: low evidence score"
            dropped.append(item)
            continue
        selected.append(item)

    selected.sort(key=lambda it: (it.source_type != SOURCE_GENERAL, -it.final_score))
    selected_dicts = [it.to_dict() for it in selected]
    dropped_dicts = [it.to_dict() for it in dropped]
    policy = {
        "intent": intent,
        "max_items": max_items,
        "llm_used": False,
        "selection_mode": "deterministic_guardrails",
    }
    manifest = {
        "context_type": "evidence_manifest",
        "policy": policy,
        "selected_count": len(selected),
        "dropped_count": len(dropped),
        "selected": selected_dicts,
        "dropped": dropped_dicts,
    }
    return EvidenceSelection(selected=selected, dropped=dropped, policy=policy, manifest=manifest)


def build_evidence_manifest(
    *,
    query: str,
    retrieval_context: Any = None,
    profile_meta: dict[str, Any] | None = None,
    query_preanalysis_meta: dict[str, Any] | None = None,
    max_items: int = 8,
) -> EvidenceSelection:
    general_items = normalize_general_retrieval(query, retrieval_context)
    student_items = normalize_student_status(query, profile_meta)
    query_items = normalize_query_preanalysis(query, query_preanalysis_meta)
    return select_evidence(
        query=query,
        general_items=general_items,
        student_items=student_items,
        query_items=query_items,
        max_items=max_items,
    )
