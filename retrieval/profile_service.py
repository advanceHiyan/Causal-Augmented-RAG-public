"""
Student status library retrieval service (table-based, non-graph).

Design goals:
1) keep student-status retrieval independent from text/causal graph retrieval
2) avoid repeated slot fetches inside the same conversation
3) support backward compatibility (no bound user -> graceful no-op)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import re

from loguru import logger

import db.database as DB
from utils.llm_client import llm_client


PROFILE_SLOT_HINTS: dict[str, tuple[str, ...]] = {
    "student_id": ("学号", "学生id", "student id", "id"),
    "name": ("姓名", "名字", "name"),
    "college": ("学院", "院系", "college", "faculty"),
    "major": ("专业", "major"),
    "class_name": ("班级", "class"),
    "grade_year": ("年级", "入学", "grade"),
    "gpa": ("绩点", "gpa"),
    "score": ("成绩", "分数", "score"),
    "credits": ("学分", "credit"),
    "rank": ("排名", "rank"),
    "attendance": ("考勤", "出勤", "attendance"),
    "scholarship": ("奖学金", "scholarship"),
    "warning_status": ("预警", "挂科", "warning", "fail"),
}

PROFILE_SLOT_ALIAS: dict[str, str] = {
    "学号": "student_id",
    "stu_code": "student_id",
    "student_code": "student_id",
    "student_no": "student_id",
    "student_number": "student_id",
    "student_pid": "student_id",
    "姓名": "name",
    "stu_name": "name",
    "student_name": "name",
    "display_name": "name",
    "学院": "college",
    "专业": "major",
    "班级": "class_name",
    "年级": "grade_year",
    "绩点": "gpa",
    "gpa": "gpa",
    "成绩": "score",
    "分数": "score",
    "学分": "credits",
    "排名": "rank",
    "考勤": "attendance",
    "奖学金": "scholarship",
    "预警": "warning_status",
    "挂科": "warning_status",
}

PROFILE_SLOT_LABELS: dict[str, str] = {
    "student_id": "学号",
    "name": "姓名",
    "college": "学院",
    "major": "专业",
    "class_name": "班级",
    "grade_year": "年级",
    "gpa": "绩点",
    "score": "成绩",
    "credits": "学分",
    "rank": "排名",
    "attendance": "考勤",
    "scholarship": "奖学金",
    "warning_status": "学业预警",
}

PROFILE_BOOTSTRAP_SLOTS: tuple[str, ...] = (
    "student_id",
    "name",
    "college",
    "major",
    "class_name",
    "grade_year",
)
ALL_STUDENT_STATUS_FIELDS = "all_student_status_fields"
MAX_MATCHED_RECORDS_PER_DOC = 5000
MAX_CONTEXT_RECORDS = 24
MAX_DATE_SERIES_POINTS = 45
TS_SUMMARY_KEY_PREFIX = "student_time_series_summary::"
TIME_SERIES_ONLY_ROW_THRESHOLD = 16

REFRESH_HINTS = ("重新", "最新", "刷新", "更新", "re-check", "refresh")
FIRST_PERSON_HINTS = ("我", "本人", "我的", "my ", "i ")


@dataclass
class ProfileContextResult:
    user_id: str = ""
    state_lib_id: str = ""
    identity_ok: bool = True
    identity_note: str = ""
    requested_slots: list[str] | None = None
    fetched_slots: list[str] | None = None
    reused_slots: list[str] | None = None
    profile_facts: dict[str, str] | None = None
    relevant_facts: dict[str, str] | None = None
    source_records: list[dict] | None = None
    profile_context: str = ""
    llm_review: str = ""
    extracted_updates: dict[str, str] | None = None
    doc_sources: list[str] | None = None
    doc_count: int = 0
    relevance_filter_trace: dict | None = None
    time_series_trace: dict | None = None

    def to_meta(self) -> dict:
        return asdict(self)


def _norm_slot(raw: str) -> str:
    t = (raw or "").strip().lower()
    if not t:
        return ""
    if t in PROFILE_SLOT_HINTS:
        return t
    if raw in PROFILE_SLOT_ALIAS:
        return PROFILE_SLOT_ALIAS[raw]
    if t in PROFILE_SLOT_ALIAS:
        return PROFILE_SLOT_ALIAS[t]
    return ""


def _norm_fact_key(raw: str) -> str:
    t = (raw or "").strip()
    if not t:
        return ""
    nk = _norm_slot(t)
    if nk:
        return nk
    low = t.lower()
    low = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", low).strip("_")
    return low[:64]


def _canonical_profile_key(raw: str) -> str:
    key = str(raw or "").strip()
    if not key:
        return ""
    if key.startswith(TS_SUMMARY_KEY_PREFIX):
        return "student_time_series_summary"
    if key.startswith("_matched_records_count:"):
        return "_matched_records_count"
    nk = _norm_fact_key(key)
    return nk or key


def _dedupe_slot_keys_for_view(slots: list[str] | None) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    has_ts = False
    for raw in slots or []:
        key = str(raw or "").strip()
        if not key:
            continue
        ck = _canonical_profile_key(key)
        if ck == "_matched_records_count":
            continue
        if ck == "student_time_series_summary":
            if has_ts:
                continue
            has_ts = True
        if ck in seen:
            continue
        seen.add(ck)
        ordered.append(ck)
    return ordered


def _dedupe_fact_candidates(facts: dict[str, str]) -> tuple[dict[str, str], dict]:
    """Compress redundant fact keys before relevance selection."""
    trace = {
        "raw_candidate_keys": list((facts or {}).keys()),
        "normalized_candidate_keys": [],
        "dropped": [],
        "rules": [
            "drop _matched_records_count:* metrics",
            "collapse file-level time-series keys to one semantic slot",
            "prefer file-level time-series summary over global summary when both exist",
        ],
    }
    if not facts:
        return {}, trace

    out: dict[str, str] = {}
    ts_file_items = [(k, v) for k, v in facts.items() if str(k).startswith(TS_SUMMARY_KEY_PREFIX)]
    has_ts_file = bool(ts_file_items)
    has_global_ts = "student_time_series_summary" in facts

    for key, value in facts.items():
        k = str(key or "").strip()
        if not k:
            continue
        if k.startswith("_matched_records_count:"):
            trace["dropped"].append({"key": k, "reason": "debug_metric"})
            continue
        if has_ts_file and k == "student_time_series_summary":
            trace["dropped"].append({"key": k, "reason": "prefer_file_level_time_series"})
            continue
        if k.startswith(TS_SUMMARY_KEY_PREFIX):
            ck = "student_time_series_summary"
            if ck in out:
                trace["dropped"].append({"key": k, "reason": "time_series_duplicate"})
                continue
            out[ck] = str(value)
            continue
        ck = _canonical_profile_key(k)
        if ck in out:
            trace["dropped"].append({"key": k, "reason": f"canonical_duplicate->{ck}"})
            continue
        out[ck] = str(value)

    if has_global_ts and not has_ts_file and "student_time_series_summary" in facts:
        out["student_time_series_summary"] = str(facts["student_time_series_summary"])

    trace["normalized_candidate_keys"] = list(out.keys())
    return out, trace


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)


def _detect_requested_slots(query: str) -> list[str]:
    q = (query or "").lower()
    slots: list[str] = []
    for slot, hints in PROFILE_SLOT_HINTS.items():
        if any(h.lower() in q for h in hints):
            slots.append(slot)
    return slots


def _extract_identity(query: str) -> dict[str, str]:
    text = query or ""
    out: dict[str, str] = {}
    sid_match = re.search(r"(?:学号|student\s*id|id)\s*[:：]?\s*([A-Za-z0-9\-_]{4,30})", text, flags=re.I)
    if sid_match:
        out["student_id"] = sid_match.group(1).strip()
    name_match = re.search(r"(?:姓名|名字|name)\s*[:：]?\s*([^\s，。,.；;]{2,20})", text, flags=re.I)
    if name_match:
        out["name"] = name_match.group(1).strip()
    college_match = re.search(r"(?:学院|院系|college)\s*[:：]?\s*([^\n，。,.；;]{2,40})", text, flags=re.I)
    if college_match:
        out["college"] = college_match.group(1).strip()
    major_match = re.search(r"(?:专业|major)\s*[:：]?\s*([^\n，。,.；;]{2,40})", text, flags=re.I)
    if major_match:
        out["major"] = major_match.group(1).strip()
    return out


def _normalize_row_facts(row: dict) -> dict[str, str]:
    if isinstance(row.get("normalized_fields"), dict):
        return {
            str(k): str(v).strip()
            for k, v in row.get("normalized_fields", {}).items()
            if str(k).strip() and str(v).strip()
        }
    out: dict[str, str] = {}
    for k, v in (row or {}).items():
        nk = _norm_fact_key(str(k))
        if not nk:
            continue
        if v is None:
            continue
        sv = str(v).strip()
        if not sv or sv.lower() in {"nan", "null", "none"}:
            continue
        out[nk] = sv
    return out


def _match_row_to_user(row: dict, user: dict) -> bool:
    if not row or not user:
        return False
    sid = str(user.get("student_id", "") or "").strip()
    name = str(user.get("name", "") or "").strip()

    nf = row.get("normalized_fields", {}) if isinstance(row.get("normalized_fields"), dict) else row
    row_sid = str(
        nf.get("student_id", "")
        or nf.get("student_pid", "")
        or row.get("学号", "")
        or row.get("student_pid", "")
        or ""
    ).strip()
    row_name = str(
        nf.get("name", "")
        or nf.get("display_name", "")
        or row.get("姓名", "")
        or row.get("display_name", "")
        or ""
    ).strip()

    if sid and row_sid and sid == row_sid:
        return True
    if sid and row_sid and sid != row_sid:
        return False
    if name and row_name and name == row_name:
        return True
    return False


def _parse_datetime_value(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace("/", "-")
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y%m%d%H%M%S",
        "%Y%m%d",
    ):
        try:
            return datetime.strptime(text[:19] if fmt.endswith("%S") else text, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _parse_number_value(value: str) -> float | None:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _period_for_hour(hour: int) -> str:
    if 5 <= hour < 10:
        return "breakfast"
    if 10 <= hour < 14:
        return "lunch"
    if 16 <= hour < 20:
        return "dinner"
    return "other"


def _time_column_priority(key: str) -> int:
    key_l = (key or "").strip().lower()
    score = 0
    if any(token in key_l for token in ("consume", "transaction", "occur", "event", "dt")):
        score += 30
    if any(token in key_l for token in ("date", "time")):
        score += 20
    if any(token in key_l for token in ("消费", "发生", "交易", "日期", "时间")):
        score += 30
    if any(token in key_l for token in ("update", "updated", "create", "created", "last")):
        score -= 40
    return score


def _build_time_series_summary(source_records: list[dict]) -> dict:
    if len(source_records) < 2:
        return {}
    rows = [r.get("fields", {}) or {} for r in source_records]
    keys = sorted({str(k) for row in rows for k in row.keys()})
    if not keys:
        return {}

    time_scores: list[tuple[int, int, str]] = []
    for key in keys:
        key_l = key.lower()
        hint = any(token in key_l for token in ("time", "date", "dt")) or any(
            token in key for token in ("时间", "日期")
        )
        parsed = sum(1 for row in rows if _parse_datetime_value(str(row.get(key, ""))))
        if parsed and (hint or parsed >= max(2, len(rows) // 2)):
            time_scores.append((parsed, _time_column_priority(key), key))
    if not time_scores:
        return {}
    time_scores.sort(key=lambda item: (item[0], item[1], -len(item[2])), reverse=True)
    time_key = time_scores[0][2]

    numeric_keys: list[tuple[int, str]] = []
    excluded = {"student_id", "name", time_key}
    for key in keys:
        if key in excluded or key.lower() in {"id"}:
            continue
        parsed = sum(1 for row in rows if _parse_number_value(str(row.get(key, ""))) is not None)
        if parsed >= max(2, len(rows) // 3):
            numeric_keys.append((parsed, key))
    numeric_keys.sort(reverse=True)
    if not numeric_keys:
        return {}
    value_keys = [k for _, k in numeric_keys[:4]]

    parsed_rows: list[dict] = []
    for row in rows:
        dt = _parse_datetime_value(str(row.get(time_key, "")))
        if not dt:
            continue
        values = {
            key: _parse_number_value(str(row.get(key, "")))
            for key in value_keys
        }
        values = {k: v for k, v in values.items() if v is not None}
        if values:
            parsed_rows.append({"dt": dt, "date": dt.date().isoformat(), "hour": dt.hour, "values": values})
    if not parsed_rows:
        return {}
    parsed_rows.sort(key=lambda item: item["dt"])

    def _stats(nums: list[float]) -> dict:
        if not nums:
            return {}
        return {
            "count": len(nums),
            "sum": round(sum(nums), 4),
            "mean": round(sum(nums) / len(nums), 4),
            "min": round(min(nums), 4),
            "max": round(max(nums), 4),
        }

    numeric_summary = {}
    for key in value_keys:
        nums = [r["values"][key] for r in parsed_rows if key in r["values"]]
        if nums:
            numeric_summary[key] = _stats(nums)

    primary_key = value_keys[0]
    daily: dict[str, list[float]] = {}
    periods: dict[str, list[float]] = {"breakfast": [], "lunch": [], "dinner": [], "other": []}
    for row in parsed_rows:
        if primary_key not in row["values"]:
            continue
        val = float(row["values"][primary_key])
        daily.setdefault(row["date"], []).append(val)
        periods[_period_for_hour(int(row["hour"]))].append(val)

    sorted_dates = sorted(daily.keys())
    daily_totals = [sum(daily[d]) for d in sorted_dates]
    daily_counts = [len(daily[d]) for d in sorted_dates]
    period_summary = {
        key: _stats(vals)
        for key, vals in periods.items()
        if vals
    }
    date_series = [
        {
            "date": d,
            "total": round(sum(daily[d]), 4),
            "count": len(daily[d]),
            "mean": round(sum(daily[d]) / max(len(daily[d]), 1), 4),
        }
        for d in sorted_dates
    ]
    if len(date_series) > MAX_DATE_SERIES_POINTS:
        date_series = date_series[-MAX_DATE_SERIES_POINTS:]

    trend_note = "stable"
    if len(daily_totals) >= 2:
        first = daily_totals[0]
        last = daily_totals[-1]
        if first > 0:
            delta_ratio = (last - first) / first
            if delta_ratio >= 0.2:
                trend_note = "upward"
            elif delta_ratio <= -0.2:
                trend_note = "downward"
        elif last > 0:
            trend_note = "upward"

    return {
        "kind": "time_series_summary",
        "matched_record_sample_count": len(source_records),
        "analyzed_record_count": len(parsed_rows),
        "time_column": time_key,
        "numeric_columns": value_keys,
        "time_range": [
            parsed_rows[0]["dt"].isoformat(sep=" "),
            parsed_rows[-1]["dt"].isoformat(sep=" "),
        ],
        "numeric_summary": numeric_summary,
        "daily_summary": {
            "active_days": len(daily),
            "avg_daily_total": round(sum(daily_totals) / len(daily_totals), 4) if daily_totals else 0.0,
            "avg_daily_count": round(sum(daily_counts) / len(daily_counts), 4) if daily_counts else 0.0,
        },
        "date_series": date_series,
        "date_trend_summary": {
            "trend": trend_note,
            "first_day_total": round(daily_totals[0], 4) if daily_totals else 0.0,
            "last_day_total": round(daily_totals[-1], 4) if daily_totals else 0.0,
            "active_days": len(daily),
        },
        "period_summary_for_primary_numeric": {
            "primary_numeric_column": primary_key,
            "periods": period_summary,
        },
    }


def _build_time_series_trace(source_records: list[dict], summary: dict) -> dict:
    rows = [r.get("fields", {}) or {} for r in source_records]
    keys = sorted({str(k) for row in rows for k in row.keys()})
    time_candidates = []
    numeric_candidates = []
    for key in keys:
        key_l = key.lower()
        time_hint = any(token in key_l for token in ("time", "date", "dt")) or any(
            token in key for token in ("时间", "日期")
        )
        time_candidates.append(
            {
                "column": key,
                "parseable_count": sum(1 for row in rows if _parse_datetime_value(str(row.get(key, "")))),
                "hint": bool(time_hint),
                "priority": _time_column_priority(key),
            }
        )
        numeric_candidates.append(
            {
                "column": key,
                "parseable_count": sum(
                    1 for row in rows if _parse_number_value(str(row.get(key, ""))) is not None
                ),
            }
        )
    time_candidates.sort(key=lambda item: (item["parseable_count"], item["priority"]), reverse=True)
    numeric_candidates.sort(key=lambda item: item["parseable_count"], reverse=True)
    return {
        "input_record_count": len(source_records),
        "candidate_columns": keys,
        "time_column_candidates": time_candidates[:12],
        "numeric_column_candidates": numeric_candidates[:12],
        "selected_time_column": summary.get("time_column", "") if isinstance(summary, dict) else "",
        "selected_numeric_columns": summary.get("numeric_columns", []) if isinstance(summary, dict) else [],
        "summary_generated": bool(summary),
        "summary": summary or {},
    }


def _build_file_level_time_series_summaries(source_records: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for r in source_records or []:
        if not isinstance(r, dict):
            continue
        source_file = str(r.get("source_file", "") or "").strip()
        if not source_file:
            continue
        grouped.setdefault(source_file, []).append(r)
    out: dict[str, dict] = {}
    for source_file, rows in grouped.items():
        summary = _build_time_series_summary(rows)
        if summary:
            out[source_file] = summary
    return out


def _retrieve_doc_facts(
    kb_id: str,
    user: dict,
    requested_slots: list[str],
    state_lib_id: str = "",
) -> tuple[dict[str, str], list[str], int, list[dict]]:
    facts: dict[str, str] = {}
    doc_sources: list[str] = []
    source_records: list[dict] = []
    if not kb_id:
        return facts, doc_sources, 0, source_records

    docs = DB.student_doc_list(kb_id, state_lib_id=state_lib_id)
    req = [_norm_slot(s) for s in requested_slots if _norm_slot(s)]
    req_set = set(req)
    fetch_all = (not req_set) or (ALL_STUDENT_STATUS_FIELDS in requested_slots)
    for doc in docs:
        parsed = doc.get("parsed", {}) or {}
        hit = False
        # table rows: require user identity match to prevent leaking another student's row.
        rows = parsed.get("rows", [])
        if isinstance(rows, list) and rows:
            matched_rows: list[dict] = []
            matched_total = 0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if _match_row_to_user(row, user):
                    matched_total += 1
                    if len(matched_rows) < MAX_MATCHED_RECORDS_PER_DOC:
                        matched_rows.append(row)
            for matched_row in matched_rows:
                rf = _normalize_row_facts(matched_row)
                record_fields: dict[str, str] = {}
                if fetch_all:
                    for key, value in rf.items():
                        if value:
                            if key not in facts:
                                facts[key] = value
                            record_fields[key] = value
                            hit = True
                else:
                    for key in req_set:
                        if key in rf and rf[key]:
                            if key not in facts:
                                facts[key] = rf[key]
                            record_fields[key] = rf[key]
                            hit = True
                if record_fields:
                    source_records.append(
                        {
                            "source_file": str(matched_row.get("source_file") or doc.get("name", "")),
                            "sheet": str(matched_row.get("sheet", "")),
                            "orientation": str(matched_row.get("orientation", "horizontal")),
                            "fields": _normalize_row_facts(matched_row),
                        }
                    )
            if matched_total > 1:
                facts[f"_matched_records_count:{doc.get('id', doc.get('name', 'doc'))}"] = str(matched_total)

        # kv facts: shared keys, direct use.
        kv = parsed.get("kv", {})
        if isinstance(kv, dict):
            nkv = _normalize_row_facts(kv)
            if fetch_all:
                for key, value in nkv.items():
                    if value:
                        facts[key] = value
                        hit = True
            else:
                for key in req_set:
                    if key in nkv and nkv[key]:
                        facts[key] = nkv[key]
                        hit = True

        if hit:
            doc_sources.append(str(doc.get("name", "")))
    return facts, doc_sources, len(docs), source_records


def _collect_previously_retrieved_slots(prior_messages: list[dict]) -> set[str]:
    slots: set[str] = set()
    for msg in prior_messages or []:
        if str(msg.get("role", "")).strip().lower() != "assistant":
            continue
        meta = msg.get("meta", {}) or {}
        pr = meta.get("profile_retrieval", {}) or {}
        for k in pr.get("requested_slots", []) or []:
            nk = _norm_slot(str(k))
            if nk:
                slots.add(nk)
    return slots


def _collect_previous_profile_facts(prior_messages: list[dict]) -> dict[str, str]:
    facts: dict[str, str] = {}
    for msg in prior_messages or []:
        if str(msg.get("role", "")).strip().lower() != "assistant":
            continue
        meta = msg.get("meta", {}) or {}
        pr = meta.get("profile_retrieval", {}) or {}
        raw_facts = pr.get("profile_facts", {}) or {}
        if not isinstance(raw_facts, dict):
            continue
        for k, v in raw_facts.items():
            nk = _norm_fact_key(str(k))
            vv = str(v).strip()
            if nk and vv:
                facts[nk] = vv
    return facts


def _extract_updates_regex(query: str) -> dict[str, str]:
    text = query or ""
    updates: dict[str, str] = {}

    patterns: list[tuple[str, str]] = [
        ("student_id", r"(?:学号|student\s*id|id)\s*[:：]?\s*([A-Za-z0-9\-_]{4,30})"),
        ("name", r"(?:姓名|名字|name)\s*[:：]?\s*([^\s，。,.；;]{2,20})"),
        ("college", r"(?:学院|院系|college)\s*[:：]?\s*([^\n，。,.；;]{2,40})"),
        ("major", r"(?:专业|major)\s*[:：]?\s*([^\n，。,.；;]{2,40})"),
        ("class_name", r"(?:班级|class)\s*[:：]?\s*([A-Za-z0-9\u4e00-\u9fff\-_]{2,30})"),
        ("grade_year", r"(?:年级|入学)\s*[:：]?\s*([0-9]{2,4})"),
        ("gpa", r"(?:绩点|gpa)\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)"),
        ("score", r"(?:成绩|分数)\s*[:：]?\s*([0-9]{1,3}(?:\.[0-9]+)?)"),
        ("credits", r"(?:学分|credit)\s*[:：]?\s*([0-9]{1,4}(?:\.[0-9]+)?)"),
    ]
    for key, pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            updates[key] = m.group(1).strip()
    return updates


def _extract_updates_llm(query: str) -> dict[str, str]:
    if not _contains_any(query, FIRST_PERSON_HINTS):
        return {}
    system = (
        "Extract explicit self-reported student-profile facts from the user text. "
        "Output JSON only: {\"facts\":{\"slot\":\"value\",...}}. "
        "Only keep concrete profile facts, no inference."
    )
    user = (
        "Allowed slots: student_id,name,college,major,class_name,grade_year,gpa,score,credits,rank,attendance,scholarship,warning_status.\n"
        f"User text:\n{query}\n"
    )
    try:
        data = llm_client.chat_json(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=512,
            parse_retry=1,
        )
        facts = (data or {}).get("facts", {}) if isinstance(data, dict) else {}
        out: dict[str, str] = {}
        if isinstance(facts, dict):
            for k, v in facts.items():
                nk = _norm_slot(str(k))
                vv = str(v).strip()
                if nk and vv:
                    out[nk] = vv
        return out
    except Exception as e:
        logger.debug(f"LLM profile fact extraction skipped: {e}")
        return {}


def _llm_review_profile_context(query: str, facts: dict[str, str]) -> str:
    if not facts:
        return ""
    payload = ", ".join(f"{PROFILE_SLOT_LABELS.get(k, k)}={v}" for k, v in facts.items())
    system = (
        "You validate whether retrieved student-profile facts are relevant to the question. "
        "Output one short sentence only."
    )
    user = (
        f"Question: {query}\n"
        f"Profile facts: {payload}\n"
        "Reply format: [high|medium|low] + short reason. "
        "No extra text."
    )
    try:
        text = llm_client.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=160,
        ).strip()
        return text[:200]
    except Exception as e:
        logger.debug(f"LLM profile review skipped: {e}")
        return ""


def _select_relevant_facts_llm(
    query: str,
    facts: dict[str, str],
    source_records: list[dict],
) -> tuple[dict[str, str], str, dict]:
    normalized_facts, dedupe_trace = _dedupe_fact_candidates(facts or {})
    base_trace = {
        "question": query,
        "candidate_count": len(normalized_facts or {}),
        "candidate_keys": list((normalized_facts or {}).keys()),
        "source_record_preview_count": min(len(source_records or []), 5),
        "llm_used": False,
        "mode": "",
        "selected_keys": [],
        "dropped_keys": [],
        "reason": "",
        "key_dedupe_trace": dedupe_trace,
    }
    if not normalized_facts:
        base_trace["mode"] = "empty"
        return {}, "", base_trace
    if len(normalized_facts) <= 12:
        selected = dict(normalized_facts)
        base_trace.update(
            {
                "mode": "all_small_payload",
                "selected_keys": list(selected.keys()),
                "dropped_keys": [],
                "reason": "candidate fact count <= 12",
            }
        )
        return selected, "relevance_filter=all_small_payload", base_trace
    payload = {
        "question": query,
        "facts": normalized_facts,
        "source_records": [
            {
                "source_file": r.get("source_file", ""),
                "sheet": r.get("sheet", ""),
                "orientation": r.get("orientation", ""),
                "fields": r.get("fields", {}),
            }
            for r in source_records[:5]
        ],
    }
    system = (
        "Select student-status fields relevant to the current question. "
        "Output JSON only: {\"relevant_keys\":[\"...\"],\"reason\":\"...\"}. "
        "Keep identity/source fields if useful. Do not invent keys."
    )
    try:
        base_trace["llm_input"] = payload
        data = llm_client.chat_json(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.0,
            max_tokens=800,
            parse_retry=1,
        )
        raw_keys = data.get("relevant_keys", []) if isinstance(data, dict) else []
        keys = [str(k).strip() for k in raw_keys if str(k).strip() in normalized_facts]
        for identity_key in ("student_id", "name"):
            if identity_key in normalized_facts and identity_key not in keys:
                keys.insert(0, identity_key)
        ts_keys = [k for k in normalized_facts.keys() if str(k).startswith(TS_SUMMARY_KEY_PREFIX) or str(k) == "student_time_series_summary"]
        for ts_key in ts_keys:
            if ts_key not in keys:
                keys.insert(0, ts_key)
        if keys:
            reason = str((data or {}).get("reason", "")).strip()[:200] if isinstance(data, dict) else ""
            picked = {k: normalized_facts[k] for k in keys}
            base_trace.update(
                {
                    "llm_used": True,
                    "mode": "llm",
                    "llm_output": data,
                    "selected_keys": keys,
                    "dropped_keys": [k for k in normalized_facts.keys() if k not in keys],
                    "reason": reason,
                }
            )
            return picked, f"relevance_filter=llm; {reason}".strip(), base_trace
    except Exception as e:
        base_trace["llm_error"] = str(e)
        logger.debug(f"LLM relevance filter skipped: {e}")

    q = (query or "").lower()
    picked = {
        k: v
        for k, v in normalized_facts.items()
        if (
            k in {"student_id", "name"}
            or str(k).startswith(TS_SUMMARY_KEY_PREFIX)
            or k.lower() in q
            or any(ch in q for ch in k if "\u4e00" <= ch <= "\u9fff")
        )
    }
    if not picked:
        picked = dict(list(facts.items())[:24])
    base_trace.update(
        {
            "mode": "fallback",
            "selected_keys": list(picked.keys()),
            "dropped_keys": [k for k in normalized_facts.keys() if k not in picked],
            "reason": "keyword overlap fallback or first 24 facts",
        }
    )
    return picked, "relevance_filter=fallback", base_trace


def _build_student_status_context_json(
    *,
    user: dict,
    state_lib_id: str,
    facts: dict[str, str],
    relevant_facts: dict[str, str],
    source_records: list[dict],
    doc_sources: list[str],
    relevance_note: str,
) -> tuple[str, dict]:
    relevant_keys = set(relevant_facts.keys())
    records = []
    time_series_summary = _build_time_series_summary(source_records)
    file_time_series_summaries = _build_file_level_time_series_summaries(source_records)
    time_series_trace = _build_time_series_trace(source_records, time_series_summary)
    file_record_counts: dict[str, int] = {}
    for r in source_records:
        source_file = str(r.get("source_file", "") or "").strip()
        if source_file:
            file_record_counts[source_file] = file_record_counts.get(source_file, 0) + 1
    time_series_only_files = {
        source_file
        for source_file, count in file_record_counts.items()
        if count > TIME_SERIES_ONLY_ROW_THRESHOLD and source_file in file_time_series_summaries
    }
    for r in source_records[:MAX_CONTEXT_RECORDS]:
        source_file = str(r.get("source_file", "") or "").strip()
        if source_file in time_series_only_files:
            # For dense time-series files, only keep the summary in context.
            continue
        all_fields = r.get("fields", {}) or {}
        records.append(
            {
                "source_file": source_file,
                "sheet": r.get("sheet", ""),
                "orientation": r.get("orientation", ""),
                "fields_for_current_question": {
                    k: v for k, v in all_fields.items() if k in relevant_keys
                },
                "matched_student_fields": all_fields,
            }
        )
    payload = {
        "context_type": "student_status",
        "student_identity": {
            "student_id": str(user.get("student_id", "") or ""),
            "name": str(user.get("name", "") or ""),
            "college": str(user.get("college", "") or ""),
            "major": str(user.get("major", "") or ""),
        },
        "state_lib_id": state_lib_id,
        "source_files": doc_sources,
        "relevance_note": relevance_note,
        "matched_record_count": len(source_records),
        "context_record_limit": MAX_CONTEXT_RECORDS,
        "time_series_only_files": sorted(time_series_only_files),
        "time_series_summary": time_series_summary,
        "file_time_series_summaries": file_time_series_summaries,
        "relevant_fields": relevant_facts,
        "matched_records": records,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2), time_series_trace


def prepare_profile_context(
    *,
    kb_id: str = "",
    conv_id: str,
    user_query: str,
    prior_messages: list[dict],
    bound_user_id: str = "",
    state_lib_id: str = "",
    enable_llm_review: bool = True,
    force_refresh: bool = False,
) -> ProfileContextResult:
    query = (user_query or "").strip()
    result = ProfileContextResult(
        state_lib_id=str(state_lib_id or "").strip(),
        requested_slots=[],
        fetched_slots=[],
        reused_slots=[],
        profile_facts={},
        relevant_facts={},
        source_records=[],
        extracted_updates={},
        doc_sources=[],
        doc_count=0,
    )
    if not conv_id or not query:
        return result

    conv = DB.conv_get(conv_id) or {}
    kb_id = (kb_id or "").strip() or str(conv.get("kb_id", "")).strip()
    user_id = str(conv.get("user_id") or "").strip() or str(bound_user_id or "").strip()

    identity = _extract_identity(query)
    if not user_id and (identity.get("student_id") or identity.get("name")):
        u = DB.user_upsert(
            student_id=identity.get("student_id", ""),
            name=identity.get("name", ""),
            college=identity.get("college", ""),
            major=identity.get("major", ""),
        )
        user_id = str(u.get("id", "")).strip()
        if user_id:
            DB.conv_bind_user(conv_id, user_id)
            result.identity_note = "auto_bound_from_query_identity"

    if user_id and str(conv.get("user_id") or "").strip() != user_id:
        DB.conv_bind_user(conv_id, user_id)

    result.user_id = user_id
    if not user_id:
        return result

    user = DB.user_get(user_id) or {}
    result.identity_ok = True
    if identity.get("student_id") and user.get("student_id"):
        if identity["student_id"].strip() != str(user.get("student_id", "")).strip():
            result.identity_ok = False
            result.identity_note = "student_id_mismatch"
    if result.identity_ok and identity.get("name") and user.get("name"):
        if identity["name"].strip() != str(user.get("name", "")).strip():
            result.identity_ok = False
            result.identity_note = "name_mismatch"

    prev_slots = _collect_previously_retrieved_slots(prior_messages)
    prev_facts = _collect_previous_profile_facts(prior_messages)
    refresh = bool(force_refresh) or _contains_any(query, REFRESH_HINTS)
    req_slots = set(_detect_requested_slots(query))
    if state_lib_id:
        # Strong-cache mode for student-state retrieval:
        # after first successful import into profile_facts, do not re-scan
        # state-library files unless user explicitly refreshes.
        state_facts = {
            str(item.get("fact_key", "")): str(item.get("fact_value", ""))
            for item in DB.profile_fact_list(user_id)
            if str(item.get("fact_key", "")).strip()
            and str(item.get("source", "")).startswith(f"student_state_lib:{state_lib_id}")
        }
        if state_facts and not refresh:
            context_slots = sorted(state_facts.keys())
            fetched_slots = []
            reused_slots = list(context_slots)
            prev_facts.update(state_facts)
        else:
            context_slots = [ALL_STUDENT_STATUS_FIELDS]
            fetched_slots = context_slots
            reused_slots = []
    else:
        bootstrap = set(PROFILE_BOOTSTRAP_SLOTS) if not prev_slots else set()
        context_slots = sorted(req_slots | bootstrap)
        fetched_slots = sorted(set(context_slots) if refresh else (set(context_slots) - prev_slots))
        reused_slots = sorted(set(context_slots) & prev_slots)

    # Parse self-reported updates first (regex + optional LLM), then persist.
    updates = _extract_updates_regex(query)
    if _contains_any(query, FIRST_PERSON_HINTS):
        llm_updates = _extract_updates_llm(query)
        updates.update({k: v for k, v in llm_updates.items() if k and v})
    if updates:
        DB.profile_fact_bulk_upsert(
            user_id=user_id,
            facts=updates,
            source="query_extract",
            confidence=0.75,
        )
    result.extracted_updates = dict(sorted(updates.items()))

    facts = {
        k: v
        for k, v in prev_facts.items()
        if (k in set(context_slots) or ALL_STUDENT_STATUS_FIELDS in context_slots) and v and not refresh
    }
    db_lookup_slots = [s for s in context_slots if s != ALL_STUDENT_STATUS_FIELDS]
    if state_lib_id and not refresh:
        db_facts = {
            str(item.get("fact_key", "")): str(item.get("fact_value", ""))
            for item in DB.profile_fact_list(user_id)
            if str(item.get("fact_key", "")).strip()
        }
    else:
        db_facts = DB.profile_fact_get_many(user_id, db_lookup_slots)
    facts.update({k: v for k, v in db_facts.items() if v})
    if state_lib_id and (not refresh) and (not fetched_slots):
        # Reuse-only round: avoid re-scanning state-library documents.
        doc_facts, doc_sources, doc_count, source_records = {}, [], 0, []
    else:
        doc_facts, doc_sources, doc_count, source_records = _retrieve_doc_facts(
            kb_id=kb_id,
            user=user,
            requested_slots=fetched_slots,
            state_lib_id=state_lib_id,
        )
    result.doc_sources = doc_sources
    result.doc_count = int(doc_count)
    result.source_records = source_records

    # Student-doc facts fill missing slots; explicit/manual facts keep higher priority.
    doc_updates: dict[str, str] = {}
    for k, v in doc_facts.items():
        if not facts.get(k):
            facts[k] = v
            doc_updates[k] = v
    # Fill identity base fields if DB facts are absent.
    base_map = {
        "student_id": str(user.get("student_id", "") or ""),
        "name": str(user.get("name", "") or ""),
        "college": str(user.get("college", "") or ""),
        "major": str(user.get("major", "") or ""),
    }
    for key, value in base_map.items():
        if key in context_slots and value and not facts.get(key):
            facts[key] = value

    result.requested_slots = _dedupe_slot_keys_for_view(context_slots)
    result.fetched_slots = _dedupe_slot_keys_for_view(fetched_slots)
    result.reused_slots = _dedupe_slot_keys_for_view(reused_slots)
    result.profile_facts = dict(sorted(facts.items()))

    if not result.identity_ok:
        result.profile_context = (
            "用户身份信息与当前会话绑定学生不一致，已停止注入学生状态。"
        )
        return result

    if not facts:
        return result

    if doc_updates:
        DB.profile_fact_bulk_upsert(
            user_id=user_id,
            facts=doc_updates,
            source=f"student_state_lib:{state_lib_id or 'all'}",
            confidence=0.9,
        )

    relevant_facts, relevance_note, relevance_trace = _select_relevant_facts_llm(query, result.profile_facts, source_records)
    result.relevant_facts = relevant_facts
    result.relevance_filter_trace = relevance_trace
    if relevant_facts:
        result.reused_slots = _dedupe_slot_keys_for_view(list((result.reused_slots or [])) + list(relevant_facts.keys()))
    context, time_series_trace = _build_student_status_context_json(
        user=user,
        state_lib_id=state_lib_id,
        facts=result.profile_facts,
        relevant_facts=relevant_facts,
        source_records=source_records,
        doc_sources=result.doc_sources or [],
        relevance_note=relevance_note,
    )
    result.time_series_trace = time_series_trace
    # Persist date-aggregated time-series summary as profile fact for reuse across turns.
    ts_summary = ((time_series_trace or {}).get("summary") or {}) if isinstance(time_series_trace, dict) else {}
    file_ts_summaries = _build_file_level_time_series_summaries(source_records)
    ts_fact_updates: dict[str, str] = {}
    for source_file, summary in (file_ts_summaries or {}).items():
        ts_key = f"{TS_SUMMARY_KEY_PREFIX}{source_file}"
        ts_fact_updates[ts_key] = json.dumps(summary, ensure_ascii=False)
    if ts_fact_updates:
        DB.profile_fact_bulk_upsert(
            user_id=user_id,
            facts=ts_fact_updates,
            source=f"student_state_lib:{state_lib_id or 'all'}",
            confidence=0.85,
        )
        for k, v in ts_fact_updates.items():
            if k not in result.profile_facts:
                result.profile_facts[k] = v
    if ts_summary:
        try:
            DB.profile_fact_upsert(
                user_id=user_id,
                fact_key="student_time_series_summary",
                fact_value=json.dumps(ts_summary, ensure_ascii=False),
                source=f"student_state_lib:{state_lib_id or 'all'}",
                confidence=0.85,
            )
            if "student_time_series_summary" not in result.profile_facts:
                result.profile_facts["student_time_series_summary"] = json.dumps(ts_summary, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"time-series summary upsert skipped: {e}")
    if enable_llm_review:
        review = _llm_review_profile_context(query, relevant_facts)
        result.llm_review = review
    result.profile_context = context
    return result
