from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    return " ".join(remove_articles(remove_punc(str(text).lower())).split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def answer_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def chunk_id_from_text_or_meta(item: dict[str, Any]) -> str:
    meta = item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
    for key in ("hotpot_chunk_id", "chunk_id"):
        if meta.get(key):
            return str(meta[key])
    text = str(item.get("text", "") or item.get("content", ""))
    match = re.search(r"hotpot_chunk_id:\s*([^\s]+)", text)
    return match.group(1).strip() if match else ""


def retrieval_scores(retrieved_ids: list[str], gold_ids: list[str]) -> dict[str, float]:
    retrieved = {str(x) for x in retrieved_ids if str(x).strip()}
    gold = {str(x) for x in gold_ids if str(x).strip()}
    hit = retrieved & gold
    precision = len(hit) / len(retrieved) if retrieved else 0.0
    recall = len(hit) / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "retrieval_precision": precision,
        "retrieval_recall": recall,
        "retrieval_f1": f1,
        "retrieved_count": float(len(retrieved)),
        "gold_count": float(len(gold)),
        "hit_count": float(len(hit)),
    }


def load_questions_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_gold_map(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("gold map must be a JSON object keyed by sample id")
    return data


def summarize_records(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(str(rec.get("method", "")), []).append(rec)
    out: dict[str, dict[str, float]] = {}
    for method, rows in grouped.items():
        n = len(rows)
        if not n:
            continue
        out[method] = {
            "n": float(n),
            "failures": float(sum(1 for r in rows if r.get("error"))),
            "em": sum(float(r.get("em", 0.0) or 0.0) for r in rows) / n,
            "answer_f1": sum(float(r.get("answer_f1", 0.0) or 0.0) for r in rows) / n,
            "retrieval_precision": sum(float(r.get("retrieval_precision", 0.0) or 0.0) for r in rows) / n,
            "retrieval_recall": sum(float(r.get("retrieval_recall", 0.0) or 0.0) for r in rows) / n,
            "retrieval_f1": sum(float(r.get("retrieval_f1", 0.0) or 0.0) for r in rows) / n,
            "avg_retrieved_count": sum(float(r.get("retrieved_count", 0.0) or 0.0) for r in rows) / n,
            "avg_elapsed_s": sum(float(r.get("elapsed_s", 0.0) or 0.0) for r in rows) / n,
        }
    return out

