"""
Evaluation module.

Supports:
- Exact match (EM)
- F1 token overlap
- LLM-as-judge scoring (correctness + causal_quality)
- Batch evaluation over a QA dataset (JSON lines format)

Dataset format (JSONL, one per line):
  {"question": "...", "answer": "...", "context": "optional gold context"}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from loguru import logger

from utils.llm_client import llm_client


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = set(_normalize(pred).split())
    gold_tokens = set(_normalize(gold).split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


_JUDGE_SYSTEM = """\
你是一个严格、客观的答案评分专家。

给定问题、参考答案和模型回答，请从两个维度打分（0-10分）：
1. correctness: 回答与参考答案内容的一致程度
2. causal_quality: 回答中因果推理的清晰度、准确性和深度（若问题不涉及因果则给5分）

输出 JSON：{"correctness": 0-10, "causal_quality": 0-10, "comment": "简短理由"}
"""


def llm_judge(question: str, reference: str, prediction: str) -> dict:
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"问题：{question}\n\n"
                f"参考答案：{reference}\n\n"
                f"模型回答：{prediction}"
            ),
        },
    ]
    try:
        return llm_client.chat_json(messages)
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return {"correctness": 0, "causal_quality": 0, "comment": str(e)}


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    question: str
    gold_answer: str
    prediction: str
    em: float = 0.0
    f1: float = 0.0
    correctness: float = 0.0
    causal_quality: float = 0.0
    judge_comment: str = ""


@dataclass
class EvalSummary:
    n: int
    avg_em: float
    avg_f1: float
    avg_correctness: float
    avg_causal_quality: float
    samples: list[EvalSample] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"N={self.n} | EM={self.avg_em:.3f} | F1={self.avg_f1:.3f} | "
            f"Correctness={self.avg_correctness:.2f}/10 | "
            f"CausalQuality={self.avg_causal_quality:.2f}/10"
        )


class Evaluator:
    def __init__(self, use_llm_judge: bool = True):
        self.use_llm_judge = use_llm_judge

    def evaluate_sample(self, question: str, gold: str, prediction: str) -> EvalSample:
        sample = EvalSample(
            question=question,
            gold_answer=gold,
            prediction=prediction,
        )
        sample.em = exact_match(prediction, gold)
        sample.f1 = f1_score(prediction, gold)

        if self.use_llm_judge:
            scores = llm_judge(question, gold, prediction)
            sample.correctness = float(scores.get("correctness", 0))
            sample.causal_quality = float(scores.get("causal_quality", 0))
            sample.judge_comment = str(scores.get("comment", ""))

        return sample

    def evaluate_dataset(
        self,
        dataset_path: str,
        predict_fn: Callable[[str], str],
        output_path: str | None = None,
    ) -> EvalSummary:
        """
        Args:
            dataset_path: path to JSONL file with {"question", "answer"} records
            predict_fn: callable str → str (system under test)
            output_path: optional path to save per-sample results as JSONL
        """
        samples: list[EvalSample] = []

        with open(dataset_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        logger.info(f"Evaluating {len(lines)} examples from {dataset_path}...")

        for i, line in enumerate(lines):
            record = json.loads(line)
            question = record["question"]
            gold = record["answer"]

            logger.debug(f"[{i+1}/{len(lines)}] {question[:60]}...")
            try:
                prediction = predict_fn(question)
            except Exception as e:
                logger.error(f"predict_fn failed on sample {i}: {e}")
                prediction = ""

            sample = self.evaluate_sample(question, gold, prediction)
            samples.append(sample)

            if (i + 1) % 10 == 0:
                logger.info(
                    f"Progress {i+1}/{len(lines)} | "
                    f"avg_F1={sum(s.f1 for s in samples)/len(samples):.3f}"
                )

        summary = self._summarize(samples)
        logger.info(f"Evaluation complete: {summary}")

        if output_path:
            self._save_results(samples, output_path)

        return summary

    @staticmethod
    def _summarize(samples: list[EvalSample]) -> EvalSummary:
        n = len(samples)
        if n == 0:
            return EvalSummary(0, 0.0, 0.0, 0.0, 0.0)
        return EvalSummary(
            n=n,
            avg_em=sum(s.em for s in samples) / n,
            avg_f1=sum(s.f1 for s in samples) / n,
            avg_correctness=sum(s.correctness for s in samples) / n,
            avg_causal_quality=sum(s.causal_quality for s in samples) / n,
            samples=samples,
        )

    @staticmethod
    def _save_results(samples: list[EvalSample], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps({
                    "question": s.question,
                    "gold": s.gold_answer,
                    "prediction": s.prediction,
                    "em": s.em,
                    "f1": s.f1,
                    "correctness": s.correctness,
                    "causal_quality": s.causal_quality,
                    "comment": s.judge_comment,
                }, ensure_ascii=False) + "\n")
        logger.info(f"Results saved to {path}")
