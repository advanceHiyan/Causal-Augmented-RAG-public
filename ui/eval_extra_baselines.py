from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Any

from config.settings import settings
from utils.llm_client import llm_client


EXTRA_METHODS = {
    "bm25": "BM25",
    "cosine_shuffle": "Cosine",
    "vector_analysis": "Vector+Analysis",
    "external_placeholder": "External-HopRAG-Placeholder",
}


def _tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    zh = re.findall(r"[\u4e00-\u9fff]", text)
    words = re.findall(r"[a-z0-9_]{2,}", text)
    return zh + words


def _all_vector_chunks(vector_store: Any) -> list[dict[str, Any]]:
    collection = getattr(vector_store, "_collection", None)
    if collection is None:
        return []
    try:
        got = collection.get(include=["documents", "metadatas"])
    except Exception:
        return []
    ids = got.get("ids", []) or []
    docs = got.get("documents", []) or []
    metas = got.get("metadatas", []) or []
    out = []
    for idx, doc in enumerate(docs):
        out.append(
            {
                "id": ids[idx] if idx < len(ids) else f"bm25-{idx}",
                "text": doc or "",
                "score": 0.0,
                "metadata": metas[idx] if idx < len(metas) else {},
            }
        )
    return out


def bm25_retrieve(vector_store: Any, query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
    chunks = _all_vector_chunks(vector_store)
    if not chunks:
        return []
    q_terms = _tokenize(query)
    if not q_terms:
        return chunks[:top_k]
    q_set = set(q_terms)
    doc_terms = [_tokenize(str(c.get("text", ""))) for c in chunks]
    n_docs = len(doc_terms)
    avgdl = sum(len(t) for t in doc_terms) / max(n_docs, 1)
    df: dict[str, int] = {}
    for terms in doc_terms:
        for term in set(terms):
            if term in q_set:
                df[term] = df.get(term, 0) + 1
    k1 = 1.5
    b = 0.75
    scored = []
    for chunk, terms in zip(chunks, doc_terms):
        tf: dict[str, int] = {}
        for term in terms:
            if term in q_set:
                tf[term] = tf.get(term, 0) + 1
        dl = len(terms) or 1
        score = 0.0
        for term, freq in tf.items():
            idf = math.log(1.0 + (n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5))
            score += idf * (freq * (k1 + 1.0)) / (freq + k1 * (1.0 - b + b * dl / max(avgdl, 1e-6)))
        row = dict(chunk)
        row["score"] = float(score)
        scored.append(row)
    scored.sort(key=lambda c: float(c.get("score", 0.0) or 0.0), reverse=True)
    return scored[:top_k]


def chunk_ids(chunks: list[dict[str, Any]]) -> list[str]:
    return [str(c.get("id", "")) for c in chunks if str(c.get("id", "")).strip()]


def build_context(chunks: list[dict[str, Any]], *, max_chars: int | None = None) -> str:
    parts = []
    used = 0
    for idx, chunk in enumerate(chunks, 1):
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        piece = f"[片段 {idx}]\n{text}"
        if max_chars is not None and used + len(piece) > max_chars:
            break
        parts.append(piece)
        used += len(piece)
    return "\n\n".join(parts)


def generate_answer(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    mode: str,
    answer_instruction: str = "",
) -> str:
    context = build_context(chunks, max_chars=None)
    if mode == "vector_analysis":
        system = (
            "你是简化版RAG问答助手。只用给定片段作答。"
            "先在心中判断哪些片段最相关，再直接给出答案；不要输出分析过程。"
            "证据不足时简洁说明无法确定。"
        )
    else:
        system = "你是RAG问答助手。只使用给定片段作答；证据不足时说明无法确定。"
    instruction = f"\n\n额外输出要求：\n{answer_instruction}" if answer_instruction.strip() else ""
    user = f"问题：\n{question}\n\n检索片段：\n{context or '(无)'}{instruction}"
    return str(
        llm_client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=max(2048, int(settings.llm.max_tokens)),
        )
        or ""
    ).strip()


def make_branch_payload(
    *,
    mode: str,
    title: str,
    answer: str,
    chunks: list[dict[str, Any]],
    elapsed_s: float,
    steps: list[str],
) -> dict[str, Any]:
    return {
        "mode": mode,
        "title": title,
        "answer": answer,
        "elapsed_s": elapsed_s,
        "l1": False,
        "l2": False,
        "chunks": len(chunks),
        "l3_used_reranker": False,
        "causal_graph_text": "",
        "causal_path_summary": "",
        "vector_snippets": chunks,
        "steps": steps,
    }


def run_extra_baselines(
    *,
    mgr: Any,
    question: str,
    retrieval_query: str,
    vector_only_branch: dict[str, Any],
    top_k: int = 5,
    answer_instruction: str = "",
    include_external_placeholder: bool = True,
    only_methods: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    selected = set(only_methods or {"bm25", "cosine_shuffle", "vector_analysis"})

    vector_chunks: list[dict[str, Any]] = []
    vector_source = "existing_vector_only"
    if selected & {"cosine_shuffle", "vector_analysis"}:
        vector_chunks = [dict(c) for c in (vector_only_branch.get("vector_snippets", []) or []) if isinstance(c, dict)]
    if (selected & {"cosine_shuffle", "vector_analysis"}) and not vector_chunks:
        try:
            vector_chunks = [dict(c) for c in (mgr.vector_store.search(retrieval_query, top_k=top_k) or [])]
            vector_source = "direct_cosine_search_no_reranker"
        except Exception:
            vector_chunks = []
            vector_source = "direct_cosine_search_failed"

    def _run_bm25() -> tuple[str, dict[str, Any]]:
        t0 = time.perf_counter()
        bm25_chunks = bm25_retrieve(mgr.vector_store, retrieval_query, top_k=top_k)
        bm25_answer = generate_answer(question, bm25_chunks, mode="bm25", answer_instruction=answer_instruction)
        return "bm25", make_branch_payload(
            mode="bm25",
            title=EXTRA_METHODS["bm25"],
            answer=bm25_answer,
            chunks=bm25_chunks,
            elapsed_s=time.perf_counter() - t0,
            steps=["retrieve: BM25 lexical over indexed chunks", "generate: same simple RAG prompt"],
        )

    def _run_cosine_shuffle() -> tuple[str, dict[str, Any]]:
        t0 = time.perf_counter()
        shuffled = list(vector_chunks)
        if vector_source == "existing_vector_only":
            random.Random(str(question)).shuffle(shuffled)
        cosine_answer = generate_answer(question, shuffled, mode="cosine_shuffle", answer_instruction=answer_instruction)
        steps = (
            ["reuse: existing vector_only retrieved snippets", "shuffle snippets to remove rerank/order advantage"]
            if vector_source == "existing_vector_only"
            else ["retrieve: direct vector-store cosine search without reranker"]
        )
        return "cosine_shuffle", make_branch_payload(
            mode="cosine_shuffle",
            title=EXTRA_METHODS["cosine_shuffle"],
            answer=cosine_answer,
            chunks=shuffled,
            elapsed_s=time.perf_counter() - t0,
            steps=steps + ["generate: simple RAG prompt"],
        )

    def _run_vector_analysis() -> tuple[str, dict[str, Any]]:
        t0 = time.perf_counter()
        va_answer = generate_answer(question, vector_chunks, mode="vector_analysis", answer_instruction=answer_instruction)
        return "vector_analysis", make_branch_payload(
            mode="vector_analysis",
            title=EXTRA_METHODS["vector_analysis"],
            answer=va_answer,
            chunks=vector_chunks,
            elapsed_s=time.perf_counter() - t0,
            steps=[f"reuse vector context source: {vector_source}", "generate: one-stage concise context engineering prompt"],
        )

    jobs = []
    if "bm25" in selected:
        jobs.append(_run_bm25)
    if "cosine_shuffle" in selected:
        jobs.append(_run_cosine_shuffle)
    if "vector_analysis" in selected:
        jobs.append(_run_vector_analysis)
    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as ex:
        futs = [ex.submit(fn) for fn in jobs]
        for fut in futs:
            key, payload = fut.result()
            out[key] = payload

    if include_external_placeholder:
        out["external_placeholder"] = make_branch_payload(
            mode="external_placeholder",
            title=EXTRA_METHODS["external_placeholder"],
            answer="",
            chunks=[],
            elapsed_s=0.0,
            steps=["placeholder: external HopRAG/project reproduction not connected yet"],
        )
    return out
