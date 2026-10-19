from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from config.settings import settings
from knowledge_base.causal_graph import _normalize_polarity, _normalize_relation_type
from knowledge_base.preprocessor import Chunk
from utils.embedding_client import embedding_client


ProgressFn = Callable[[str], None]


def _console(msg: str) -> None:
    print(f"[HopRAG Build] {msg}", flush=True)


def load_chunks_jsonl(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunk_id = str(rec.get("chunk_id", "") or "").strip()
            text = str(rec.get("text", "") or "").strip()
            if not chunk_id or not text:
                raise ValueError(f"{path.name}:{line_no} missing chunk_id/text")
            metadata = rec.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    source=str(rec.get("source", path.name) or path.name),
                    chunk_index=int(rec.get("chunk_index", len(chunks)) or len(chunks)),
                    metadata=metadata,
                )
            )
    return chunks


def _task_result(name: str, start: float, stats: dict[str, Any]) -> dict[str, Any]:
    stats = dict(stats)
    stats["task"] = name
    stats["elapsed_s"] = round(time.perf_counter() - start, 3)
    return stats


def _chunks_by_size(items: list[Any], size: int) -> list[list[Any]]:
    size = max(1, int(size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _add_vector_chunks_parallel(
    vector_store: Any,
    chunks: list[Chunk],
    workers: int,
    progress_cb: ProgressFn | None = None,
) -> dict[str, Any]:
    def emit(msg: str) -> None:
        _console(msg)
        if progress_cb:
            progress_cb(msg)

    existing_ids = set(vector_store._collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
    if not new_chunks:
        emit(f"向量库：{len(chunks)} 个 chunks 已存在，跳过 embedding")
        return {"chunks": len(chunks), "new_chunks": 0, "embedding_batches": 0}

    batch_size = int(settings.embedding.batch_size)
    batches = _chunks_by_size(new_chunks, batch_size)
    emit(
        f"向量库：准备 embedding {len(new_chunks)} 个新 chunks，"
        f"{len(batches)} 批，batch_size={batch_size}，workers={max(1, int(workers))}"
    )
    vectors_by_batch: dict[int, list[list[float]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        future_map = {
            ex.submit(embedding_client.embed, [c.text for c in batch]): idx
            for idx, batch in enumerate(batches)
        }
        done = 0
        for fut in as_completed(future_map):
            idx = future_map[fut]
            vectors_by_batch[idx] = fut.result()
            done += 1
            emit(f"向量库：embedding batch {done}/{len(batches)} 完成")

    vectors: list[list[float]] = []
    for idx in range(len(batches)):
        vectors.extend(vectors_by_batch[idx])

    vector_store._collection.add(
        ids=[c.chunk_id for c in new_chunks],
        embeddings=vectors,
        documents=[c.text for c in new_chunks],
        metadatas=[
            {"source": c.source, "chunk_index": c.chunk_index, **c.metadata}
            for c in new_chunks
        ],
    )
    emit(f"向量库：写入 Chroma 完成，新增 {len(new_chunks)} 个 chunks")
    return {
        "chunks": len(chunks),
        "new_chunks": len(new_chunks),
        "embedding_batches": len(batches),
        "embedding_batch_size": batch_size,
        "embedding_workers": max(1, int(workers)),
    }


def _build_text_graph_parallel(
    text_graph: Any,
    chunks: list[Chunk],
    workers: int,
    progress_cb: ProgressFn | None = None,
) -> dict[str, Any]:
    def emit(msg: str) -> None:
        _console(msg)
        if progress_cb:
            progress_cb(msg)

    triples = []
    emit(f"文本图：开始并行抽取 {len(chunks)} 个 article units，workers={max(1, int(workers))}")
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        future_map = {ex.submit(text_graph._extract_triples, chunk): chunk.chunk_id for chunk in chunks}
        done = 0
        for fut in as_completed(future_map):
            triples.extend(fut.result())
            done += 1
            if done == len(chunks) or done % 10 == 0:
                emit(f"文本图：LLM 抽取 {done}/{len(chunks)} 完成")

    emit(f"文本图：抽取完成，开始串行合并 {len(triples)} 条 triples")
    text_graph._add_triples(triples)
    text_graph.save()
    emit(f"文本图：保存完成，nodes={text_graph.graph.number_of_nodes()} edges={text_graph.graph.number_of_edges()}")
    return {
        "chunks": len(chunks),
        "raw_triples": len(triples),
        "llm_workers": max(1, int(workers)),
        "tg_nodes": text_graph.graph.number_of_nodes(),
        "tg_edges": text_graph.graph.number_of_edges(),
    }


def _build_causal_graph_parallel(
    causal_graph: Any,
    chunks: list[Chunk],
    *,
    workers: int,
    enable_implicit_causal_discovery: bool,
    implicit_causal_max_chunks: int,
    progress_cb: ProgressFn | None = None,
) -> dict[str, Any]:
    def emit(msg: str) -> None:
        _console(msg)
        if progress_cb:
            progress_cb(msg)

    explicit_candidates = [c for c in chunks if _has_causal_signal_proxy(c.text)]
    implicit_candidates: list[Chunk] = []
    if enable_implicit_causal_discovery and implicit_causal_max_chunks > 0:
        explicit_ids = {c.chunk_id for c in explicit_candidates}
        remaining = [c for c in chunks if c.chunk_id not in explicit_ids]
        implicit_candidates = causal_graph._select_implicit_candidates(remaining, int(implicit_causal_max_chunks))
    candidates = explicit_candidates + implicit_candidates
    emit(
        f"因果图：候选 chunks explicit={len(explicit_candidates)} "
        f"implicit={len(implicit_candidates)} total={len(candidates)}/{len(chunks)}，"
        f"workers={max(1, int(workers))}"
    )

    triples = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        future_map = {ex.submit(causal_graph._extract_triples, chunk): chunk.chunk_id for chunk in candidates}
        done = 0
        for fut in as_completed(future_map):
            triples.extend(fut.result())
            done += 1
            if done == len(candidates) or done % 10 == 0:
                emit(f"因果图：LLM 抽取 {done}/{len(candidates)} 完成")

    emit(f"因果图：抽取完成，开始串行去重/过滤/归一化，raw_triples={len(triples)}")
    seen = set()
    deduped = []
    for t in triples:
        key = (
            causal_graph._clean_entity(t.cause).lower(),
            causal_graph._clean_entity(t.relation).lower(),
            causal_graph._clean_entity(t.effect).lower(),
            _normalize_relation_type(t.relation_type),
            _normalize_polarity(t.polarity),
            str(t.source_chunk_id),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)

    min_conf = settings.graph.causal_confidence_threshold
    filtered = [t for t in deduped if t.confidence >= min_conf]
    causal_graph._normalize_and_build(filtered)
    causal_graph.save()
    emit(
        f"因果图：保存完成，filtered_triples={len(filtered)} "
        f"nodes={causal_graph.graph.number_of_nodes()} edges={causal_graph.graph.number_of_edges()}"
    )
    return {
        "chunks": len(chunks),
        "candidate_chunks": len(candidates),
        "explicit_candidates": len(explicit_candidates),
        "implicit_candidates": len(implicit_candidates),
        "raw_triples": len(triples),
        "deduped_triples": len(deduped),
        "filtered_triples": len(filtered),
        "llm_workers": max(1, int(workers)),
        "cg_nodes": causal_graph.graph.number_of_nodes(),
        "cg_edges": causal_graph.graph.number_of_edges(),
    }


def _has_causal_signal_proxy(text: str) -> bool:
    # Keep this proxy local so the HopRAG builder does not change normal graph code.
    from knowledge_base.causal_graph import _has_causal_signal

    return _has_causal_signal(text)


def build_hoprag_kb_parallel(
    *,
    mgr: Any,
    sentence_chunks_path: Path,
    article_units_path: Path,
    build_vector: bool = True,
    build_text_graph: bool = True,
    build_causal_graph: bool = True,
    enable_implicit_causal_discovery: bool = False,
    implicit_causal_max_chunks: int = 24,
    embedding_workers: int = 8,
    graph_llm_workers: int = 16,
    progress_cb: ProgressFn | None = None,
) -> dict[str, Any]:
    """Build a HopRAG-aligned KB with separated sentence/vector and article/graph inputs."""

    def emit(msg: str) -> None:
        _console(msg)
        if progress_cb:
            progress_cb(msg)

    sentence_chunks = load_chunks_jsonl(sentence_chunks_path) if build_vector else []
    article_units = load_chunks_jsonl(article_units_path) if (build_text_graph or build_causal_graph) else []
    started = time.perf_counter()
    results: dict[str, Any] = {
        "sentence_chunks": len(sentence_chunks),
        "article_units": len(article_units),
        "tasks": {},
    }

    def vector_task() -> dict[str, Any]:
        t0 = time.perf_counter()
        emit(f"向量库开始：{len(sentence_chunks)} 个句子级 chunks，并发 {int(embedding_workers)}")
        stats = _add_vector_chunks_parallel(
            mgr.vector_store,
            sentence_chunks,
            int(embedding_workers),
            progress_cb=progress_cb,
        )
        stats["vector_chunks"] = mgr.vector_store.count()
        return _task_result("vector", t0, stats)

    def text_graph_task() -> dict[str, Any]:
        t0 = time.perf_counter()
        emit(f"文本图开始：{len(article_units)} 个文章级 units，LLM 并发 {int(graph_llm_workers)}")
        stats = _build_text_graph_parallel(
            mgr.text_graph,
            article_units,
            int(graph_llm_workers),
            progress_cb=progress_cb,
        )
        return _task_result("text_graph", t0, stats)

    def causal_graph_task() -> dict[str, Any]:
        t0 = time.perf_counter()
        emit(f"因果图开始：{len(article_units)} 个文章级 units，LLM 并发 {int(graph_llm_workers)}")
        stats = _build_causal_graph_parallel(
            mgr.causal_graph,
            article_units,
            workers=int(graph_llm_workers),
            enable_implicit_causal_discovery=bool(enable_implicit_causal_discovery),
            implicit_causal_max_chunks=int(implicit_causal_max_chunks),
            progress_cb=progress_cb,
        )
        return _task_result("causal_graph", t0, stats)

    jobs = []
    if build_vector:
        jobs.append(("vector", vector_task))
    if build_text_graph:
        jobs.append(("text_graph", text_graph_task))
    if build_causal_graph:
        jobs.append(("causal_graph", causal_graph_task))
    if not jobs:
        raise ValueError("At least one build target must be enabled")

    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        future_map = {ex.submit(fn): name for name, fn in jobs}
        for fut in as_completed(future_map):
            name = future_map[fut]
            try:
                result = fut.result()
                results["tasks"][name] = result
                emit(f"{name} 完成：{result.get('elapsed_s')}s")
            except Exception as e:
                results["tasks"][name] = {"task": name, "error": str(e)}
                emit(f"{name} 失败：{e}")
                raise

    mgr._retriever = None
    results["elapsed_s"] = round(time.perf_counter() - started, 3)
    return results


def append_hoprag_vector_chunks(
    *,
    mgr: Any,
    sentence_chunks_path: Path,
    embedding_workers: int = 8,
    progress_cb: ProgressFn | None = None,
) -> dict[str, Any]:
    """Append HopRAG sentence/text chunks to the vector store without touching either graph."""

    started = time.perf_counter()
    chunks = load_chunks_jsonl(sentence_chunks_path)
    stats = _add_vector_chunks_parallel(
        mgr.vector_store,
        chunks,
        int(embedding_workers),
        progress_cb=progress_cb,
    )
    stats["task"] = "append_vector_chunks"
    stats["sentence_chunks_path"] = str(sentence_chunks_path)
    stats["vector_chunks"] = mgr.vector_store.count()
    stats["elapsed_s"] = round(time.perf_counter() - started, 3)
    return stats
