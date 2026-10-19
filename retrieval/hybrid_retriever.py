"""
Hybrid retriever orchestrator.

Runs the three-level retrieval pipeline:
  L1 (causal graph) ──► success? use as primary context
        │ fail
        ▼
  L2 (text graph)   ──► success? use as primary context
        │ fail
        ▼
  (fall through to L3 only)

L3 (vector) always runs in parallel with L1/L2.
When L1 or L2 succeeds, L3 results are appended as supplementary context.
When both L1 and L2 fail, L3 is the sole context.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from loguru import logger

from retrieval.causal_retriever import CausalRetriever
from retrieval.graph_retriever import GraphRetriever
from retrieval.vector_retriever import VectorRetriever


@dataclass
class RetrievalContext:
    # Causal / graph context (structured text)
    causal_graph_text: str = ""          # from L1
    causal_path_summary: str = ""        # from L2

    # Vector chunks
    vector_chunks: list[dict] = field(default_factory=list)

    # Provenance flags
    l1_success: bool = False
    l2_success: bool = False
    l3_used_reranker: bool = False

    @property
    def has_causal_context(self) -> bool:
        return bool(
            (self.l1_success and self.causal_graph_text)
            or (self.l2_success and self.causal_path_summary)
        )

    def build_context_string(self) -> str:
        """Assemble all retrieved information into a single context block."""
        parts: list[str] = []

        if self.causal_graph_text:
            parts.append("【因果图谱信息】\n" + self.causal_graph_text)

        if self.causal_path_summary:
            if self.l2_success:
                parts.append("【因果路径摘要】\n" + self.causal_path_summary)
            else:
                parts.append("【检索退化说明】\n" + self.causal_path_summary)

        if self.vector_chunks:
            chunk_texts = "\n\n".join(
                f"[文档片段 {i+1}]\n{str(c.get('text', '')).strip()}"
                for i, c in enumerate(self.vector_chunks)
                if str(c.get("text", "")).strip()
            )
            if chunk_texts:
                parts.append("【相关文档片段】\n" + chunk_texts)

        return "\n\n".join(parts)


class HybridRetriever:
    def __init__(
        self,
        causal_ret: CausalRetriever,
        graph_ret: GraphRetriever,
        vector_ret: VectorRetriever,
    ):
        self._cr = causal_ret
        self._gr = graph_ret
        self._vr = vector_ret

    def retrieve(
        self,
        query: str,
        persist_causal_updates: bool = False,
        top_k_override: int | None = None,
    ) -> RetrievalContext:
        ctx = RetrievalContext()

        # Run L1 + L3 in parallel (L3 always runs)
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_l1 = ex.submit(self._cr.retrieve, query)
            fut_l3 = ex.submit(self._vr.retrieve, query, "", top_k_override)   # no causal context yet

            l1_result = fut_l1.result()
            l3_result = fut_l3.result()

        ctx.l1_success = l1_result.success
        ctx.l3_used_reranker = l3_result.used_reranker

        if l1_result.success:
            ctx.causal_graph_text = l1_result.subgraph_text
            logger.info("L1 causal graph retrieval succeeded")
        else:
            # L1 failed → try L2
            logger.info("L1 failed, trying L2 text graph retrieval...")
            l2_result = self._gr.retrieve(query, persist_updates=persist_causal_updates)
            ctx.l2_success = l2_result.success
            if l2_result.success:
                ctx.causal_path_summary = l2_result.causal_summary
                logger.info("L2 text graph retrieval succeeded")
            else:
                logger.info("L2 failed, using L3 vector results only")

        # If we have causal context now, re-run L3 reranker with causal signal
        causal_ctx = ctx.causal_graph_text or ctx.causal_path_summary
        if causal_ctx and l3_result.used_reranker is False and ctx.has_causal_context:
            # Redo the vector retrieval with causal context to improve reranking
            l3_result = self._vr.retrieve(query, causal_context=causal_ctx, top_k_override=top_k_override)
            ctx.l3_used_reranker = l3_result.used_reranker

        ctx.vector_chunks = l3_result.chunks

        # Last-resort explanation: keep a visible retrieval fallback note for observability.
        if not ctx.causal_graph_text and not ctx.causal_path_summary:
            ctx.causal_path_summary = self._build_fallback_summary_from_vector(ctx.vector_chunks)

        logger.info(
            f"Hybrid retrieval done | L1={ctx.l1_success} L2={ctx.l2_success} "
            f"vector_chunks={len(ctx.vector_chunks)} reranked={ctx.l3_used_reranker}"
        )
        return ctx

    @staticmethod
    def _build_fallback_summary_from_vector(chunks: list[dict]) -> str:
        if not chunks:
            return "未检索到可用因果候选证据。"
        hints = []
        for i, c in enumerate(chunks[:3]):
            text = str(c.get("text", "")).strip().replace("\n", " ")
            if text:
                hints.append(f"片段{i+1}:{text[:140]}")
        if hints:
            return (
                "未检索到高置信显式因果子图，已回退到向量证据。"
                "可基于以下片段继续因果判别：" + " | ".join(hints)
            )
        return "未检索到高置信显式因果子图，已回退到向量检索。"
