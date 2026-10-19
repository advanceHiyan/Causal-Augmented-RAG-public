"""
Level-3 Retriever: vector search with optional reranking.

Always runs in parallel with graph retrieval. Results can be:
  - Primary context (when graph retrieval fails)
  - Supplementary context (reranked using graph results as signal)
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from config.settings import settings
from utils.embedding_client import rerank_client
from knowledge_base.vector_store import VectorStore


@dataclass
class VectorRetrievalResult:
    chunks: list[dict]   # [{"id", "text", "score", "metadata"}, ...]
    used_reranker: bool


class VectorRetriever:
    def __init__(self, store: VectorStore):
        self._store = store
        self._cfg = settings.retrieval

    def retrieve(
        self,
        query: str,
        causal_context: str = "",
        top_k_override: int | None = None,
    ) -> VectorRetrievalResult:
        """
        1. Fetch top-k candidates from vector store
        2. Optionally rerank using the combined query + causal_context as rerank query
        """
        top_k = int(top_k_override) if top_k_override else int(self._cfg.vector_top_k)
        fetch_k = max(int(self._cfg.vector_fetch_k), top_k)

        candidates = self._store.search(query, top_k=fetch_k)
        if not candidates:
            return VectorRetrievalResult([], used_reranker=False)

        if not self._cfg.use_reranker or len(candidates) <= top_k:
            return VectorRetrievalResult(candidates[:top_k], used_reranker=False)

        # Build rerank query: original query optionally augmented by causal context
        rerank_query = query
        if causal_context:
            rerank_query = f"{query}\n\n相关因果背景：{causal_context[:300]}"

        texts = [str(c.get("text", "")) for c in candidates]
        try:
            reranked = rerank_client.rerank(rerank_query, texts, top_n=top_k)
            # Map back to original candidate dicts using returned indices
            result_chunks = []
            for item in reranked:
                idx = item.get("index", 0)
                if idx < len(candidates):
                    chunk = candidates[idx].copy()
                    chunk["rerank_score"] = item.get("relevance_score", 0.0)
                    result_chunks.append(chunk)
            return VectorRetrievalResult(result_chunks, used_reranker=True)
        except Exception as e:
            logger.warning(f"Reranker failed, falling back to vector scores: {e}")
            return VectorRetrievalResult(candidates[:top_k], used_reranker=False)
