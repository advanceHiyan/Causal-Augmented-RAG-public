"""Embedding and reranking via SiliconFlow OpenAI-compatible API."""

from __future__ import annotations

import sys
import time

import httpx
from openai import OpenAI
from loguru import logger

from config.settings import settings, EmbeddingConfig, RerankConfig

# Max characters per text before embedding (bge-large-zh-v1.5 limit: 512 tokens)
# Chinese ≈ 1 char/token; leave margin for special tokens → 400 chars is safe
_MAX_EMBED_CHARS = 400


def _truncate_for_embed(text: str) -> str:
    """Truncate text to fit within the embedding model's token limit."""
    if len(text) <= _MAX_EMBED_CHARS:
        return text
    return text[:_MAX_EMBED_CHARS]


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig | None = None):
        self._cfg = config or settings.embedding
        self._client = OpenAI(
            api_key=self._cfg.api_key,
            base_url=self._cfg.base_url,
        )

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Embed a list of texts, truncating to token limit and batching."""
        model = model or self._cfg.model
        batch_size = self._cfg.batch_size
        # Truncate all texts first
        truncated = [_truncate_for_embed(t) for t in texts]
        all_vectors: list[list[float]] = []

        for i in range(0, len(truncated), batch_size):
            batch = truncated[i : i + batch_size]
            t0 = time.time()
            try:
                resp = self._client.embeddings.create(model=model, input=batch)
                sorted_data = sorted(resp.data, key=lambda x: x.index)
                vecs = [d.embedding for d in sorted_data]
                all_vectors.extend(vecs)
            except Exception as e:
                logger.error(f"Embedding failed for batch [{i}:{i+batch_size}]: {e}")
                print(f"[EMBEDDING API ERROR] model={model} batch={i}:{i+batch_size} error={e}", file=sys.stderr, flush=True)
                raise
            finally:
                duration_ms = (time.time() - t0) * 1000

            # Debug logging
            try:
                from utils.api_logger import APILogger
                APILogger.log(
                    call_type="embed",
                    model=model,
                    input_data=batch,
                    output_data=f"[{len(vecs)} vectors, dim={len(vecs[0]) if vecs else 0}]",
                    duration_ms=duration_ms,
                )
            except Exception:
                pass

        return all_vectors

    def embed_one(self, text: str, model: str | None = None) -> list[float]:
        return self.embed([text], model=model)[0]


class RerankClient:
    def __init__(self, config: RerankConfig | None = None):
        self._cfg = config or settings.rerank
        self._base_url = self._cfg.base_url.rstrip("/")

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
        model: str | None = None,
    ) -> list[dict]:
        top_n = top_n or self._cfg.top_n
        model = model or self._cfg.model
        url = f"{self._base_url}/rerank"
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": True,
        }
        headers = {
            "Authorization": f"Bearer {self._cfg.api_key}",
            "Content-Type": "application/json",
        }
        t0 = time.time()
        try:
            with httpx.Client(timeout=60) as client:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            results = data.get("results", [])
            sorted_results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            print(f"[RERANK API ERROR] model={model} docs={len(documents)} error={e}", file=sys.stderr, flush=True)
            raise
        finally:
            duration_ms = (time.time() - t0) * 1000

        # Debug logging
        try:
            from utils.api_logger import APILogger
            APILogger.log(
                call_type="rerank",
                model=model,
                input_data={"query": query, "docs_count": len(documents)},
                output_data=[{"index": r.get("index"), "score": r.get("relevance_score")} for r in sorted_results],
                duration_ms=duration_ms,
            )
        except Exception:
            pass

        return sorted_results


embedding_client = EmbeddingClient()
rerank_client = RerankClient()
