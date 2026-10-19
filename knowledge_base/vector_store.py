"""Vector store backed by ChromaDB with SiliconFlow embeddings.

Each knowledge base gets its own ChromaDB collection: kb_<kb_id_prefix>.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from config.settings import settings, DATA_DIR
from utils.embedding_client import embedding_client
from knowledge_base.preprocessor import Chunk


def _collection_name(kb_id: str) -> str:
    # ChromaDB collection names must be 3-63 chars, alphanumeric + hyphens
    safe = kb_id.replace("-", "")[:32]
    return f"kb{safe}"


class VectorStore:
    def __init__(self, kb_id: str):
        self.kb_id = kb_id
        chroma_dir = str(DATA_DIR / "chroma_db")
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_collection_name(kb_id),
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk]) -> None:
        existing_ids = set(self._collection.get(include=[])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            logger.info("All chunks already indexed, skipping.")
            return

        texts = [c.text for c in new_chunks]
        logger.info(f"Embedding {len(texts)} new chunks...")
        vectors = embedding_client.embed(texts)

        self._collection.add(
            ids=[c.chunk_id for c in new_chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                {"source": c.source, "chunk_index": c.chunk_index, **c.metadata}
                for c in new_chunks
            ],
        )
        logger.info(f"Indexed {len(new_chunks)} chunks.")

    def delete_all(self) -> None:
        """Delete all chunks in this KB's collection."""
        ids = self._collection.get(include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks from KB {self.kb_id}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or settings.vector_store.top_k
        if self.count() == 0:
            return []
        query_vec = embedding_client.embed_one(query)
        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        return [
            {"id": id_, "text": doc, "score": float(1 - dist / 2), "metadata": meta}
            for id_, doc, meta, dist in zip(ids, docs, metas, distances)
        ]

    def count(self) -> int:
        return self._collection.count()
