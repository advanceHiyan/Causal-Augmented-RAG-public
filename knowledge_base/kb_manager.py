"""
KnowledgeBaseManager — orchestrates all per-KB resources.

For each kb_id, creates and caches:
  - VectorStore (ChromaDB collection)
  - TextGraph   (JSON file in data/graphs/<kb_id>/)
  - CausalGraph (JSON file in data/graphs/<kb_id>/)
  - HybridRetriever (assembled from the three above)
"""

from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger

from config.settings import DATA_DIR
from knowledge_base.preprocessor import DocumentPreprocessor, Chunk
from knowledge_base.vector_store import VectorStore
from knowledge_base.text_graph import TextGraph
from knowledge_base.causal_graph import CausalGraph
from retrieval.causal_retriever import CausalRetriever
from retrieval.graph_retriever import GraphRetriever
from retrieval.vector_retriever import VectorRetriever
from retrieval.hybrid_retriever import HybridRetriever
from generation.generator_v2 import CausalGenerator


class KBManager:
    """All resources scoped to a single knowledge base."""

    def __init__(self, kb_id: str):
        self.kb_id = kb_id
        self.vector_store = VectorStore(kb_id)
        self.text_graph = TextGraph(kb_id)
        self.causal_graph = CausalGraph(kb_id)
        self._retriever: HybridRetriever | None = None
        self._generator = CausalGenerator()
        self._preprocessor = DocumentPreprocessor(chunk_size=350, chunk_overlap=50)

    def load_graphs(self) -> None:
        """Load text/causal graphs from JSON files."""
        self.text_graph.load()
        self.causal_graph.load()
        self._retriever = None  # invalidate cached retriever

    def get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(
                causal_ret=CausalRetriever(graph=self.causal_graph),
                graph_ret=GraphRetriever(tgraph=self.text_graph, cgraph=self.causal_graph),
                vector_ret=VectorRetriever(store=self.vector_store),
            )
        return self._retriever

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def add_text(
        self,
        text: str,
        doc_name: str,
        build_vector: bool = True,
        build_text_graph: bool = True,
        build_causal_graph: bool = True,
        text_graph_max_workers: int = 10,
        enable_implicit_causal_discovery: bool | None = None,
        implicit_causal_max_chunks: int | None = None,
    ) -> dict:
        """
        Ingest raw text, update knowledge structures.
        Returns stats dict.
        """
        chunks = self._preprocessor.load_texts([text], source=doc_name)
        stats = {"chunks": len(chunks), "tg_nodes": 0, "tg_edges": 0, "cg_nodes": 0, "cg_edges": 0}
        logger.info(f"Ingesting '{doc_name}' → {len(chunks)} chunks into KB {self.kb_id[:8]}…")

        if build_vector:
            self.vector_store.add_chunks(chunks)

        if build_text_graph:
            self.text_graph.build_from_chunks(
                chunks,
                save=True,
                max_workers=max(1, int(text_graph_max_workers or 10)),
            )
            stats["tg_nodes"] = self.text_graph.graph.number_of_nodes()
            stats["tg_edges"] = self.text_graph.graph.number_of_edges()

        if build_causal_graph:
            self.causal_graph.build_from_chunks(
                chunks,
                save=True,
                enable_implicit_discovery=enable_implicit_causal_discovery,
                implicit_max_chunks=implicit_causal_max_chunks,
            )
            stats["cg_nodes"] = self.causal_graph.graph.number_of_nodes()
            stats["cg_edges"] = self.causal_graph.graph.number_of_edges()

        self._retriever = None  # invalidate
        return stats

    def add_chunks(
        self,
        chunks: list[Chunk],
        build_vector: bool = True,
        build_text_graph: bool = True,
        build_causal_graph: bool = True,
        text_graph_max_workers: int = 10,
        enable_implicit_causal_discovery: bool | None = None,
        implicit_causal_max_chunks: int | None = None,
    ) -> dict:
        """
        Ingest pre-split chunks without running the default text chunker.
        Used by HopRAG-aligned experiments where chunk IDs must stay stable.
        """
        chunks = [c for c in chunks if str(c.chunk_id).strip() and str(c.text).strip()]
        stats = {"chunks": len(chunks), "tg_nodes": 0, "tg_edges": 0, "cg_nodes": 0, "cg_edges": 0}
        logger.info(f"Ingesting {len(chunks)} pre-split chunks into KB {self.kb_id[:8]}")

        if build_vector:
            self.vector_store.add_chunks(chunks)

        if build_text_graph:
            self.text_graph.build_from_chunks(
                chunks,
                save=True,
                max_workers=max(1, int(text_graph_max_workers or 10)),
            )
            stats["tg_nodes"] = self.text_graph.graph.number_of_nodes()
            stats["tg_edges"] = self.text_graph.graph.number_of_edges()

        if build_causal_graph:
            self.causal_graph.build_from_chunks(
                chunks,
                save=True,
                enable_implicit_discovery=enable_implicit_causal_discovery,
                implicit_max_chunks=implicit_causal_max_chunks,
            )
            stats["cg_nodes"] = self.causal_graph.graph.number_of_nodes()
            stats["cg_edges"] = self.causal_graph.graph.number_of_edges()

        self._retriever = None
        return stats

    # ------------------------------------------------------------------
    # Q&A
    # ------------------------------------------------------------------

    def retrieve(self, retrieval_query: str, persist_causal_updates: bool = False, top_k_override: int | None = None):
        """Run retrieval only and return RetrievalContext."""
        retriever = self.get_retriever()
        return retriever.retrieve(
            retrieval_query,
            persist_causal_updates=persist_causal_updates,
            top_k_override=top_k_override,
        )

    def discover_structured_relations_from_text_graph(
        self,
        max_edges: int = 600,
        use_llm: bool = True,
        save: bool = True,
        keep_unknown: bool = False,
    ) -> dict:
        """
        Discover structured causal-compatible relations from text graph edges.
        Keeps old relation labels while adding structured relation_type metadata.
        """
        result = self.causal_graph.discover_structured_relations_from_text_graph(
            text_graph=self.text_graph.graph,
            max_edges=max(1, int(max_edges)),
            use_llm=bool(use_llm),
            save=bool(save),
            keep_unknown=bool(keep_unknown),
        )
        self._retriever = None
        return result

    def suggest_structured_relations_from_text_graph(
        self,
        max_edges: int = 600,
        use_llm: bool = True,
        keep_unknown: bool = False,
    ) -> dict:
        return self.causal_graph.suggest_structured_relations_from_text_graph(
            text_graph=self.text_graph.graph,
            max_edges=max(1, int(max_edges)),
            use_llm=bool(use_llm),
            keep_unknown=bool(keep_unknown),
        )

    def apply_structured_relation_candidates(self, candidates: list[dict], save: bool = True) -> dict:
        result = self.causal_graph.apply_structured_relation_candidates(
            candidates=candidates,
            save=bool(save),
            normalize=True,
        )
        self._retriever = None
        return result

    def generate(
        self,
        question: str,
        ctx,
        conversation_history: list[dict[str, str]] | None = None,
        retrieval_query: str = "",
        profile_context: str = "",
        query_preanalysis_context: str = "",
        answer_instruction: str = "",
    ) -> str:
        result = self.generate_with_meta(
            question,
            ctx,
            conversation_history=conversation_history,
            retrieval_query=retrieval_query,
            profile_context=profile_context,
            query_preanalysis_context=query_preanalysis_context,
            answer_instruction=answer_instruction,
        )
        return result.answer

    def generate_with_meta(
        self,
        question: str,
        ctx,
        conversation_history: list[dict[str, str]] | None = None,
        retrieval_query: str = "",
        profile_context: str = "",
        query_preanalysis_context: str = "",
        answer_instruction: str = "",
    ):
        result = self._generator.generate(
            question,
            ctx,
            conversation_history=conversation_history,
            retrieval_query=retrieval_query,
            profile_context=profile_context,
            query_preanalysis_context=query_preanalysis_context,
            answer_instruction=answer_instruction,
        )
        return result

    def ask(
        self,
        question: str,
        persist_causal_updates: bool = False,
        conversation_history: list[dict[str, str]] | None = None,
        retrieval_query: str = "",
        profile_context: str = "",
        query_preanalysis_context: str = "",
        answer_instruction: str = "",
    ) -> tuple[str, object]:
        """Returns (answer_text, RetrievalContext)."""
        retrieval_query = retrieval_query or question
        ctx = self.retrieve(retrieval_query, persist_causal_updates=persist_causal_updates)
        answer = self.generate(
            question,
            ctx,
            conversation_history=conversation_history,
            retrieval_query=retrieval_query,
            profile_context=profile_context,
            query_preanalysis_context=query_preanalysis_context,
            answer_instruction=answer_instruction,
        )
        return answer, ctx

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "vector_chunks": self.vector_store.count(),
            "tg_nodes": self.text_graph.graph.number_of_nodes(),
            "tg_edges": self.text_graph.graph.number_of_edges(),
            "cg_nodes": self.causal_graph.graph.number_of_nodes(),
            "cg_edges": self.causal_graph.graph.number_of_edges(),
        }

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete_all_data(self) -> None:
        """Delete all indexed data for this KB (vector index + graph files)."""
        self.vector_store.delete_all()
        self.text_graph.delete_file()
        self.causal_graph.delete_file()
        graph_dir = DATA_DIR / "graphs" / self.kb_id
        if graph_dir.exists():
            shutil.rmtree(graph_dir, ignore_errors=True)
        logger.info(f"All data deleted for KB {self.kb_id[:8]}")


# ---------------------------------------------------------------------------
# Global cache: one KBManager per kb_id (process-level, Streamlit-safe)
# ---------------------------------------------------------------------------

_manager_cache: dict[str, KBManager] = {}


def get_kb_manager(kb_id: str) -> KBManager:
    """Return cached manager; loads graphs from disk on first access."""
    if kb_id not in _manager_cache:
        mgr = KBManager(kb_id)
        mgr.load_graphs()
        _manager_cache[kb_id] = mgr
    return _manager_cache[kb_id]


def invalidate_kb_cache(kb_id: str) -> None:
    """Call after building a new knowledge base to force reload."""
    _manager_cache.pop(kb_id, None)
