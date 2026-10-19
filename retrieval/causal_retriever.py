"""
Level-1 Retriever: direct causal graph search.

Finds causal nodes semantically similar to the query, returns their
surrounding causal subgraph as structured text context.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from loguru import logger

from config.settings import settings
from utils.embedding_client import embedding_client
from knowledge_base.causal_graph import CausalGraph


@dataclass
class CausalRetrievalResult:
    matched_nodes: list[str]
    subgraph_text: str          # human-readable causal chain
    success: bool


class CausalRetriever:
    def __init__(self, graph: CausalGraph):
        self._graph = graph
        self._threshold = settings.retrieval.causal_match_threshold

    def retrieve(self, query: str) -> CausalRetrievalResult:
        """
        1. Embed the query
        2. Find structurally similar causal nodes
        3. Extract their subgraph
        4. Convert to text
        """
        if self._graph.graph.number_of_nodes() == 0:
            return CausalRetrievalResult([], "", success=False)

        query_vec = embedding_client.embed_one(query)
        similar_nodes = self._graph.find_similar_nodes(query_vec, top_k=5)

        # Filter by threshold
        matched = [
            (node, score)
            for node, score in similar_nodes
            if score >= self._threshold
        ]

        if not matched:
            logger.debug(
                f"L1 causal retrieval: no nodes above threshold {self._threshold} "
                f"(best: {similar_nodes[0][1]:.3f} for '{similar_nodes[0][0]}')"
                if similar_nodes else "L1: causal graph empty"
            )
            return CausalRetrievalResult([], "", success=False)

        matched_node_names = [n for n, _ in matched]
        logger.debug(f"L1 causal retrieval: matched nodes {matched_node_names}")

        # Build merged subgraph from all matched nodes
        merged = nx.DiGraph()
        for node, _ in matched:
            sub = self._graph.get_causal_subgraph(node, depth=2)
            merged = nx.compose(merged, sub)

        # Trim to max nodes if too large
        max_nodes = settings.graph.max_causal_subgraph_nodes
        if merged.number_of_nodes() > max_nodes:
            # Keep highest-degree nodes
            top_nodes = sorted(
                merged.nodes(), key=lambda n: merged.degree(n), reverse=True
            )[:max_nodes]
            merged = merged.subgraph(top_nodes).copy()

        subgraph_text = self._graph.subgraph_to_text(merged)

        return CausalRetrievalResult(
            matched_nodes=matched_node_names,
            subgraph_text=subgraph_text,
            success=bool(subgraph_text),
        )
