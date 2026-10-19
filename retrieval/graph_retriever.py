"""
Level-2 Retriever: text graph causal path discovery.

When the causal graph has no direct match, we fall back to the richer
text graph, traverse it to find a relevant subgraph, then ask the LLM
to filter genuine causal paths and summarize them.

Discovered causal knowledge is fed back to update the causal graph.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import networkx as nx
import numpy as np
from loguru import logger

from config.settings import settings
from utils.embedding_client import embedding_client
from utils.llm_client import llm_client
from knowledge_base.text_graph import TextGraph
from knowledge_base.causal_graph import CausalGraph, CausalTriple


_FILTER_SYSTEM = """\
你是一个因果关系鉴别专家。给定一个知识图谱子图（以"实体A --[关系]--> 实体B"格式表示）
和一个用户查询，请：
1. 识别子图中真正属于因果关系的路径
2. 生成一段简洁的因果结构描述（自然语言）
3. 提取新发现的因果三元组（可选）

输出 JSON：
{
  "causal_summary": "描述发现的因果关系的自然语言段落",
  "new_triples": [
    {
      "cause": "...",
      "relation": "...",
      "effect": "...",
      "confidence": 0.0-1.0,
      "relation_type": "causal|necessary|sufficient|precondition|trigger|disqualify|exception|temporal_before|unknown",
      "polarity": "positive|negative"
    }
  ]
}

若子图中找不到有效因果关系，causal_summary 返回空字符串，new_triples 返回 []。

说明：
- 结构化关系（例如 necessary/sufficient/precondition/trigger/disqualify/exception/temporal_before）
 也是因果关系的存储形式，应纳入因果链判断。
- 若同时出现通用关系词与结构化标签，以结构化标签优先。
- relation_type 拿不准可用 unknown，但不要臆造。
"""


@dataclass
class GraphRetrievalResult:
    causal_summary: str
    new_triples_added: int
    success: bool


class GraphRetriever:
    def __init__(
        self,
        tgraph: TextGraph,
        cgraph: CausalGraph,
    ):
        self._tg = tgraph
        self._cg = cgraph
        self._threshold = settings.retrieval.causal_match_threshold

    def retrieve(self, query: str, persist_updates: bool = False) -> GraphRetrievalResult:
        if self._tg.graph.number_of_nodes() == 0:
            return GraphRetrievalResult("", 0, success=False)

        # 1. Find entry nodes in text graph by embedding similarity
        query_vec = embedding_client.embed_one(query)
        entry_nodes = self._find_entry_nodes(query_vec)

        if not entry_nodes:
            return GraphRetrievalResult("", 0, success=False)

        logger.debug(f"L2 text graph: entry nodes {[n for n, _ in entry_nodes]}")

        # 2. BFS traversal to build candidate subgraph
        subgraph = self._expand_subgraph_with_budget(query_vec, entry_nodes)

        if subgraph.number_of_edges() == 0:
            return GraphRetrievalResult("", 0, success=False)

        # 3. Ask LLM to filter causal paths and summarize
        subgraph_text = self._tg.subgraph_to_text(subgraph)
        causal_summary, new_triples = self._llm_filter(query, subgraph_text)

        # 4. Optional feed-back to causal graph (guarded by caller flag).
        added = 0
        if persist_updates:
            for t in new_triples:
                if t.confidence >= settings.graph.causal_confidence_threshold:
                    self._cg.add_triple(t, normalize=True)
                    added += 1
            if added > 0:
                self._cg.save()
                logger.info(f"L2 retrieval: added {added} new triples to causal graph")

        return GraphRetrievalResult(
            causal_summary=causal_summary,
            new_triples_added=added,
            success=bool(causal_summary),
        )

    def _find_entry_nodes(self, query_vec: list[float], top_k: int | None = None) -> list[tuple[str, float]]:
        """Find text graph nodes most similar to query vector."""
        nodes = list(self._tg.graph.nodes())
        if not nodes:
            return []
        if top_k is None:
            top_k = settings.graph.l2_entry_top_k

        # Embed nodes that aren't cached yet
        self._tg.embed_nodes(nodes)

        q = np.array(query_vec)
        scores = []
        for node in nodes:
            vec = self._tg.get_node_vector(node)
            if vec is None:
                continue
            v = np.array(vec)
            sim = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
            scores.append((node, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        above = [(n, s) for n, s in scores[:top_k] if s >= settings.graph.l2_entry_min_similarity]
        if above:
            return above
        if settings.graph.l2_force_top_k_when_threshold_miss:
            # Avoid hard-empty L2 when similarity thresholds are too strict on noisy node vocab.
            fallback = scores[:top_k]
            if fallback:
                logger.debug(
                    "L2 entry threshold miss, fallback to top_k nodes: "
                    + ", ".join([f"{n}({s:.3f})" for n, s in fallback])
                )
            return fallback
        return []

    def _expand_subgraph_with_budget(
        self,
        query_vec: list[float],
        entry_nodes: list[tuple[str, float]],
    ) -> nx.DiGraph:
        """
        Expand from entry nodes with bounded BFS, then keep the most query-relevant
        nodes/edges under budgets. This improves relation coverage without exploding
        graph size and token cost.
        """
        graph = self._tg.graph
        if graph.number_of_nodes() == 0:
            return nx.DiGraph()

        max_depth = settings.graph.max_graph_depth
        node_limit = max(10, settings.graph.l2_candidate_node_limit)
        edge_limit = max(20, settings.graph.l2_candidate_edge_limit)
        depth_penalty = max(0.0, settings.graph.l2_depth_penalty)

        undirected = graph.to_undirected()
        best_depth: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque()

        for node, _ in entry_nodes:
            if node in graph:
                best_depth[node] = 0
                queue.append((node, 0))

        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for nb in undirected.neighbors(node):
                next_depth = depth + 1
                old_depth = best_depth.get(nb)
                if old_depth is not None and old_depth <= next_depth:
                    continue
                best_depth[nb] = next_depth
                queue.append((nb, next_depth))

        candidate_nodes = list(best_depth.keys())
        if not candidate_nodes:
            return nx.DiGraph()

        self._tg.embed_nodes(candidate_nodes)
        q = np.array(query_vec)

        scored_nodes: list[tuple[str, float]] = []
        for node in candidate_nodes:
            vec = self._tg.get_node_vector(node)
            if vec is None:
                continue
            v = np.array(vec)
            sim = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
            score = sim - depth_penalty * best_depth.get(node, max_depth)
            scored_nodes.append((node, score))

        scored_nodes.sort(key=lambda item: item[1], reverse=True)
        selected_nodes = [node for node, _ in scored_nodes[:node_limit]]
        if not selected_nodes:
            return nx.DiGraph()

        node_score_map = {node: score for node, score in scored_nodes}
        subgraph = graph.subgraph(selected_nodes).copy()

        if subgraph.number_of_edges() <= edge_limit:
            return subgraph

        ranked_edges = sorted(
            subgraph.edges(data=True),
            key=lambda e: (
                node_score_map.get(e[0], -1.0) + node_score_map.get(e[1], -1.0)
            ) / 2.0,
            reverse=True,
        )
        trimmed = nx.DiGraph()
        for node in selected_nodes:
            trimmed.add_node(node, **subgraph.nodes[node])
        for u, v, data in ranked_edges[:edge_limit]:
            trimmed.add_edge(u, v, **data)
        return trimmed

    def _llm_filter(
        self, query: str, subgraph_text: str
    ) -> tuple[str, list[CausalTriple]]:
        messages = [
            {"role": "system", "content": _FILTER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"用户查询：{query}\n\n"
                    f"知识图谱子图：\n{subgraph_text}"
                ),
            },
        ]
        try:
            data = llm_client.chat_json(messages)
        except Exception as e:
            logger.warning(f"L2 LLM filter failed: {e}")
            return self._rule_fallback_summary(subgraph_text), []

        if not isinstance(data, dict):
            data = {}
        summary = data.get("causal_summary", "")
        raw_triples = data.get("new_triples", [])
        if not isinstance(raw_triples, list):
            raw_triples = []

        triples = []
        for t in raw_triples:
            cause = str(t.get("cause", "")).strip()
            relation = str(t.get("relation", "")).strip()
            effect = str(t.get("effect", "")).strip()
            try:
                confidence = float(t.get("confidence", 0.5))
            except (ValueError, TypeError):
                confidence = 0.5
            relation_type = str(t.get("relation_type", "unknown")).strip()
            polarity = str(t.get("polarity", "positive")).strip()
            if cause and relation and effect:
                triples.append(
                    CausalTriple(
                        cause=cause,
                        relation=relation,
                        effect=effect,
                        confidence=confidence,
                        relation_type=relation_type,
                        polarity=polarity,
                        extraction_method="l2_llm",
                        evidence_text=f"{cause} --[{relation}]--> {effect}",
                    )
                )

        if not str(summary).strip():
            summary = self._rule_fallback_summary(subgraph_text)
        return summary, triples

    def _rule_fallback_summary(self, subgraph_text: str) -> str:
        """
        Cheap fallback when LLM JSON/filter fails or returns empty summary.
        Extract likely causal edges by relation keywords so L2 doesn't collapse to empty.
        """
        causal_tokens = (
            "导致", "引起", "造成", "影响", "促进", "抑制", "increase", "decrease",
            "cause", "causes", "caused", "lead to", "leads to", "result in", "results in",
            "depend", "depends", "related to", "关联", "依赖", "驱动",
            "necessary", "sufficient", "precondition", "trigger", "disqualify",
            "exception", "temporal_before",
        )
        picked: list[str] = []
        for raw in subgraph_text.splitlines():
            line = raw.strip()
            if not line or "--[" not in line:
                continue
            if any(tok in line.lower() for tok in causal_tokens):
                picked.append(line)
            if len(picked) >= 6:
                break
        if picked:
            return "；".join(picked)
        return ""
