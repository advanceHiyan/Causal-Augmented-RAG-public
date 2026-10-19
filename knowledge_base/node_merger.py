"""Helpers for suggesting and applying graph node merges."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

import networkx as nx
import numpy as np


@dataclass
class MergeSuggestion:
    canonical: str
    aliases: list[str]
    average_similarity: float
    reason: str


class _UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[\s\-_()/,.:;，。、“”‘’《》【】\[\]·]+", "", text)
    return text


def _lexical_score(a: str, b: str) -> float:
    na = _normalize(a)
    nb = _normalize(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if len(na) >= 2 and len(nb) >= 2 and (na in nb or nb in na):
        shorter = min(len(na), len(nb))
        longer = max(len(na), len(nb))
        return shorter / max(longer, 1)
    return SequenceMatcher(None, na, nb).ratio()


def _cosine_similarity_matrix(vectors: dict[str, list[float]]) -> dict[tuple[str, str], float]:
    if not vectors:
        return {}
    names = list(vectors)
    matrix = np.array([vectors[name] for name in names], dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    matrix = matrix / norms
    sims = matrix @ matrix.T
    result: dict[tuple[str, str], float] = {}
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            result[(a, names[j])] = float(sims[i, j])
    return result


def suggest_merge_groups(
    nodes: list[str],
    vectors: dict[str, list[float]] | None = None,
    degree_map: dict[str, int] | None = None,
    semantic_threshold: float = 0.92,
    lexical_threshold: float = 0.82,
    max_groups: int = 20,
) -> list[MergeSuggestion]:
    if len(nodes) < 2:
        return []

    uf = _UnionFind(nodes)
    pair_scores: dict[tuple[str, str], tuple[float, str]] = {}
    semantic_scores = _cosine_similarity_matrix(vectors or {})

    for i, left in enumerate(nodes):
        for right in nodes[i + 1 :]:
            lexical = _lexical_score(left, right)
            semantic = semantic_scores.get((left, right), semantic_scores.get((right, left), 0.0))

            should_merge = False
            reason = ""
            score = max(lexical, semantic)

            if lexical >= 0.995:
                should_merge = True
                reason = "规范化后完全一致"
            elif lexical >= 0.60 and semantic >= semantic_threshold:
                should_merge = True
                reason = "语义高度相似且文本接近"
            elif lexical >= lexical_threshold and semantic >= 0.70:
                should_merge = True
                reason = "文本相似且语义接近"
            elif lexical >= 0.92:
                should_merge = True
                reason = "文本高度相似"

            if should_merge:
                uf.union(left, right)
                pair_scores[(left, right)] = (score, reason)

    grouped: dict[str, list[str]] = {}
    for node in nodes:
        grouped.setdefault(uf.find(node), []).append(node)

    suggestions: list[MergeSuggestion] = []
    for members in grouped.values():
        if len(members) <= 1:
            continue

        degrees = degree_map or {}
        canonical = sorted(
            members,
            key=lambda node: (-degrees.get(node, 0), len(_normalize(node)), len(node), node),
        )[0]
        aliases = [node for node in members if node != canonical]

        scores = []
        reasons = []
        for alias in aliases:
            pair = (canonical, alias)
            rev_pair = (alias, canonical)
            score_reason = pair_scores.get(pair) or pair_scores.get(rev_pair)
            if score_reason:
                scores.append(score_reason[0])
                reasons.append(score_reason[1])

        suggestions.append(
            MergeSuggestion(
                canonical=canonical,
                aliases=sorted(aliases),
                average_similarity=sum(scores) / len(scores) if scores else 1.0,
                reason="；".join(sorted(set(reasons))) if reasons else "候选同义节点",
            )
        )

    suggestions.sort(key=lambda item: (len(item.aliases), item.average_similarity), reverse=True)
    return suggestions[:max_groups]


def merge_graph_nodes(
    graph: nx.DiGraph,
    canonical: str,
    aliases: list[str],
    merge_edge_attrs,
) -> int:
    if canonical not in graph:
        raise ValueError(f"节点不存在: {canonical}")

    merged = 0
    for alias in aliases:
        if alias == canonical or alias not in graph:
            continue

        alias_node = copy.deepcopy(graph.nodes[alias])
        canonical_node = graph.nodes[canonical]
        canonical_node["sources"] = set(canonical_node.get("sources", set())) | set(
            alias_node.get("sources", set())
        )

        for src, _, data in list(graph.in_edges(alias, data=True)):
            new_src = canonical if src == alias else src
            if graph.has_edge(new_src, canonical):
                merge_edge_attrs(graph[new_src][canonical], data)
            else:
                graph.add_edge(new_src, canonical, **copy.deepcopy(data))

        for _, dst, data in list(graph.out_edges(alias, data=True)):
            new_dst = canonical if dst == alias else dst
            if graph.has_edge(canonical, new_dst):
                merge_edge_attrs(graph[canonical][new_dst], data)
            else:
                graph.add_edge(canonical, new_dst, **copy.deepcopy(data))

        graph.remove_node(alias)
        merged += 1

    return merged
