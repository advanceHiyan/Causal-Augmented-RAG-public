"""
Text graph builder — per-KB isolated storage.
File: data/graphs/<kb_id>/text_graph.json
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from pathlib import Path
from typing import NamedTuple

import networkx as nx
from loguru import logger

from config.settings import settings, DATA_DIR
from utils.llm_client import llm_client
from utils.embedding_client import embedding_client
from knowledge_base.preprocessor import Chunk
from knowledge_base.node_merger import MergeSuggestion, merge_graph_nodes, suggest_merge_groups


class TextTriple(NamedTuple):
    head: str
    relation: str
    tail: str
    source_chunk_id: str


_EXTRACT_SYSTEM = """\
你是一个知识图谱构建助手。给定一段文本，抽取其中所有实体和语义关系。

输出 JSON 格式，结构如下：
{
  "triples": [
    {"head": "实体A", "relation": "关系类型", "tail": "实体B"},
    ...
  ]
}

关系类型示例（不限于此）：影响、导致、关联、属于、包含、使用、定义、依赖、对比、支持、反对。
只提取文本中明确存在的关系，不推断。实体名称尽量简洁。

抽取目标：
- 优先抽取“概念、机制、制度、事件、流程、政策、风险、问题、措施、组织、系统、技术对象”
- 当一句话里既有抽象概念又有表格/名单信息时，只保留能支撑语义理解和推理的概念或事件
- 节点应尽量是可复用的语义单元，而不是某个表格单元格里的取值

额外约束：
- 优先保留具有稳定语义的概念、制度、事件、机构、项目、技术对象
- 不要把纯数字、比例、金额、日期、年份、序号、编号、附件名、章节名、列表项当作节点
- 不要把“附件1”“表2”“第3条”“2025年”“30%”“1.5万元/年”“61人”这类低信息量短语当作节点
- 人名只有在文本核心语义确实围绕该人物时才保留；普通名单、署名、评委名、作者名不要抽
- 如果一个关系的头实体或尾实体属于上述低价值对象，则直接忽略这条 triple

特别注意：
- 对通知、公告、名单、表格、制度文件类文本，不要抽取“获奖等级、名额、金额、日期、名单成员、附件标题”这类节点
- 若文本只是说明某机构发布了一份名单或附件，应优先抽取“奖学金评选”“评审机制”“名额分配规则”这类事件或制度，而不是名单项本身
- 对“某某同学”“某某老师”“某某委员”这类孤立人名默认忽略，除非该人物是论述核心
- 若无法找到高质量实体，宁可少抽，也不要为了凑数量输出噪声节点
"""


_EXTRACT_SYSTEM += """
Additional constraints:
1) Do not output bare generic entities like 部门/学校/学院/平台/中心/单位/机构/系统/项目/政策/文件.
   If needed, expand them into specific phrases (e.g., 制度管理主责部门, 学校相关规定).
2) You may optionally output entity types:
   {"head":"...", "head_type":"...", "relation":"...", "tail":"...", "tail_type":"..."}.
3) Prefer concept/event/institution entities, and avoid field/list/number/date/attachment style entities.
"""


_LOW_VALUE_PATTERNS = [
    r"^\d+(\.\d+)?$",
    r"^\d+(\.\d+)?%$",
    r"^\d+(\.\d+)?[万千百十]?(元|万元|亿元|人|名|项|条|个|份|页|章|节|篇|所|次|年|月|日)(/年|/月|/天)?$",
    r"^\d{4}年$",
    r"^\d{4}级.*$",
    r"^第[\d一二三四五六七八九十百千]+[章节条款项部分].*$",
    r"^(附件|附表|表|图)[\d一二三四五六七八九十]+.*$",
    r"^(名单|名录|目录|序号|编号|备注|说明)$",
    r"^[A-Za-z]?\d+[A-Za-z0-9\-]*$",
]
_LOW_VALUE_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in _LOW_VALUE_PATTERNS]
_GENERIC_LOW_VALUE_TERMS = {
    "研究生名单",
    "附件",
    "附件1",
    "附件2",
    "附件3",
    "名单",
    "名额分配",
    "评选等级",
    "一等奖",
    "二等奖",
    "三等奖",
}

_BARE_AMBIGUOUS_TERMS = {
    "部门",
    "学校",
    "学院",
    "平台",
    "中心",
    "单位",
    "机构",
    "系统",
    "项目",
    "政策",
    "文件",
}

_LOW_VALUE_ENTITY_TYPES = {
    "field",
    "table_item",
    "list_item",
    "number",
    "date",
    "amount",
    "code",
    "attachment",
}

_SHORT_NODE_WHITELIST = {
    "政策",
    "制度",
    "法规",
    "规章",
    "文件",
    "规则",
    "机制",
    "平台",
    "项目",
    "系统",
    "流程",
    "风险",
    "方案",
    "计划",
}

_SHORT_NODE_LOW_INFO_HINTS = {
    "一份",
    "一个",
    "一种",
    "一些",
    "相关",
    "事项",
    "内容",
    "情况",
    "说明",
    "问题",
    "代号",
    "附注",
    "型号",
    "尺寸",
    "日期",
    "名牌",
    "专家",
}


def _clean_entity(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("，。；：、“”‘’()（）[]【】")
    return text


def _normalize_entity_type(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    return text


def _extract_contextual_alias(term: str, chunk_text: str) -> str:
    """Try to expand a bare ambiguous term using local phrase context."""
    term = _clean_entity(term)
    if not term or not chunk_text:
        return ""
    token = r"[\u4e00-\u9fa5A-Za-z0-9（）()《》·\-]"
    patterns = [
        rf"({token}{{2,30}}{re.escape(term)})",
        rf"({re.escape(term)}{token}{{1,30}})",
    ]
    candidates: list[str] = []
    for pat in patterns:
        for match in re.findall(pat, chunk_text):
            c = _clean_entity(str(match))
            if not c or c == term:
                continue
            if len(c) > 40 or len(c) < len(term) + 1:
                continue
            if any(sym in c for sym in ("/", "\\", "|")):
                continue
            candidates.append(c)
    if not candidates:
        return ""
    uniq = sorted(set(candidates), key=lambda x: (len(x), x), reverse=True)
    for c in uniq:
        if c in _BARE_AMBIGUOUS_TERMS:
            continue
        if c in _GENERIC_LOW_VALUE_TERMS:
            continue
        return c
    return ""


def _resolve_entity(entity: str, chunk_text: str) -> str:
    entity = _clean_entity(entity)
    if not entity:
        return ""
    if entity in _BARE_AMBIGUOUS_TERMS:
        alias = _extract_contextual_alias(entity, chunk_text)
        if alias:
            return alias
        return ""
    return entity


def _is_low_value_entity(text: str) -> bool:
    text = _clean_entity(text)
    if not text:
        return True
    if text in _BARE_AMBIGUOUS_TERMS:
        return True
    if text in _GENERIC_LOW_VALUE_TERMS:
        return True
    if len(text) <= 1:
        return True
    if all(ch.isdigit() for ch in text):
        return True
    if any(pattern.match(text) for pattern in _LOW_VALUE_COMPILED):
        return True
    if re.fullmatch(r"[0-9.\-/%年月日万元人名项条个份页章节]+", text):
        return True
    if re.fullmatch(r"[A-Za-z0-9\-_/.]+", text) and any(ch.isdigit() for ch in text):
        return True
    return False


def _is_noise_relation(text: str) -> bool:
    text = text.strip()
    return not text or len(text) > 20


def _low_value_reason(text: str) -> str | None:
    text = _clean_entity(text)
    if not text:
        return "空节点"
    if text in _BARE_AMBIGUOUS_TERMS:
        return "未消歧的泛化节点"
    if text in _GENERIC_LOW_VALUE_TERMS:
        return "通用表格/名单术语"
    if len(text) <= 1:
        return "长度过短"
    if all(ch.isdigit() for ch in text):
        return "纯数字"
    for pattern in _LOW_VALUE_COMPILED:
        if pattern.match(text):
            return "命中低价值模式"
    if re.fullmatch(r"[0-9.\-/%年月日万元人名项条个份页章节]+", text):
        return "数值或计量短语"
    if re.fullmatch(r"[A-Za-z0-9\-_/.]+", text) and any(ch.isdigit() for ch in text):
        return "编号样式"
    return None


def _structural_low_value_reason(text: str, degree: int) -> str | None:
    text = _clean_entity(text)
    if not text:
        return "空节点"
    if degree > 1:
        return None
    if text in _SHORT_NODE_WHITELIST:
        return None
    if text in _SHORT_NODE_LOW_INFO_HINTS:
        return "低连通低信息短词"
    if len(text) <= 2:
        return "低连通短节点"
    if re.fullmatch(r"[A-Za-z0-9\\-_/\\.]{1,8}", text):
        return "低连通编码样式节点"
    if text.upper() in {"XX", "XXX", "N/A", "NA"}:
        return "占位符节点"
    return None


def _graph_to_json(graph: nx.DiGraph) -> dict:
    data = nx.node_link_data(graph, edges="links")
    for node in data.get("nodes", []):
        for k, v in node.items():
            if isinstance(v, set):
                node[k] = sorted(v)
    for link in data.get("links", []):
        for k, v in link.items():
            if isinstance(v, set):
                link[k] = sorted(v)
    return data


def _graph_from_json(data: dict) -> nx.DiGraph:
    graph = nx.node_link_graph(data, directed=True, multigraph=False, edges="links")
    for node, attrs in graph.nodes(data=True):
        if "sources" in attrs and isinstance(attrs["sources"], list):
            attrs["sources"] = set(attrs["sources"])
    for u, v, attrs in graph.edges(data=True):
        if "relations" in attrs and isinstance(attrs["relations"], list):
            attrs["relations"] = set(attrs["relations"])
        if "sources" in attrs and isinstance(attrs["sources"], list):
            attrs["sources"] = set(attrs["sources"])
    return graph


class TextGraph:
    def __init__(self, kb_id: str):
        self.kb_id = kb_id
        self._path = DATA_DIR / "graphs" / kb_id / "text_graph.json"
        self.graph: nx.DiGraph = nx.DiGraph()
        self._node_vectors: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_from_chunks(
        self,
        chunks: list[Chunk],
        save: bool = True,
        max_workers: int = 10,
    ) -> None:
        logger.info(f"Building text graph from {len(chunks)} chunks...")
        if not chunks:
            return

        # Chunk-level rolling parallelism.
        # Keep output merge order identical to input chunk order for stability.
        max_workers = min(max(1, int(max_workers or 1)), len(chunks))
        indexed_results: dict[int, list[TextTriple]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {
                ex.submit(self._extract_triples, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            for i, fut in enumerate(as_completed(fut_map), start=1):
                idx = fut_map[fut]
                try:
                    indexed_results[idx] = fut.result() or []
                except Exception as e:
                    logger.warning(f"TextGraph parallel extract failed at chunk#{idx}: {e}")
                    indexed_results[idx] = []
                if i % 20 == 0:
                    logger.debug(f"  Parallel processed {i}/{len(chunks)} chunks...")

        all_triples: list[TextTriple] = []
        for idx in range(len(chunks)):
            all_triples.extend(indexed_results.get(idx, []))

        self._add_triples(all_triples)
        logger.info(
            f"Text graph [{self.kb_id[:8]}]: "
            f"{self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges"
        )
        if save:
            self.save()

    def _extract_triples(self, chunk: Chunk) -> list[TextTriple]:
        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": chunk.text},
        ]
        try:
            data = llm_client.chat_json(messages)
            raw = data.get("triples", [])
        except Exception as e:
            logger.warning(f"Triple extraction failed for {chunk.chunk_id}: {e}")
            return []
        result = []
        for t in raw:
            head_type = _normalize_entity_type(str(t.get("head_type", "")))
            tail_type = _normalize_entity_type(str(t.get("tail_type", "")))
            if head_type in _LOW_VALUE_ENTITY_TYPES or tail_type in _LOW_VALUE_ENTITY_TYPES:
                continue

            h = _resolve_entity(str(t.get("head", "")), chunk.text)
            r = str(t.get("relation", "")).strip()
            tail = _resolve_entity(str(t.get("tail", "")), chunk.text)
            if not h or not r or not tail:
                continue
            if h == tail:
                continue
            if _is_noise_relation(r):
                continue
            if _is_low_value_entity(h) or _is_low_value_entity(tail):
                continue
            if len(h) > 40 or len(tail) > 40:
                continue
            if len(h) <= 2 and len(tail) <= 2:
                continue
            if re.search(r"(名单|附件|附表|序号|编号|名额)", h + tail):
                continue
            if h and r and tail:
                result.append(TextTriple(h, r, tail, chunk.chunk_id))
        return result

    def _add_triples(self, triples: list[TextTriple]) -> None:
        for t in triples:
            for entity in (t.head, t.tail):
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity, sources=set())
                self.graph.nodes[entity]["sources"].add(t.source_chunk_id)
            if self.graph.has_edge(t.head, t.tail):
                self.graph[t.head][t.tail]["relations"].add(t.relation)
                self.graph[t.head][t.tail]["sources"].add(t.source_chunk_id)
            else:
                self.graph.add_edge(
                    t.head, t.tail,
                    relations={t.relation},
                    sources={t.source_chunk_id},
                )

    # ------------------------------------------------------------------
    # Node embeddings
    # ------------------------------------------------------------------

    def embed_nodes(self, nodes: list[str] | None = None) -> None:
        targets = nodes or list(self.graph.nodes())
        to_embed = [n for n in targets if n not in self._node_vectors]
        if not to_embed:
            return
        logger.info(f"Embedding {len(to_embed)} text graph nodes...")
        vectors = embedding_client.embed(to_embed)
        for node, vec in zip(to_embed, vectors):
            self._node_vectors[node] = vec

    def get_node_vector(self, node: str) -> list[float] | None:
        if node not in self._node_vectors:
            self.embed_nodes([node])
        return self._node_vectors.get(node)

    def suggest_merges(
        self,
        semantic_threshold: float = 0.92,
        lexical_threshold: float = 0.82,
        max_groups: int = 20,
    ) -> list[MergeSuggestion]:
        nodes = list(self.graph.nodes())
        if not nodes:
            return []
        self.embed_nodes(nodes)
        degree_map = {node: int(self.graph.degree(node)) for node in nodes}
        vectors = {node: self._node_vectors[node] for node in nodes if node in self._node_vectors}
        return suggest_merge_groups(
            nodes=nodes,
            vectors=vectors,
            degree_map=degree_map,
            semantic_threshold=semantic_threshold,
            lexical_threshold=lexical_threshold,
            max_groups=max_groups,
        )

    def merge_nodes(self, canonical: str, aliases: list[str], save: bool = True) -> int:
        merged = merge_graph_nodes(
            graph=self.graph,
            canonical=canonical,
            aliases=aliases,
            merge_edge_attrs=self._merge_edge_attrs,
        )
        for alias in aliases:
            self._node_vectors.pop(alias, None)
        if canonical in self.graph:
            self._node_vectors.pop(canonical, None)
            self.embed_nodes([canonical])
        if save and merged:
            self.save()
        return merged

    def suggest_low_value_nodes(self, max_nodes: int = 50) -> list[dict]:
        candidates = []
        for node in self.graph.nodes():
            degree = int(self.graph.degree(node))
            reason = _low_value_reason(node)
            if not reason:
                reason = _structural_low_value_reason(str(node), degree)
            if not reason:
                continue
            candidates.append(
                {
                    "node": node,
                    "reason": reason,
                    "degree": degree,
                }
            )
        # Prioritize low-degree noisy nodes first.
        candidates.sort(key=lambda item: (item["degree"], len(item["node"])))
        return candidates[:max_nodes]

    def diagnose_quality(self, max_examples: int = 20) -> dict:
        bare_ambiguous: list[str] = []
        structural_noise: list[str] = []
        for node in self.graph.nodes():
            node_text = str(node)
            degree = int(self.graph.degree(node))
            if node_text in _BARE_AMBIGUOUS_TERMS:
                bare_ambiguous.append(node_text)
            if _structural_low_value_reason(node_text, degree):
                structural_noise.append(node_text)
        return {
            "node_count": int(self.graph.number_of_nodes()),
            "edge_count": int(self.graph.number_of_edges()),
            "bare_ambiguous_count": len(bare_ambiguous),
            "bare_ambiguous_examples": sorted(set(bare_ambiguous))[:max_examples],
            "structural_noise_count": len(structural_noise),
            "structural_noise_examples": sorted(set(structural_noise), key=lambda x: (len(x), x))[:max_examples],
        }

    def suggest_singleton_component_nodes(self, max_nodes: int = 500) -> list[str]:
        """Return nodes that form one-node weakly connected components."""
        if self.graph.number_of_nodes() == 0:
            return []
        nodes: list[str] = []
        for comp in nx.weakly_connected_components(self.graph):
            if len(comp) == 1:
                node = next(iter(comp))
                nodes.append(str(node))
        # Stable ordering for frontend display
        nodes.sort(key=lambda n: (len(n), n))
        return nodes[:max_nodes]

    def delete_singleton_components(self, save: bool = True) -> int:
        nodes = self.suggest_singleton_component_nodes(max_nodes=self.graph.number_of_nodes())
        if not nodes:
            return 0
        return self.delete_nodes(nodes, save=save)

    def delete_nodes(self, nodes: list[str], save: bool = True) -> int:
        deleted = 0
        for node in nodes:
            if node not in self.graph:
                continue
            self.graph.remove_node(node)
            self._node_vectors.pop(node, None)
            deleted += 1
        if save and deleted:
            self.save()
        return deleted

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_neighbors_subgraph(self, node: str, depth: int | None = None) -> nx.DiGraph:
        depth = depth or settings.graph.max_graph_depth
        nodes = nx.single_source_shortest_path_length(
            self.graph.to_undirected(), node, cutoff=depth
        ).keys()
        return self.graph.subgraph(nodes).copy()

    def subgraph_to_text(self, subgraph: nx.DiGraph) -> str:
        lines = []
        for u, v, data in subgraph.edges(data=True):
            rels = ", ".join(data.get("relations", []))
            lines.append(f"{u} --[{rels}]--> {v}")
        return "\n".join(lines) if lines else ""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "graph": _graph_to_json(self.graph),
            "node_vectors": self._node_vectors,
        }
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Text graph saved: {self._path}")

    def load(self) -> bool:
        if not self._path.exists():
            return False
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        self.graph = _graph_from_json(payload["graph"])
        self._node_vectors = payload.get("node_vectors", {})
        logger.info(
            f"Text graph loaded [{self.kb_id[:8]}]: "
            f"{self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges"
        )
        return True

    def delete_file(self) -> None:
        if self._path.exists():
            self._path.unlink()

    @staticmethod
    def _merge_edge_attrs(target: dict, incoming: dict) -> None:
        target["relations"] = set(target.get("relations", set())) | set(incoming.get("relations", set()))
        target["sources"] = set(target.get("sources", set())) | set(incoming.get("sources", set()))
