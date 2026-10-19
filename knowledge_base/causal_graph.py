"""
Causal graph builder — per-KB isolated storage.
File: data/graphs/<kb_id>/causal_graph.json
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
from loguru import logger

from config.settings import settings, DATA_DIR
from utils.llm_client import llm_client
from utils.embedding_client import embedding_client
from knowledge_base.preprocessor import Chunk
from knowledge_base.node_merger import MergeSuggestion, merge_graph_nodes, suggest_merge_groups


# ---------------------------------------------------------------------------
# Causal signal patterns
# ---------------------------------------------------------------------------

_CAUSAL_PATTERNS = [
    r"因为.+?(?:所以|导致|引起|造成)",
    r"由于.+?(?:因此|从而|使得|导致)",
    r"(?:导致|引起|造成|引发|促使|触发).+",
    r"(?:因此|所以|从而|进而).+",
    r"causes?\s+\w+",
    r"leads?\s+to\s+\w+",
    r"results?\s+in\s+\w+",
    r"because\s+of\s+\w+",
    r"due\s+to\s+\w+",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _CAUSAL_PATTERNS]

_IMPLICIT_CAUSAL_HINT_PATTERNS = [
    r"(抑制|促进|推动|加剧|缓解|削弱|恶化|改善)",
    r"(风险|后果|结果|效应|作用|机制)",
    r"(上升|下降|增长|减少|波动|失衡|短缺|过剩)",
    r"(increase|decrease|rise|drop|effect|mechanism|risk)",
]
_IMPLICIT_COMPILED = [re.compile(p, re.IGNORECASE) for p in _IMPLICIT_CAUSAL_HINT_PATTERNS]


def _has_causal_signal(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED)


def _implicit_causal_signal_score(text: str) -> int:
    return sum(1 for pattern in _IMPLICIT_COMPILED if pattern.search(text))


def _normalize_relation_type(value: str) -> str:
    key = re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())
    return key if key in _STRUCTURED_RELATION_TYPES else "unknown"


def _normalize_polarity(value: str) -> str:
    p = str(value or "").strip().lower()
    return "negative" if p in {"negative", "neg", "否定", "no", "not"} else "positive"


def _guess_relation_type_regex(text: str) -> str:
    source = str(text or "")
    if _is_non_causal_relation(source):
        return "unknown"
    lower = source.lower()
    for relation_type, patterns in _RELATION_TYPE_PATTERNS:
        for pat in patterns:
            if re.search(pat, lower, re.IGNORECASE) or re.search(pat, source):
                return relation_type
    return "unknown"


def _guess_polarity(text: str) -> str:
    source = str(text or "")
    if re.search(r"(不|无|未|否|not|no|without|never)", source, re.IGNORECASE):
        return "negative"
    return "positive"


def _is_structured_relation_candidate(relation: str) -> bool:
    value = str(relation or "").strip()
    if not value:
        return False
    if len(value) > 64:
        return False
    return any(p.search(value) for p in _STRUCTURED_RELATION_CANDIDATE_COMPILED)


def _is_non_causal_relation(relation: str) -> bool:
    value = str(relation or "").strip()
    if not value:
        return True
    return any(p.search(value) for p in _NON_CAUSAL_RELATION_COMPILED)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class CausalTriple:
    cause: str
    relation: str
    effect: str
    confidence: float
    source_chunk_id: str = ""
    relation_type: str = "unknown"
    polarity: str = "positive"
    extraction_method: str = "llm"
    evidence_text: str = ""


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """\
你是一个因果关系抽取专家。给定一段文本，从中抽取所有明确的因果关系。

输出 JSON，格式如下：
{
  "triples": [
    {
      "cause": "原因实体或事件",
      "relation": "关系词（如：导致、引起、造成、使得、因此）",
      "effect": "结果实体或事件",
      "confidence": 0.0 到 1.0 之间的置信度,
      "relation_type": "结构化关系类型（如 causal/necessary/sufficient/precondition/trigger/disqualify/exception/temporal_before/unknown）",
      "polarity": "positive 或 negative"
    }
  ]
}

规则：
- 只抽取文本中明确存在的因果关系，不进行推断
- 原因和结果应是简洁的名词短语或动词短语
- confidence 反映因果关系的明确程度：直接陈述=0.9+，隐含但合理=0.6-0.8
- relation_type 必须从上述枚举中选一个；拿不准用 unknown
- polarity 表示该关系是正向成立还是否定条件（例如“不会导致”“不满足则不能”）
- 过滤功能型或泛化关系：如“支持、用于、创建、配置、管理、查看、安装、连接、包含、属于、影响”等，不要输出
- 若文本中没有因果关系，返回 {"triples": []}
"""


_STRUCTURED_RELATION_TYPES = {
    "causal",
    "necessary",
    "sufficient",
    "precondition",
    "trigger",
    "disqualify",
    "exception",
    "temporal_before",
    "unknown",
}

_RELATION_TYPE_PATTERNS: list[tuple[str, list[str]]] = [
    ("disqualify", [r"不予", r"不得", r"不能", r"禁止", r"取消资格", r"disqualif", r"not eligible"]),
    ("exception", [r"除非", r"例外", r"豁免", r"except", r"unless"]),
    ("necessary", [r"必须", r"需", r"必要条件", r"前提是", r"must", r"required", r"requirement"]),
    ("sufficient", [r"即可", r"足以", r"充分条件", r"只要", r"sufficient"]),
    ("precondition", [r"前提", r"先决", r"prior to", r"before .* can", r"条件"]),
    ("trigger", [r"触发", r"引发", r"trigger"]),
    ("temporal_before", [r"之前", r"之后", r"先于", r"晚于", r"before", r"after"]),
    ("causal", [r"导致", r"引起", r"造成", r"使得", r"because", r"due to", r"lead to", r"result in", r"cause"]),
]

_REGEX_STRUCTURED_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # Chinese: 因为A，所以B / 由于A，B
    (re.compile(r"(?:因为|由于)\s*(?P<cause>[^，。；;]+?)\s*(?:所以|因此|从而|进而)\s*(?P<effect>[^，。；;]+)"), "causal", "positive"),
    # Chinese: A导致B
    (re.compile(r"(?P<cause>[^，。；;]+?)\s*(?:导致|引起|造成|使得|触发)\s*(?P<effect>[^，。；;]+)"), "causal", "positive"),
    # Chinese: 只有A才B / 必须A才B
    (re.compile(r"(?:只有|必须|需|需要)\s*(?P<cause>[^，。；;]+?)\s*(?:才|方可|才能)\s*(?P<effect>[^，。；;]+)"), "necessary", "positive"),
    # Chinese: 若不A则B(negative precondition/disqualify)
    (re.compile(r"(?:若|如果|如|当)\s*不(?P<cause>[^，。；;]+?)\s*(?:则|将|会)?\s*(?P<effect>[^，。；;]+)"), "disqualify", "negative"),
    # English: if A, then B
    (re.compile(r"\bif\s+(?P<cause>[^,.!;]+?)\s*,?\s*then\s+(?P<effect>[^,.!;]+)", re.IGNORECASE), "precondition", "positive"),
    # English: A leads to B / results in B / causes B
    (re.compile(r"(?P<cause>[^,.!;]+?)\s+(?:leads?\s+to|results?\s+in|causes?)\s+(?P<effect>[^,.!;]+)", re.IGNORECASE), "causal", "positive"),
]

_RELATION_CLASSIFY_SYSTEM = """\
你是关系类型分类器。给定关系词（可能中英文混合），映射到以下枚举之一：
causal, necessary, sufficient, precondition, trigger, disqualify, exception, temporal_before, unknown

输出 JSON：
{
  "items": [
    {"relation":"...", "relation_type":"...", "polarity":"positive|negative", "drop": true|false}
  ]
}
严格约束（宁缺毋滥）：
- 只有当关系词明确表达“因果、条件、前置、触发、否决、例外、时序约束”时，drop=false。
- 语义泛化关系（如“包含、属于、使用、相关介绍、并列、描述性连接”）必须 drop=true。
- “支持、用于、创建、配置、管理、查看、安装、连接”等功能/操作关系，一律 drop=true。
- 拿不准时优先 drop=true。
仅输出 JSON。
"""


_STRUCTURED_RELATION_CANDIDATE_PATTERNS = [
    r"导致|引起|造成|使得|触发|驱动|抑制|促进|因果|因为|由于|因此|所以|从而|进而",
    r"必须|需要|需|前提|先决|条件|方可|才能|只有|否则|若|如果|当|除非|例外",
    r"不予|不得|不能|禁止|取消资格|不符合|不满足",
    r"\bif\b|\bthen\b|\bunless\b|\bmust\b|\brequired\b|\bprecondition\b|\btrigger\b",
    r"\blead(s)?\s+to\b|\bresult(s|ed)?\s+in\b|\bcause(s|d)?\b|\bdue to\b|\bbecause\b",
    r"\bbefore\b|\bafter\b|\bprior to\b|\bnot eligible\b|\bdisqualif",
]
_STRUCTURED_RELATION_CANDIDATE_COMPILED = [
    re.compile(p, re.IGNORECASE) for p in _STRUCTURED_RELATION_CANDIDATE_PATTERNS
]

_NON_CAUSAL_RELATION_PATTERNS = [
    r"支持|用于|用来|使用|包含|属于|提供|创建|新增|添加|删除|编辑|修改|配置|管理|查看|显示|安装|连接|同步|导出|导入|升级|发布|预览|打开|关闭|影响|影响到",
    r"\bsupport(s|ed)?\b|\bused?\s+for\b|\bfor\b|\binclude(s|d)?\b|\bbelong(s|ed)?\s+to\b|\bprovide(s|d)?\b",
    r"\bcreate(s|d)?\b|\badd(s|ed)?\b|\bdelete(s|d)?\b|\bedit(s|ed)?\b|\bconfig(ure|ures|ured)?\b",
    r"\bmanage(s|d)?\b|\bview(s|ed)?\b|\bshow(s|ed)?\b|\binstall(s|ed)?\b|\bconnect(s|ed)?\b|\bsync(s|ed)?\b|\bimpact(s|ed)?\b|\baffect(s|ed)?\b",
]
_NON_CAUSAL_RELATION_COMPILED = [re.compile(p, re.IGNORECASE) for p in _NON_CAUSAL_RELATION_PATTERNS]


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CausalGraph
# ---------------------------------------------------------------------------

class CausalGraph:
    def __init__(self, kb_id: str):
        self.kb_id = kb_id
        self._path = DATA_DIR / "graphs" / kb_id / "causal_graph.json"
        self.graph: nx.DiGraph = nx.DiGraph()
        self._entity_map: dict[str, str] = {}
        self._entity_vectors: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_from_chunks(
        self,
        chunks: list[Chunk],
        save: bool = True,
        enable_implicit_discovery: bool | None = None,
        implicit_max_chunks: int | None = None,
    ) -> None:
        logger.info(f"Building causal graph from {len(chunks)} chunks...")
        explicit_candidates = [c for c in chunks if _has_causal_signal(c.text)]

        if enable_implicit_discovery is None:
            enable_implicit_discovery = settings.graph.implicit_causal_discovery_default
        if implicit_max_chunks is None:
            implicit_max_chunks = settings.graph.implicit_causal_max_chunks

        implicit_candidates: list[Chunk] = []
        if enable_implicit_discovery and implicit_max_chunks > 0:
            explicit_ids = {c.chunk_id for c in explicit_candidates}
            remaining = [c for c in chunks if c.chunk_id not in explicit_ids]
            implicit_candidates = self._select_implicit_candidates(remaining, implicit_max_chunks)

        candidates = explicit_candidates + implicit_candidates
        logger.info(
            "  candidates: explicit={} implicit={} total={}/{}"
            .format(
                len(explicit_candidates),
                len(implicit_candidates),
                len(candidates),
                len(chunks),
            )
        )

        all_triples: list[CausalTriple] = []
        if candidates:
            # Per requirement: causal graph chunk extraction uses full parallel workers.
            max_workers = max(1, len(candidates))
            indexed_results: dict[int, list[CausalTriple]] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut_map = {
                    ex.submit(self._extract_triples, chunk): idx
                    for idx, chunk in enumerate(candidates)
                }
                for i, fut in enumerate(as_completed(fut_map), start=1):
                    idx = fut_map[fut]
                    try:
                        indexed_results[idx] = fut.result() or []
                    except Exception as e:
                        logger.warning(f"CausalGraph parallel extract failed at chunk#{idx}: {e}")
                        indexed_results[idx] = []
                    if i % 10 == 0:
                        logger.debug(f"  Parallel extracted causal triples {i}/{len(candidates)}...")

            # Preserve deterministic merge order by original candidate index.
            for idx in range(len(candidates)):
                all_triples.extend(indexed_results.get(idx, []))

        # De-duplicate by semantics while preserving old relation text.
        seen = set()
        deduped: list[CausalTriple] = []
        for t in all_triples:
            key = (
                self._clean_entity(t.cause).lower(),
                self._clean_entity(t.relation).lower(),
                self._clean_entity(t.effect).lower(),
                _normalize_relation_type(t.relation_type),
                _normalize_polarity(t.polarity),
                str(t.source_chunk_id),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(t)
        all_triples = deduped

        logger.info(f"  Extracted {len(all_triples)} raw causal triples")
        min_conf = settings.graph.causal_confidence_threshold
        filtered = [t for t in all_triples if t.confidence >= min_conf]
        logger.info(f"  {len(filtered)} triples pass confidence ≥ {min_conf}")

        self._normalize_and_build(filtered)
        logger.info(
            f"Causal graph [{self.kb_id[:8]}]: "
            f"{self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges"
        )
        if save:
            self.save()

    def _select_implicit_candidates(self, chunks: list[Chunk], limit: int) -> list[Chunk]:
        if not chunks or limit <= 0:
            return []
        min_score = settings.graph.implicit_causal_min_signal_score
        scored: list[tuple[int, int, Chunk]] = []
        for chunk in chunks:
            score = _implicit_causal_signal_score(chunk.text)
            if score < min_score:
                continue
            # Prefer denser weak signals and slightly longer chunks.
            scored.append((score, len(chunk.text), chunk))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = [item[2] for item in scored[:limit]]
        if selected:
            logger.info(
                f"  implicit discovery selected {len(selected)} weak-signal chunks "
                f"(limit={limit}, min_score={min_score})"
            )
        return selected

    def _extract_triples(self, chunk: Chunk) -> list[CausalTriple]:
        result: list[CausalTriple] = []

        # Fast path: regex structured extraction (cheap and deterministic).
        result.extend(self._extract_structured_triples_regex(chunk))

        # LLM extraction (captures implicit/complex wording).
        messages = [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": chunk.text},
        ]
        try:
            data = llm_client.chat_json(messages)
            raw = data.get("triples", [])
        except Exception as e:
            logger.warning(f"Causal extraction failed for {chunk.chunk_id}: {e}")
            return result
        for t in raw:
            cause = str(t.get("cause", "")).strip()
            relation = str(t.get("relation", "")).strip()
            effect = str(t.get("effect", "")).strip()
            try:
                confidence = float(t.get("confidence", 0.5))
            except (ValueError, TypeError):
                confidence = 0.5
            relation_type = _normalize_relation_type(t.get("relation_type", "unknown"))
            polarity = _normalize_polarity(t.get("polarity", "positive"))
            if relation_type == "unknown":
                relation_type = _guess_relation_type_regex(relation)
            if not relation_type:
                relation_type = "unknown"
            if polarity == "positive":
                polarity = _guess_polarity(f"{cause} {relation} {effect}")
            if _is_non_causal_relation(relation):
                continue
            if relation_type == "unknown" and not _is_structured_relation_candidate(relation):
                continue
            if cause and relation and effect:
                result.append(
                    CausalTriple(
                        cause=cause,
                        relation=relation,
                        effect=effect,
                        confidence=max(0.0, min(1.0, confidence)),
                        source_chunk_id=chunk.chunk_id,
                        relation_type=relation_type,
                        polarity=polarity,
                        extraction_method="llm",
                        evidence_text=chunk.text[:240],
                    )
                )
        return result

    def _extract_structured_triples_regex(self, chunk: Chunk) -> list[CausalTriple]:
        text = str(chunk.text or "")
        if not text:
            return []
        triples: list[CausalTriple] = []
        for pattern, relation_type, polarity in _REGEX_STRUCTURED_PATTERNS:
            for m in pattern.finditer(text):
                cause = self._clean_entity(m.groupdict().get("cause", ""))
                effect = self._clean_entity(m.groupdict().get("effect", ""))
                if not cause or not effect or cause == effect:
                    continue
                snippet_start = max(0, m.start() - 20)
                snippet_end = min(len(text), m.end() + 20)
                snippet = text[snippet_start:snippet_end].strip()
                relation = self._infer_relation_label_from_snippet(snippet, relation_type)
                triples.append(
                    CausalTriple(
                        cause=cause,
                        relation=relation,
                        effect=effect,
                        confidence=0.72,
                        source_chunk_id=chunk.chunk_id,
                        relation_type=relation_type,
                        polarity=polarity,
                        extraction_method="regex",
                        evidence_text=snippet,
                    )
                )
        return triples

    @staticmethod
    def _infer_relation_label_from_snippet(snippet: str, relation_type: str) -> str:
        relation_keywords: list[str] = []
        if relation_type == "causal":
            relation_keywords = ["导致", "引起", "造成", "使得", "because", "lead to", "result in", "cause"]
        elif relation_type == "necessary":
            relation_keywords = ["必须", "需", "required", "must"]
        elif relation_type == "sufficient":
            relation_keywords = ["即可", "足以", "sufficient", "enough"]
        elif relation_type == "precondition":
            relation_keywords = ["前提", "if", "条件", "before"]
        elif relation_type == "disqualify":
            relation_keywords = ["不予", "不得", "不能", "disqualify", "not eligible"]
        elif relation_type == "exception":
            relation_keywords = ["除非", "例外", "except", "unless"]
        elif relation_type == "temporal_before":
            relation_keywords = ["之前", "之后", "before", "after"]
        elif relation_type == "trigger":
            relation_keywords = ["触发", "trigger"]
        lower = snippet.lower()
        for keyword in relation_keywords:
            if keyword.lower() in lower:
                return keyword
        return relation_type

    # ------------------------------------------------------------------
    # Entity normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _structured_record(triple: CausalTriple) -> dict:
        return {
            "relation": str(triple.relation or "").strip(),
            "relation_type": _normalize_relation_type(triple.relation_type),
            "polarity": _normalize_polarity(triple.polarity),
            "confidence": float(max(0.0, min(1.0, triple.confidence))),
            "source_chunk_id": str(triple.source_chunk_id or ""),
            "method": str(triple.extraction_method or "unknown"),
            "evidence_text": str(triple.evidence_text or "")[:280],
        }

    @staticmethod
    def _append_structured_relation(edge_attrs: dict, item: dict) -> None:
        cur = edge_attrs.get("typed_relations")
        if not isinstance(cur, list):
            cur = []
        sig = (
            str(item.get("relation", "")),
            str(item.get("relation_type", "")),
            str(item.get("polarity", "")),
            str(item.get("source_chunk_id", "")),
            str(item.get("method", "")),
        )
        exists = False
        for old in cur:
            old_sig = (
                str(old.get("relation", "")),
                str(old.get("relation_type", "")),
                str(old.get("polarity", "")),
                str(old.get("source_chunk_id", "")),
                str(old.get("method", "")),
            )
            if old_sig == sig:
                old["confidence"] = max(float(old.get("confidence", 0.0)), float(item.get("confidence", 0.0)))
                if not old.get("evidence_text") and item.get("evidence_text"):
                    old["evidence_text"] = str(item.get("evidence_text", ""))[:280]
                exists = True
                break
        if not exists:
            cur.append(item)
        edge_attrs["typed_relations"] = cur

    def _normalize_and_build(self, triples: list[CausalTriple]) -> None:
        entities = list({t.cause for t in triples} | {t.effect for t in triples})
        if not entities:
            return
        logger.info(f"  Embedding {len(entities)} unique entities for normalization...")
        vectors = embedding_client.embed(entities)
        entity_vecs = {e: np.array(v) for e, v in zip(entities, vectors)}
        threshold = settings.graph.entity_merge_threshold
        canonical: dict[str, str] = {}
        for entity in entities:
            if entity in canonical:
                continue
            canonical[entity] = entity
            vec_a = entity_vecs[entity]
            for other in entities:
                if other == entity or other in canonical:
                    continue
                vec_b = entity_vecs[other]
                sim = float(
                    np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-9)
                )
                if sim >= threshold:
                    canonical[other] = entity
        self._entity_map.update(canonical)
        for entity, canon in canonical.items():
            if canon not in self._entity_vectors:
                self._entity_vectors[canon] = entity_vecs[canon].tolist()
        for t in triples:
            cause_c = canonical.get(t.cause, t.cause)
            effect_c = canonical.get(t.effect, t.effect)
            structured_item = self._structured_record(t)
            for node in (cause_c, effect_c):
                if not self.graph.has_node(node):
                    self.graph.add_node(node, sources=set())
                self.graph.nodes[node]["sources"].add(t.source_chunk_id)
            if self.graph.has_edge(cause_c, effect_c):
                edge = self.graph[cause_c][effect_c]
                edge["confidence"] = max(edge["confidence"], t.confidence)
                edge["relations"].add(t.relation)
                edge["sources"].add(t.source_chunk_id)
                self._append_structured_relation(edge, structured_item)
            else:
                self.graph.add_edge(
                    cause_c, effect_c,
                    confidence=t.confidence,
                    relations={t.relation},
                    sources={t.source_chunk_id},
                    typed_relations=[structured_item],
                )

    # ------------------------------------------------------------------
    # Dynamic update
    # ------------------------------------------------------------------

    def add_triple(self, triple: CausalTriple, normalize: bool = True) -> None:
        cause = self._clean_entity(triple.cause)
        effect = self._clean_entity(triple.effect)
        if not cause or not effect:
            return
        triple = CausalTriple(
            cause=cause,
            relation=str(triple.relation or "").strip() or "causal",
            effect=effect,
            confidence=float(max(0.0, min(1.0, triple.confidence))),
            source_chunk_id=str(triple.source_chunk_id or ""),
            relation_type=_normalize_relation_type(triple.relation_type),
            polarity=_normalize_polarity(triple.polarity),
            extraction_method=str(triple.extraction_method or "runtime"),
            evidence_text=str(triple.evidence_text or "")[:280],
        )
        if triple.relation_type == "unknown":
            triple.relation_type = _guess_relation_type_regex(triple.relation)
        if triple.polarity == "positive":
            triple.polarity = _guess_polarity(f"{triple.cause} {triple.relation} {triple.effect}")

        if normalize:
            cause = self._normalize_entity_online(cause)
            effect = self._normalize_entity_online(effect)

        for node in (cause, effect):
            if not self.graph.has_node(node):
                self.graph.add_node(node, sources=set())
            self.graph.nodes[node]["sources"].add(triple.source_chunk_id)
        if self.graph.has_edge(cause, effect):
            edge = self.graph[cause][effect]
            edge["confidence"] = max(edge["confidence"], triple.confidence)
            edge["relations"].add(triple.relation)
            edge["sources"].add(triple.source_chunk_id)
            self._append_structured_relation(edge, self._structured_record(triple))
        else:
            self.graph.add_edge(
                cause, effect,
                confidence=triple.confidence,
                relations={triple.relation},
                sources={triple.source_chunk_id},
                typed_relations=[self._structured_record(triple)],
            )

    @staticmethod
    def _clean_entity(text: str) -> str:
        text = re.sub(r"\s+", " ", str(text or "")).strip()
        return text.strip("，。；：、“”‘’\"'()（）[]【】")

    @staticmethod
    def _normalized_key(text: str) -> str:
        return re.sub(r"[\s\-_()/,.:;，。、“”‘’\"'\[\]【】]+", "", text.strip().lower())

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _normalize_entity_online(self, entity: str) -> str:
        # Direct map hit from previous build/manual merge.
        mapped = self._entity_map.get(entity)
        if mapped:
            return mapped

        # Exact node match.
        if entity in self.graph:
            self._entity_map[entity] = entity
            if entity not in self._entity_vectors:
                self._embed_missing_nodes([entity])
            return entity

        # Lexical fallback for near-identical aliases.
        key_entity = self._normalized_key(entity)
        lexical_threshold = settings.graph.dynamic_entity_lexical_threshold
        best_lexical_node = ""
        best_lexical_score = 0.0
        candidate_nodes = list(self.graph.nodes())
        max_candidates = settings.graph.dynamic_entity_max_candidates
        if len(candidate_nodes) > max_candidates:
            candidate_nodes = sorted(
                candidate_nodes,
                key=lambda node: self.graph.degree(node),
                reverse=True,
            )[:max_candidates]

        for node in candidate_nodes:
            node_key = self._normalized_key(node)
            if not node_key:
                continue
            if key_entity == node_key:
                self._entity_map[entity] = node
                return node
            score = SequenceMatcher(None, key_entity, node_key).ratio()
            if score > best_lexical_score:
                best_lexical_score = score
                best_lexical_node = node

        if best_lexical_node and best_lexical_score >= lexical_threshold:
            self._entity_map[entity] = best_lexical_node
            return best_lexical_node

        # Semantic fallback by embedding similarity.
        if self._entity_vectors:
            entity_vec = np.array(embedding_client.embed_one(entity))
            best_semantic_node = ""
            best_semantic_score = -1.0
            for node, vec in self._entity_vectors.items():
                sim = self._cosine_similarity(entity_vec, np.array(vec))
                if sim > best_semantic_score:
                    best_semantic_score = sim
                    best_semantic_node = node
            if (
                best_semantic_node
                and best_semantic_score >= settings.graph.entity_merge_threshold
            ):
                self._entity_map[entity] = best_semantic_node
                return best_semantic_node

            # Keep vector for newly created node if it doesn't merge.
            self._entity_vectors[entity] = entity_vec.tolist()
        else:
            # First runtime update for old graphs without cached vectors.
            self._entity_vectors[entity] = embedding_client.embed_one(entity)

        self._entity_map[entity] = entity
        return entity

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_causal_subgraph(self, node: str, depth: int = 2) -> nx.DiGraph:
        if not self.graph.has_node(node):
            return nx.DiGraph()
        reachable = nx.single_source_shortest_path_length(self.graph, node, cutoff=depth).keys()
        pred_reachable = nx.single_source_shortest_path_length(
            self.graph.reverse(), node, cutoff=depth
        ).keys()
        return self.graph.subgraph(set(reachable) | set(pred_reachable)).copy()

    def subgraph_to_text(self, subgraph: nx.DiGraph) -> str:
        lines = []
        for u, v, data in subgraph.edges(data=True):
            rels = ", ".join(data.get("relations", []))
            conf = data.get("confidence", 0.0)
            # Legacy view (kept for backward compatibility and historical summaries).
            base_line = f"{u} --[{rels} (置信度:{conf:.2f})]--> {v}"

            # Structured relation view (new format for causal reasoning).
            typed = data.get("typed_relations", [])
            typed_items: list[str] = []
            if isinstance(typed, list):
                for item in typed[:4]:
                    if not isinstance(item, dict):
                        continue
                    r_type = str(item.get("relation_type", "unknown"))
                    pol = str(item.get("polarity", "positive"))
                    r = str(item.get("relation", "")).strip()
                    c = float(item.get("confidence", conf) or conf)
                    typed_items.append(f"{r_type}/{pol}:{r or '-'}@{c:.2f}")
            if typed_items:
                lines.append(base_line + " | 结构化关系: " + " ; ".join(typed_items))
            else:
                lines.append(base_line)
        return "\n".join(lines) if lines else ""

    def find_similar_nodes(self, query_vec: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        if not self._entity_vectors:
            return []
        q = np.array(query_vec)
        scores = []
        for node, vec in self._entity_vectors.items():
            v = np.array(vec)
            sim = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9))
            scores.append((node, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ------------------------------------------------------------------
    # Structured relation discovery from text graph
    # ------------------------------------------------------------------

    def suggest_structured_relations_from_text_graph(
        self,
        text_graph: nx.DiGraph,
        max_edges: int = 600,
        use_llm: bool = True,
        keep_unknown: bool = False,
    ) -> dict:
        if text_graph.number_of_edges() == 0:
            return {
                "processed_edges": 0,
                "candidate_triples": [],
                "candidate_count": 0,
                "skipped_unknown": 0,
                "llm_classified_relations": 0,
            }

        typed_candidates: list[CausalTriple] = []
        unknown_relations: set[str] = set()
        processed = 0
        skipped_non_causal = 0

        for u, v, data in text_graph.edges(data=True):
            if processed >= max_edges:
                break
            rels = data.get("relations", [])
            rel_iter: Iterable[str] = rels if isinstance(rels, (list, set, tuple)) else [str(rels or "")]
            for rel in rel_iter:
                relation = str(rel or "").strip()
                if not relation:
                    continue
                if _is_non_causal_relation(relation):
                    skipped_non_causal += 1
                    continue
                relation_type = _guess_relation_type_regex(relation)
                if relation_type == "unknown":
                    if not _is_structured_relation_candidate(relation):
                        skipped_non_causal += 1
                        continue
                    unknown_relations.add(relation)
                typed_candidates.append(
                    CausalTriple(
                        cause=str(u),
                        relation=relation,
                        effect=str(v),
                        confidence=0.66,
                        source_chunk_id="text_graph_discovery",
                        relation_type=relation_type,
                        polarity=_guess_polarity(relation),
                        extraction_method="regex",
                        evidence_text=f"{u} --[{relation}]--> {v}",
                    )
                )
                processed += 1
                if processed >= max_edges:
                    break

        llm_map: dict[str, tuple[str, str]] = {}
        if use_llm and unknown_relations:
            llm_map = self._classify_relation_labels_llm(sorted(unknown_relations))

        skipped_unknown = 0
        kept: list[CausalTriple] = []
        for triple in typed_candidates:
            if triple.relation_type == "unknown" and triple.relation in llm_map:
                typ, pol = llm_map[triple.relation]
                triple.relation_type = _normalize_relation_type(typ)
                triple.polarity = _normalize_polarity(pol)
                triple.extraction_method = "llm+regex"
            if triple.relation_type == "unknown" and not keep_unknown:
                skipped_unknown += 1
                continue
            kept.append(triple)

        return {
            "processed_edges": processed,
            "candidate_triples": [self._structured_record(item) | {
                "cause": item.cause,
                "effect": item.effect,
            } for item in kept],
            "candidate_count": len(kept),
            "skipped_non_causal": skipped_non_causal,
            "skipped_unknown": skipped_unknown,
            "llm_classified_relations": len(llm_map),
        }

    def apply_structured_relation_candidates(
        self,
        candidates: list[dict],
        save: bool = True,
        normalize: bool = True,
    ) -> dict:
        added = 0
        skipped = 0
        for item in candidates:
            if not isinstance(item, dict):
                skipped += 1
                continue
            cause = str(item.get("cause", "")).strip()
            effect = str(item.get("effect", "")).strip()
            relation = str(item.get("relation", "")).strip()
            relation_type = _normalize_relation_type(item.get("relation_type", "unknown"))
            if not cause or not effect or not relation or relation_type == "unknown":
                skipped += 1
                continue
            triple = CausalTriple(
                cause=cause,
                relation=relation,
                effect=effect,
                confidence=float(item.get("confidence", 0.66) or 0.66),
                source_chunk_id=str(item.get("source_chunk_id", "text_graph_discovery") or "text_graph_discovery"),
                relation_type=relation_type,
                polarity=_normalize_polarity(item.get("polarity", "positive")),
                extraction_method=str(item.get("method", "regex") or "regex"),
                evidence_text=str(item.get("evidence_text", "")),
            )
            self.add_triple(triple, normalize=normalize)
            added += 1
        if added and save:
            self.save()
        return {"added": added, "skipped": skipped}

    def discover_structured_relations_from_text_graph(
        self,
        text_graph: nx.DiGraph,
        max_edges: int = 600,
        use_llm: bool = True,
        save: bool = True,
        keep_unknown: bool = False,
    ) -> dict:
        suggestion = self.suggest_structured_relations_from_text_graph(
            text_graph=text_graph,
            max_edges=max_edges,
            use_llm=use_llm,
            keep_unknown=keep_unknown,
        )
        applied = self.apply_structured_relation_candidates(
            candidates=suggestion.get("candidate_triples", []),
            save=save,
            normalize=True,
        )
        return {
            "processed_edges": int(suggestion.get("processed_edges", 0)),
            "discovered_triples": int(applied.get("added", 0)),
            "skipped_non_causal": int(suggestion.get("skipped_non_causal", 0)),
            "skipped_unknown": int(suggestion.get("skipped_unknown", 0)),
            "skipped_invalid": int(applied.get("skipped", 0)),
            "llm_classified_relations": int(suggestion.get("llm_classified_relations", 0)),
        }

    def _classify_relation_labels_llm(self, labels: list[str]) -> dict[str, tuple[str, str]]:
        if not labels:
            return {}
        # Batch labels to control cost.
        batch_size = 40
        out: dict[str, tuple[str, str]] = {}
        for i in range(0, len(labels), batch_size):
            batch = labels[i:i + batch_size]
            user = (
                "关系词列表：\n- " + "\n- ".join(batch) + "\n\n"
                "请逐项映射 relation_type 与 polarity，并给出 drop。"
            )
            try:
                data = llm_client.chat_json(
                    messages=[
                        {"role": "system", "content": _RELATION_CLASSIFY_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=max(1024, int(getattr(settings.llm, "max_tokens", 8192))),
                    parse_retry=1,
                )
            except Exception as e:
                logger.warning(f"Relation classify LLM failed: {e}")
                continue
            items = data.get("items", []) if isinstance(data, dict) else []
            for item in items:
                rel = str(item.get("relation", "")).strip()
                if not rel:
                    continue
                if _is_non_causal_relation(rel):
                    continue
                drop = bool(item.get("drop", False))
                if drop:
                    continue
                r_type = _normalize_relation_type(item.get("relation_type", "unknown"))
                pol = _normalize_polarity(item.get("polarity", "positive"))
                out[rel] = (r_type, pol)
        return out

    def suggest_merges(
        self,
        semantic_threshold: float = 0.92,
        lexical_threshold: float = 0.82,
        max_groups: int = 20,
    ) -> list[MergeSuggestion]:
        nodes = list(self.graph.nodes())
        if not nodes:
            return []
        self._embed_missing_nodes(nodes)
        degree_map = {node: int(self.graph.degree(node)) for node in nodes}
        vectors = {node: self._entity_vectors[node] for node in nodes if node in self._entity_vectors}
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
            self._entity_map[alias] = canonical
            self._entity_vectors.pop(alias, None)
        if canonical in self.graph:
            self._entity_vectors.pop(canonical, None)
            self._embed_missing_nodes([canonical])
        if save and merged:
            self.save()
        return merged

    def delete_nodes(self, nodes: list[str], save: bool = True) -> int:
        deleted = 0
        for node in nodes:
            if node not in self.graph:
                continue
            self.graph.remove_node(node)
            self._entity_vectors.pop(node, None)
            self._entity_map.pop(node, None)
            deleted += 1
        if save and deleted:
            self.save()
        return deleted

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "graph": _graph_to_json(self.graph),
            "entity_map": self._entity_map,
            "entity_vectors": self._entity_vectors,
        }
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Causal graph saved: {self._path}")

    def load(self) -> bool:
        if not self._path.exists():
            return False
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        self.graph = _graph_from_json(payload["graph"])
        self._entity_map = payload.get("entity_map", {})
        self._entity_vectors = payload.get("entity_vectors", {})
        logger.info(
            f"Causal graph loaded [{self.kb_id[:8]}]: "
            f"{self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges"
        )
        return True

    def delete_file(self) -> None:
        if self._path.exists():
            self._path.unlink()

    def _embed_missing_nodes(self, nodes: list[str]) -> None:
        missing = [node for node in nodes if node not in self._entity_vectors]
        if not missing:
            return
        logger.info(f"Embedding {len(missing)} causal graph nodes for merge suggestions...")
        vectors = embedding_client.embed(missing)
        for node, vec in zip(missing, vectors):
            self._entity_vectors[node] = vec

    @staticmethod
    def _merge_edge_attrs(target: dict, incoming: dict) -> None:
        target["relations"] = set(target.get("relations", set())) | set(incoming.get("relations", set()))
        target["sources"] = set(target.get("sources", set())) | set(incoming.get("sources", set()))
        target["confidence"] = max(float(target.get("confidence", 0.0)), float(incoming.get("confidence", 0.0)))
        cur = target.get("typed_relations")
        if not isinstance(cur, list):
            cur = []
        inc = incoming.get("typed_relations")
        if isinstance(inc, list):
            for item in inc:
                if not isinstance(item, dict):
                    continue
                CausalGraph._append_structured_relation(target, item)
        else:
            target["typed_relations"] = cur
