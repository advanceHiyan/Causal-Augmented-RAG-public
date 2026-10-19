import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root = directory containing this file's parent (config/)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


@dataclass
class LLMConfig:
    # DeepSeek
    deepseek_api_key: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-pro"
    deepseek_v4_pro_model: str = "deepseek-v4-pro"
    deepseek_v4_flash_model: str = "deepseek-v4-flash"
    deepseek_v4_pro_reasoning_effort: str = "high"
    deepseek_v4_pro_thinking_type: str = "enabled"
    deepseek_v4_flash_reasoning_effort: str = "high"
    deepseek_v4_flash_thinking_type: str = "enabled"
    # Default replacement mapping for deprecated aliases:
    # deepseek-chat -> deepseek-v4-flash (thinking mode)
    # deepseek-reasoner -> deepseek-v4-pro (thinking mode)
    deepseek_chat_model: str = "deepseek-v4-flash"
    deepseek_reasoner_model: str = "deepseek-v4-pro"

    # OpenRouter (optional, more model variety)
    openrouter_api_key: str = field(
        default_factory=lambda: (
            os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("openrouter_api_key", "")
        )
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "deepseek/deepseek-chat"
    openrouter_gpt5mini_model: str = "openai/gpt-5-mini"

    # Gemini (free-tier key is often rate-limited)
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    gemini_paid_api_key: str = field(
        default_factory=lambda: (
            os.environ.get("GEMINI_PAID_API_KEY", "")
            or os.environ.get("GEMINI_API_KEY_PAID", "")
        )
    )
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_model: str = "gemini-2.0-flash"
    gemini_flash_preview_model: str = "gemini-3-flash-preview"
    gemini_flash_lite_model: str = "gemini-2.0-flash-lite"
    gemini_flash_lite_preview_model: str = "gemini-3.1-flash-lite-preview"
    gemini_retry_max_attempts: int = 4
    gemini_retry_base_delay_s: float = 1.2

    # Default provider: "deepseek" | "openrouter" | "gemini"
    default_provider: str = "deepseek"

    # Generation params
    temperature: float = 0.0
    max_tokens: int = 131072
    llm_retry_max_attempts: int = 2
    llm_retry_base_delay_s: float = 3.0
    llm_rate_limit_cooldown_s: float = 30.0
    llm_rate_limit_max_wait_s: float = 300.0
    llm_max_concurrent_calls: int = 256


@dataclass
class EmbeddingConfig:
    # SiliconFlow embeddings
    api_key: str = field(default_factory=lambda: os.environ.get("SILICON_API_KEY", ""))
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "BAAI/bge-large-zh-v1.5"
    dimension: int = 1024
    batch_size: int = 32


@dataclass
class RerankConfig:
    # SiliconFlow reranking
    api_key: str = field(default_factory=lambda: os.environ.get("SILICON_API_KEY", ""))
    base_url: str = "https://api.siliconflow.cn/v1"
    model: str = "BAAI/bge-reranker-v2-m3"
    top_n: int = 5


@dataclass
class VectorStoreConfig:
    # ChromaDB — stored inside project data dir
    persist_dir: str = field(default_factory=lambda: str(DATA_DIR / "chroma_db"))
    collection_name: str = "documents"
    top_k: int = 10


@dataclass
class GraphConfig:
    # JSON files for graphs — stored inside project data dir
    text_graph_path: str = field(default_factory=lambda: str(DATA_DIR / "text_graph.json"))
    causal_graph_path: str = field(default_factory=lambda: str(DATA_DIR / "causal_graph.json"))

    # Causal extraction
    causal_confidence_threshold: float = 0.6
    # Entity normalization: cosine similarity threshold for merging entities
    entity_merge_threshold: float = 0.85
    # Dynamic write-back normalization: lexical similarity fallback threshold
    dynamic_entity_lexical_threshold: float = 0.94
    # Dynamic write-back normalization: cap candidate nodes for similarity matching
    dynamic_entity_max_candidates: int = 400

    # Optional implicit causal discovery during build
    implicit_causal_discovery_default: bool = False
    implicit_causal_max_chunks: int = 24
    implicit_causal_min_signal_score: int = 1

    # Retrieval
    max_graph_depth: int = 3
    l2_entry_top_k: int = 3
    l2_entry_min_similarity: float = 0.45
    l2_force_top_k_when_threshold_miss: bool = True
    l2_candidate_node_limit: int = 120
    l2_candidate_edge_limit: int = 220
    l2_depth_penalty: float = 0.08
    max_causal_subgraph_nodes: int = 20
    min_causal_path_confidence: float = 0.5


@dataclass
class RetrievalConfig:
    causal_match_threshold: float = 0.7
    vector_fetch_k: int = 20
    vector_top_k: int = 5
    use_reranker: bool = True


@dataclass
class GenerationConfig:
    use_counterfactual: bool = True
    use_self_reflection: bool = True
    max_context_length: int = 3000
    max_chat_history_turns: int = 8
    rag_query_max_tokens: int = 128


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


settings = Settings()
