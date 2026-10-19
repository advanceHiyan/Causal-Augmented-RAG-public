"""
Causal-Augmented RAG — main entry point.

Usage:
  # Build knowledge base from a directory of documents
  python main.py build --docs ./data/documents

  # Interactive Q&A
  python main.py chat

  # Evaluate on a JSONL dataset
  python main.py eval --dataset ./data/datasets/causal_qa.jsonl --output ./data/eval_results.jsonl

  # Single question (non-interactive)
  python main.py ask "经济危机是如何导致失业率上升的？"
"""

from __future__ import annotations

import argparse
import os
import sys

from loguru import logger

from evaluation.evaluator import Evaluator
from config.settings import settings
from knowledge_base.kb_manager import get_kb_manager
from knowledge_base.preprocessor import DocumentPreprocessor


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class CausalRAGPipeline:
    def __init__(self, kb_id: str = "cli_default"):
        self._kb_id = kb_id
        self._mgr = get_kb_manager(kb_id)
        self._preprocessor = DocumentPreprocessor()

    def build(
        self,
        docs_dir: str,
        enable_implicit_causal_discovery: bool = False,
        implicit_max_chunks: int | None = None,
    ) -> None:
        """Full knowledge base construction pipeline."""
        logger.info(f"=== Building knowledge base [{self._kb_id[:8]}] from: {docs_dir} ===")

        # 1. Load and chunk documents
        chunks = self._preprocessor.load_directory(docs_dir)
        if not chunks:
            logger.error(f"No .txt or .md files found in {docs_dir}")
            return

        # 2. Build vector index
        logger.info("--- Step 1/3: Building vector store ---")
        self._mgr.vector_store.add_chunks(chunks)
        logger.info(f"Vector store: {self._mgr.vector_store.count()} chunks indexed")

        # 3. Build text graph
        logger.info("--- Step 2/3: Building text graph ---")
        self._mgr.text_graph.build_from_chunks(chunks, save=True)

        # 4. Build causal graph
        logger.info("--- Step 3/3: Building causal graph ---")
        self._mgr.causal_graph.build_from_chunks(
            chunks,
            save=True,
            enable_implicit_discovery=enable_implicit_causal_discovery,
            implicit_max_chunks=implicit_max_chunks,
        )

        logger.info("=== Knowledge base build complete ===")

    def load(self) -> None:
        """Load pre-built graphs from disk (vector store auto-loads via ChromaDB)."""
        self._mgr.load_graphs()

    def ask(self, question: str, persist_causal_updates: bool = False) -> str:
        answer, _ = self._mgr.ask(question, persist_causal_updates=persist_causal_updates)
        return answer

    def chat(self, persist_causal_updates: bool = False) -> None:
        logger.info("Loading knowledge graphs...")
        self.load()
        print("\nCausal-Augmented RAG interactive mode")
        print("Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                question = input("Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                return
            if not question:
                continue
            if question.lower() in {"quit", "exit"}:
                return
            answer = self.ask(question, persist_causal_updates=persist_causal_updates)
            print(f"\nA:\n{answer}\n")
        return
        logger.info("Loading knowledge graphs...")
        self.load()
        print("\nCausal-Augmented RAG — 交互问答模式")
        print("输入 'quit' 或 'exit' 退出\n")
        while True:
            try:
                question = input("问题: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出。")
                break
            if not question:
                continue
            if question.lower() in {"quit", "exit", "退出"}:
                break
            answer = self.ask(question, persist_causal_updates=persist_causal_updates)
            print(f"\n回答:\n{answer}\n")

    def evaluate(self, dataset_path: str, output_path: str | None = None) -> None:
        logger.info("Loading knowledge graphs for evaluation...")
        self.load()
        evaluator = Evaluator(use_llm_judge=True)
        summary = evaluator.evaluate_dataset(
            dataset_path=dataset_path,
            predict_fn=self.ask,
            output_path=output_path,
        )
        print(f"\n=== Evaluation Results ===\n{summary}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _configure_console_encoding() -> None:
    """Best-effort UTF-8 console configuration for Windows terminals."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleOutputCP(65001)
    except Exception:
        pass

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main() -> None:
    _configure_console_encoding()
    parser = argparse.ArgumentParser(description="Causal-Augmented RAG system")
    parser.add_argument("--kb-id", default="cli_default", help="Knowledge base ID")
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = sub.add_parser("build", help="Build knowledge base from documents")
    p_build.add_argument("--docs", default="./data/documents", help="Document directory")
    p_build.add_argument(
        "--enable-implicit-causal-discovery",
        action="store_true",
        help="Include weak-signal chunks for optional implicit causal discovery",
    )
    p_build.add_argument(
        "--implicit-max-chunks",
        type=int,
        default=settings.graph.implicit_causal_max_chunks,
        help="Max weak-signal chunks for implicit causal discovery",
    )

    # chat
    p_chat = sub.add_parser("chat", help="Interactive Q&A")
    p_chat.add_argument(
        "--persist-causal-updates",
        action="store_true",
        help="Persist L2 dynamic causal updates to causal_graph.json",
    )

    # ask
    p_ask = sub.add_parser("ask", help="Single question")
    p_ask.add_argument("question", type=str)
    p_ask.add_argument(
        "--persist-causal-updates",
        action="store_true",
        help="Persist L2 dynamic causal updates to causal_graph.json",
    )

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate on a JSONL dataset")
    p_eval.add_argument("--dataset", required=True)
    p_eval.add_argument("--output", default=None)

    args = parser.parse_args()
    pipeline = CausalRAGPipeline(kb_id=args.kb_id)

    if args.command == "build":
        pipeline.build(
            args.docs,
            enable_implicit_causal_discovery=args.enable_implicit_causal_discovery,
            implicit_max_chunks=max(0, int(args.implicit_max_chunks)),
        )

    elif args.command == "chat":
        pipeline.chat(persist_causal_updates=args.persist_causal_updates)

    elif args.command == "ask":
        pipeline.load()
        answer = pipeline.ask(
            args.question,
            persist_causal_updates=args.persist_causal_updates,
        )
        print(answer)

    elif args.command == "eval":
        pipeline.evaluate(args.dataset, args.output)


if __name__ == "__main__":
    main()
