"""Document loading and chunking for the knowledge base."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str          # file path or document name
    chunk_index: int     # position within source document
    metadata: dict = field(default_factory=dict)


class DocumentPreprocessor:
    """
    Loads plain text / markdown files and splits them into overlapping chunks.
    No local ML models — pure text processing.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size        # chars
        self.chunk_overlap = chunk_overlap  # chars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_directory(self, directory: str) -> list[Chunk]:
        """Recursively load all .txt and .md files from a directory."""
        chunks: list[Chunk] = []
        for path in Path(directory).rglob("*"):
            if path.suffix in {".txt", ".md"}:
                try:
                    chunks.extend(self.load_file(str(path)))
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
        logger.info(f"Loaded {len(chunks)} chunks from {directory}")
        return chunks

    def load_file(self, filepath: str) -> list[Chunk]:
        """Load a single file and return its chunks."""
        text = Path(filepath).read_text(encoding="utf-8")
        text = self._clean(text)
        raw_chunks = self._split(text)
        source = os.path.basename(filepath)
        return [
            Chunk(
                chunk_id=f"{source}_{i}",
                text=c,
                source=source,
                chunk_index=i,
                metadata={"filepath": filepath},
            )
            for i, c in enumerate(raw_chunks)
        ]

    def load_texts(self, texts: list[str], source: str = "inline") -> list[Chunk]:
        """Load raw text strings directly."""
        chunks: list[Chunk] = []
        for doc_idx, text in enumerate(texts):
            text = self._clean(text)
            raw_chunks = self._split(text)
            for chunk_idx, c in enumerate(raw_chunks):
                chunks.append(
                    Chunk(
                        chunk_id=f"{source}_{doc_idx}_{chunk_idx}",
                        text=c,
                        source=source,
                        chunk_index=chunk_idx,
                        metadata={"doc_index": doc_idx},
                    )
                )
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _split(self, text: str) -> list[str]:
        """Sentence-aware sliding window split."""
        # Split into sentences first then reassemble into chunks
        sentences = re.split(r"(?<=[。！？.!?])\s*", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[str] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) <= self.chunk_size:
                current = (current + " " + sent).strip()
            else:
                if current:
                    chunks.append(current)
                # Start new chunk with overlap from end of previous
                if len(current) > self.chunk_overlap:
                    overlap_text = current[-self.chunk_overlap :]
                    current = (overlap_text + " " + sent).strip()
                else:
                    current = sent
        if current:
            chunks.append(current)
        return chunks
