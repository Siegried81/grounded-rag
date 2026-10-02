"""Persistent BM25 keyword index over chunks.

Dense embeddings miss exact terms (identifiers, rare names, acronyms); BM25
catches them. This module is self-contained so it can be fused with dense
results without depending on the vector store.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag.types import Chunk, Retrieved

_TOKEN_SPLIT = re.compile(r"[\W_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase `text` and split on non-alphanumerics, dropping empties.

    One shared tokeniser for documents and queries guarantees both sides of the
    BM25 match are normalised identically.
    """
    return [t for t in _TOKEN_SPLIT.split(text.lower()) if t]


class BM25Index:
    """BM25Okapi index over an ordered list of chunks, savable to JSON.

    Tokenised documents are persisted next to the chunks so reloading skips
    re-tokenising; the BM25 statistics are cheap to rebuild from them.
    """

    def __init__(self, chunks: list[Chunk], tokenized: list[list[str]]) -> None:
        """Wrap chunks with their token lists and build the BM25 model."""
        self._chunks = list(chunks)
        self._tokenized = tokenized
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "BM25Index":
        """Tokenise every chunk's text and build the index, keeping chunk order."""
        return cls(chunks, [tokenize(c.text) for c in chunks])

    def search(self, query: str, top_k: int) -> list[Retrieved]:
        """Return up to `top_k` chunks by BM25 score, best first.

        Empty index, empty query or non-positive `top_k` yield []. Chunks sharing
        no term with the query score 0 and are excluded, since they are noise
        for a keyword channel.
        """
        tokens = tokenize(query)
        if self._bm25 is None or not tokens or top_k <= 0:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [
            Retrieved(self._chunks[i], float(scores[i]))
            for i in order[:top_k]
            if scores[i] > 0
        ]

    def save(self, path: Path) -> None:
        """Write chunks and tokenised docs to `path` as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [asdict(c) for c in self._chunks],
            "tokenized": self._tokenized,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Reload an index saved by `save`; raise FileNotFoundError if absent."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"BM25 index file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls([Chunk(**c) for c in data["chunks"]], data["tokenized"])

    def __len__(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)
