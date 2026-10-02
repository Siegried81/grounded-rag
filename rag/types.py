"""Shared data structures and the Embedder contract for the RAG pipeline.

These types are the single source of truth every module builds against: chunking
produces `Chunk`s, retrieval returns `Retrieved`s, and anything that turns text
into vectors satisfies the `Embedder` protocol. Keeping the contract in one tiny,
dependency-free module is what lets the tests inject a fake embedder and lets the
store, retriever and ingester stay decoupled from any specific model or vendor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage of a source document.

    `source` is the file it came from and `ordinal` its position within that file,
    so a citation can point a reader back to the exact passage. `meta` carries any
    extra per-chunk context (e.g. a PDF page number) without widening this type.
    """

    id: str
    text: str
    source: str
    ordinal: int
    corpus: str
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Retrieved:
    """A chunk paired with its cosine similarity to the query (1.0 == identical)."""

    chunk: Chunk
    score: float


@runtime_checkable
class Embedder(Protocol):
    """Anything that maps texts to fixed-length vectors.

    The pipeline depends only on this protocol, never on a concrete model, so the
    embedding backend (local Ollama, a hosted endpoint, or a test fake) is a
    swap-in. `model` identifies the backing model; the store records it to refuse
    mixing vectors from two different embedding spaces.
    """

    @property
    def model(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...
