"""Persistent embedding cache so re-ingesting an unchanged corpus costs nothing.

Embedding is the slow/expensive step of ingestion. Keying vectors by a hash of
(model, text) means unchanged passages are never re-embedded, and switching model
never returns a vector from the wrong embedding space.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class EmbeddingCache:
    """JSON-backed map from hash(model, text) to an embedding vector.

    JSON keeps it human-inspectable and dependency-free; fine at this project's
    scale.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict[str, list[float]] = {}

    @staticmethod
    def _key(model: str, text: str) -> str:
        """Hash (model, text); the NUL separator avoids ambiguous concatenations."""
        return hashlib.sha256(f"{model}\0{text}".encode("utf-8")).hexdigest()

    def get(self, model: str, text: str) -> list[float] | None:
        """Return the cached vector, or None on a miss."""
        return self._data.get(self._key(model, text))

    def put(self, model: str, text: str, vector: list[float]) -> None:
        """Store a vector in memory (call `save` to persist)."""
        self._data[self._key(model, text)] = list(vector)

    def save(self) -> None:
        """Write the cache to disk, creating parent folders if needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EmbeddingCache":
        """Load a cache from disk; a missing file yields an empty cache.

        A first run has no cache yet, so that must not be an error.
        """
        cache = cls(path)
        p = Path(path)
        if p.exists():
            cache._data = json.loads(p.read_text(encoding="utf-8"))
        return cache

    def embed_with_cache(self, embedder, texts: list[str]) -> list[list[float]]:
        """Return a vector per text, embedding only the cache misses.

        Misses are sent to `embedder.embed` in one batch (duplicates once), then
        stored; output order always matches input order.
        """
        model = embedder.model
        missing: list[str] = []
        for t in texts:
            if self.get(model, t) is None and t not in missing:
                missing.append(t)
        if missing:
            for t, v in zip(missing, embedder.embed(missing)):
                self.put(model, t, v)
        return [self.get(model, t) for t in texts]
