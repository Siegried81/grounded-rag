"""Embedder implementations (fake, Ollama, hosted) and a config-driven factory.

Each class satisfies the `Embedder` protocol from `rag.types`, so the rest of the
pipeline never knows which backend is in use. `get_embedder` is the single place
that maps the configured provider name to a concrete embedder, which keeps
provider selection out of the ingester and retriever.
"""

from __future__ import annotations

import hashlib
import math
import random

import requests

import config
from rag.types import Embedder

FAKE_DIM = 64
_TIMEOUT = 60


class FakeEmbedder:
    """Deterministic, offline embedder used by the whole test suite.

    It seeds a PRNG from a hash of the text and L2-normalises the result, so the
    same text always yields the same unit vector without any network or model.
    """

    @property
    def model(self) -> str:
        """Identify the backing model; always "fake"."""
        return "fake"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one deterministic, normalised 64-dim vector per text."""
        out = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            rng = random.Random(seed)
            vec = [rng.uniform(-1.0, 1.0) for _ in range(FAKE_DIM)]
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


class OllamaEmbedder:
    """Embedder backed by a local Ollama server (one request per text).

    Ollama's /api/embeddings endpoint accepts a single prompt, hence the loop.
    """

    def __init__(self, url: str, model: str):
        self._url = url.rstrip("/")
        self._model = model

    @property
    def model(self) -> str:
        """Identify the backing model name."""
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each text with a separate POST and return the vectors in order."""
        vectors = []
        for text in texts:
            resp = requests.post(
                f"{self._url}/api/embeddings",
                json={"model": self._model, "prompt": text},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
        return vectors


class HostedEmbedder:
    """Embedder for an OpenAI-compatible hosted /embeddings endpoint.

    The whole batch goes in one request, which is cheaper than per-text calls.
    """

    def __init__(self, base_url: str, model: str, api_key: str):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key

    @property
    def model(self) -> str:
        """Identify the backing model name."""
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed all texts in one POST and return the vectors in input order."""
        resp = requests.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": texts},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return [d["embedding"] for d in resp.json()["data"]]


def get_embedder(provider: str | None = None) -> Embedder:
    """Build the embedder for `provider` (default: config.EMBED_PROVIDER).

    Raises ValueError on an unknown provider so a config typo fails loudly
    instead of silently falling back to another embedding space.
    """
    provider = provider or config.EMBED_PROVIDER
    if provider == "fake":
        return FakeEmbedder()
    if provider == "ollama":
        return OllamaEmbedder(config.OLLAMA_URL, config.OLLAMA_EMBED_MODEL)
    if provider == "hosted":
        return HostedEmbedder(
            config.HOSTED_EMBED_BASE_URL,
            config.HOSTED_EMBED_MODEL,
            config.HOSTED_EMBED_API_KEY,
        )
    raise ValueError(f"Unknown embed provider: {provider!r}")
