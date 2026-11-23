"""Embedder implementations (fake, Ollama, hosted, Azure OpenAI, in-process) and a config-driven factory.

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
    `batch_size` splits it for a provider that caps one request (Azure OpenAI
    limits both inputs and tokens per call); None keeps the single request.
    """

    def __init__(self, base_url: str, model: str, api_key: str,
                 batch_size: int | None = None):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._batch_size = batch_size

    @property
    def model(self) -> str:
        """Identify the backing model name."""
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed the texts (one POST per batch) and return the vectors in input order."""
        size = self._batch_size or max(len(texts), 1)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), size):
            vectors.extend(self._post(texts[start:start + size]))
        return vectors

    def _post(self, texts: list[str]) -> list[list[float]]:
        """One /embeddings request, its vectors put back in input order.

        The API tags each vector with the position of its input; sorting on it
        guards against a server that does not answer in order, since a swapped
        vector would silently attach a passage to another passage's meaning.
        """
        resp = requests.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": texts},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        if len(data) != len(texts):
            raise RuntimeError(f"embeddings endpoint returned {len(data)} vectors for "
                               f"{len(texts)} texts")
        data = sorted(data, key=lambda d: d.get("index", 0))
        return [d["embedding"] for d in data]


class SentenceTransformerEmbedder:
    """Embedder that runs a sentence-transformers model in this process.

    The other two backends are servers: `OllamaEmbedder` needs a daemon listening
    on a port, `HostedEmbedder` needs the network and a key. Both are things that
    can be down when it matters. This one holds the model in memory, which makes
    retrieval work with nothing else running - and removes one HTTP round trip per
    batch, which is the whole cost of embedding a corpus.

    What it costs instead: a torch install and a model download the first time the
    name is seen. So the import is LAZY - the module must stay importable, and the
    test suite must stay runnable, on a machine that has neither.

    The model name travels into `self.model`, which the store records beside every
    vector. Two embedding spaces are not comparable, and the store refuses to mix
    them rather than returning a plausible nearest neighbour from the wrong one.
    """

    def __init__(self, model: str):
        self._name = model
        self._encoder = None  # built on first use, see embed()

    @property
    def model(self) -> str:
        """Identify the backing model, exactly as configured."""
        return self._name

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one normalised vector per input text, in the same order.

        `normalize_embeddings=True` because the rest of the pipeline compares with
        a dot product and treats it as cosine similarity: unnormalised vectors
        would make that a length-weighted score, which ranks long chunks higher
        for being long.
        """
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "EMBED_PROVIDER=sentence_transformers needs the "
                    "`sentence-transformers` package (which pulls in torch). "
                    "Install it, or use EMBED_PROVIDER=ollama for the daemon "
                    "backend or `fake` for tests."
                ) from exc
            self._encoder = SentenceTransformer(self._name)
        vectors = self._encoder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return [[float(x) for x in row] for row in vectors]


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
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(config.ST_EMBED_MODEL)
    if provider == "hosted":
        return HostedEmbedder(
            config.HOSTED_EMBED_BASE_URL,
            config.HOSTED_EMBED_MODEL,
            config.HOSTED_EMBED_API_KEY,
        )
    if provider == "azure_openai":
        # The v1 API of an Azure OpenAI resource is OpenAI-compatible, so it is
        # the hosted embedder on the resource's URL, batched. The store records
        # the deployment name as the model: a renamed deployment is a new
        # embedding space as far as the index is concerned, which is the safe side.
        if not (config.AZURE_OPENAI_ENDPOINT and config.AZURE_OPENAI_API_KEY
                and config.AZURE_OPENAI_EMBED_DEPLOYMENT):
            raise ValueError(
                "EMBED_PROVIDER=azure_openai needs AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_EMBED_DEPLOYMENT"
            )
        return HostedEmbedder(
            config.azure_openai_v1_url(),
            config.AZURE_OPENAI_EMBED_DEPLOYMENT,
            config.AZURE_OPENAI_API_KEY,
            batch_size=config.AZURE_OPENAI_EMBED_BATCH,
        )
    raise ValueError(f"Unknown embed provider: {provider!r}")
