"""Offline tests for rag.embed: the fake embedder, HTTP parsing and the factory.

All HTTP is mocked by monkeypatching requests.post, so no network is touched.
"""

import math

import pytest

from rag import embed
import config
from rag.embed import (
    FakeEmbedder,
    HostedEmbedder,
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_fake_deterministic_and_dimension():
    e = FakeEmbedder()
    a = e.embed(["hello"])[0]
    assert a == e.embed(["hello"])[0]
    assert len(a) == 64
    assert e.model == "fake"


def test_fake_normalised_and_distinct():
    vecs = FakeEmbedder().embed(["one", "two"])
    for v in vecs:
        assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-9)
    assert vecs[0] != vecs[1]


def test_ollama_parses_response(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        return FakeResponse({"embedding": [0.1, 0.2]})

    monkeypatch.setattr(embed.requests, "post", fake_post)
    e = OllamaEmbedder("http://x:11434", "m")
    assert e.model == "m"
    assert e.embed(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert calls[0] == ("http://x:11434/api/embeddings", {"model": "m", "prompt": "a"})
    assert len(calls) == 2


def test_hosted_parses_response(monkeypatch):
    seen = {}

    def fake_post(url, **kwargs):
        seen["url"] = url
        seen["headers"] = kwargs["headers"]
        seen["json"] = kwargs["json"]
        return FakeResponse({"data": [{"embedding": [1.0]}, {"embedding": [2.0]}]})

    monkeypatch.setattr(embed.requests, "post", fake_post)
    e = HostedEmbedder("http://h/v1", "hm", "key")
    assert e.model == "hm"
    assert e.embed(["a", "b"]) == [[1.0], [2.0]]
    assert seen["url"] == "http://h/v1/embeddings"
    assert seen["headers"]["Authorization"] == "Bearer key"
    assert seen["json"] == {"model": "hm", "input": ["a", "b"]}


def test_get_embedder_providers():
    assert isinstance(get_embedder("fake"), FakeEmbedder)
    assert isinstance(get_embedder("ollama"), OllamaEmbedder)
    assert isinstance(get_embedder("hosted"), HostedEmbedder)


def test_get_embedder_default_uses_config(monkeypatch):
    monkeypatch.setattr(embed.config, "EMBED_PROVIDER", "fake")
    assert isinstance(get_embedder(), FakeEmbedder)


def test_get_embedder_unknown():
    with pytest.raises(ValueError):
        get_embedder("nope")


# --- the in-process backend -----------------------------------------------------

def test_sentence_transformers_is_built_by_the_factory(monkeypatch):
    """A fourth provider, selected the same way as the other three."""
    monkeypatch.setattr(config, "ST_EMBED_MODEL", "some-org/some-model")
    e = get_embedder("sentence_transformers")
    assert isinstance(e, SentenceTransformerEmbedder)
    assert e.model == "some-org/some-model"


def test_the_model_is_not_loaded_until_something_is_embedded(monkeypatch):
    """The import is lazy on purpose: `sentence-transformers` pulls in torch, and
    this module has to stay importable - and the suite runnable - on a machine
    that has neither. Constructing the embedder must therefore touch nothing."""
    e = SentenceTransformerEmbedder("some-org/some-model")
    assert e.model == "some-org/some-model"  # no download, no import, no error


def test_a_missing_package_names_the_alternatives(monkeypatch):
    """The failure a reader meets first, so it has to say what to do instead
    rather than surfacing an ImportError from three frames down."""
    import builtins

    real_import = builtins.__import__

    def no_sentence_transformers(name, *args, **kwargs):
        if name.startswith("sentence_transformers"):
            raise ImportError("no module named sentence_transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sentence_transformers)
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        SentenceTransformerEmbedder("some-org/some-model").embed(["hello"])


def test_vectors_are_normalised_and_ordered(monkeypatch):
    """Two properties the rest of the pipeline depends on.

    Order, because `embed` is called on a batch and the vectors are zipped back
    onto their chunks positionally. Normalisation, because retrieval compares with
    a dot product and calls the result cosine similarity - unnormalised vectors
    would make it a length-weighted score and rank long chunks higher for being
    long.
    """

    class _Encoder:
        def encode(self, texts, normalize_embeddings=False, convert_to_numpy=False):
            assert normalize_embeddings, "the pipeline treats the dot product as cosine"
            return [[1.0, 0.0, 0.0] if t == "a" else [0.0, 1.0, 0.0] for t in texts]

    e = SentenceTransformerEmbedder("some-org/some-model")
    e._encoder = _Encoder()
    out = e.embed(["a", "b", "a"])
    assert out == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
