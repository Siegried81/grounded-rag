"""Offline tests for rag.embed: the fake embedder, HTTP parsing and the factory.

All HTTP is mocked by monkeypatching requests.post, so no network is touched.
"""

import math

import pytest

from rag import embed
from rag.embed import FakeEmbedder, HostedEmbedder, OllamaEmbedder, get_embedder


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
