"""Azure OpenAI as an LLM provider (rag/llm.py) and as an embedder (rag/embed.py).

All offline: `requests.post` is replaced by a stub, so no Azure resource and no
credit is needed. The checks: the v1 URL and the deployment name go where Azure
expects them, a half-configured resource is skipped instead of called, a
filtered (null) reply falls through like an empty one, the deployment is part of
the LLM cache key, and embeddings are batched and kept in input order.
"""

import pytest
import requests

import config
from rag import embed, llm


class _Resp:
    """Just enough of `requests.Response` for the chat and embeddings helpers."""

    def __init__(self, status: int = 200, payload: dict | None = None):
        self.status_code, self.headers, self._payload = status, {}, payload or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def json(self) -> dict:
        return self._payload


def _chat(text):
    return _Resp(200, {"choices": [{"message": {"content": text}}]})


@pytest.fixture
def azure(monkeypatch):
    """A fully configured Azure resource (fake values)."""
    monkeypatch.setattr(config, "AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com")
    monkeypatch.setattr(config, "AZURE_OPENAI_API_KEY", "az_key")
    monkeypatch.setattr(config, "AZURE_OPENAI_DEPLOYMENT", "chat-dep")
    monkeypatch.setattr(config, "AZURE_OPENAI_EMBED_DEPLOYMENT", "embed-dep")


def test_calls_the_v1_api_with_the_deployment_as_model(monkeypatch, azure):
    calls = []
    monkeypatch.setattr(llm.requests, "post",
                        lambda url, **k: calls.append((url, k)) or _chat("from azure"))

    assert llm.complete("q", order=["azure_openai"]) == "from azure"
    url, kwargs = calls[0]
    assert url == "https://res.openai.azure.com/openai/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer az_key"
    assert kwargs["json"]["model"] == "chat-dep" and kwargs["json"]["temperature"] == 0
    assert llm.last_provider() == "azure_openai"


@pytest.mark.parametrize("endpoint", ["https://res.openai.azure.com/openai/v1",
                                      "https://res.openai.azure.com/"])
def test_endpoint_with_or_without_the_v1_suffix_gives_one_url(monkeypatch, endpoint):
    monkeypatch.setattr(config, "AZURE_OPENAI_ENDPOINT", endpoint.rstrip("/"))
    assert config.azure_openai_v1_url() == "https://res.openai.azure.com/openai/v1"


def test_no_endpoint_means_no_url(monkeypatch):
    monkeypatch.setattr(config, "AZURE_OPENAI_ENDPOINT", "")
    assert config.azure_openai_v1_url() == ""


@pytest.mark.parametrize("missing", ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
                                     "AZURE_OPENAI_DEPLOYMENT"])
def test_half_configured_resource_is_skipped_not_called(monkeypatch, azure, missing):
    monkeypatch.setattr(config, missing, "")
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: pytest.fail("Azure was called"))
    with pytest.raises(llm.LLMError, match="azure_openai: no API key"):
        llm.complete("q", order=["azure_openai"])


def test_filtered_null_reply_falls_through_and_is_not_cached(monkeypatch, azure, tmp_path):
    """Azure's content filter can answer 200 with a null content."""
    monkeypatch.setattr(config, "LLM_CACHE", True)
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "OLLAMA_URL", "http://ollama")
    monkeypatch.setattr(llm.requests, "post", lambda url, **k: (
        _chat(None) if "azure" in url else _Resp(200, {"message": {"content": "from ollama"}})))

    assert llm.complete("q", order=["azure_openai", "ollama"]) == "from ollama"
    assert llm.last_provider() == "ollama"


def test_rejected_prompt_falls_through(monkeypatch, azure):
    """A prompt the content filter rejects is an HTTP 400, a failure like any other."""
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "or")
    monkeypatch.setattr(llm.requests, "post", lambda url, **k: (
        _Resp(400) if "azure" in url else _chat("from openrouter")))
    assert llm.complete("q", order=["azure_openai", "openrouter"]) == "from openrouter"


def test_streams_through_the_same_url(monkeypatch, azure):
    class _Stream(_Resp):
        def iter_lines(self):
            yield b'data: {"choices":[{"delta":{"content":"Hel"}}]}'
            yield b'data: {"choices":[{"delta":{"content":"lo"}}]}'
            yield b"data: [DONE]"

    urls = []
    monkeypatch.setattr(llm.requests, "post", lambda url, **k: urls.append(url) or _Stream())
    assert "".join(llm.stream("q", order=["azure_openai"])) == "Hello"
    assert urls == ["https://res.openai.azure.com/openai/v1/chat/completions"]


def test_deployment_is_part_of_the_cache_key_only_when_azure_is_in_the_order(monkeypatch, azure):
    monkeypatch.setattr(config, "LLM_CACHE", True)
    with_azure = llm._cache_path(["groq", "azure_openai"], "p", None)
    without = llm._cache_path(["groq"], "p", None)
    monkeypatch.setattr(config, "AZURE_OPENAI_DEPLOYMENT", "other-dep")
    assert llm._cache_path(["groq", "azure_openai"], "p", None) != with_azure
    assert llm._cache_path(["groq"], "p", None) == without


def test_fallback_order_adds_azure_before_ollama_only_with_a_key():
    assert config.fallback_order("groq", True, "", "") == ["groq", "groq", "openrouter", "ollama"]
    assert config.fallback_order("groq", True, "", "k") == [
        "groq", "groq", "openrouter", "azure_openai", "ollama"]
    assert config.fallback_order("azure_openai", False, "", "k") == ["azure_openai"]


def test_embedder_batches_and_restores_input_order(monkeypatch, azure):
    monkeypatch.setattr(config, "AZURE_OPENAI_EMBED_BATCH", 2)
    calls = []

    def fake_post(url, **kwargs):
        texts = kwargs["json"]["input"]
        calls.append((url, kwargs["json"]["model"], texts))
        # Answer out of order, tagged with each input's index.
        data = [{"index": i, "embedding": [float(len(t))]} for i, t in enumerate(texts)]
        return _Resp(200, {"data": list(reversed(data))})

    monkeypatch.setattr(embed.requests, "post", fake_post)
    e = embed.get_embedder("azure_openai")
    assert e.model == "embed-dep"
    assert e.embed(["a", "bb", "ccc"]) == [[1.0], [2.0], [3.0]]
    assert [c[2] for c in calls] == [["a", "bb"], ["ccc"]]
    assert calls[0][0] == "https://res.openai.azure.com/openai/v1/embeddings"


def test_embedder_refuses_a_vector_count_mismatch(monkeypatch, azure):
    monkeypatch.setattr(embed.requests, "post",
                        lambda url, **k: _Resp(200, {"data": [{"index": 0, "embedding": [1.0]}]}))
    with pytest.raises(RuntimeError, match="1 vectors for 2 texts"):
        embed.get_embedder("azure_openai").embed(["a", "b"])


def test_embedder_needs_the_three_settings(monkeypatch, azure):
    monkeypatch.setattr(config, "AZURE_OPENAI_EMBED_DEPLOYMENT", "")
    with pytest.raises(ValueError, match="AZURE_OPENAI_EMBED_DEPLOYMENT"):
        embed.get_embedder("azure_openai")


def test_hosted_embedder_without_batch_size_still_sends_one_request(monkeypatch):
    calls = []
    monkeypatch.setattr(embed.requests, "post", lambda url, **k: calls.append(k) or _Resp(
        200, {"data": [{"embedding": [0.0]} for _ in k["json"]["input"]]}))
    embed.HostedEmbedder("https://h/v1", "m", "k").embed(["a", "b", "c"])
    assert len(calls) == 1
