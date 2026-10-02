"""Offline tests for the LLM fallback client and the cite-or-refuse answer step.

All HTTP is replaced by monkeypatching `requests.post`, so no network or API key
is needed; this keeps the suite fast and deterministic.
"""

import pytest

import config
import rag.answer as answer
import rag.llm as llm
from rag.types import Chunk, Retrieved


class FakeResp:
    """Minimal stand-in for requests.Response."""

    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


OPENAI_JSON = {"choices": [{"message": {"content": "hello from openai-style"}}]}
OLLAMA_JSON = {"message": {"content": "hello from ollama"}}


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    """Give both hosted providers a dummy key so they are attempted by default.

    GROQ_API_KEYS is set explicitly so the suite stays deterministic even when a
    real .env with live Groq keys is present (config.py loads it at import).
    """
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["g"])
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "o")


def test_groq_extraction_and_payload(monkeypatch):
    seen = {}

    def post(url, **kw):
        seen["url"], seen["kw"] = url, kw
        return FakeResp(OPENAI_JSON)

    monkeypatch.setattr(llm.requests, "post", post)
    assert llm.complete("hi", system="sys", order=["groq"]) == "hello from openai-style"
    assert "groq.com" in seen["url"]
    assert seen["kw"]["headers"]["Authorization"] == "Bearer g"
    assert seen["kw"]["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["kw"]["json"]["temperature"] == 0


def test_ollama_extraction(monkeypatch):
    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: FakeResp(OLLAMA_JSON))
    assert llm.complete("hi", order=["ollama"]) == "hello from ollama"


def test_fallback_on_failure_and_dedup(monkeypatch):
    calls = []

    def post(url, **kw):
        calls.append(url)
        if "groq" in url:
            raise RuntimeError("boom")
        return FakeResp(OPENAI_JSON)

    monkeypatch.setattr(llm.requests, "post", post)
    out = llm.complete("hi", order=["groq", "groq", "openrouter"])
    assert out == "hello from openai-style"
    assert len(calls) == 2 and "openrouter" in calls[1]


def test_groq_rotates_keys_on_failure(monkeypatch):
    # First key is rate-limited (429-like), rotation must retry with the second.
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["k1", "k2"])
    seen = []

    def post(url, **kw):
        key = kw["headers"]["Authorization"]
        seen.append(key)
        if key == "Bearer k1":
            raise RuntimeError("429 rate limit")
        return FakeResp(OPENAI_JSON)

    monkeypatch.setattr(llm.requests, "post", post)
    assert llm.complete("hi", order=["groq"]) == "hello from openai-style"
    assert seen == ["Bearer k1", "Bearer k2"]  # rotated to the second key


def test_skips_provider_without_key(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEYS", [])
    calls = []

    def post(url, **kw):
        calls.append(url)
        return FakeResp(OPENAI_JSON)

    monkeypatch.setattr(llm.requests, "post", post)
    llm.complete("hi", order=["groq", "openrouter"])
    assert len(calls) == 1 and "openrouter" in calls[0]


def test_all_fail_raises(monkeypatch):
    def post(url, **kw):
        raise RuntimeError("down")

    monkeypatch.setattr(llm.requests, "post", post)
    with pytest.raises(llm.LLMError) as e:
        llm.complete("hi", order=["groq", "openrouter", "ollama"])
    assert "down" in str(e.value)


def _retrieved():
    c1 = Chunk(id="a", text="Alpha text.", source="a.md", ordinal=0, corpus="c")
    c2 = Chunk(id="b", text="Beta text.", source="b.md", ordinal=1, corpus="c")
    return [Retrieved(c1, 0.9), Retrieved(c2, 0.8)]


def test_refuses_without_calling_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(answer.rag.llm, "complete", lambda *a, **k: calls.append(1) or "x")
    res = answer.answer_question("q?", [])
    assert res.refused and res.text == answer.REFUSAL_MESSAGE and res.sources == []
    assert calls == []


def test_answers_with_sources(monkeypatch):
    seen = {}

    def fake(prompt, *, system=None, order=None):
        seen["prompt"], seen["system"] = prompt, system
        return "Answer [S1]"

    monkeypatch.setattr(answer.rag.llm, "complete", fake)
    r = _retrieved()
    res = answer.answer_question("What?", r)
    assert not res.refused and res.text == "Answer [S1]" and res.sources == r
    for s in ("[S1]", "[S2]", "Alpha text.", "Beta text.", "a.md", "What?"):
        assert s in seen["prompt"]
    assert "ONLY" in seen["system"]
