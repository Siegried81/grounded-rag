"""Free-tier protections in rag/llm.py: the Groq rate-limit wait and the response cache.

All offline: `requests.post` is replaced by a stub and `time.sleep` by a recorder.
"""

import json

import pytest
import requests

import config
from rag import llm


class _Resp:
    """Just enough of `requests.Response` for `_openai_style`."""

    def __init__(self, status: int, text: str = "", headers: dict | None = None):
        self.status_code, self.headers = status, headers or {}
        self._text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._text}}]}


@pytest.mark.parametrize("raw,seconds", [("7.66s", 7.66), ("1m2.5s", 62.5), ("250ms", 0.25),
                                         ("3", 3.0), ("", None), ("soon", None)])
def test_reset_header_parsing(raw, seconds):
    assert llm._seconds(raw) == (pytest.approx(seconds) if seconds is not None else None)


def test_all_keys_limited_waits_for_the_window_then_retries(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["k1", "k2"])
    monkeypatch.setattr(config, "GROQ_MAX_WAIT_S", 65.0)
    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    replies = iter([
        _Resp(429, headers={"x-ratelimit-reset-tokens": "12s"}),
        _Resp(429, headers={"x-ratelimit-reset-tokens": "8s"}),
        _Resp(200, "answer"),
    ])
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: next(replies))

    assert llm.complete("q", order=["groq"]) == "answer"
    assert slept == [pytest.approx(8.5)]  # the shortest announced window, plus a margin


def test_the_wait_is_capped_and_can_be_disabled(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["k1"])
    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    limited = _Resp(429, headers={"retry-after": "600"})

    monkeypatch.setattr(config, "GROQ_MAX_WAIT_S", 30.0)
    replies = iter([limited, _Resp(200, "ok")])
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: next(replies))
    assert llm.complete("q", order=["groq"]) == "ok"
    assert slept == [30.0]

    monkeypatch.setattr(config, "GROQ_MAX_WAIT_S", 0.0)
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: limited)
    with pytest.raises(llm.LLMError):
        llm.complete("q", order=["groq"])
    assert slept == [30.0]  # no second wait when waiting is off


def test_a_non_rate_limit_failure_does_not_wait(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["k1"])
    monkeypatch.setattr(config, "GROQ_MAX_WAIT_S", 65.0)
    slept: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Resp(500))
    with pytest.raises(llm.LLMError):
        llm.complete("q", order=["groq"])
    assert slept == []


def test_identical_requests_are_served_from_the_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LLM_CACHE", True)
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["k1"])
    calls: list[dict] = []
    monkeypatch.setattr(llm.requests, "post",
                        lambda url, **k: calls.append(k["json"]) or _Resp(200, "cached answer"))

    first = llm.complete("same prompt", system="sys", order=["groq"])
    second = llm.complete("same prompt", system="sys", order=["groq"])
    llm.complete("other prompt", system="sys", order=["groq"])

    assert first == second == "cached answer"
    assert len(calls) == 2  # the repeat never reached the network


def test_changing_the_model_misses_the_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LLM_CACHE", True)
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["k1"])
    calls: list[str] = []
    monkeypatch.setattr(llm.requests, "post",
                        lambda url, **k: calls.append(k["json"]["model"]) or _Resp(200, "x"))

    llm.complete("p", order=["groq"])
    monkeypatch.setattr(config, "GROQ_MODEL", "another-model")
    llm.complete("p", order=["groq"])
    assert len(calls) == 2


def test_an_empty_reply_falls_through_to_the_next_provider_and_is_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LLM_CACHE", True)
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["k1"])
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "or-key")
    urls: list[str] = []

    def post(url, **k):
        urls.append(url)
        return _Resp(200, "") if "groq" in url else _Resp(200, "from openrouter")

    monkeypatch.setattr(llm.requests, "post", post)
    assert llm.complete("p", order=["groq", "openrouter"]) == "from openrouter"
    assert len(urls) == 2 and "groq" in urls[0] and "openrouter" in urls[1]
    assert llm.last_provider() == "openrouter"
    # The cache holds the real reply and who gave it, never the empty one.
    [entry] = list((tmp_path / "cache").iterdir())
    assert json.loads(entry.read_text(encoding="utf-8")) == {
        "text": "from openrouter", "provider": "openrouter",
    }
    # A cache hit reports the provider that produced the stored reply.
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Resp(200, "fresh"))
    assert llm.complete("p", order=["groq", "openrouter"]) == "from openrouter"
    assert llm.last_provider() == "openrouter"


def test_an_empty_reply_from_every_provider_is_an_error(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["k1"])
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Resp(200, "   "))
    with pytest.raises(llm.LLMError, match="groq: empty reply"):
        llm.complete("p", order=["groq"])


def test_groq_with_no_keys_is_skipped_not_crashed(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEYS", [])
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: pytest.fail("no HTTP expected"))
    with pytest.raises(llm.LLMError, match="groq: no API key"):
        llm.complete("p", order=["groq"])
    with pytest.raises(RuntimeError, match="no Groq API keys"):
        llm._groq("p", None)  # called directly: a clear error, not an UnboundLocalError


def test_a_failed_request_is_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LLM_CACHE", True)
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["k1"])
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Resp(500))
    with pytest.raises(llm.LLMError):
        llm.complete("p", order=["groq"])
    assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").iterdir())
