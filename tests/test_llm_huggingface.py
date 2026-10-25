"""The Hugging Face provider in rag/llm.py: how it is called, skipped and cached.

All offline: `requests.post` is replaced by a stub. The fallback-order check runs
`config` in a fresh interpreter, because the order is computed at import.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests

import config
from rag import llm

ROOT = Path(__file__).resolve().parents[1]


class _Resp:
    """Just enough of `requests.Response` for `_openai_style`."""

    def __init__(self, status: int, text: str = ""):
        self.status_code, self.headers, self._text = status, {}, text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._text}}]}


def test_calls_the_router_with_the_token_and_model(monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE", False)
    monkeypatch.setattr(config, "HF_TOKEN", "hf_test")
    monkeypatch.setattr(config, "HF_MODEL", "org/model")
    calls: list[tuple] = []
    monkeypatch.setattr(llm.requests, "post",
                        lambda url, **k: calls.append((url, k)) or _Resp(200, "from hf"))

    assert llm.complete("q", order=["huggingface"]) == "from hf"
    url, kwargs = calls[0]
    assert url == "https://router.huggingface.co/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == "Bearer hf_test"
    assert kwargs["json"]["model"] == "org/model" and kwargs["json"]["temperature"] == 0
    assert llm.last_provider() == "huggingface"


def test_no_credits_falls_through_to_the_next_provider(monkeypatch):
    """The router answers 402 once the account's credits are spent."""
    monkeypatch.setattr(config, "LLM_CACHE", False)
    monkeypatch.setattr(config, "HF_TOKEN", "hf_test")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "or_test")
    monkeypatch.setattr(llm.requests, "post",
                        lambda url, **k: _Resp(402) if "huggingface" in url else _Resp(200, "from openrouter"))

    assert llm.complete("q", order=["huggingface", "openrouter"]) == "from openrouter"
    assert llm.last_provider() == "openrouter"


def test_no_token_is_skipped_without_a_call(monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE", False)
    monkeypatch.setattr(config, "HF_TOKEN", "")
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    with pytest.raises(llm.LLMError, match="huggingface: no API key"):
        llm.complete("q", order=["huggingface"])


def test_streaming_and_whole_replies_know_the_same_providers():
    assert set(llm._stream_table()) == set(llm._provider_table())


def test_cache_keys_without_huggingface_are_unchanged(monkeypatch, tmp_path):
    """Answers cached before the provider existed must still be found."""
    monkeypatch.setattr(config, "LLM_CACHE", True)
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path)
    names = ["groq", "openrouter", "ollama"]
    before = {"order": names,
              "models": [config.GROQ_MODEL, config.OPENROUTER_MODEL, config.OLLAMA_LLM_MODEL],
              "system": "sys", "prompt": "p"}
    digest = hashlib.sha256(json.dumps(before, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    assert llm._cache_path(names, "p", "sys") == tmp_path / f"{digest}.json"
    # With the provider in the order, its model is part of the key.
    with_hf = llm._cache_path(["huggingface", *names], "p", "sys")
    monkeypatch.setattr(config, "HF_MODEL", "another/model")
    assert llm._cache_path(["huggingface", *names], "p", "sys") != with_hf


def _fallback_order(**env) -> list[str]:
    """The chain as rag/llm.py walks it: the configured provider is listed twice, once first."""
    script = "import config, json; print(json.dumps(list(dict.fromkeys(config.LLM_FALLBACK_ORDER))))"
    full_env = {**os.environ, "LLM_PROVIDER": "groq", "HF_TOKEN": "", **env}
    out = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=full_env,
                         capture_output=True, text=True, check=True, timeout=60).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_the_chain_lists_huggingface_only_with_a_token():
    assert _fallback_order() == ["groq", "openrouter", "ollama"]
    assert _fallback_order(HF_TOKEN="hf_test") == ["groq", "openrouter", "huggingface", "ollama"]
    assert _fallback_order(LLM_PROVIDER="huggingface", HF_TOKEN="hf_test")[0] == "huggingface"
