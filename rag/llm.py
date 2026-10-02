"""Provider-agnostic single-shot chat client with ordered fallback.

Each provider attempt is exactly one HTTP call with a fixed payload and no tool
use or looping: the answer step needs text, not an agent. Falling back across
providers (hosted free tiers first, local Ollama last) keeps the app answering
when one backend is rate-limited or down.
"""

from __future__ import annotations

import requests

import config

TIMEOUT_S = 60


class LLMError(Exception):
    """Raised when every provider in the fallback order failed or was skipped."""


def _messages(prompt: str, system: str | None) -> list[dict]:
    """Build the chat message list; the system turn is optional."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def _openai_style(url: str, key: str, model: str, prompt: str, system: str | None) -> str:
    """Call an OpenAI-compatible chat endpoint (Groq, OpenRouter) and return the text.

    Both hosted providers share one wire format, so one helper avoids duplication.
    """
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": _messages(prompt, system), "temperature": 0},
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _groq(prompt: str, system: str | None) -> str:
    """Call Groq and return the reply text (the missing-key skip is done by complete)."""
    return _openai_style(
        "https://api.groq.com/openai/v1/chat/completions",
        config.GROQ_API_KEY, config.GROQ_MODEL, prompt, system,
    )


def _openrouter(prompt: str, system: str | None) -> str:
    """Call OpenRouter and return the reply text."""
    return _openai_style(
        "https://openrouter.ai/api/v1/chat/completions",
        config.OPENROUTER_API_KEY, config.OPENROUTER_MODEL, prompt, system,
    )


def _ollama(prompt: str, system: str | None) -> str:
    """Call the local Ollama chat endpoint (non-streaming) and return the reply text."""
    resp = requests.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": config.OLLAMA_LLM_MODEL,
            "messages": _messages(prompt, system),
            "stream": False,
        },
        timeout=TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _provider_table() -> dict:
    """Map provider name -> (callable, key); read at call time so config changes apply.

    A falsy key means the provider is skipped; Ollama needs none, hence True.
    """
    return {
        "groq": (_groq, config.GROQ_API_KEY),
        "openrouter": (_openrouter, config.OPENROUTER_API_KEY),
        "ollama": (_ollama, True),
    }


def complete(prompt: str, *, system: str | None = None, order: list[str] | None = None) -> str:
    """Return the first successful completion across providers, else raise LLMError.

    Providers are tried in priority order (de-duplicated); those missing an API key
    are skipped rather than treated as errors, and the final error lists why each
    one did not produce text.
    """
    names = list(dict.fromkeys(order if order is not None else config.LLM_FALLBACK_ORDER))
    table = _provider_table()
    reasons: list[str] = []
    for name in names:
        if name not in table:
            reasons.append(f"{name}: unknown provider")
            continue
        fn, key = table[name]
        if not key:
            reasons.append(f"{name}: no API key")
            continue
        try:
            return fn(prompt, system)
        except Exception as exc:  # any failure -> try the next provider
            reasons.append(f"{name}: {exc}")
    raise LLMError("All LLM providers failed: " + "; ".join(reasons or ["none configured"]))
