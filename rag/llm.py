"""Provider-agnostic single-shot chat client with ordered fallback.

Each provider attempt is exactly one HTTP call with a fixed payload and no tool
use or looping: the answer step needs text, not an agent. Falling back across
providers (hosted free tiers first, local Ollama last) keeps the app answering
when one backend is rate-limited or down.

Two protections for free tiers: when every Groq key is rate-limited, the client
waits for the window Groq announces (``config.GROQ_MAX_WAIT_S``) and retries
once before falling back; and successful completions are cached on disk by an
exact hash of the request (``config.LLM_CACHE``), so a repeated question or a
re-run evaluation costs no quota.

`stream` is the same request delivered token by token, for the UIs: same
payload apart from the provider's stream flag, same fallback order, same cache
entry. It only changes when the reader sees the text, never what the text is.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections.abc import Iterator

import requests

import config

TIMEOUT_S = 60


class LLMError(Exception):
    """Raised when every provider in the fallback order failed or was skipped."""


# Which provider produced the last completion on this thread. The fallback chain
# can silently move from Groq to OpenRouter or Ollama, so evaluation rows record
# it: a result is only "from gpt-oss-120b" if it says so.
_last = threading.local()


def last_provider() -> str | None:
    """Provider name of this thread's last successful `complete` call, or None."""
    return getattr(_last, "provider", None)


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


def _openai_style_stream(url: str, key: str, model: str, prompt: str,
                         system: str | None) -> Iterator[str]:
    """Open a streamed OpenAI-compatible completion and return an iterator of text deltas.

    The request is sent and its status checked HERE, before the iterator is
    returned, so a 429 or a bad key raises at the same point as in
    `_openai_style` and key rotation and provider fallback still apply. The
    body is server-sent events: one `data: {json}` line per delta, ending with
    `data: [DONE]`.
    """
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": _messages(prompt, system), "temperature": 0,
              "stream": True},
        timeout=TIMEOUT_S,
        stream=True,
    )
    resp.raise_for_status()

    def deltas() -> Iterator[str]:
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                return
            choices = json.loads(data).get("choices") or [{}]
            text = (choices[0].get("delta") or {}).get("content")
            if text:
                yield text

    return deltas()


_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def _seconds(value: str | None) -> float | None:
    """Parse a Groq reset header ("7.66s", "1m2.5s", "250ms") or plain seconds."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    parts = _DURATION_PART.findall(value)
    if not parts:
        return None
    scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(n) * scale[unit] for n, unit in parts)


def _rate_limit_wait(exc: Exception) -> float | None:
    """Seconds a 429 error says to wait (retry-after, else the reset headers)."""
    resp = getattr(exc, "response", None)
    if resp is None or getattr(resp, "status_code", None) != 429:
        return None
    headers = getattr(resp, "headers", None) or {}
    waits = [
        _seconds(headers.get(h))
        for h in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests")
    ]
    waits = [w for w in waits if w is not None and w > 0]
    return min(waits) if waits else None


def _groq(prompt: str, system: str | None, call=_openai_style):
    """Call Groq, rotating through the configured keys until one succeeds.

    `call` is `_openai_style` for a whole reply or `_openai_style_stream` for a
    streamed one; both raise on a bad status before returning, so rotation and
    the rate-limit wait below are shared by the two.

    The free tier is rate-limited per key, so when a key fails (typically a 429)
    we try the next one. When every key was rate-limited and Groq announced when
    its window resets, we wait that long (capped by ``config.GROQ_MAX_WAIT_S``)
    and go round the keys once more; only then do we raise, letting `complete`
    fall back to the next provider. The empty-keys case is handled by
    `complete`, which skips Groq when no key is set.
    """
    last_exc: Exception | None = None
    for round_ in range(2):
        waits: list[float] = []
        for key in config.GROQ_API_KEYS:
            try:
                return call(
                    "https://api.groq.com/openai/v1/chat/completions",
                    key, config.GROQ_MODEL, prompt, system,
                )
            except Exception as exc:  # rate-limited/failed key -> try the next one
                last_exc = exc
                wait = _rate_limit_wait(exc)
                if wait is not None:
                    waits.append(wait)
        all_limited = bool(waits) and len(waits) == len(config.GROQ_API_KEYS)
        if round_ or not all_limited or config.GROQ_MAX_WAIT_S <= 0:
            break
        time.sleep(min(min(waits) + 0.5, config.GROQ_MAX_WAIT_S))
    raise last_exc if last_exc else RuntimeError("no Groq API keys configured")


def _openrouter(prompt: str, system: str | None, call=_openai_style):
    """Call OpenRouter and return the reply text (or, with `call` streaming, its deltas)."""
    return call(
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


def _ollama_stream(prompt: str, system: str | None) -> Iterator[str]:
    """Open a streamed Ollama chat and return an iterator of text deltas.

    Ollama streams newline-delimited JSON objects, each carrying a piece of
    `message.content`, the last one with `done: true`. As in
    `_openai_style_stream`, the status is checked before the iterator is
    returned, so a daemon that is down fails over like the non-streamed call.
    """
    resp = requests.post(
        f"{config.OLLAMA_URL}/api/chat",
        json={
            "model": config.OLLAMA_LLM_MODEL,
            "messages": _messages(prompt, system),
            "stream": True,
        },
        timeout=TIMEOUT_S,
        stream=True,
    )
    resp.raise_for_status()

    def deltas() -> Iterator[str]:
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            event = json.loads(line)
            text = (event.get("message") or {}).get("content")
            if text:
                yield text
            if event.get("done"):
                return

    return deltas()


def _provider_table() -> dict:
    """Map provider name -> (callable, key); read at call time so config changes apply.

    A falsy key means the provider is skipped; Ollama needs none, hence True.
    """
    return {
        "groq": (_groq, config.GROQ_API_KEYS),
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
    cache_path = _cache_path(names, prompt, system)
    cached = _cache_get(cache_path)
    if cached is not None:
        _last.provider = cached[1]
        return cached[0]
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
            text = fn(prompt, system)
        except Exception as exc:  # any failure -> try the next provider
            reasons.append(f"{name}: {exc}")
            continue
        if not isinstance(text, str) or not text.strip():
            # gpt-oss sometimes returns null or "" content. That is a failed
            # attempt, not an answer: try the next provider and never cache it.
            reasons.append(f"{name}: empty reply")
            continue
        _cache_put(cache_path, text, name)
        _last.provider = name
        return text
    raise LLMError("All LLM providers failed: " + "; ".join(reasons or ["none configured"]))


def _stream_table() -> dict:
    """Map provider name -> function opening a streamed reply; same keys as `_provider_table`."""
    return {
        "groq": lambda p, s: _groq(p, s, call=_openai_style_stream),
        "openrouter": lambda p, s: _openrouter(p, s, call=_openai_style_stream),
        "ollama": _ollama_stream,
    }


def stream(prompt: str, *, system: str | None = None,
           order: list[str] | None = None) -> Iterator[str]:
    """Yield the completion of `complete(prompt, ...)` as text deltas, as they arrive.

    Same request, provider order, key skipping and cache as `complete`: a cached
    reply is yielded whole, and a streamed reply is cached under the same key,
    so streaming and non-streaming answers to one question are interchangeable.

    Fallback only happens while nothing has been yielded: once a provider has
    sent text, the reader has seen it, so a failure after that point raises
    LLMError instead of silently continuing with another model's answer. An
    empty reply falls through to the next provider, as in `complete`.
    """
    names = list(dict.fromkeys(order if order is not None else config.LLM_FALLBACK_ORDER))
    cache_path = _cache_path(names, prompt, system)
    cached = _cache_get(cache_path)
    if cached is not None:
        _last.provider = cached[1]
        yield cached[0]
        return
    keys = {name: key for name, (_, key) in _provider_table().items()}
    openers = _stream_table()
    reasons: list[str] = []
    for name in names:
        if name not in openers:
            reasons.append(f"{name}: unknown provider")
            continue
        if not keys[name]:
            reasons.append(f"{name}: no API key")
            continue
        parts: list[str] = []
        try:
            for delta in openers[name](prompt, system):
                parts.append(delta)
                yield delta
        except Exception as exc:  # before any text: next provider; after: give up
            if any(p.strip() for p in parts):
                raise LLMError(f"{name}: stream interrupted: {exc}") from exc
            reasons.append(f"{name}: {exc}")
            continue
        text = "".join(parts)
        if not text.strip():
            reasons.append(f"{name}: empty reply")
            continue
        _cache_put(cache_path, text, name)
        _last.provider = name
        return
    raise LLMError("All LLM providers failed: " + "; ".join(reasons or ["none configured"]))


def _cache_path(names: list[str], prompt: str, system: str | None):
    """Cache file for this exact request, or None when caching is off.

    The key covers the provider order and every model in it, so changing the
    configured model or provider never serves an answer from another one.
    """
    if not config.LLM_CACHE:
        return None
    request = {
        "order": names,
        "models": [config.GROQ_MODEL, config.OPENROUTER_MODEL, config.OLLAMA_LLM_MODEL],
        "system": system,
        "prompt": prompt,
    }
    digest = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode())
    return config.LLM_CACHE_DIR / f"{digest.hexdigest()}.json"


def _cache_get(path) -> tuple[str, str | None] | None:
    """A stored (completion, provider) for this request; a missing, broken or empty entry is a miss.

    Entries written before the provider was recorded have none.
    """
    if path is None or not path.exists():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
        text = entry["text"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    return text, entry.get("provider")


def _cache_put(path, text: str, provider: str) -> None:
    """Store a completion and the provider that produced it; a write failure never fails the answer."""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text, "provider": provider}, ensure_ascii=False),
                        encoding="utf-8")
    except OSError:
        pass
