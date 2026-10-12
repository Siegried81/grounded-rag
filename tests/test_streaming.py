"""Offline tests for answer streaming: the LLM client, the answer step, the API and the CLI.

Streaming must change when the reader sees the text, never what the text is.
So beyond the wire formats (OpenAI-style server-sent events, Ollama NDJSON),
these tests pin down the equivalences: a streamed reply is cached under the
same key as a whole one, the final answer is normalised on the complete text
(a citation split across two deltas still counts), and POST /api/ask/stream
ends with exactly the body POST /api/ask returns. All HTTP is faked.
"""

import json

import pytest

import cli
import config
import rag.answer as answer
import rag.llm as llm
from tests.test_api import _r, client, setup  # noqa: F401 - shared fixtures


class StreamResp:
    """Stand-in for a streamed requests.Response: a status and an iterable body."""

    def __init__(self, lines, status_error=None, fail_after=None):
        self._lines = lines
        self._status_error = status_error
        self._fail_after = fail_after

    def raise_for_status(self):
        if self._status_error:
            raise self._status_error

    def iter_lines(self, decode_unicode=False):
        for i, line in enumerate(self._lines):
            if self._fail_after is not None and i == self._fail_after:
                raise ConnectionError("connection reset mid-stream")
            yield line


def _sse_lines(*pieces):
    """OpenAI-style stream body: one data line per delta, blank separators, [DONE]."""
    lines = []
    for p in pieces:
        lines += [f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}", ""]
    return lines + ["data: [DONE]", ""]


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["g"])
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "o")


# --- rag.llm.stream -------------------------------------------------------------


def test_openai_style_stream_yields_deltas_with_the_same_payload_plus_stream(monkeypatch):
    seen = {}

    def post(url, **kw):
        seen["url"], seen["kw"] = url, kw
        return StreamResp(_sse_lines("Logs ", "are kept", " [S1]."))

    monkeypatch.setattr(llm.requests, "post", post)
    assert list(llm.stream("hi", system="sys", order=["groq"])) == ["Logs ", "are kept", " [S1]."]
    payload = seen["kw"]["json"]
    assert "groq.com" in seen["url"] and seen["kw"]["stream"] is True
    assert payload["stream"] is True and payload["temperature"] == 0
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert llm.last_provider() == "groq"


def test_ollama_stream_reads_ndjson_until_done(monkeypatch):
    lines = [json.dumps({"message": {"content": "a"}, "done": False}), "",
             json.dumps({"message": {"content": "b"}, "done": True}),
             json.dumps({"message": {"content": "never read"}})]
    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: StreamResp(lines))
    assert list(llm.stream("hi", order=["ollama"])) == ["a", "b"]


def test_stream_falls_back_while_nothing_was_yielded(monkeypatch):
    calls = []

    def post(url, **kw):
        calls.append(url)
        if "groq" in url:
            return StreamResp([], status_error=RuntimeError("429"))
        return StreamResp(_sse_lines("from openrouter"))

    monkeypatch.setattr(llm.requests, "post", post)
    assert "".join(llm.stream("hi", order=["groq", "openrouter"])) == "from openrouter"
    assert len(calls) == 2 and llm.last_provider() == "openrouter"


def test_stream_does_not_switch_model_after_text_was_shown(monkeypatch):
    calls = []

    def post(url, **kw):
        calls.append(url)
        return StreamResp(_sse_lines("half an ", "answer"), fail_after=2)

    monkeypatch.setattr(llm.requests, "post", post)
    got = []
    with pytest.raises(llm.LLMError, match="interrupted"):
        for d in llm.stream("hi", order=["groq", "openrouter"]):
            got.append(d)
    assert got == ["half an "] and len(calls) == 1


def test_stream_empty_reply_falls_through_and_all_failing_raises(monkeypatch):
    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: StreamResp(_sse_lines()))
    with pytest.raises(llm.LLMError, match="empty reply"):
        list(llm.stream("hi", order=["groq", "openrouter"]))
    monkeypatch.setattr(config, "GROQ_API_KEYS", [])
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    with pytest.raises(llm.LLMError, match="no API key"):
        list(llm.stream("hi", order=["groq", "openrouter"]))


def test_a_streamed_reply_is_the_cached_reply_of_complete(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LLM_CACHE", True)
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: StreamResp(_sse_lines("A ", "B")))
    assert "".join(llm.stream("q", system="s", order=["groq"])) == "A B"

    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: pytest.fail("must be a cache hit"))
    assert llm.complete("q", system="s", order=["groq"]) == "A B"
    assert list(llm.stream("q", system="s", order=["groq"])) == ["A B"]


# --- rag.answer.stream_answer ---------------------------------------------------


def test_stream_answer_normalises_a_citation_split_across_deltas(monkeypatch):
    monkeypatch.setattr(llm, "stream", lambda prompt, system=None: iter(["Logs are kept 【S", "1】."]))
    events = list(answer.stream_answer("q", [_r(1, 0.9)]))
    assert events[:2] == [("delta", "Logs are kept 【S"), ("delta", "1】.")]
    kind, final = events[-1]
    assert kind == "answer" and final.text == "Logs are kept [S1]." and not final.refused


def test_stream_answer_equals_answer_question(monkeypatch):
    text = "Logs are kept [**S1**, **S2**]."
    monkeypatch.setattr(llm, "complete", lambda prompt, system=None: text)
    monkeypatch.setattr(llm, "stream", lambda prompt, system=None: iter([text[:9], text[9:]]))
    retrieved = [_r(1, 0.9), _r(2, 0.8)]
    whole = answer.answer_question("q", retrieved, language="fr")
    streamed = list(answer.stream_answer("q", retrieved, language="fr"))[-1][1]
    assert streamed == whole


def test_stream_answer_refuses_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(llm, "stream", lambda *a, **k: pytest.fail("LLM must not run"))
    events = list(answer.stream_answer("q", [], language="fr"))
    assert len(events) == 1 and events[0][0] == "answer"
    assert events[0][1].refused and events[0][1].text == answer.REFUSAL_MESSAGES["fr"]


# --- POST /api/ask/stream -------------------------------------------------------


def _events(resp):
    """Parse a text/event-stream body into [(event, data)]."""
    out = []
    for block in resp.text.split("\n\n"):
        if not block.strip():
            continue
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def _without_latency(body):
    return {k: v for k, v in body.items() if k != "latency"}


def test_stream_endpoint_ends_with_the_ask_body(setup, client, monkeypatch):
    text = "High-risk systems must keep logs and documentation 【S1】."
    monkeypatch.setattr(llm, "complete", lambda prompt, system=None: text)
    monkeypatch.setattr(llm, "stream", lambda prompt, system=None: iter([text[:20], text[20:]]))
    req = {"corpus": "demo", "question": "What must high-risk systems keep?", "top_k": 2}

    resp = client.post("/api/ask/stream", json=req)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _events(resp)
    assert [e for e, _ in events] == ["sources", "delta", "delta", "done"]
    assert [s["sid"] for s in events[0][1]] == [1, 2] and not any(s["cited"] for s in events[0][1])
    assert "".join(d["text"] for e, d in events if e == "delta") == text

    done = events[-1][1]
    whole = client.post("/api/ask", json=req).json()
    assert _without_latency(done) == _without_latency(whole)
    assert done["answer"].endswith("[S1].") and done["verification"]["ok"] is True
    assert done["sources"][0]["cited"] is True


def test_stream_endpoint_refusal_is_a_single_done_event(setup, client, monkeypatch):
    setup["store"].results = [_r(1, 0.1)]
    monkeypatch.setattr(llm, "stream", lambda *a, **k: pytest.fail("LLM must not run"))
    events = _events(client.post("/api/ask/stream", json={"corpus": "demo", "question": "weather?"}))
    assert [e for e, _ in events] == ["done"]
    assert events[0][1]["refused"] is True and events[0][1]["refusal"]["best_score"] == 0.1


def test_stream_endpoint_llm_failure_mid_stream_keeps_the_passages(setup, client, monkeypatch):
    def broken(prompt, system=None):
        yield "partial "
        raise llm.LLMError("groq: stream interrupted: reset")

    monkeypatch.setattr(llm, "stream", broken)
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    events = _events(client.post("/api/ask/stream", json={"corpus": "demo", "question": "logs?"}))
    assert [e for e, _ in events] == ["sources", "delta", "done"]
    done = events[-1][1]
    assert done["error"]["kind"] == "llm" and done["answer"] is None
    assert len(done["sources"]) == 2


@pytest.mark.parametrize("payload, status", [
    ({"corpus": "demo", "question": "   "}, 422),
    ({"corpus": "nope", "question": "logs?"}, 404),
])
def test_stream_endpoint_rejects_bad_requests_before_streaming(setup, client, payload, status):
    assert client.post("/api/ask/stream", json=payload).status_code == status


def test_config_reports_whether_uis_stream(setup, client, monkeypatch):
    monkeypatch.setattr(config, "STREAM_ANSWERS", False)
    assert client.get("/api/config").json()["stream_answers"] is False
    monkeypatch.setattr(config, "STREAM_ANSWERS", True)
    assert client.get("/api/config").json()["stream_answers"] is True


# --- cli.py ask --stream --------------------------------------------------------


def test_cli_prints_deltas_then_the_normalised_answer(monkeypatch, capsys):
    monkeypatch.setattr(llm, "stream", lambda prompt, system=None: iter(["Kept ", "【S1】."]))
    final = cli._print_streamed("q", [_r(1, 0.9)])
    out = capsys.readouterr().out
    assert out.startswith("Kept 【S1】.\n")
    assert "With citations normalised:\nKept [S1]." in out
    assert final.text == "Kept [S1]."


def test_cli_stream_refusal_prints_the_refusal(monkeypatch, capsys):
    monkeypatch.setattr(llm, "stream", lambda *a, **k: pytest.fail("LLM must not run"))
    final = cli._print_streamed("q", [])
    assert capsys.readouterr().out.strip() == answer.REFUSAL_MESSAGE and final.refused
