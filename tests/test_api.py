"""Offline tests for the FastAPI layer (api/main.py) used by the React UI.

The index, embedder and LLM are replaced by in-file fakes: `load_resources` and
`indexed_corpora` are monkeypatched on the api module and `rag.llm.complete` is
stubbed, so no index files, network or API key are needed.
"""

import json

import pytest
from fastapi.testclient import TestClient

import config
import rag.llm
from api import main as api
from rag.answer import system_prompt
from rag.types import Chunk, Retrieved


class FakeEmbedder:
    """Returns a fixed vector; `fail=True` simulates an unreachable backend."""

    model = "fake"

    def __init__(self, fail=False):
        self.fail = fail

    def embed(self, texts):
        if self.fail:
            raise ConnectionError("embedding backend down")
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeStore:
    """Returns a preset ranked list for any query and exposes its chunks for stats."""

    def __init__(self, results):
        self.results = results
        self._chunks = [r.chunk for r in results]

    def search(self, query_vec, top_k):
        return self.results[:top_k]


def _r(i, score, text=None, page=None):
    meta = {"page": page} if page else {}
    text = text or f"The high-risk systems must keep logs and documentation number {i}."
    return Retrieved(Chunk(f"c{i}", text, f"doc{i}.txt", i, "demo", meta), score)


@pytest.fixture
def setup(monkeypatch, tmp_path):
    """Install one fake corpus "demo" and return a helper to change its results."""
    state = {"store": FakeStore([_r(1, 0.9, page=3), _r(2, 0.8)]), "embedder": FakeEmbedder()}

    monkeypatch.setattr(api, "indexed_corpora", lambda: ["demo"])
    monkeypatch.setattr(api, "load_resources", lambda c: (state["store"], None, state["embedder"]))
    api.load_stats.cache_clear()
    rows = [json.dumps({"question": f"Question {i}?"}) for i in range(8)]
    (tmp_path / "demo_eval.jsonl").write_text("\n".join(rows), encoding="utf-8")
    monkeypatch.setattr(api, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(config, "SCORE_THRESHOLD", 0.35)
    yield state
    api.load_stats.cache_clear()


@pytest.fixture
def client():
    return TestClient(api.app)


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200 and resp.json() == {"status": "ok"}


def test_config_has_badges_and_no_secrets(setup, client, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "GROQ_API_KEY", "sk-secret-groq")
    monkeypatch.setattr(config, "GROQ_API_KEYS", ["sk-secret-groq"])
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-secret-or")
    monkeypatch.setattr(config, "HOSTED_EMBED_API_KEY", "sk-secret-embed")
    resp = client.get("/api/config")
    assert resp.status_code == 200
    body = resp.json()
    assert "sk-secret" not in resp.text
    assert body["llm_key_configured"] is True
    assert body["llm_provider"] == "groq" and body["llm_model"] == config.GROQ_MODEL
    assert body["refusal_threshold"] == 0.35
    assert len(body["pipeline_steps"]) == 5
    # Fake corpus has no BM25 index, so the limitations must not mention it.
    assert body["limitations"] and "BM25" not in " ".join(body["limitations"])


def test_config_reports_missing_key(setup, client, monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(config, "GROQ_API_KEYS", [])
    assert client.get("/api/config").json()["llm_key_configured"] is False


def test_config_unknown_corpus_is_404(setup, client):
    assert client.get("/api/config", params={"corpus": "nope"}).status_code == 404


def test_corpora_shape(setup, client):
    body = client.get("/api/corpora").json()
    [c] = body["corpora"]
    assert c["name"] == "demo"
    assert c["documents"] == 2 and c["chunks"] == 2
    assert c["sources"] == ["doc1.txt", "doc2.txt"]
    assert c["languages"] == ["English"]
    assert c["has_bm25"] is False and c["modes"] == ["dense"]
    assert c["examples"] == ["Question 0?", "Question 2?", "Question 4?", "Question 6?"]


def test_ask_answer_path(setup, client, monkeypatch):
    monkeypatch.setattr(
        rag.llm, "complete",
        lambda prompt, system=None: "High-risk systems must keep logs and documentation [S1].",
    )
    resp = client.post("/api/ask", json={"corpus": "demo", "question": "What must high-risk systems keep?",
                                         "mode": "hybrid", "top_k": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is None and body["refused"] is False
    assert body["answer"].endswith("[S1].")
    assert body["settings"]["mode"] == "dense"  # no BM25 index -> dense, as app.py offers
    s1, s2 = body["sources"]
    assert s1["sid"] == 1 and s1["cited"] is True and s1["page"] == 3
    assert s1["location"] == "doc1.txt · p.3"
    assert s2["cited"] is False
    assert "<mark>" in s1["excerpt_html"]
    ver = body["verification"]
    assert ver["ok"] is True and ver["invalid_citations"] == []
    assert "passes" in ver["explanation"]
    assert body["latency"]["retrieval_s"] is not None and body["latency"]["answer_s"] is not None


def test_ask_flags_invalid_citation(setup, client, monkeypatch):
    monkeypatch.setattr(rag.llm, "complete", lambda prompt, system=None: "Logs are kept [S9].")
    body = client.post("/api/ask", json={"corpus": "demo", "question": "logs?"}).json()
    assert body["verification"]["ok"] is False
    assert body["verification"]["invalid_citations"] == [9]


def test_ask_refusal_path_does_not_call_llm(setup, client, monkeypatch):
    setup["store"] = FakeStore([_r(1, 0.2), _r(2, 0.1)])

    def boom(*a, **k):
        raise AssertionError("LLM must not be called on refusal")

    monkeypatch.setattr(rag.llm, "complete", boom)
    body = client.post("/api/ask", json={"corpus": "demo", "question": "Unrelated topic?"}).json()
    assert body["refused"] is True and body["answer"] is None and body["sources"] == []
    ref = body["refusal"]
    assert ref["best_score"] == 0.2 and ref["threshold"] == 0.35
    assert "0.20" in ref["explanation"]
    assert [c["sid"] for c in ref["closest"]] == [1, 2]
    assert body["latency"]["answer_s"] is None


def test_ask_llm_failure_returns_hint_not_500(setup, client, monkeypatch):
    def fail(prompt, system=None):
        raise rag.llm.LLMError("All LLM providers failed: groq: no API key; ollama: ConnectionError")

    monkeypatch.setattr(rag.llm, "complete", fail)
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    resp = client.post("/api/ask", json={"corpus": "demo", "question": "What must be kept?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["error"]["kind"] == "llm"
    assert "GROQ_API_KEY" in body["error"]["hint"]
    assert len(body["sources"]) == 2  # passages are still shown
    assert body["answer"] is None and body["verification"] is None


def test_ask_embedding_failure_returns_hint(setup, client, monkeypatch):
    setup["embedder"] = FakeEmbedder(fail=True)
    monkeypatch.setattr(config, "EMBED_PROVIDER", "ollama")
    resp = client.post("/api/ask", json={"corpus": "demo", "question": "anything"})
    assert resp.status_code == 200
    err = resp.json()["error"]
    assert err["kind"] == "embedding" and "ollama serve" in err["hint"]


@pytest.mark.parametrize("payload", [
    {"corpus": "demo", "question": ""},
    {"corpus": "demo", "question": "   "},
    {"corpus": "demo", "question": "ok?", "mode": "sparse"},
    {"corpus": "demo", "question": "ok?", "top_k": 0},
    {"corpus": "demo", "question": "ok?", "top_k": 11},
    {"question": "missing corpus"},
])
def test_ask_input_validation_is_422(setup, client, payload):
    assert client.post("/api/ask", json=payload).status_code == 422


def test_ask_unknown_corpus_is_404(setup, client):
    resp = client.post("/api/ask", json={"corpus": "../index", "question": "hello?"})
    assert resp.status_code == 404


def test_config_exposes_languages(setup, client, monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_ANSWER_LANGUAGE", "en")
    body = client.get("/api/config").json()
    assert body["languages"] == ["auto", "en", "fr", "nl"]
    assert body["default_language"] == "en"
    assert body["ui_languages"] == ["en", "fr"]
    assert body["pipeline_steps"][0].startswith("**Chunk**")


def test_config_ui_lang_french_and_invalid(setup, client):
    body = client.get("/api/config", params={"ui_lang": "fr"}).json()
    assert body["pipeline_steps"][0].startswith("**Découpage**")
    assert "corpus indexé" in body["limitations"][0]
    assert client.get("/api/config", params={"ui_lang": "de"}).status_code == 422


def test_corpora_examples_by_language(setup, client, monkeypatch):
    fr_q = "Quels sont les quatre niveaux de risque définis par l'AI Act ?"
    (api.EVAL_DIR / "demo_eval.jsonl").write_text(json.dumps({"question": fr_q}), encoding="utf-8")
    c = client.get("/api/corpora").json()["corpora"][0]
    assert c["examples_by_language"] == {
        "en": ["What are the four risk levels defined by the AI Act?"], "fr": [fr_q],
    }
    assert c["examples"] == c["examples_by_language"]["en"]
    c_fr = client.get("/api/corpora", params={"ui_lang": "fr"}).json()["corpora"][0]
    assert c_fr["examples"] == [fr_q]
    assert client.get("/api/corpora", params={"ui_lang": "nl"}).status_code == 422


@pytest.mark.parametrize("lang", ["auto", "en", "fr", "nl"])
def test_ask_language_is_passed_and_echoed(setup, client, monkeypatch, lang):
    systems = []

    def fake(prompt, system=None):
        systems.append(system)
        return "High-risk systems must keep logs and documentation [S1]."

    monkeypatch.setattr(rag.llm, "complete", fake)
    body = client.post("/api/ask", json={"corpus": "demo", "question": "logs?", "language": lang}).json()
    assert body["language"] == lang
    assert systems == [system_prompt(lang)]


def test_ask_language_defaults_to_config(setup, client, monkeypatch):
    monkeypatch.setattr(config, "DEFAULT_ANSWER_LANGUAGE", "fr")
    monkeypatch.setattr(rag.llm, "complete", lambda prompt, system=None: "Logs [S1].")
    body = client.post("/api/ask", json={"corpus": "demo", "question": "logs?"}).json()
    assert body["language"] == "fr"


@pytest.mark.parametrize("payload", [
    {"corpus": "demo", "question": "logs?", "language": "de"},
    {"corpus": "demo", "question": "logs?", "ui_lang": "nl"},
])
def test_ask_unknown_language_is_422(setup, client, payload):
    assert client.post("/api/ask", json=payload).status_code == 422


def test_ask_ui_lang_localizes_explanations(setup, client, monkeypatch):
    monkeypatch.setattr(rag.llm, "complete",
                        lambda prompt, system=None: "High-risk systems must keep logs and documentation [S1].")
    body = client.post("/api/ask", json={"corpus": "demo", "question": "logs?", "ui_lang": "fr"}).json()
    assert "contrôle lexical" in body["verification"]["explanation"]
    setup["store"] = FakeStore([_r(1, 0.1)])
    body = client.post("/api/ask", json={"corpus": "demo", "question": "weather?", "ui_lang": "fr"}).json()
    assert body["refused"] and "seuil" in body["refusal"]["explanation"]


# --- GET /api/document ----------------------------------------------------------
# The source cards link to the whole file a passage came from, because an excerpt
# alone cannot be checked. The filename comes back from the server inside an
# answer, so these tests pin down that it can only ever resolve to an indexed
# file of the named corpus.


@pytest.fixture
def corpus_files(setup, monkeypatch, tmp_path):
    """Give the fake "demo" corpus a real folder on disk and return it."""
    folder = tmp_path / "data" / "demo"
    folder.mkdir(parents=True)
    monkeypatch.setattr(config, "corpus_dir", lambda name: tmp_path / "data" / name)
    return folder


def test_document_returns_full_text_with_highlights(corpus_files, client):
    (corpus_files / "doc1.txt").write_text(
        "Logs must be kept.\nDocumentation too.", encoding="utf-8"
    )
    body = client.get(
        "/api/document", params={"corpus": "demo", "source": "doc1.txt", "q": "which logs?"}
    ).json()
    assert body["source"] == "doc1.txt" and body["corpus"] == "demo"
    assert body["chars"] == 37 and body["shown_chars"] == 37 and body["truncated"] is False
    # The query term is marked; the rest of the document is returned intact.
    assert "<mark>Logs</mark> must be kept." in body["text_html"]
    assert "Documentation too." in body["text_html"]


def test_document_without_question_has_no_marks(corpus_files, client):
    (corpus_files / "doc1.txt").write_text("Logs must be kept.", encoding="utf-8")
    body = client.get("/api/document", params={"corpus": "demo", "source": "doc1.txt"}).json()
    assert body["text_html"] == "Logs must be kept."


def test_document_escapes_markup(corpus_files, client):
    """The browser injects text_html as HTML, so the document must not be able to."""
    (corpus_files / "doc1.txt").write_text("<script>alert('x')</script> logs", encoding="utf-8")
    body = client.get(
        "/api/document", params={"corpus": "demo", "source": "doc1.txt", "q": "logs"}
    ).json()
    assert "<script>" not in body["text_html"]
    assert "&lt;script&gt;" in body["text_html"]


def test_document_truncates_long_file(corpus_files, client, monkeypatch):
    monkeypatch.setattr(api, "MAX_DOCUMENT_CHARS", 10)
    (corpus_files / "doc1.txt").write_text("x" * 50, encoding="utf-8")
    body = client.get("/api/document", params={"corpus": "demo", "source": "doc1.txt"}).json()
    assert body["chars"] == 50 and body["shown_chars"] == 10 and body["truncated"] is True
    assert body["text_html"] == "x" * 10


@pytest.mark.parametrize("source", [
    "nope.txt",
    "../secret.txt",          # path traversal out of the corpus
    "/etc/passwd",
    "README.md",              # present but never indexed
    "notes.csv",              # present but an unsupported suffix
    "raw/dump.txt",           # present but in a subfolder, so not indexed
])
def test_document_only_serves_indexed_files(corpus_files, client, source):
    (corpus_files / "doc1.txt").write_text("logs", encoding="utf-8")
    (corpus_files / "README.md").write_text("about", encoding="utf-8")
    (corpus_files / "notes.csv").write_text("x,y", encoding="utf-8")
    (corpus_files / "raw").mkdir()
    (corpus_files / "raw" / "dump.txt").write_text("raw", encoding="utf-8")
    (corpus_files.parent / "secret.txt").write_text("CLASSIFIED", encoding="utf-8")

    resp = client.get("/api/document", params={"corpus": "demo", "source": source})
    assert resp.status_code == 404
    # The 404 detail echoes the name the client asked for; no file content leaks.
    assert "CLASSIFIED" not in resp.text


def test_document_unknown_corpus_is_404(corpus_files, client):
    (corpus_files / "doc1.txt").write_text("logs", encoding="utf-8")
    resp = client.get("/api/document", params={"corpus": "nope", "source": "doc1.txt"})
    assert resp.status_code == 404


def test_document_requires_corpus_and_source(setup, client):
    assert client.get("/api/document", params={"corpus": "demo"}).status_code == 422
    assert client.get("/api/document", params={"source": "doc1.txt"}).status_code == 422


def test_document_links_bare_urls(corpus_files, client):
    """Both `text_html` and `excerpt_html` carry real links, so a passage naming
    its source as a URL can be followed from either UI."""
    (corpus_files / "doc1.txt").write_text(
        "Logs: see https://eur-lex.europa.eu/ai for the text.", encoding="utf-8"
    )
    body = client.get(
        "/api/document", params={"corpus": "demo", "source": "doc1.txt", "q": "logs"}
    ).json()
    assert '<a href="https://eur-lex.europa.eu/ai"' in body["text_html"]
    assert 'rel="noopener noreferrer"' in body["text_html"]
    assert "<mark>Logs</mark>" in body["text_html"]


def test_ask_sources_link_bare_urls(setup, client, monkeypatch):
    setup["store"] = FakeStore([_r(1, 0.9, text="Logs at https://europa.eu/ai are required.")])
    monkeypatch.setattr(rag.llm, "complete", lambda prompt, system=None: "Logs [S1].")
    body = client.post("/api/ask", json={"corpus": "demo", "question": "logs?"}).json()
    assert '<a href="https://europa.eu/ai"' in body["sources"][0]["excerpt_html"]
