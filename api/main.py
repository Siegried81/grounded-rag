"""JSON API over the grounded RAG pipeline, consumed by the React UI in web/.

It mirrors app.py (the Streamlit UI) call for call: the same `retrieve` ->
`answer_question` -> `verify_answer` sequence, the same per-corpus caching of the
store, BM25 index and embedder, and the same refusal and error explanations from
`rag/ui_helpers.py`. Nothing here decides what an answer, a refusal or a grounding
score means; it only serialises what the pipeline already produces, so both UIs
show identical behaviour.

Failures of the embedding backend or of every LLM provider are returned as a
normal 200 response with an `error` object and an actionable hint, not as a 500:
the retrieved passages are still useful to the reader, exactly as in app.py.

Run with `uvicorn api.main:app --port 8002`. When `web/dist` exists (after
`npm run build` in web/), the built UI is served from `/` by the same process.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import config
from rag import ui_helpers as ui
from rag.answer import answer_question
from rag.embed import get_embedder
from rag.ingest import document_paths, read_document
from rag.lexical import BM25Index
from rag.retrieve import retrieve
from rag.store import VectorStore
from rag.verify import extract_citations, verify_answer

log = logging.getLogger("api")

EVAL_DIR = config.ROOT / "eval"
WEB_DIST = config.ROOT / "web" / "dist"
# Interface languages for server-generated texts (examples, explanations,
# limitations, pipeline steps); see config.SUPPORTED_UI_LANGUAGES.
UiLang = Literal["en", "fr"]
AnswerLang = Literal["auto", "en", "fr", "nl"]
# Cap on what /api/document sends back. The shipped corpora top out around 70 kB,
# so this only bites if someone drops a very large file into a corpus folder.
MAX_DOCUMENT_CHARS = 400_000
# The Vite dev server; in production the UI is served by this same app, so no
# other origin needs to be allowed.
DEV_ORIGINS = ["http://localhost:5180", "http://127.0.0.1:5180"]

app = FastAPI(title="Grounded RAG API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# --- Resources ------------------------------------------------------------------


def indexed_corpora() -> list[str]:
    """Corpus folders under DATA_DIR that already have a built index (as app.py)."""
    if not config.DATA_DIR.exists():
        return []
    return sorted(
        p.name for p in config.DATA_DIR.iterdir()
        if p.is_dir() and config.index_path(p.name).exists()
    )


@lru_cache(maxsize=None)
def load_resources(corpus: str):
    """Load the store, BM25 index and embedder once per corpus (expensive, reusable).

    Same contract as app.py's `load_resources`: a corpus indexed before the
    keyword channel existed has no BM25 file and runs dense-only.
    """
    try:
        bm25 = BM25Index.load(config.bm25_path(corpus))
    except FileNotFoundError:
        bm25 = None
    return VectorStore.load(config.index_path(corpus)), bm25, get_embedder()


@lru_cache(maxsize=None)
def load_stats(corpus: str) -> dict:
    """Document/chunk/language counts for a corpus, computed once.

    VectorStore exposes no public chunk accessor, so this reads its chunk list
    directly, as app.py does; it is read-only and display-only.
    """
    store, _, _ = load_resources(corpus)
    return ui.corpus_stats(store._chunks)


def llm_model_name() -> str:
    """Model name for the configured generation provider (never a key)."""
    return {
        "groq": config.GROQ_MODEL,
        "openrouter": config.OPENROUTER_MODEL,
        "ollama": config.OLLAMA_LLM_MODEL,
    }.get(config.LLM_PROVIDER, "unknown")


def llm_key_configured() -> bool:
    """False when the primary hosted provider has no key (Ollama needs none).

    Only this boolean leaves the server; key values are never serialised.
    """
    return not (
        (config.LLM_PROVIDER == "groq" and not config.GROQ_API_KEYS)
        or (config.LLM_PROVIDER == "openrouter" and not config.OPENROUTER_API_KEY)
    )


def _require_corpus(corpus: str) -> None:
    """Raise 404 unless `corpus` is an indexed corpus (also blocks path tricks)."""
    if corpus not in indexed_corpora():
        raise HTTPException(status_code=404, detail=f"Unknown or unindexed corpus: {corpus!r}")


# --- Serialisation ----------------------------------------------------------------


def _source_dict(index: int, r, terms: list[str], cited: list[int]) -> dict:
    """One passage as JSON: label, score, plain + highlighted excerpt, cited flag.

    `excerpt_html` comes from `ui.highlight_and_linkify`, which escapes the passage
    before adding <mark> tags and http(s) links, so it is safe to inject as HTML in
    the browser.
    """
    excerpt = ui.snippet(r.chunk.text, terms)
    return {
        "sid": index,
        "source": r.chunk.source,
        "page": (r.chunk.meta or {}).get("page"),
        "location": ui.format_location(r.chunk.source, r.chunk.meta),
        "score": round(float(r.score), 4),
        "excerpt": excerpt,
        "excerpt_html": ui.highlight_and_linkify(excerpt, terms),
        "cited": index in cited,
    }


def _sources(results, question: str, cited: list[int]) -> list[dict]:
    """Serialise a ranked list of passages, numbered S1..Sn like the prompt."""
    terms = ui.query_terms(question)
    return [_source_dict(i, r, terms, cited) for i, r in enumerate(results, 1)]


def _resolve_document(corpus: str, source: str):
    """Map a citation's filename back to the document on disk, or raise 404.

    The name is compared against the corpus listing instead of being joined onto
    a path, so a crafted `source` ("../.env", an absolute path, a file in a
    sibling corpus) resolves to nothing rather than to a file outside the corpus.
    Matching `ingest.document_paths` also means only indexed files can be opened.
    """
    _require_corpus(corpus)
    for path in document_paths(corpus):
        if path.name == source:
            return path
    raise HTTPException(status_code=404, detail=f"Unknown source: {source!r}")


# --- Routes -----------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict:
    """Liveness check; touches no index or provider so it never fails on config."""
    return {"status": "ok"}


@app.get("/api/config")
def get_config(
    corpus: str | None = Query(default=None),
    ui_lang: UiLang = Query(default="en"),
) -> dict:
    """Badges, pipeline steps, limitations and language options, from the live settings.

    Limitations mention BM25 only when the corpus has a keyword index, so the
    optional `corpus` parameter makes that list match the selected corpus;
    without it, BM25 is assumed available when any corpus has it. `ui_lang`
    picks the language of the pipeline steps and limitations (English default).
    """
    corpora = indexed_corpora()
    if corpus is not None:
        _require_corpus(corpus)
        has_bm25 = load_resources(corpus)[1] is not None
    else:
        has_bm25 = any(load_resources(c)[1] is not None for c in corpora)
    return {
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": llm_model_name(),
        "embed_provider": config.EMBED_PROVIDER,
        "retrieval_mode": config.RETRIEVAL_MODE,
        "top_k": min(config.TOP_K, 10),
        "use_mmr": config.USE_MMR,
        "refusal_threshold": config.SCORE_THRESHOLD,
        "min_grounding": config.VERIFY_MIN_GROUNDING,
        "llm_key_configured": llm_key_configured(),
        "pipeline_steps": ui.pipeline_steps(ui_lang),
        "limitations": ui.build_limitations(
            config.SCORE_THRESHOLD, config.VERIFY_MIN_GROUNDING, has_bm25, lang=ui_lang
        ),
        "languages": list(config.SUPPORTED_ANSWER_LANGUAGES),
        "default_language": config.DEFAULT_ANSWER_LANGUAGE,
        "ui_languages": list(config.SUPPORTED_UI_LANGUAGES),
    }


@app.get("/api/corpora")
def get_corpora(ui_lang: UiLang = Query(default="en")) -> dict:
    """Every indexed corpus with its stats, available modes and example questions.

    Loading the index needs no network (the embedder is only built, not called),
    so this works without any LLM key or running Ollama. `examples_by_language`
    carries the examples in every UI language; `examples` is the `ui_lang` entry
    (English by default), so a client that ignores languages keeps working.
    """
    items = []
    for name in indexed_corpora():
        _, bm25, _ = load_resources(name)
        stats = load_stats(name)
        eval_path = EVAL_DIR / f"{name}_eval.jsonl"
        by_lang = {
            lang: ui.localized_examples(eval_path, lang) for lang in config.SUPPORTED_UI_LANGUAGES
        }
        items.append({
            "name": name,
            "documents": stats["documents"],
            "chunks": stats["chunks"],
            "pages": stats["pages"],
            "languages": stats["languages"],
            "sources": stats["sources"],
            "has_bm25": bm25 is not None,
            "modes": ["hybrid", "dense"] if bm25 is not None else ["dense"],
            "examples": by_lang[ui_lang],
            "examples_by_language": by_lang,
        })
    return {"corpora": items}


@app.get("/api/document")
def get_document(
    corpus: str = Query(min_length=1),
    source: str = Query(min_length=1),
    q: str = Query(default="", max_length=2000),
) -> dict:
    """Return one cited document in full, with the question's terms highlighted.

    A source card only shows a window around the match (`ui.snippet`), which is
    not enough to check a citation: this is what lets the reader open the file the
    passage came from and read around it. `q` is the question, so the same words
    are marked here as on the card and the UI can scroll to the first match; it
    is optional because a document can be opened without one.

    `text_html` is escaped by `ui.highlight_and_linkify` before any <mark> or link
    is added, so
    a document can never inject markup into the page. Long documents are cut at
    `MAX_DOCUMENT_CHARS` and say so, rather than sending megabytes to the browser.
    """
    path = _resolve_document(corpus, source)
    text = read_document(path)
    shown = text[:MAX_DOCUMENT_CHARS]
    return {
        "corpus": corpus,
        "source": source,
        "chars": len(text),
        "shown_chars": len(shown),
        "truncated": len(shown) < len(text),
        "text_html": ui.highlight_and_linkify(shown, ui.query_terms(q)),
    }


class AskRequest(BaseModel):
    """Body of POST /api/ask. Only the knobs `retrieve()` takes are exposed.

    The refusal threshold is deliberately not a parameter: changing it changes
    what a refusal means, so it stays a reviewed config value (as in app.py).
    `language` only sets the language the answer is written in (None means
    config.DEFAULT_ANSWER_LANGUAGE); `ui_lang` sets the language of the refusal
    and grounding explanations generated here.
    """

    corpus: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["hybrid", "dense"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=10)
    use_mmr: bool | None = None
    language: AnswerLang | None = None
    ui_lang: UiLang = "en"

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        """Reject whitespace-only questions; there is nothing to retrieve on."""
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    """Retrieve, answer and verify one question, mirroring app.py's `run_question`.

    Retrieval and generation are timed separately because they fail and slow
    down for different reasons. On refusal the closest passages come from a plain
    dense search so the user sees what was found and how far below the threshold
    it was. Hybrid falls back to dense when the corpus has no BM25 index, which is
    the only mode app.py would offer for it.
    """
    _require_corpus(req.corpus)
    store, bm25, embedder = load_resources(req.corpus)
    mode = req.mode or config.RETRIEVAL_MODE
    if bm25 is None:
        mode = "dense"
    top_k = req.top_k or min(config.TOP_K, 10)
    use_mmr = config.USE_MMR if req.use_mmr is None else req.use_mmr
    question = req.question
    language = req.language or config.DEFAULT_ANSWER_LANGUAGE

    out = {
        "question": question,
        "language": language,
        "settings": {"mode": mode, "top_k": top_k, "use_mmr": use_mmr},
        "answer": None,
        "refused": False,
        "refusal": None,
        "sources": [],
        "verification": None,
        "latency": {"retrieval_s": None, "answer_s": None},
        "error": None,
    }

    t0 = time.perf_counter()
    try:
        retrieved = retrieve(
            question, store, embedder, top_k=top_k, bm25=bm25, mode=mode, use_mmr=use_mmr,
        )
    except Exception:  # embedding backend down or misconfigured
        log.exception("retrieval failed")
        out["latency"]["retrieval_s"] = round(time.perf_counter() - t0, 3)
        out["error"] = {
            "kind": "embedding",
            "message": "The question could not be embedded, so no passages were retrieved.",
            "hint": ui.embed_error_hint(config.EMBED_PROVIDER),
        }
        return out
    out["latency"]["retrieval_s"] = round(time.perf_counter() - t0, 3)

    if not retrieved:
        try:
            closest = store.search(embedder.embed([question])[0], top_k)
        except Exception:
            closest = []
        best = closest[0].score if closest else None
        out["refused"] = True
        out["refusal"] = {
            "best_score": None if best is None else round(float(best), 4),
            "threshold": config.SCORE_THRESHOLD,
            "explanation": ui.explain_refusal(best, config.SCORE_THRESHOLD, lang=req.ui_lang),
            "closest": _sources(closest, question, []),
        }
        return out

    t1 = time.perf_counter()
    try:
        answer = answer_question(question, retrieved, language=language)
        report = verify_answer(answer.text, retrieved, min_grounding=config.VERIFY_MIN_GROUNDING)
    except Exception as exc:  # LLMError or any provider failure: still show the passages
        log.warning("answer step failed: %s", exc)
        out["latency"]["answer_s"] = round(time.perf_counter() - t1, 3)
        out["sources"] = _sources(retrieved, question, [])
        out["error"] = {
            "kind": "llm",
            "message": "No language model could be reached.",
            "hint": ui.llm_error_hint(str(exc), config.LLM_PROVIDER),
        }
        return out
    out["latency"]["answer_s"] = round(time.perf_counter() - t1, 3)

    out["answer"] = answer.text
    out["sources"] = _sources(retrieved, question, extract_citations(answer.text))
    out["verification"] = {
        "ok": report.ok,
        "grounding_score": round(report.grounding_score, 4),
        "min_grounding": config.VERIFY_MIN_GROUNDING,
        "valid_citations": report.valid_citations,
        "invalid_citations": report.invalid_citations,
        "uncited_sentences": report.uncited_sentences,
        "explanation": ui.explain_grounding(
            report.grounding_score, config.VERIFY_MIN_GROUNDING, report.ok, lang=req.ui_lang
        ),
    }
    return out


# Single-process production mode: serve the built React app from "/". Mounted
# last so every /api route above takes precedence over the static files.
if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
