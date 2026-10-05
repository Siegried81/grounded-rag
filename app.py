"""Streamlit chat UI over the grounded RAG pipeline.

Lets a user pick an indexed corpus, ask questions, and inspect exactly which
passages (and scores) each answer rested on, which is what makes the cite-or-refuse
behaviour verifiable rather than a claim. Sources, the verification report and the
reason for any refusal are always on screen, never hidden behind a click.

Layout only: every decision about what to show (highlighting, labels, example
questions, refusal and error explanations) lives in `rag/ui_helpers.py`, where it
is unit-tested without a Streamlit runtime.
"""

from __future__ import annotations

import html
import time

import streamlit as st

import config
from rag import ui_helpers as ui
from rag.answer import answer_question
from rag.embed import get_embedder
from rag.lexical import BM25Index
from rag.retrieve import retrieve
from rag.store import VectorStore
from rag.verify import extract_citations, verify_answer

EVAL_DIR = config.ROOT / "eval"
# Labels for the answer-language picker, in config.SUPPORTED_ANSWER_LANGUAGES order.
LANGUAGE_LABELS = {"auto": "Auto (follow the question)", "en": "English",
                   "fr": "Français", "nl": "Nederlands"}

_CSS = """
<style>
.badges span { display:inline-block; padding:2px 10px; margin:0 6px 6px 0; border-radius:999px;
  font-size:0.78rem; border:1px solid rgba(128,128,128,0.35); background:rgba(128,128,128,0.10); }
.badges span.warn { border-color:rgba(230,140,0,0.6); background:rgba(230,140,0,0.12); }
span.cite { display:inline-block; padding:0 6px; margin:0 1px; border-radius:6px; font-size:0.75rem;
  font-weight:600; background:rgba(255,98,0,0.15); color:inherit; border:1px solid rgba(255,98,0,0.45); }
span.cite-bad { display:inline-block; padding:0 6px; border-radius:6px; font-size:0.75rem;
  font-weight:600; background:rgba(220,0,0,0.15); border:1px solid rgba(220,0,0,0.6);
  text-decoration:line-through; }
.src { border-left:3px solid rgba(128,128,128,0.35); padding:4px 0 4px 10px; margin:6px 0 10px 0; }
.src.cited { border-left-color:rgba(255,98,0,0.8); }
.src .head { font-size:0.85rem; font-weight:600; }
.src .meta { font-size:0.75rem; opacity:0.7; }
.src .body { font-size:0.85rem; opacity:0.9; }
.src mark { background:rgba(255,200,0,0.35); color:inherit; padding:0 1px; border-radius:2px; }
.refusal { border:1px solid rgba(128,128,128,0.45); border-left:4px solid rgba(120,120,120,0.9);
  border-radius:6px; padding:10px 14px; background:rgba(128,128,128,0.08); }
</style>
"""


def indexed_corpora() -> list[str]:
    """Corpus folders under DATA_DIR that already have a built index."""
    if not config.DATA_DIR.exists():
        return []
    return sorted(
        p.name for p in config.DATA_DIR.iterdir()
        if p.is_dir() and config.index_path(p.name).exists()
    )


@st.cache_resource
def load_resources(corpus: str):
    """Load the store, BM25 index and embedder once per corpus (expensive, reusable)."""
    try:
        bm25 = BM25Index.load(config.bm25_path(corpus))
    except FileNotFoundError:
        bm25 = None  # older index without a keyword channel -> dense-only
    return VectorStore.load(config.index_path(corpus)), bm25, get_embedder()


@st.cache_data
def load_stats(corpus: str) -> dict:
    """Document/chunk/language counts for the sidebar, computed once per corpus.

    VectorStore exposes no public chunk accessor, so this reads its chunk list
    directly; it is read-only and display-only.
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


def llm_key_missing() -> bool:
    """True when the primary hosted provider has no key (Ollama needs none)."""
    return (config.LLM_PROVIDER == "groq" and not config.GROQ_API_KEYS) or (
        config.LLM_PROVIDER == "openrouter" and not config.OPENROUTER_API_KEY
    )


def render_header(corpus: str, mode: str) -> None:
    """Title, the one rule, and badges for corpus / retrieval / model providers."""
    st.title("Grounded RAG assistant")
    st.markdown(
        "Answers **only from your documents**, cites every claim as `[S#]`, "
        "and **refuses** when nothing relevant is found."
    )
    key_badge = '<span class="warn">no LLM key set</span>' if llm_key_missing() else ""
    st.markdown(
        '<div class="badges">'
        f"<span>corpus: {html.escape(corpus)}</span>"
        f"<span>retrieval: {html.escape(mode)}</span>"
        f"<span>LLM: {html.escape(config.LLM_PROVIDER)} · {html.escape(llm_model_name())}</span>"
        f"<span>embeddings: {html.escape(config.EMBED_PROVIDER)}</span>"
        f"{key_badge}</div>",
        unsafe_allow_html=True,
    )


def render_sidebar(corpora: list[str]) -> tuple[str, dict]:
    """Corpus picker, corpus stats, retrieval settings and the about/limits panels.

    Only knobs `retrieve()` actually takes are exposed (mode, top_k, MMR), plus
    the answer language, which only changes the language the answer is written
    in. The refusal threshold is shown but not editable: changing it changes
    what a refusal means, so it stays a reviewed config value.
    """
    sb = st.sidebar
    corpus = sb.selectbox("Corpus", corpora)
    _, bm25, _ = load_resources(corpus)
    stats = load_stats(corpus)
    c1, c2 = sb.columns(2)
    c1.metric("Documents", stats["documents"])
    c2.metric("Chunks", stats["chunks"])
    langs = ", ".join(stats["languages"]) or "unknown"
    extra = f" · {stats['pages']} PDF pages" if stats["pages"] else ""
    sb.caption(f"Language (detected): {langs}{extra}")
    with sb.expander("Indexed files"):
        st.markdown("\n".join(f"- `{s}`" for s in stats["sources"]))

    sb.subheader("Retrieval")
    modes = ["hybrid", "dense"] if bm25 is not None else ["dense"]
    default_mode = config.RETRIEVAL_MODE if config.RETRIEVAL_MODE in modes else modes[0]
    mode = sb.radio(
        "Mode", modes, index=modes.index(default_mode), horizontal=True,
        help="hybrid = dense embeddings + BM25 keywords fused by rank (RRF). "
             "dense = embeddings only.",
    )
    if bm25 is None:
        sb.caption("No BM25 index for this corpus: re-ingest to enable hybrid mode.")
    top_k = sb.slider("Passages per answer (top_k)", 1, 10, min(config.TOP_K, 10))
    use_mmr = sb.checkbox(
        "Diversify passages (MMR)", value=config.USE_MMR,
        help="Drops near-duplicate passages so the sources cover more ground.",
    )
    sb.caption(
        f"Refusal threshold: {config.SCORE_THRESHOLD:.2f} dense cosine "
        f"(set `SCORE_THRESHOLD` in .env)."
    )

    answer_langs = list(config.SUPPORTED_ANSWER_LANGUAGES)
    language = sb.selectbox(
        "Answer language", answer_langs, index=answer_langs.index(config.DEFAULT_ANSWER_LANGUAGE),
        format_func=LANGUAGE_LABELS.get,
        help="Language the answer is written in, whatever the language of the documents. "
             "Quotes and [S#] citations are kept as they are.",
    )

    if sb.button("Clear conversation", use_container_width=True):
        st.session_state.setdefault("history", {})[corpus] = []

    with sb.expander("About / How it works"):
        st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(ui.PIPELINE_STEPS, 1)))
    with sb.expander("Limitations"):
        items = ui.build_limitations(config.SCORE_THRESHOLD, config.VERIFY_MIN_GROUNDING, bm25 is not None)
        st.markdown("\n".join(f"- {s}" for s in items))
    return corpus, {"mode": mode, "top_k": top_k, "use_mmr": use_mmr, "language": language}


def render_sources(sources, question: str, cited: list[int], title: str) -> None:
    """List every passage with location, score, highlighted snippet and cited flag."""
    terms = ui.query_terms(question)
    st.markdown(f"**{title}**")
    for i, r in enumerate(sources, 1):
        is_cited = i in cited
        flag = " · cited" if is_cited else ""
        body = ui.highlight_terms(ui.snippet(r.chunk.text, terms), terms)
        st.markdown(
            f'<div class="src{" cited" if is_cited else ""}">'
            f'<div class="head">[S{i}] {html.escape(ui.format_location(r.chunk.source, r.chunk.meta))}</div>'
            f'<div class="meta">score {r.score:.3f}{flag}</div>'
            f'<div class="body">{body}</div></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        "Score = cosine similarity to the question (1.0 = identical). In hybrid mode a "
        "passage found only by keywords shows its BM25 score instead."
    )


def render_assistant(msg: dict) -> None:
    """Render one assistant turn: answer or refusal, verification, sources, timings."""
    if msg.get("embed_error"):
        st.error(msg["embed_error"])
        return
    if msg["refused"]:
        st.markdown(
            '<div class="refusal"><b>No answer: not enough evidence in this corpus.</b><br>'
            f"{html.escape(msg['refusal_reason'])}</div>",
            unsafe_allow_html=True,
        )
        if msg["closest"]:
            render_sources(msg["closest"], msg["question"], [], "Closest passages found (below threshold)")
    else:
        if msg.get("llm_error"):
            st.warning(msg["llm_error"])
        else:
            st.markdown(ui.style_citations(msg["text"], len(msg["sources"])), unsafe_allow_html=True)
            rep = msg["report"]
            badge = ":green-badge[verified]" if rep.ok else ":orange-badge[issues found]"
            st.markdown(f"{badge} grounding **{rep.grounding_score:.2f}**")
            st.caption(ui.explain_grounding(rep.grounding_score, config.VERIFY_MIN_GROUNDING, rep.ok))
            if rep.invalid_citations:
                st.markdown(
                    "Invalid citations (no such source): "
                    + ", ".join(f"[S{n}]" for n in rep.invalid_citations)
                )
            if rep.uncited_sentences:
                st.markdown("Claims without a citation:")
                st.markdown("\n".join(f"- {s}" for s in rep.uncited_sentences))
        cited = extract_citations(msg.get("text") or "")
        render_sources(msg["sources"], msg["question"], cited, "Sources")
    timing = f"retrieval {msg['t_retrieve']:.2f}s"
    if msg.get("t_answer") is not None:
        timing += f" · answer {msg['t_answer']:.2f}s"
    st.caption(timing)


def run_question(question: str, store, bm25, embedder, settings: dict) -> dict:
    """Retrieve, answer and verify one question; turn failures into messages.

    Retrieval and generation are timed separately because they fail and slow down
    for different reasons (embedding backend vs LLM provider). On refusal the
    closest passages are fetched with a plain dense search so the user sees what
    was found and how far below the threshold it was.
    """
    msg = {"role": "assistant", "question": question, "refused": False, "text": "",
           "sources": [], "closest": [], "report": None, "t_answer": None}
    t0 = time.perf_counter()
    try:
        retrieved = retrieve(
            question, store, embedder, top_k=settings["top_k"], bm25=bm25,
            mode=settings["mode"], use_mmr=settings["use_mmr"],
        )
    except Exception:  # embedding backend down or misconfigured
        msg["t_retrieve"] = time.perf_counter() - t0
        msg["embed_error"] = ui.embed_error_hint(config.EMBED_PROVIDER)
        return msg
    msg["t_retrieve"] = time.perf_counter() - t0

    if not retrieved:
        msg["refused"] = True
        try:
            msg["closest"] = store.search(embedder.embed([question])[0], settings["top_k"])
        except Exception:
            msg["closest"] = []
        best = msg["closest"][0].score if msg["closest"] else None
        msg["refusal_reason"] = ui.explain_refusal(best, config.SCORE_THRESHOLD)
        return msg

    msg["sources"] = retrieved
    t1 = time.perf_counter()
    try:
        answer = answer_question(question, retrieved, language=settings["language"])
    except Exception as exc:  # LLMError or any provider failure: still show the passages
        msg["llm_error"] = ui.llm_error_hint(str(exc), config.LLM_PROVIDER)
        msg["t_answer"] = time.perf_counter() - t1
        return msg
    msg["t_answer"] = time.perf_counter() - t1
    msg["text"] = answer.text
    # Outside the try: verification is offline code, so a failure here is a bug
    # to surface, not a provider outage to explain with an API-key hint.
    msg["report"] = verify_answer(answer.text, retrieved, min_grounding=config.VERIFY_MIN_GROUNDING)
    return msg


def render_empty_state(corpus: str, language: str) -> None:
    """Short explanation plus clickable example questions from the corpus eval set.

    Examples are shown in English unless French answers were picked, so the
    suggested questions match the language the user is working in.
    """
    st.info(
        "Ask a question about the selected corpus. Each answer cites the passages it "
        "uses, which are listed underneath with their scores; if nothing in the "
        "documents is relevant enough, the assistant says so instead of guessing."
    )
    examples = ui.localized_examples(
        EVAL_DIR / f"{corpus}_eval.jsonl", "fr" if language == "fr" else "en"
    )
    if examples:
        st.markdown("**Try one of these:**")
        cols = st.columns(2)
        for i, q in enumerate(examples):
            if cols[i % 2].button(q, key=f"ex-{corpus}-{i}", use_container_width=True):
                st.session_state["pending_question"] = q
                st.rerun()


def main() -> None:
    """Render header and sidebar, replay history and handle a new question."""
    st.set_page_config(page_title="Grounded RAG assistant", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    corpora = indexed_corpora()
    if not corpora:
        st.title("Grounded RAG assistant")
        st.info("No indexed corpus found. Run `python cli.py ingest --corpus NAME` first.")
        return
    corpus, settings = render_sidebar(corpora)
    store, bm25, embedder = load_resources(corpus)
    render_header(corpus, settings["mode"])

    history = st.session_state.setdefault("history", {}).setdefault(corpus, [])
    for msg in history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["text"])
            else:
                render_assistant(msg)

    typed = st.chat_input("Ask a question about this corpus")
    question = typed or st.session_state.pop("pending_question", None)
    if not history and not question:
        render_empty_state(corpus, settings["language"])
    if question:
        history.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving passages and drafting a cited answer..."):
                msg = run_question(question, store, bm25, embedder, settings)
            history.append(msg)
            render_assistant(msg)


main()
