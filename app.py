"""Streamlit chat UI over the grounded RAG pipeline.

Lets a user pick an indexed corpus, ask questions, and inspect exactly which
passages (and scores) each answer rested on, which is what makes the cite-or-refuse
behaviour verifiable rather than a claim.
"""

from __future__ import annotations

import streamlit as st

import config
from rag.answer import answer_question
from rag.embed import get_embedder
from rag.lexical import BM25Index
from rag.retrieve import retrieve
from rag.store import VectorStore
from rag.verify import verify_answer


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


def _show_sources(retrieved) -> None:
    """Expander listing each retrieved passage with its source, page and score."""
    with st.expander("Sources"):
        for i, r in enumerate(retrieved, 1):
            page = r.chunk.meta.get("page")
            loc = r.chunk.source + (f" · p.{page}" if page else "")
            st.markdown(f"**[S{i}] {loc}** (score {r.score:.2f})")
            st.caption(r.chunk.text)


def main() -> None:
    """Render the sidebar, replay history and handle a new question."""
    st.title("Grounded RAG assistant")
    corpora = indexed_corpora()
    if not corpora:
        st.info("No indexed corpus found. Run `python cli.py ingest --corpus NAME` first.")
        return
    corpus = st.sidebar.selectbox("Corpus", corpora)
    store, bm25, embedder = load_resources(corpus)

    history = st.session_state.setdefault("history", {}).setdefault(corpus, [])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])
            if msg.get("retrieved"):
                _show_sources(msg["retrieved"])

    question = st.chat_input("Ask a question about this corpus")
    if question:
        history.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.write(question)
        retrieved = retrieve(question, store, embedder, bm25=bm25)
        answer = answer_question(question, retrieved)
        # A refusal is shown plainly, without a sources expander.
        shown = [] if answer.refused else retrieved
        history.append({"role": "assistant", "text": answer.text, "retrieved": shown})
        with st.chat_message("assistant"):
            st.write(answer.text)
            if shown:
                report = verify_answer(answer.text, shown, min_grounding=config.VERIFY_MIN_GROUNDING)
                badge = "✅ grounded" if report.ok else "⚠️ check sources"
                st.caption(f"{badge} · grounding {report.grounding_score:.2f}")
                _show_sources(shown)


main()
