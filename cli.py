"""Command-line entry point: build an index (`ingest`) and query it (`ask`).

A thin wrapper over the library so the pipeline can be exercised without the UI;
all logic lives in the `rag` package.
"""

from __future__ import annotations

import argparse
import sys

import config


def cmd_ingest(args) -> int:
    """Build and save the index for a corpus, then report its size and location."""
    from rag.ingest import ingest_corpus

    store = ingest_corpus(args.corpus)
    print(f"Indexed {len(store)} chunks for '{args.corpus}' in {config.index_path(args.corpus)}")
    return 0


def cmd_ask(args) -> int:
    """Answer a question from a saved index, with hybrid retrieval and verification."""
    from rag.answer import answer_question
    from rag.embed import get_embedder
    from rag.lexical import BM25Index
    from rag.logging_utils import log_query
    from rag.retrieve import retrieve
    from rag.store import VectorStore
    from rag.verify import format_report, verify_answer

    try:
        store = VectorStore.load(config.index_path(args.corpus))
    except FileNotFoundError:
        print(f"No index for corpus '{args.corpus}'. Run: python cli.py ingest --corpus {args.corpus}")
        return 1
    # BM25 is optional: without it (older index) retrieval falls back to dense.
    try:
        bm25 = BM25Index.load(config.bm25_path(args.corpus))
    except FileNotFoundError:
        bm25 = None

    question = " ".join(args.question)
    embedder = get_embedder()
    retrieved = retrieve(question, store, embedder, bm25=bm25)
    answer = answer_question(question, retrieved)

    print(answer.text)
    if not answer.refused:
        print("\nSources:")
        for i, src in enumerate(answer.sources, 1):
            page = src.chunk.meta.get("page")
            loc = f"{src.chunk.source}" + (f" p.{page}" if page else "")
            print(f"[S{i}] {loc} (score {src.score:.2f})")
        report = verify_answer(answer.text, answer.sources, min_grounding=config.VERIFY_MIN_GROUNDING)
        print("\n" + format_report(report))

    top_score = retrieved[0].score if retrieved else None
    log_query(config.QUERY_LOG_PATH, question=question, corpus=args.corpus,
              n_retrieved=len(retrieved), top_score=top_score, refused=answer.refused)
    return 0


def main() -> None:
    """Parse arguments and dispatch to the chosen subcommand."""
    parser = argparse.ArgumentParser(description="Grounded RAG assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="build the index for a corpus")
    p_ingest.add_argument("--corpus", required=True)
    p_ingest.set_defaults(func=cmd_ingest)
    p_ask = sub.add_parser("ask", help="ask a question about a corpus")
    p_ask.add_argument("--corpus", required=True)
    p_ask.add_argument("question", nargs="+")
    p_ask.set_defaults(func=cmd_ask)
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
