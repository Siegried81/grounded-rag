"""Load a corpus from disk and build its persisted indexes.

Ingestion is the offline half of the pipeline: read every document, chunk it,
embed all chunks in one batch and save both a dense vector store and a BM25
keyword index. Doing it as a single explicit step (rather than lazily at question
time) keeps `ask` fast and makes the indexes reproducible artifacts of the corpus
folder. Embeddings go through an on-disk cache so re-ingesting an unchanged corpus
costs nothing.
"""

from __future__ import annotations

from pathlib import Path

import config

SUFFIXES = {".txt", ".md", ".pdf"}


def _read_pdf_pages(path) -> list[str]:
    """Return one text string per PDF page (empty pages kept for page numbering)."""
    from pypdf import PdfReader  # imported lazily: only needed for PDF corpora

    reader = PdfReader(str(path))
    return [(page.extract_text() or "") for page in reader.pages]


def document_paths(corpus: str) -> list[Path]:
    """The files of a corpus the indexer reads, sorted by filename.

    Only files directly inside `data/<corpus>/` count: a subfolder is not walked,
    so raw dumps kept beside a corpus are not indexed. README.md is skipped
    because it documents the folder rather than being evidence, and sorting keeps
    the index deterministic. Single source of this rule, shared by chunking and
    by the API's document viewer, so a reader can only open a file that was
    actually indexed.
    """
    return [
        p
        for p in sorted(config.corpus_dir(corpus).iterdir(), key=lambda p: p.name)
        if p.is_file() and p.suffix.lower() in SUFFIXES and p.name != "README.md"
    ]


def read_document(path) -> str:
    """A document's full text, PDF pages joined with newlines.

    `ingest_corpus` reads PDFs page by page instead, so it can record the page
    number a chunk came from; this flat form is for callers that just want the
    text.
    """
    if path.suffix.lower() == ".pdf":
        return "\n".join(_read_pdf_pages(path))
    return path.read_text(encoding="utf-8")


def load_documents(corpus: str) -> list[tuple[str, str]]:
    """Return (filename, text) for each supported, non-empty file of a corpus."""
    docs = []
    for path in document_paths(corpus):
        text = read_document(path)
        if text.strip():
            docs.append((path.name, text))
    return docs


def _chunk_corpus(corpus: str, chunker: str):
    """Chunk every document of a corpus, page-aware for PDFs.

    Uses the sentence-aware splitter by default (better passages and citations),
    falling back to the original fixed-character splitter when chunker == "char".
    """
    from rag.chunk import chunk_text
    from rag.smart_chunk import chunk_pages, smart_chunk_text

    chunks = []
    for path in document_paths(corpus):
        source = path.name
        if path.suffix.lower() == ".pdf":
            # Page-aware so a citation can point to a PDF page.
            chunks.extend(chunk_pages(_read_pdf_pages(path), source, corpus))
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        if chunker == "char":
            chunks.extend(chunk_text(text, source, corpus))
        else:
            chunks.extend(smart_chunk_text(text, source, corpus))
    return chunks


def ingest_corpus(corpus: str, embedder=None, chunker: str | None = None):
    """Chunk, embed and persist a corpus; return the built VectorStore.

    Builds two indexes from the same chunks — a dense vector store and a BM25
    keyword index — so retrieval can run in hybrid mode. Raises ValueError when
    the corpus has no usable documents, since an empty index would silently make
    every question refuse.
    """
    from rag.cache import EmbeddingCache
    from rag.embed import get_embedder
    from rag.lexical import BM25Index
    from rag.store import VectorStore

    chunker = chunker or config.CHUNKER
    embedder = embedder or get_embedder()

    chunks = _chunk_corpus(corpus, chunker)
    if not chunks:
        raise ValueError(
            f"No documents (.txt/.md/.pdf) found for corpus '{corpus}' in "
            f"{config.corpus_dir(corpus)}"
        )

    texts = [c.text for c in chunks]
    if config.USE_EMBED_CACHE:
        cache = EmbeddingCache.load(config.embed_cache_path(corpus))
        vectors = cache.embed_with_cache(embedder, texts)
        cache.save()
    else:
        vectors = embedder.embed(texts)

    store = VectorStore(config.index_path(corpus), embedder.model)
    store.add(chunks, vectors)
    store.save()

    # Keyword index alongside the vectors, so `ask` can fuse both channels.
    BM25Index.build(chunks).save(config.bm25_path(corpus))
    return store
