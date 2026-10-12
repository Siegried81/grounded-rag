"""Load a corpus from disk and build its persisted indexes.

Ingestion is the offline half of the pipeline: read every document, chunk it,
embed all chunks in one batch and save both a dense vector store and a BM25
keyword index. Doing it as a single explicit step (rather than lazily at question
time) keeps `ask` fast and makes the indexes reproducible artifacts of the corpus
folder. Embeddings go through an on-disk cache so re-ingesting an unchanged corpus
costs nothing.
"""

from __future__ import annotations

import hashlib
import json
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


def _chunk_file(path: Path, corpus: str, chunker: str):
    """Chunk one document, page-aware for PDFs; an empty text file gives no chunk.

    Uses the sentence-aware splitter by default (better passages and citations),
    falling back to the original fixed-character splitter when chunker == "char".
    A chunk's id and ordinal depend only on its own file, which is what lets an
    incremental ingest reuse an unchanged file's chunks as they are.
    """
    from rag.chunk import chunk_text
    from rag.smart_chunk import chunk_pages, smart_chunk_text

    source = path.name
    if path.suffix.lower() == ".pdf":
        # Page-aware so a citation can point to a PDF page.
        return chunk_pages(_read_pdf_pages(path), source, corpus)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if chunker == "char":
        return chunk_text(text, source, corpus)
    return smart_chunk_text(text, source, corpus)


def _chunk_corpus(corpus: str, chunker: str):
    """Chunk every document of a corpus, in `document_paths` order."""
    return [c for path in document_paths(corpus) for c in _chunk_file(path, corpus, chunker)]


def _file_hash(path: Path) -> str:
    """sha256 of a file's bytes: the identity an incremental ingest compares."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_settings(chunker: str, model: str) -> dict:
    """Everything besides a file's bytes that decides its chunks and vectors.

    A file whose hash is unchanged still chunks differently under another
    chunker or window size, and embeds differently under another model, so a
    manifest written under other settings is not reused at all.
    """
    return {
        "chunker": chunker,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "model": model,
    }


def _reusable_rows(corpus: str, settings: dict) -> tuple[dict, dict]:
    """(file hashes, {source: [(chunk, vector), ...]}) from the previous build, or empty.

    Anything that makes the previous build untrustworthy — no manifest (an
    index older than this feature), other settings, a store that does not load
    or holds another number of chunks for a file than the manifest says —
    returns nothing for the affected scope, and those files are rebuilt.
    """
    from rag.store import VectorStore

    try:
        manifest = json.loads(manifest_path(corpus).read_text(encoding="utf-8"))
        store = VectorStore.load(config.index_path(corpus))
    except (OSError, ValueError):
        return {}, {}
    if manifest.get("settings") != settings or store.model != settings["model"]:
        return {}, {}
    rows: dict[str, list] = {}
    for chunk, vector in store.rows():
        rows.setdefault(chunk.source, []).append((chunk, vector))
    files = manifest.get("files", {})
    hashes = {
        name: entry["sha256"]
        for name, entry in files.items()
        if len(rows.get(name, [])) == entry.get("chunks")
    }
    return hashes, rows


def manifest_path(corpus: str) -> Path:
    """File recording, per indexed document, its hash and chunk count."""
    return config.index_path(corpus) / "manifest.json"


def ingest_corpus(corpus: str, embedder=None, chunker: str | None = None,
                  incremental: bool | None = None, report: dict | None = None):
    """Chunk, embed and persist a corpus; return the built VectorStore.

    Builds two indexes from the same chunks — a dense vector store and a BM25
    keyword index — so retrieval can run in hybrid mode. Raises ValueError when
    the corpus has no usable documents, since an empty index would silently make
    every question refuse.

    With `incremental` (default `config.INCREMENTAL_INGEST`, off), a file whose
    sha256 matches the previous build's manifest keeps its chunks and vectors
    from the stored index instead of being re-read, re-chunked and re-embedded;
    a new or changed file is processed as in a full build, and a file no longer
    in the folder is simply not carried over. The result is the index a full
    build would write: files are assembled in the same order, chunk ids depend
    only on their own file, and BM25 is always rebuilt from all chunks, since
    its term statistics are corpus-wide. Every build writes the manifest, so a
    full build is what makes the next incremental one possible.

    `report`, when given, is filled with the file names in each category
    (`reused`, `processed`, `removed`) and the number of chunks that needed a
    vector (`new_chunks`; the embedding cache may still serve them).
    """
    from rag.cache import EmbeddingCache
    from rag.embed import get_embedder
    from rag.lexical import BM25Index
    from rag.store import VectorStore

    chunker = chunker or config.CHUNKER
    embedder = embedder or get_embedder()
    incremental = config.INCREMENTAL_INGEST if incremental is None else incremental
    settings = _manifest_settings(chunker, embedder.model)
    old_hashes, old_rows = _reusable_rows(corpus, settings) if incremental else ({}, {})

    paths = document_paths(corpus)
    hashes = {p.name: _file_hash(p) for p in paths}
    chunks, known, reused, processed = [], {}, [], []
    for path in paths:
        if old_hashes.get(path.name) == hashes[path.name]:
            for chunk, vector in old_rows.get(path.name, []):
                chunks.append(chunk)
                known[chunk.id] = vector
            reused.append(path.name)
        else:
            chunks.extend(_chunk_file(path, corpus, chunker))
            processed.append(path.name)
    if not chunks:
        raise ValueError(
            f"No documents (.txt/.md/.pdf) found for corpus '{corpus}' in "
            f"{config.corpus_dir(corpus)}"
        )

    todo = [c.text for c in chunks if c.id not in known]
    if not todo:
        fresh = []
    elif config.USE_EMBED_CACHE:
        cache = EmbeddingCache.load(config.embed_cache_path(corpus))
        fresh = cache.embed_with_cache(embedder, todo)
        cache.save()
    else:
        fresh = embedder.embed(todo)
    fresh_iter = iter(fresh)
    vectors = [known[c.id] if c.id in known else next(fresh_iter) for c in chunks]

    store = VectorStore(config.index_path(corpus), embedder.model)
    store.add(chunks, vectors)
    store.save()

    # Keyword index alongside the vectors, so `ask` can fuse both channels.
    BM25Index.build(chunks).save(config.bm25_path(corpus))

    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.source] = counts.get(c.source, 0) + 1
    manifest = {
        "settings": settings,
        "files": {name: {"sha256": h, "chunks": counts.get(name, 0)} for name, h in hashes.items()},
    }
    manifest_path(corpus).write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    if report is not None:
        report.update(
            reused=reused,
            processed=processed,
            removed=sorted((set(old_hashes) | set(old_rows)) - set(hashes)),
            new_chunks=len(todo),
        )
    return store


def ensure_index(corpus: str, embedder=None) -> bool:
    """Build the index of a corpus only when it is missing; True when it was built.

    This is the start-up path of a deployment with no build step and no
    persistent disk (Streamlit Community Cloud, Render's free plan): the corpora
    ship with the code, the index does not, and every cold start has to make it
    again. Presence is judged on the vector store's two files, so a half-written
    index from an interrupted start is rebuilt rather than loaded. An index built
    with another embedder is left alone here: the store reports that mismatch at
    query time, where the configured model is known.
    """
    from rag.store import VectorStore

    if VectorStore.exists(config.index_path(corpus)):
        return False
    ingest_corpus(corpus, embedder=embedder)
    return True
