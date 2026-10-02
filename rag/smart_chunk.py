"""Sentence-aware chunking: a drop-in alternative to fixed character windows.

Fixed windows cut sentences in half, which hurts both embedding quality and the
readability of a cited passage. This module packs whole sentences into chunks and
overlaps neighbours by whole sentences, producing the same `Chunk` type and id
scheme as `rag.chunk` so the rest of the pipeline does not care which is used.
"""

from __future__ import annotations

import re

import config
from rag.types import Chunk

_PARA_BREAK = re.compile(r"\n\s*\n")
_SENT_BREAK = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping the terminal punctuation.

    A pragmatic heuristic, not a full NLP tokenizer: it breaks after ., ! or ?
    followed by whitespace, and treats blank lines as hard breaks. It will
    wrongly split on abbreviations such as "e.g. "; that is acceptable here
    because it is dependency-free and a bad split only slightly shifts a chunk
    boundary.
    """
    sentences: list[str] = []
    for para in _PARA_BREAK.split(text):
        for s in _SENT_BREAK.split(para):
            s = " ".join(s.split())  # collapse inner whitespace/newlines
            if s:
                sentences.append(s)
    return sentences


def _joined_len(sentences: list[str]) -> int:
    """Length of the sentences joined by single spaces."""
    return sum(len(s) for s in sentences) + max(len(sentences) - 1, 0)


def _overlap_tail(sentences: list[str], overlap: int) -> list[str]:
    """Return the trailing sentences whose joined length fits within `overlap`.

    Sentence-level overlap keeps context across a boundary without repeating a
    half sentence.
    """
    tail: list[str] = []
    for s in reversed(sentences):
        if _joined_len([s, *tail]) > overlap:
            break
        tail.insert(0, s)
    return tail


def smart_chunk_text(
    text: str,
    source: str,
    corpus: str,
    size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
    page: int | None = None,
) -> list[Chunk]:
    """Pack whole sentences into chunks of at most `size` characters.

    When a chunk is full the next one starts with the trailing sentences that fit
    in `overlap` characters. A single sentence longer than `size` becomes its own
    chunk rather than being split, so a word is never cut. Ids follow
    f"{corpus}:{source}:{ordinal}" like `rag.chunk`.
    """
    chunks: list[Chunk] = []
    current: list[str] = []

    def emit() -> None:
        body = " ".join(current).strip()
        if not body:
            return
        meta: dict = {"n_sentences": len(current)}
        if page is not None:
            meta["page"] = page
        n = len(chunks)
        chunks.append(Chunk(f"{corpus}:{source}:{n}", body, source, n, corpus, meta))

    for sent in split_sentences(text):
        if current and _joined_len([*current, sent]) > size:
            emit()
            current = _overlap_tail(current, overlap)
            # Drop carried sentences until the new one fits (or none are left).
            while current and _joined_len([*current, sent]) > size:
                current.pop(0)
        current.append(sent)
    if current:
        emit()
    return chunks


def chunk_pages(pages: list[str], source: str, corpus: str, **kw) -> list[Chunk]:
    """Chunk a list of page texts, tagging each chunk with its 1-based page.

    Chunks never span pages, so a citation can point to one page; ordinals keep
    increasing across pages so ids stay unique within the source.
    """
    out: list[Chunk] = []
    for page_no, page_text in enumerate(pages, start=1):
        for c in smart_chunk_text(page_text, source, corpus, page=page_no, **kw):
            n = len(out)
            out.append(Chunk(f"{corpus}:{source}:{n}", c.text, source, n, corpus, c.meta))
    return out
