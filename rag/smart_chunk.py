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


# --- Context-headed documents (financial statements) -------------------------
# The sentence packer above is wrong for a table flattened to one `label: values`
# line per row: the lines hold no sentence punctuation, so a whole statement
# became ONE sentence (a 2,300-character chunk of unrelated rows) and the file's
# header paragraph became a chunk of its own, pure boilerplate that matched
# every question about "Apple fiscal 2025" and carried no figure. A document
# opts in by starting with a `Context:` line; this chunker then packs whole
# lines, and puts the context, the statement or note title, the column names and
# the nearest group heading at the top of every chunk, so a chunk read alone
# still says which statement, which year and which line its figure belongs to.
CONTEXT_PREFIX = "Context:"
_ROW = re.compile(r"^[^:]{1,200}: \S")
_COLS = re.compile(r"^(?:\d{4}|Years ended)$")
_HEADING_MAX = 90
_ROW_MAX = 700
_TOTAL = re.compile(r"(?i)^(?:sub)?total")


def has_context_header(text: str) -> bool:
    """True when the document starts with a `Context:` line (see above)."""
    return text.lstrip().startswith(CONTEXT_PREFIX)


def _line_kind(line: str) -> str:
    """"heading" (a label ending in a colon), "row" (`label: values`), "cols" or "prose".

    Column lines are the header of a table: a bare year, "Years ended", or cells
    separated by " | " with no label. They are sticky: a chunk that starts
    mid-table repeats the last one seen, otherwise its numbers have no year.
    """
    if line.endswith(":") and len(line) <= _HEADING_MAX:
        return "heading"
    if len(line) <= _ROW_MAX and _ROW.match(line) and re.search(r"[\d—]", line):
        return "row"
    if len(line) < 250 and ": " not in line and (
        _COLS.match(line) or (" | " in line and not line.endswith("."))
    ):
        return "cols"
    return "prose"


def chunk_contextual_text(
    text: str, source: str, corpus: str, size: int = config.CHUNK_SIZE
) -> list[Chunk]:
    """Pack whole lines of a `Context:`-headed document into chunks of at most `size` characters.

    Every chunk starts with the context line and the title (the first line after
    it), then the last column line and, when the chunk begins on a table row,
    the last group heading ("Operating expenses:"). A sub-total or total row
    closes its group, so a heading is never carried past its own rows. Lines are
    never split, so a figure stays next to its label; a line longer than the
    room is cut at sentence boundaries by `smart_chunk_text` and each piece gets
    the same top (only a single sentence longer than the room runs over `size`). Ids follow f"{corpus}:{source}:{ordinal}" and
    `meta["kind"]` is "lines" for packed lines and "prose" for a cut paragraph.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    context = lines[0][len(CONTEXT_PREFIX):].strip()
    body = lines[1:]
    title = body[0] if body else ""
    top = f"{context}\n{title}"
    room = max(size - len(top) - 1, 1)
    chunks: list[Chunk] = []
    cur: list[str] = []
    cols: str | None = None
    heading: str | None = None
    head_at_start: list[str] = []  # the sticky lines to put above `cur`

    def add(body_text: str, kind: str) -> None:
        n = len(chunks)
        chunks.append(Chunk(f"{corpus}:{source}:{n}", f"{top}\n{body_text}", source, n,
                            corpus, {"kind": kind}))

    def emit() -> None:
        nonlocal cur
        if cur:
            add("\n".join([*head_at_start, *cur]), "table")
        cur = []

    for line in body[1:]:
        kind = _line_kind(line)
        if len(line) > room:
            emit()
            for piece in smart_chunk_text(line, source, corpus, size=room, overlap=0):
                add(piece.text, "prose")
            continue
        used = sum(len(s) + 1 for s in (*head_at_start, *cur))
        if cur and used + len(line) > room:
            emit()
        if not cur:
            head_at_start = [s for s in (
                cols if kind != "cols" else None,
                heading if kind == "row" else None) if s]
            if sum(len(s) + 1 for s in head_at_start) + len(line) > room:
                head_at_start = []  # a line this long stands alone, without the extras
        cur.append(line)
        if kind == "cols":
            cols = line
        elif kind == "heading":
            heading = line
        elif kind == "row" and _TOTAL.match(line):
            heading = None
    emit()
    return chunks
