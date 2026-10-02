"""Tests for rag.chunk.chunk_text: windowing, overlap, ids and edge cases."""

import pytest

from rag.chunk import chunk_text


def test_long_text_yields_multiple_chunks():
    """A text longer than one window must be split into several chunks."""
    text = "abcdefghij" * 50
    chunks = chunk_text(text, "a.txt", "c", size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 100 for c in chunks)


def test_overlap_shares_text_between_consecutive_chunks():
    """The tail of a chunk must reappear at the head of the next one."""
    text = "".join(chr(97 + i % 26) + str(i % 10) for i in range(200))
    chunks = chunk_text(text, "a.txt", "c", size=60, overlap=15)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.text[-15:] == nxt.text[:15]


def test_ids_and_ordinals_are_sequential():
    """Ids follow corpus:source:ordinal and ordinals count up from zero."""
    chunks = chunk_text("x" * 500, "doc.md", "corp", size=100, overlap=10)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.id == f"corp:doc.md:{c.ordinal}"
        assert c.corpus == "corp"
        assert c.source == "doc.md"


@pytest.mark.parametrize("text", ["", "   ", "\n\t \n"])
def test_blank_input_yields_empty_list(text):
    assert chunk_text(text, "a.txt", "c", size=10, overlap=2) == []


def test_whitespace_only_windows_are_skipped_and_ordinals_stay_contiguous():
    """A blank middle window is dropped without leaving a gap in the ordinals."""
    text = "a" * 10 + " " * 10 + "b" * 10
    chunks = chunk_text(text, "a.txt", "c", size=10, overlap=0)
    assert [c.text for c in chunks] == ["a" * 10, "b" * 10]
    assert [c.ordinal for c in chunks] == [0, 1]


@pytest.mark.parametrize("overlap", [10, 11])
def test_overlap_not_smaller_than_size_raises(overlap):
    with pytest.raises(ValueError):
        chunk_text("hello world", "a.txt", "c", size=10, overlap=overlap)
