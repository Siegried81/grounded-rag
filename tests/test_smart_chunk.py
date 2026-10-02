"""Offline tests for sentence-aware chunking."""

from rag.smart_chunk import chunk_pages, smart_chunk_text, split_sentences

TEXT = ("Alpha is first. Bravo is second! Charlie is third? Delta is fourth. "
        "Echo is fifth. Foxtrot is sixth.")


def test_split_sentences_paragraph():
    assert split_sentences(TEXT) == [
        "Alpha is first.", "Bravo is second!", "Charlie is third?",
        "Delta is fourth.", "Echo is fifth.", "Foxtrot is sixth.",
    ]


def test_split_sentences_double_newline_is_hard_break():
    assert split_sentences("No terminal punctuation\n\nNext para here") == [
        "No terminal punctuation", "Next para here"]


def test_size_respected_and_ends_at_boundary():
    chunks = smart_chunk_text(TEXT, "a.txt", "c", size=40, overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 40
        assert c.text[-1] in ".!?"


def test_oversized_sentence_is_own_chunk():
    long = "Word " * 30 + "end."
    chunks = smart_chunk_text(f"Short one. {long} Tail.", "a", "c", size=40, overlap=0)
    assert any(c.text == long for c in chunks)
    assert all(len(c.text) <= 40 for c in chunks if c.text != long)


def test_overlap_carries_trailing_sentence():
    chunks = smart_chunk_text(TEXT, "a", "c", size=40, overlap=20)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        last = split_sentences(prev.text)[-1]
        assert nxt.text.startswith(last)


def test_ids_ordinals_meta():
    chunks = smart_chunk_text(TEXT, "doc.txt", "corp", size=40, overlap=0, page=3)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert chunks[1].id == "corp:doc.txt:1"
    assert all(c.meta["page"] == 3 and c.meta["n_sentences"] >= 1 for c in chunks)
    assert "page" not in smart_chunk_text(TEXT, "a", "c")[0].meta


def test_empty_text():
    assert smart_chunk_text("  \n\n ", "a", "c") == []


def test_chunk_pages_tags_pages_and_global_ordinals():
    chunks = chunk_pages([TEXT, "", TEXT], "p.pdf", "c", size=40, overlap=0)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert [c.id for c in chunks] == [f"c:p.pdf:{i}" for i in range(len(chunks))]
    pages = [c.meta["page"] for c in chunks]
    assert set(pages) == {1, 3}
    assert pages == sorted(pages)
