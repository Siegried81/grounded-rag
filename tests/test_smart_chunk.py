"""Offline tests for sentence-aware chunking."""

from itertools import pairwise

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
    for prev, nxt in pairwise(chunks):
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


# --- Context-headed documents (flattened financial statements) ---------------

from rag.smart_chunk import chunk_contextual_text, has_context_header  # noqa: E402

STATEMENT = "\n".join([
    "Context: Acme Inc. 10-K FY2025, Item 8 (amounts in millions of dollars).",
    "",
    "CONSOLIDATED STATEMENTS OF OPERATIONS",
    "(In millions)",
    "Years ended",
    "September 27, 2025 | September 28, 2024",
    "Net sales:",
    "Products: 307,003 | 294,866",
    "Services: 109,158 | 96,169",
    "Total net sales: 416,161 | 391,035",
    "Cost of sales:",
    "Products: 194,116 | 185,233",
    "Services: 26,844 | 25,119",
    "Total cost of sales: 220,960 | 210,352",
    "Gross margin: 195,201 | 180,683",
])
TOP = ("Acme Inc. 10-K FY2025, Item 8 (amounts in millions of dollars).\n"
       "CONSOLIDATED STATEMENTS OF OPERATIONS\n")


def test_has_context_header_only_for_files_that_opt_in():
    assert has_context_header(STATEMENT)
    assert not has_context_header("Article 5\n\nThe following practices are prohibited.")


def test_every_chunk_starts_with_the_context_and_title():
    chunks = chunk_contextual_text(STATEMENT, "ops.txt", "c", size=330)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.startswith(TOP)
        assert len(c.text) <= 330
    assert [c.id for c in chunks] == [f"c:ops.txt:{i}" for i in range(len(chunks))]


def test_a_row_is_never_cut_and_every_row_is_kept():
    chunks = chunk_contextual_text(STATEMENT, "ops.txt", "c", size=330)
    lines = [ln.split("\n") for ln in (c.text for c in chunks)]
    kept = {ln for chunk in lines for ln in chunk}
    for row in STATEMENT.splitlines()[7:]:
        assert row in kept


def test_a_chunk_that_starts_mid_table_repeats_the_columns_and_the_group_heading():
    chunks = chunk_contextual_text(STATEMENT, "ops.txt", "c", size=330)
    later = [c for c in chunks if "Services: 26,844" in c.text][0]
    assert "September 27, 2025 | September 28, 2024" in later.text
    # A chunk starting on a "Services" row must say which group the row is in.
    assert "Cost of sales:" in later.text


def test_a_total_row_closes_its_group_heading():
    text = "\n".join([
        "Context: C.", "", "Title", "2025 | 2024",
        "Group A:", "x: 1 | 2", "Total A: 3 | 4", "y: 5 | 6", "z: 7 | 8",
    ])
    chunks = chunk_contextual_text(text, "t.txt", "c", size=55)
    holder = [c for c in chunks if "\ny: 5" in c.text][0]
    assert "Group A:" not in holder.text


def test_a_long_prose_line_is_cut_at_sentences_and_keeps_the_top():
    prose = " ".join(f"Sentence number {i} says something about the Company." for i in range(30))
    text = f"Context: C.\n\nNote 1 - Policies\n{prose}\nRevenue: 10 | 20"
    chunks = chunk_contextual_text(text, "n.txt", "c", size=300)
    assert all(c.text.startswith("C.\nNote 1 - Policies\n") for c in chunks)
    assert all(len(c.text) <= 300 for c in chunks)
    assert chunks[-1].text.endswith("Revenue: 10 | 20")
    assert sum(c.meta["kind"] == "prose" for c in chunks) >= 2


def test_ingest_uses_the_contextual_chunker_only_for_context_headed_files(tmp_path):
    from rag.ingest import _chunk_file

    (tmp_path / "s.txt").write_text(STATEMENT, encoding="utf-8")
    (tmp_path / "p.txt").write_text("Plain prose. Second sentence.", encoding="utf-8")
    table = _chunk_file(tmp_path / "s.txt", "c", "smart")
    assert table[0].text.startswith(TOP)
    plain = _chunk_file(tmp_path / "p.txt", "c", "smart")
    assert [c.text for c in plain] == ["Plain prose. Second sentence."]
