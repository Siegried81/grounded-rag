"""Offline tests for rag.verify: citation parsing, validity, grounding, uncited claims."""

from rag.answer import REFUSAL_MESSAGE
from rag.types import Chunk, Retrieved
from rag.verify import extract_citations, format_report, verify_answer


def _src(text: str, n: int = 0) -> Retrieved:
    """Build a Retrieved source with the given chunk text."""
    return Retrieved(Chunk(f"c{n}", text, "doc.md", n, "corp"), 0.9)


SOURCES = [
    _src("The warranty covers manufacturing defects for twenty four months.", 1),
    _src("Returns are accepted within thirty days with the original receipt.", 2),
]


def test_extract_citations_distinct_ordered():
    assert extract_citations("A [S2]. B [S1] and [S2] again [S2].") == [2, 1]


def test_extract_citations_none():
    assert extract_citations("No markers here.") == []


def test_invalid_citation_detected():
    r = verify_answer("The warranty covers defects [S3].", SOURCES)
    assert r.invalid_citations == [3]
    assert r.valid_citations == []
    assert not r.ok


def test_zero_index_is_invalid():
    assert verify_answer("Something [S0].", SOURCES).invalid_citations == [0]


def test_grounding_high_when_words_reused():
    r = verify_answer("The warranty covers manufacturing defects [S1].", SOURCES)
    assert r.grounding_score > 0.9
    assert r.ok


def test_grounding_low_when_unrelated():
    r = verify_answer("Dragons breathe purple lightning above volcanoes [S1].", SOURCES)
    assert r.grounding_score < 0.2
    assert not r.ok


def test_uncited_factual_sentence_flagged():
    text = "Returns are accepted within thirty days with receipt. Thanks [S2]."
    r = verify_answer(text, SOURCES)
    assert r.uncited_sentences == ["Returns are accepted within thirty days with receipt."]


def test_short_or_cited_sentences_not_flagged():
    r = verify_answer("Yes. Returns are accepted within thirty days [S2].", SOURCES)
    assert r.uncited_sentences == []


def test_refusal_message_is_ok():
    r = verify_answer(REFUSAL_MESSAGE, SOURCES)
    assert r.ok and r.grounding_score == 1.0
    assert r.uncited_sentences == [] and r.invalid_citations == []


def test_format_report_mentions_status():
    out = format_report(verify_answer("Hello [S9].", SOURCES))
    assert "FAILED" in out and "9" in out
