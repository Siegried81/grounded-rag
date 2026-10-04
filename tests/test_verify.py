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


def test_lenticular_citations_are_recognised():
    """gpt-oss cites as 【S1】 (sometimes with a suffix); it must count like [S1]."""
    from rag.answer import normalize_citations

    assert normalize_citations("Yes 【S1】【S3】 and ［S2］, see 【S4†L4-L9】.") == \
        "Yes [S1][S3] and [S2], see [S4]."
    assert extract_citations("Net sales rose 【S2】 then fell 【S1】.") == [2, 1]


def test_bold_grouped_and_zero_width_citations_are_recognised():
    """Forms found in real gpt-oss answers (aa-12, aa-13, fs-12) count as citations."""
    from rag.answer import normalize_citations

    assert normalize_citations("Yes [**S1**].") == "Yes [S1]."
    assert normalize_citations("Both [**S1**, **S5**] and [S2; S3].") == "Both [S1][S5] and [S2][S3]."
    assert normalize_citations("See [​S4​].") == "See [S4]."
    assert normalize_citations("Plain [S1] stays [S1].") == "Plain [S1] stays [S1]."
    # Ordinary bracketed text is not a citation and is left alone.
    assert normalize_citations("Annex [III] and [see note]") == "Annex [III] and [see note]"
    assert extract_citations("Net sales [**S2**, **S1**].") == [2, 1]
