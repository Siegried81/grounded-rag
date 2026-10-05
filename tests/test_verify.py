"""Offline tests for rag.verify: citation parsing, validity, grounding, uncited claims."""

import pytest

from rag.answer import REFUSAL_MESSAGE, REFUSAL_MESSAGES
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
    assert r.ok and r.strict_ok and r.grounding_score == 1.0
    assert r.uncited_sentences == [] and r.invalid_citations == []


@pytest.mark.parametrize("lang", sorted(REFUSAL_MESSAGES))
def test_refusal_message_in_any_language_is_ok(lang):
    assert verify_answer(REFUSAL_MESSAGES[lang], SOURCES).strict_ok


# --- strict_ok -----------------------------------------------------------------------

def test_strict_ok_false_when_a_claim_is_uncited_but_ok_stays_true():
    # Grounding only reads CITED sources, so [S1] is cited later to keep it high.
    text = ("The warranty covers manufacturing defects for twenty four months. "
            "Returns are accepted within thirty days [S1] [S2].")
    r = verify_answer(text, SOURCES)
    assert r.ok and r.grounding_score > 0.9
    assert len(r.uncited_sentences) == 1
    assert not r.strict_ok


def test_strict_ok_true_when_everything_is_cited_and_grounded():
    r = verify_answer("The warranty covers manufacturing defects [S1].", SOURCES)
    assert r.ok and r.strict_ok


def test_strict_ok_false_whenever_ok_is_false():
    r = verify_answer("The warranty covers defects [S3].", SOURCES)
    assert not r.ok and not r.strict_ok


# --- multilingual grounding ----------------------------------------------------------

EN_LAW = [_src(
    "Non-compliance with Article 5 shall be subject to administrative fines of up to "
    "35 000 000 EUR or 7 % of the total worldwide annual turnover.", 1,
)]


def test_french_answer_on_english_source_is_grounded_by_numbers_and_names():
    """Shared figures and names carry a cross-language answer over the threshold.

    Accented words and apostrophe elisions used to break tokenisation and French
    function words counted as ungrounded content, so this scored 0.00 and failed.
    """
    fr = "Selon l’article 5, l’amende est de 35 000 000 EUR ou 7 % [S1]."
    r = verify_answer(fr, EN_LAW)
    assert r.grounding_score > 0.5
    assert r.ok and r.strict_ok


def test_cross_language_answer_still_scores_below_same_language():
    """Grounding compares surface words, so vocabulary that differs is not matched."""
    en = "Under Article 5, the fine is 35 000 000 EUR or 7 % of the turnover [S1]."
    fr = "Selon l’article 5, l’amende est de 35 000 000 EUR ou 7 % du chiffre [S1]."
    assert verify_answer(fr, EN_LAW).grounding_score < verify_answer(en, EN_LAW).grounding_score


def test_french_function_words_are_not_content():
    src = [_src("amende due banques", 1)]
    r = verify_answer("L'amende n'est pas due par les banques et leurs filiales [S1].", src)
    # "filiales" is the only content word missing from the source.
    assert r.grounding_score == pytest.approx(3 / 4)


def test_dutch_function_words_are_not_content():
    src = [_src("systeem toepassing boete", 1)]
    r = verify_answer("Het systeem is niet van toepassing op de boete [S1].", src)
    assert r.grounding_score == 1.0


def test_accented_words_are_single_tokens():
    src = [_src("Les données d'entraînement ont été vérifiées.", 1)]
    r = verify_answer("Les données d’entraînement ont été vérifiées [S1].", src)
    assert r.grounding_score == 1.0


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


def test_abbreviations_do_not_end_a_sentence():
    from rag.verify import split_sentences

    assert split_sentences("Filed in the U.S. District Court. The suit alleges harm.") == [
        "Filed in the U.S. District Court.", "The suit alleges harm."]
    assert split_sentences("Some tiers, e.g. minimal risk, carry no duty. Others do.") == [
        "Some tiers, e.g. minimal risk, carry no duty.", "Others do."]
    assert split_sentences("Apple Inc. is the registrant. It files a 10-K.") == [
        "Apple Inc. is the registrant.", "It files a 10-K."]
    assert split_sentences("Voir art. 5 du règlement. Il interdit ces pratiques.") == [
        "Voir art. 5 du règlement.", "Il interdit ces pratiques."]


def test_abbreviation_inside_markdown_emphasis_is_not_a_split():
    # The case seen in the route-or-roam UI: one uncited sentence was counted twice.
    court = _src("The DOJ filed its lawsuit in the U.S. District Court for the District "
                 "of New Jersey, alleging Apple monopolized performance smartphones.", 3)
    text = ("The DOJ filed its lawsuit in the **U.S. District Court for the District of New "
            "Jersey**.  \nIt alleges that Apple monopolized the markets for performance "
            "smartphones [S1].")
    r = verify_answer(text, [court])
    assert len(r.uncited_sentences) == 1 and r.uncited_sentences[0].startswith("The DOJ filed")
