"""Offline tests for rag.verify: citation parsing, validity, grounding, uncited claims."""

import pytest

import config
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


def test_parenthesised_citations_are_recognised():
    """gpt-oss also cites as "(S9)" (route-or-roam agent sh08, 14 October run)."""
    from rag.answer import normalize_citations

    assert normalize_citations("Les pratiques prohibées (S9) et le texte (S1).") == \
        "Les pratiques prohibées [S9] et le texte [S1]."
    assert normalize_citations("Both (**S1**, S5) and （S2）.") == "Both [S1][S5] and [S2]."
    assert extract_citations("Net sales rose (S2).") == [2]
    # Ordinary parenthesised text and mismatched pairs are left alone.
    for text in ("Article 5 (see S1 above)", "(S1]", "[S1)", "Form (S-1) filing", "(III)"):
        assert normalize_citations(text) == text


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


# --- figures and negations against the sources a sentence cites ---------------
#
# A valid citation beside a wrong number used to pass: "999 hours [S1]" against a
# source saying 24 hours scored above the grounding threshold. Figures are the
# one thing a lexical check can judge with confidence, so they gate `ok`.
# Negation is a heuristic and gates `strict_ok` only.

NUMERIC_SOURCES = [
    _src("The seller must respond within 24 hours and refund up to 1,000 euros.", 1),
    _src("Article 5 applies to systems placed on the market after 2 August 2026.", 2),
]


def test_a_figure_the_cited_source_does_not_carry_fails_the_answer():
    r = verify_answer("The seller must respond within 999 hours [S1].", NUMERIC_SOURCES)
    assert r.unsupported_numbers == ["999"]
    assert not r.ok


def test_figures_present_in_the_cited_source_pass_whatever_the_separators():
    r = verify_answer("The seller must respond within 24 hours and refund up to 1000 euros [S1].", NUMERIC_SOURCES)
    assert r.unsupported_numbers == []
    assert r.ok


def test_a_whole_number_keeps_its_trailing_zeros():
    """240 is not 24 and 10 is not 1,000: only a decimal part loses its zeros."""
    for wrong in ("within 240 hours", "up to 10 euros", "up to 100 euros"):
        r = verify_answer(f"The seller must respond {wrong} [S1].", NUMERIC_SOURCES)
        assert r.unsupported_numbers, wrong
        assert not r.ok


def test_a_decimal_still_matches_its_shorter_spelling():
    sources = [_src("Revenue rose by 2.50 percent over the fiscal year.", 1)]
    r = verify_answer("Revenue rose by 2.5 percent over the fiscal year [S1].", sources)
    assert r.unsupported_numbers == []


@pytest.mark.parametrize("sep", [" ", " ", " ", ","])
def test_thousands_separated_by_any_space_read_as_one_figure(sep):
    """EUR-Lex writes 35 000 000 with a narrow no-break space (U+202F)."""
    sources = [_src(f"Fines of up to EUR 35{sep}000{sep}000 may be imposed on operators.", 1)]
    r = verify_answer("Fines of up to EUR 35000000 may be imposed on operators [S1].", sources)
    assert r.unsupported_numbers == []
    r = verify_answer("Fines of up to EUR 35 may be imposed on operators [S1].", sources)
    assert r.unsupported_numbers == ["35"]


@pytest.mark.parametrize("answer_figure", ["EUR 15 million", "EUR 15 million", "EUR 15 000 000"])
def test_a_scale_word_matches_the_spelled_out_figure(answer_figure):
    """aa-04: the answer says "15 million", Article 99 says "15 000 000"."""
    sources = [_src("Fines of up to EUR 15 000 000 may be imposed on operators.", 1)]
    r = verify_answer(f"Fines of up to {answer_figure} may be imposed on operators [S1].", sources)
    assert r.unsupported_numbers == []


def test_a_wrong_scale_is_still_caught():
    sources = [_src("Fines of up to EUR 15 000 000 may be imposed on operators.", 1)]
    r = verify_answer("Fines of up to EUR 15 billion may be imposed on operators [S1].", sources)
    assert r.unsupported_numbers == ["15000000000"]


def test_a_scaled_figure_matches_a_table_stated_in_that_unit():
    """fs-12: "$416,161 million" against a 10-K table in millions ("416,161")."""
    sources = [_src("Total net sales were $ 416,161 and $ 391,035 (in millions).", 1)]
    r = verify_answer("Net sales reached $416,161 million, up from $391,035 million [S1].", sources)
    assert r.unsupported_numbers == []


def test_a_figure_is_checked_against_the_sentence_s_own_citation_not_the_union():
    """24 is in S1; a sentence that attaches it to S2 is still wrong."""
    r = verify_answer("Article 5 applies after 24 hours [S2].", NUMERIC_SOURCES)
    assert r.unsupported_numbers == ["24"]


def test_a_citation_marker_is_not_a_figure():
    r = verify_answer("The seller must respond within 24 hours [S1].", NUMERIC_SOURCES)
    assert "1" not in r.unsupported_numbers


def test_a_negation_the_source_does_not_carry_is_reported_and_blocks_strict_only():
    r = verify_answer("The seller must not respond within 24 hours [S1].", NUMERIC_SOURCES)
    assert r.negation_mismatches and r.ok and not r.strict_ok


def test_the_report_names_the_figures_and_the_negation():
    r = verify_answer("The seller must not respond within 999 hours [S1].", NUMERIC_SOURCES)
    text = format_report(r)
    assert "figures absent" in text and "999" in text and "negation" in text


def test_a_citation_written_after_the_full_stop_belongs_to_that_sentence():
    """gpt-oss often writes "... hours. [S1]": the marker must not become its own sentence."""
    from rag.verify import split_sentences

    assert split_sentences("Refunds take 24 hours. [S1] Logs are kept. [S2][S1]") == [
        "Refunds take 24 hours. [S1]",
        "Logs are kept. [S2][S1]",
    ]


def test_a_fabricated_figure_with_a_trailing_citation_still_fails():
    r = verify_answer("The seller must respond within 999 hours. [S1]", NUMERIC_SOURCES)
    assert r.unsupported_numbers == ["999"]
    assert not r.ok
    assert r.uncited_sentences == []


def test_a_citation_alone_on_the_next_line_belongs_to_the_line_above():
    r = verify_answer("The seller must respond within 999 hours.\n[S1]", NUMERIC_SOURCES)
    assert r.unsupported_numbers == ["999"] and not r.ok


# --- regressions: negation contractions, year/count, decimals ------------------

@pytest.mark.parametrize("claim", [
    "The warranty doesn't apply to manufacturing defects",
    "The warranty doesn\u2019t apply to manufacturing defects",
    "Providers can't ignore the manufacturing defects",
])
def test_a_contraction_counts_as_a_negation(claim):
    """`n't` sat inside a \b...\b group, so it never matched anything."""
    r = verify_answer(f"{claim} [S1].", SOURCES)
    assert r.negation_mismatches
    assert not r.strict_ok


def test_a_contraction_in_the_source_clears_the_mismatch():
    sources = [_src("The warranty doesn't apply to manufacturing defects in any case.", 1)]
    r = verify_answer("The warranty doesn't apply to manufacturing defects [S1].", sources)
    assert r.negation_mismatches == []


def test_a_year_followed_by_a_three_digit_count_is_two_figures():
    """"2024 100" used to merge into 2024100, so neither figure was ever found."""
    sources = [_src("In 2024 100 providers registered their systems with the authority.", 1)]
    r = verify_answer("In 2024 100 providers registered their systems with the authority [S1].", sources)
    assert r.unsupported_numbers == []
    r = verify_answer("In 2024100 providers registered their systems with the authority [S1].", sources)
    assert r.unsupported_numbers == ["2024100"]


@pytest.mark.parametrize("answer_figure,source_figure", [("0.35", "0.350"), ("0.350", "0.35"), ("0,35", "0.350")])
def test_a_decimal_with_a_trailing_zero_matches_its_short_spelling(answer_figure, source_figure):
    """"0.350" was read as 350 thousandths merged into 0350, so 0.35 was unsupported."""
    sources = [_src(f"The measured ratio of defects was {source_figure} across the whole sample.", 1)]
    r = verify_answer(f"The measured ratio of defects was {answer_figure} across the whole sample [S1].", sources)
    assert r.unsupported_numbers == []


def test_thousands_after_a_leading_zero_group_still_read_as_decimals_only():
    sources = [_src("The ratio was 0.350 and the count was 12 000 units in total.", 1)]
    assert verify_answer("The count was 12000 units in total [S1].", sources).unsupported_numbers == []
    assert verify_answer("The ratio was 350 units in total [S1].", sources).unsupported_numbers == ["350"]


def test_the_default_grounding_bar_is_the_one_in_config(monkeypatch):
    """verify_answer used to default to 0.5 while the app runs at config.VERIFY_MIN_GROUNDING."""
    sources = [_src("alpha beta gamma delta epsilon zeta eta theta", 1)]
    text = "alpha beta gamma delta zzzone zzztwo zzzthree zzzfour [S1]."
    monkeypatch.setattr(config, "VERIFY_MIN_GROUNDING", 0.30)
    r = verify_answer(text, sources)
    assert r.grounding_score == 0.5 and r.ok
    monkeypatch.setattr(config, "VERIFY_MIN_GROUNDING", 0.60)
    assert not verify_answer(text, sources).ok


def test_numbers_helper_is_the_contract_route_or_roam_imports():
    """route-or-roam imports rag.verify._numbers to check figures in its own answers.

    It takes a text and returns the set of every spelling of every figure in it.
    Renaming it, changing the type or dropping the scale-word spellings breaks
    that consumer.
    """
    from rag.verify import _numbers

    assert _numbers("Fines up to EUR 35 000 000, or 7 % of turnover.") == {"35000000", "7"}
    assert _numbers("Net sales were $416,161 million; 2.50 percent; no figure here") == {
        "416161", "416161000000", "2.5"}
    assert _numbers("no figure at all") == set()
    assert isinstance(_numbers("1"), set)
