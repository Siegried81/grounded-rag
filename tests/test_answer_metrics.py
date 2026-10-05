"""Offline tests for rag.answer_metrics and the two answer eval datasets.

The metric tests pin down the measurement definitions (refusal as positive class,
undefined ratios as None, numeric tolerance); the dataset tests guarantee every
label is checkable against the corpus files, so a typo in a key fact or a renamed
section fails here instead of silently lowering the live scores.
"""

import json
from pathlib import Path

import pytest

from rag.answer import REFUSAL_MESSAGE, REFUSAL_MESSAGES
from rag.answer_metrics import (
    JUDGE_SYSTEM,
    build_judge_prompt,
    extract_numbers,
    is_refusal,
    key_fact_present,
    key_fact_recall,
    normalize_text,
    parse_judge_score,
    parse_number,
    refusal_metrics,
    score_answer,
    summarize,
    summarize_by_corpus,
)
from rag.types import Chunk, Retrieved

ROOT = Path(__file__).resolve().parent.parent


def _src(text: str, n: int = 1) -> Retrieved:
    """Build a Retrieved source with the given chunk text."""
    return Retrieved(Chunk(f"c{n}", text, f"doc{n}.txt", n, "corp"), 0.9)


# --- numbers and key facts ------------------------------------------------------

def test_parse_number_grouping_and_decimals():
    assert parse_number("1,234") == 1234
    assert parse_number("1 234") == 1234
    assert parse_number("1 234") == 1234
    assert parse_number("1,234,567.5") == 1234567.5
    assert parse_number("46,9") == 46.9  # French decimal comma
    assert parse_number("2.5") == 2.5


def test_extract_numbers_scale_and_percent():
    nums = extract_numbers("Sales were $416.2 billion, up 6% on September 27, 2025.")
    assert (416.2, 1e9, False) in nums
    assert (6.0, 1.0, True) in nums
    assert (27.0, 1.0, False) in nums and (2025.0, 1.0, False) in nums


def test_digits_glued_to_a_letter_are_not_numbers():
    """Citation markers and labels like Q3 must not satisfy a numeric fact."""
    assert extract_numbers("[S3]") == []
    assert extract_numbers("See [S1] and [S12].") == []
    assert extract_numbers("Q3 results, H1 guidance") == []
    assert not key_fact_present("3", "as stated in [S3]")
    # Currency signs, spaces and operators before a digit still start a number.
    nums = extract_numbers("€500, $2.4 billion and k=5")
    assert (500.0, 1.0, False) in nums and (2.4, 1e9, False) in nums
    assert (5.0, 1.0, False) in nums


@pytest.mark.parametrize("answer", [
    "about 166,000 employees", "about 166 000 employees", "about 166000 employees",
    "about 166 000 employees", "about 167,000 employees",  # within 1%
])
def test_numeric_fact_tolerant_formats(answer):
    assert key_fact_present("166,000", answer)


def test_numeric_fact_outside_tolerance():
    assert not key_fact_present("166,000", "about 170,000 employees")  # 2.4% off
    assert not key_fact_present("166,000", "no figure given")


def test_numeric_fact_scales():
    assert key_fact_present("$416,161 million", "Net sales reached $416.2 billion [S1].")
    assert key_fact_present("$416,161 million", "Net sales were $416,161 million.")
    assert key_fact_present("$416,161 million", "Total net sales $ 416,161 (in millions).")
    assert key_fact_present("€500 million", "a fine of 500 million euros")
    assert not key_fact_present("€500 million", "a fine of 5 billion euros")


def test_scale_word_in_answer_must_match_after_scaling():
    """The unscaled figure only counts when the answer's number has no scale word."""
    assert not key_fact_present("€500 million", "a fine of 500 billion euros")
    assert not key_fact_present("$590 million", "revenue of $590 billion")
    assert key_fact_present("€500 million", "a fine of 500 (in millions of euros)")
    assert key_fact_present("€500 million", "EUR 500")
    assert key_fact_present("€500 million", "0.5 billion euros")


def test_percent_fact_needs_a_percent():
    assert key_fact_present("10%", "fines up to 10 % of worldwide net sales")
    assert key_fact_present("10%", "fines up to 10 percent of sales")
    assert not key_fact_present("10%", "within 10 days")


def test_text_fact_normalisation_and_word_boundaries():
    assert normalize_text("Données d'Entraînement!") == "donnees d entrainement"
    assert key_fact_present("santé", "IA en SANTE et en crédit")
    assert key_fact_present("2 août 2026", "le 2 aout 2026, l'Article 50 s'applique")
    assert key_fact_present("2024/1689", "Règlement (EU) 2024/1689")
    assert not key_fact_present("Mac", "machine learning only")


def test_fact_alternatives():
    fact = ["2 déc. 2027", "2 décembre 2027"]
    assert key_fact_present(fact, "reporté au 2 décembre 2027")
    assert key_fact_present(fact, "reporté au 2 déc. 2027")
    assert not key_fact_present(fact, "reporté à 2028")


def test_key_fact_recall():
    assert key_fact_recall(["Apple Card", "Apple Pay"], "Apple Pay only") == 0.5
    assert key_fact_recall(["Apple Card", "Apple Pay"], "Apple Card and Apple Pay") == 1.0
    assert key_fact_recall([], "anything") is None


# --- refusal detection ------------------------------------------------------------

def test_refusal_gate_and_exact_message():
    assert is_refusal("whatever", gate_refused=True)
    assert is_refusal(REFUSAL_MESSAGE)
    assert is_refusal("")


@pytest.mark.parametrize("lang", sorted(REFUSAL_MESSAGES))
def test_exact_refusal_message_in_any_language(lang):
    assert is_refusal(REFUSAL_MESSAGES[lang])
    assert is_refusal(f"  {REFUSAL_MESSAGES[lang]}\n")


@pytest.mark.parametrize("text", [
    "I do not know based on the provided sources.",
    "I don't know.",
    "The sources do not mention Apple's net income.",
    "Je ne sais pas.",
    "Les sources fournies ne précisent pas le montant des amendes.",
    "Les sources sont insuffisantes pour répondre.",
    "Based on the context, it is not stated. The documents do not specify a figure.",
    "No lo sé.",
    "No puedo responder a esa pregunta con las fuentes dadas.",
    "No tengo información sobre eso.",
])
def test_model_worded_refusals(text):
    assert is_refusal(text)


@pytest.mark.parametrize("text", [
    "Apple has five segments: Americas, Europe, Greater China, Japan, Rest of Asia Pacific [S1].",
    # A later hedge does not turn a cited answer into a refusal.
    "Le report vise les systèmes Annexe III [S1]. Les sources ne précisent pas le reste.",
    # A negation about the subject (not the sources) is content, not a refusal.
    "Spam filters fall under minimal risk and the AI Act does not mention specific duties [S1].",
    # A hedge in the first sentence followed by a cited claim is an answer: on an
    # unanswerable question it must be scored as a hallucination, not a refusal.
    "Je ne sais pas exactement, mais l'amende est de 35 millions [S1].",
    "I don't know for sure, but the fine is 35 million [S1].",
    # Even a refusal-worded sentence is a claim once it cites a source.
    "Les sources fournies ne précisent pas le montant des amendes [S2].",
])
def test_answers_are_not_refusals(text):
    assert not is_refusal(text)


def test_uncited_answer_with_later_refusal_phrase_is_a_refusal():
    assert is_refusal("Apple discloses many risks. The sources do not mention the figure.")


def test_cited_hedge_counts_as_answered_on_unanswerable_question():
    item = {"id": "u-01", "answerable": False, "key_facts": []}
    text = "Je ne sais pas exactement, mais l'amende est de 35 millions [S1]."
    row = score_answer(item, text, False, SOURCES)
    assert row["refused"] is False and row["valid_citations"] == [1]
    assert refusal_metrics([row])["refusal_recall"] == 0.0


# --- refusal metrics -------------------------------------------------------------

def _row(answerable, refused, **kw):
    return {"answerable": answerable, "refused": refused, **kw}


def test_refusal_metrics_math():
    rows = [
        _row(False, True), _row(False, True), _row(False, False),  # tp, tp, fn
        _row(True, True), _row(True, False), _row(True, False), _row(True, False),  # fp, tn x3
    ]
    m = refusal_metrics(rows)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 3)
    assert m["refusal_precision"] == pytest.approx(2 / 3)
    assert m["refusal_recall"] == pytest.approx(2 / 3)
    assert m["refusal_f1"] == pytest.approx(2 / 3)
    assert m["false_refusal_rate"] == pytest.approx(1 / 4)


def test_refusal_metrics_undefined_ratios_are_none():
    m = refusal_metrics([_row(True, False)])  # nothing refused, nothing to refuse
    assert m["refusal_precision"] is None and m["refusal_recall"] is None
    assert m["refusal_f1"] is None
    assert m["false_refusal_rate"] == 0.0
    assert refusal_metrics([])["false_refusal_rate"] is None


def test_refusal_metrics_zero_f1():
    m = refusal_metrics([_row(False, False), _row(True, True)])
    assert m["refusal_precision"] == 0.0 and m["refusal_recall"] == 0.0
    assert m["refusal_f1"] == 0.0
    assert m["false_refusal_rate"] == 1.0


# --- scoring and aggregation ---------------------------------------------------------

ITEM = {"id": "x-01", "answerable": True, "key_facts": ["Apple Card", "Apple Pay"]}
SOURCES = [_src("Apple offers Apple Card, a credit card, and Apple Pay, a payment service.")]


def test_score_answer_uses_verify():
    row = score_answer(ITEM, "Apple offers Apple Card and Apple Pay [S1] [S4].", False, SOURCES)
    assert row["refused"] is False and row["key_fact_recall"] == 1.0
    assert row["valid_citations"] == [1] and row["invalid_citations"] == [4]
    assert row["verify_ok"] is False  # a hallucinated [S4] fails verification
    assert 0.0 < row["grounding_score"] <= 1.0


def test_score_answer_refusal_skips_verify_and_scores_zero_facts():
    row = score_answer(ITEM, REFUSAL_MESSAGE, True, [])
    assert row["refused"] and row["key_fact_recall"] == 0.0 and row["facts_found"] == []
    assert row["grounding_score"] is None and row["valid_citations"] is None


def test_score_answer_unanswerable_has_no_fact_recall():
    item = {"id": "x-02", "answerable": False, "key_facts": []}
    row = score_answer(item, "It is 42 [S1].", False, SOURCES)
    assert row["key_fact_recall"] is None and row["refused"] is False


def _scored(answerable, refused, kfr=None, valid=(), invalid=(), grounding=None, ok=None, **kw):
    return {"answerable": answerable, "refused": refused, "key_fact_recall": kfr,
            "valid_citations": None if refused else list(valid),
            "invalid_citations": None if refused else list(invalid),
            "grounding_score": grounding, "verify_ok": ok, **kw}


def test_summarize_aggregates():
    rows = [
        _scored(True, False, 1.0, valid=[1, 2], grounding=0.8, ok=True, judge_score=2),
        _scored(True, False, 0.5, valid=[1], invalid=[3], grounding=0.4, ok=False,
                judge_score=None),
        _scored(True, True, 0.0),
        _scored(False, True),
        _scored(False, False, valid=[1], grounding=0.6, ok=True, judge_score=0),
        {"answerable": True, "refused": False, "error": "LLM down"},
    ]
    s = summarize(rows)
    assert s["n"] == 5 and s["n_errors"] == 1
    assert (s["n_answerable"], s["n_unanswerable"]) == (3, 2)
    assert s["key_fact_recall"] == pytest.approx(0.5)  # (1 + 0.5 + 0) / 3
    assert s["key_fact_recall_answered"] == pytest.approx(0.75)
    assert s["citation_validity"] == pytest.approx(4 / 5)  # pooled over citations
    assert s["mean_grounding"] == pytest.approx(0.6)
    assert s["verify_ok_rate"] == pytest.approx(2 / 3)
    assert s["refusal_recall"] == 0.5 and s["false_refusal_rate"] == pytest.approx(1 / 3)
    assert s["judge_mean"] == 1.0 and s["judge_unparsed"] == 1


def test_summarize_empty_is_all_none():
    s = summarize([])
    assert s["n"] == 0
    for key in ("key_fact_recall", "citation_validity", "mean_grounding", "judge_mean"):
        assert s[key] is None


def test_summarize_by_corpus():
    rows = [_scored(True, False, 1.0, corpus="a"), _scored(False, True, corpus="b")]
    out = summarize_by_corpus(rows)
    assert list(out) == ["a", "b", "all"]
    assert out["a"]["n"] == 1 and out["all"]["n"] == 2
    assert list(summarize_by_corpus(rows[:1])) == ["a"]  # no "all" for one corpus


# --- judge -------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ('{"score": 2, "reason": "ok"}', 2),
    ('```json\n{"score": 1, "reason": "partly"}\n```', 1),
    ('Here is my grade: {"score": "0", "reason": "no"} thanks', 0),
    ('{"score": 2.0}', 2),
    ('{"score": 3}', None),
    ('{"score": -1}', None),
    ('{"score": true}', None),
    ('{"score": "high"}', None),
    ('{"reason": "missing"}', None),
    ("score: 2", None),
    ("{not json}", None),
    ("", None),
    (None, None),
])
def test_parse_judge_score(raw, expected):
    assert parse_judge_score(raw) == expected


def test_build_judge_prompt_numbers_sources():
    prompt = build_judge_prompt("Q?", "A [S1].", [_src("alpha", 1), _src("beta", 2)])
    for part in ("[S1] (doc1.txt)", "alpha", "[S2] (doc2.txt)", "beta", "Q?", "A [S1]."):
        assert part in prompt
    assert '"score"' in JUDGE_SYSTEM


# --- datasets ------------------------------------------------------------------------

DATASETS = ["ai_act_sections", "filings_sections"]
FIELDS = {"id", "question", "answerable", "gold_sources", "key_facts", "note"}


def _load(corpus):
    path = ROOT / "eval" / f"{corpus}_answers_eval.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


@pytest.mark.parametrize("corpus", DATASETS)
def test_dataset_schema_and_composition(corpus):
    items = _load(corpus)
    for it in items:
        assert set(it) == FIELDS, it["id"]
        assert isinstance(it["answerable"], bool)
        assert it["question"].strip() and it["note"].strip()
    answerable = [it for it in items if it["answerable"]]
    unanswerable = [it for it in items if not it["answerable"]]
    assert len(answerable) >= 12 and len(unanswerable) >= 5


def test_dataset_ids_unique_across_files():
    ids = [it["id"] for c in DATASETS for it in _load(c)]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("corpus", DATASETS)
def test_unanswerable_entries_have_no_labels(corpus):
    for it in _load(corpus):
        if not it["answerable"]:
            assert it["key_facts"] == [] and it["gold_sources"] == [], it["id"]


@pytest.mark.parametrize("corpus", DATASETS)
def test_answerable_facts_exist_in_gold_sources(corpus):
    """Each fact's canonical (first) spelling must be found in one of its gold sections."""
    for it in _load(corpus):
        if not it["answerable"]:
            continue
        assert it["gold_sources"], it["id"]
        assert 1 <= len(it["key_facts"]) <= 3, it["id"]
        texts = []
        for src in it["gold_sources"]:
            path = ROOT / "data" / corpus / src
            assert path.is_file(), f"{it['id']}: missing gold source {src}"
            texts.append(path.read_text(encoding="utf-8"))
        for fact in it["key_facts"]:
            canonical = fact if isinstance(fact, str) else fact[0]
            assert any(key_fact_present(canonical, t) for t in texts), f"{it['id']}: {fact}"
