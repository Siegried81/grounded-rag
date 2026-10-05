"""Offline tests for rag.ui_helpers: highlighting, labels, examples, explanations."""

import json

from rag import ui_helpers as ui
from rag.types import Chunk


def _chunk(text: str, source: str = "doc.txt", page=None) -> Chunk:
    """Build a Chunk with an optional page number in its meta."""
    return Chunk(f"c:{source}:{text[:5]}", text, source, 0, "corp", {"page": page} if page else {})


def test_query_terms_drops_stopwords_short_words_and_duplicates():
    assert ui.query_terms("What are the risks of the iPhone, the iPhone risks?") == ["risks", "iphone"]
    assert ui.query_terms("Quels sont les niveaux de risque ?") == ["niveaux", "risque"]


def test_highlight_terms_marks_whole_words_case_insensitive():
    out = ui.highlight_terms("Risk and risky RISK", ["risk"])
    assert out == "<mark>Risk</mark> and risky <mark>RISK</mark>"


def test_highlight_terms_escapes_html_before_marking():
    out = ui.highlight_terms("<script>x</script> santé", ["santé"])
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "<mark>santé</mark>" in out


def test_highlight_terms_without_terms_only_escapes():
    assert ui.highlight_terms("a < b", []) == "a &lt; b"


# --- highlight_and_linkify ------------------------------------------------------
# What the two UIs actually render for a passage or a document: the same escaping
# and <mark> tags as highlight_terms, plus real links on bare URLs so a passage
# that names its source can be followed.


def test_linkify_makes_bare_url_a_link_and_still_highlights():
    out = ui.highlight_and_linkify("Risk rules at https://eur-lex.europa.eu/ai", ["risk"])
    assert "<mark>Risk</mark> rules at " in out
    assert '<a href="https://eur-lex.europa.eu/ai" target="_blank" rel="noopener noreferrer">' in out
    assert out.endswith("https://eur-lex.europa.eu/ai</a>")


def test_linkify_leaves_text_without_url_identical_to_highlight_terms():
    text = "Risk tiers and <b>obligations</b>"
    assert ui.highlight_and_linkify(text, ["risk"]) == ui.highlight_terms(text, ["risk"])


def test_linkify_drops_sentence_punctuation_from_the_url():
    out = ui.highlight_and_linkify("See http://x.eu/page. Next.", [])
    assert '<a href="http://x.eu/page"' in out
    assert out.endswith("http://x.eu/page</a>. Next.")


def test_linkify_never_marks_inside_an_href():
    """A query term appearing in the URL must not put a <mark> tag in the href."""
    out = ui.highlight_and_linkify("https://europa.eu/risk-tiers", ["risk"])
    assert "<mark>" not in out
    assert out == (
        '<a href="https://europa.eu/risk-tiers" target="_blank" '
        'rel="noopener noreferrer">https://europa.eu/risk-tiers</a>'
    )


def test_linkify_only_links_http_schemes():
    """javascript:, data: and file: are not linked, so no passage can hand the
    reader a scripted target; they are escaped as plain text."""
    text = "javascript:alert(1) data:text/html,x file:///etc/passwd ftp://h/x"
    out = ui.highlight_and_linkify(text, [])
    assert "<a " not in out and out == text


def test_linkify_escapes_quotes_so_a_url_cannot_break_out_of_the_attribute():
    out = ui.highlight_and_linkify('http://x.eu/a"onmouseover=alert(1)', [])
    assert '"onmouseover' not in out
    assert "&quot;onmouseover" in out


def test_linkify_handles_several_urls_and_escapes_between_them():
    out = ui.highlight_and_linkify("http://a.eu <b> http://b.eu", [])
    assert out.count("<a href=") == 2
    assert "&lt;b&gt;" in out


def test_snippet_short_text_unchanged_and_whitespace_collapsed():
    assert ui.snippet("one  two\nthree", ["two"]) == "one two three"


def test_snippet_centres_on_first_match_with_ellipses():
    text = "filler " * 100 + "TARGET word here " + "tail " * 100
    out = ui.snippet(text, ["target"], max_chars=120)
    assert "TARGET" in out
    assert out.startswith("...") and out.endswith("...")
    assert len(out) <= 120 + 6


def test_snippet_without_match_starts_at_beginning():
    out = ui.snippet("abc " * 200, ["zzz"], max_chars=50)
    assert out.startswith("abc") and out.endswith("...")


def test_format_location_with_and_without_page():
    assert ui.format_location("a.pdf", {"page": 7}) == "a.pdf · p.7"
    assert ui.format_location("a.txt", {}) == "a.txt"
    assert ui.format_location("a.txt", None) == "a.txt"


def test_style_citations_marks_valid_and_invalid_and_escapes():
    out = ui.style_citations("Yes [S1] <b>no</b> [S3].", n_sources=2)
    assert '<span class="cite">S1</span>' in out
    assert '<span class="cite-bad">S3</span>' in out
    assert "<b>" not in out


def test_example_questions_spread_evenly(tmp_path):
    path = tmp_path / "c_eval.jsonl"
    rows = [json.dumps({"question": f"Q{i}", "relevant_sources": []}) for i in range(8)]
    path.write_text("\n".join(rows) + "\n\nnot json\n", encoding="utf-8")
    assert ui.example_questions(path, n=4) == ["Q0", "Q2", "Q4", "Q6"]


def test_example_questions_short_or_missing_file(tmp_path):
    path = tmp_path / "c_eval.jsonl"
    path.write_text(json.dumps({"question": "Only"}), encoding="utf-8")
    assert ui.example_questions(path, n=4) == ["Only"]
    assert ui.example_questions(tmp_path / "missing.jsonl") == []


def test_example_questions_shipped_eval_files_load():
    from config import ROOT

    for name in ("ai_act", "ai_act_sections", "filings_sections"):
        qs = ui.example_questions(ROOT / "eval" / f"{name}_eval.jsonl")
        assert len(qs) == 4 and all(isinstance(q, str) and q for q in qs)


def test_detect_language():
    assert ui.detect_language("Les obligations de la loi sont dans le texte et les annexes.") == "French"
    assert ui.detect_language("The obligations of the act are in the text and the annexes.") == "English"
    assert ui.detect_language("12345") == "unknown"


def test_corpus_stats_counts_documents_pages_languages():
    chunks = [
        _chunk("The company reports risks in the filing.", "a.pdf", page=1),
        _chunk("The company reports more of the same.", "a.pdf", page=2),
        _chunk("Les risques sont dans le rapport de la société.", "b.txt"),
    ]
    stats = ui.corpus_stats(chunks)
    assert stats["documents"] == 2
    assert stats["chunks"] == 3
    assert stats["pages"] == 2
    assert stats["languages"] == ["English", "French"]
    assert stats["sources"] == ["a.pdf", "b.txt"]


def test_explain_refusal_quotes_scores_and_distinguishes_near_miss():
    near = ui.explain_refusal(0.33, 0.35)
    far = ui.explain_refusal(0.10, 0.35)
    assert "0.33" in near and "0.35" in near and "rephrasing" in near
    assert "does not cover" in far
    assert "no passages" in ui.explain_refusal(None, 0.35)


def test_explain_grounding_states_verdict():
    assert "passes the 30%" in ui.explain_grounding(0.8, 0.3, True)
    assert "does not pass" in ui.explain_grounding(0.1, 0.3, False)


def test_build_limitations_uses_live_settings():
    items = ui.build_limitations(0.42, 0.3, has_bm25=True)
    text = " ".join(items)
    assert "0.42" in text and "30%" in text
    assert "BM25" in text
    assert "multi-hop" in text and "scanned" in text and "re-ranker" in text
    assert "BM25" not in " ".join(ui.build_limitations(0.42, 0.3, has_bm25=False))


def test_llm_error_hint_names_env_vars():
    hint = ui.llm_error_hint(
        "All LLM providers failed: groq: no API key; openrouter: no API key; "
        "ollama: ConnectionError", "groq",
    )
    assert "GROQ_API_KEY" in hint and "OPENROUTER_API_KEY" in hint and "OLLAMA_URL" in hint


def test_llm_error_hint_rate_limit_and_fallback():
    assert "rate limit" in ui.llm_error_hint("groq: 429 Too Many Requests", "groq")
    assert "LLM_PROVIDER" in ui.llm_error_hint("something odd", "groq")


def test_embed_error_hint_per_provider():
    assert "ollama serve" in ui.embed_error_hint("ollama")
    assert "HOSTED_EMBED_" in ui.embed_error_hint("hosted")


def test_localized_examples_translate_known_and_keep_unknown(tmp_path):
    fr_q = "Quels sont les quatre niveaux de risque définis par l'AI Act ?"
    rows = [json.dumps({"question": q}) for q in (fr_q, "Untranslated question?")]
    path = tmp_path / "x_eval.jsonl"
    path.write_text("\n".join(rows), encoding="utf-8")
    assert ui.localized_examples(path, "en") == [
        "What are the four risk levels defined by the AI Act?", "Untranslated question?",
    ]
    assert ui.localized_examples(path, "fr") == [fr_q, "Untranslated question?"]
    assert ui.localized_examples(tmp_path / "missing.jsonl", "en") == []


def test_french_ui_texts_mirror_english():
    assert len(ui.pipeline_steps("fr")) == len(ui.pipeline_steps("en")) == len(ui.PIPELINE_STEPS)
    assert ui.pipeline_steps("xx") == ui.PIPELINE_STEPS
    fr = ui.build_limitations(0.42, 0.3, has_bm25=True, lang="fr")
    assert len(fr) == len(ui.build_limitations(0.42, 0.3, has_bm25=True))
    assert "0.42" in " ".join(fr) and "BM25" in " ".join(fr)
    assert "BM25" not in " ".join(ui.build_limitations(0.42, 0.3, has_bm25=False, lang="fr"))
    assert "seuil de 0.35" in ui.explain_refusal(0.10, 0.35, lang="fr")
    assert "proche" in ui.explain_refusal(0.33, 0.35, lang="fr")
    assert "aucun passage" in ui.explain_refusal(None, 0.35, lang="fr")
    assert "n'atteint pas" in ui.explain_grounding(0.1, 0.3, False, lang="fr")
    assert "atteint le seuil de 30%" in ui.explain_grounding(0.8, 0.3, True, lang="fr")
