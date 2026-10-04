"""Pure presentation helpers for the Streamlit UI.

Everything the UI needs to *decide* what to show (which words to highlight, how to
label a source, which example questions to offer, why an answer was refused, what
to tell the user when no LLM is reachable) lives here as plain functions over plain
data. Keeping them free of Streamlit and of any network call is what lets them be
unit-tested offline, and keeps app.py a thin layout file.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

_CITATION_RE = re.compile(r"\[S(\d+)\]")
# Unicode-aware word pattern so accented French terms ("santé") stay whole.
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_MIN_TERM_LEN = 3  # shorter tokens ("de", "of", "AI") would light up half the text

# Function words for the two languages the shipped corpora use. They are dropped
# from highlighting and double as the signal for the language heuristic below.
_EN_STOPWORDS = frozenset(
    "the and or but of to in on at by for with from as is are was were be been "
    "it its this that these those which who what when where how not do does did "
    "has have had can could will would should may might also than then there their "
    "they into about over under any all".split()
)
_FR_STOPWORDS = frozenset(
    "le la les un une des du de et ou mais au aux en dans par pour avec sur sous "
    "est sont été être qui que quoi quel quelle quels quelles dont ce cette ces "
    "il elle ils elles leur leurs pas ne plus comme aussi entre vers chez lors "
    "son sa ses qu".split()
)
_STOPWORDS = _EN_STOPWORDS | _FR_STOPWORDS

PIPELINE_STEPS = [
    "**Chunk** - documents are split into sentence-aligned passages (PDFs page by page, "
    "so a citation can point to a page).",
    "**Retrieve** - the question is embedded and compared to every passage (cosine); "
    "in hybrid mode a BM25 keyword ranking is fused in (RRF) and MMR drops near-duplicates.",
    "**Gate** - if no passage clears the dense cosine threshold, the assistant refuses "
    "without calling the LLM.",
    "**Answer** - one single LLM call sees only the numbered passages and must cite "
    "them inline as [S1], [S2]...",
    "**Verify** - an offline check confirms every [S#] points to a real source, flags "
    "uncited claims and measures lexical grounding.",
]

PIPELINE_STEPS_FR = [
    "**Découpage** - les documents sont découpés en passages alignés sur les phrases "
    "(les PDF page par page, pour qu'une citation puisse renvoyer à une page).",
    "**Recherche** - la question est vectorisée et comparée à chaque passage (cosinus) ; "
    "en mode hybride, un classement par mots-clés BM25 est fusionné (RRF) et MMR écarte "
    "les quasi-doublons.",
    "**Filtre** - si aucun passage ne dépasse le seuil de similarité cosinus, l'assistant "
    "refuse sans appeler le LLM.",
    "**Réponse** - un seul appel au LLM, qui ne voit que les passages numérotés et doit "
    "les citer dans le texte sous la forme [S1], [S2]...",
    "**Vérification** - un contrôle hors ligne confirme que chaque [S#] renvoie à une vraie "
    "source, signale les affirmations non citées et mesure l'ancrage lexical.",
]

# English renderings of the French example questions picked from the ai_act and
# ai_act_sections eval files. The eval files stay French (they are what the
# retrieval and answer evaluations measure); only the questions shown as clickable
# examples are translated, so an English-speaking user sees what they can ask.
# Keyed by the exact French text: a question added to an eval file later simply
# shows untranslated until a translation is added here.
EXAMPLE_TRANSLATIONS_EN = {
    "Quels sont les quatre niveaux de risque définis par l'AI Act ?":
        "What are the four risk levels defined by the AI Act?",
    "Quelles obligations s'appliquent aux systèmes d'IA à haut risque ?":
        "What obligations apply to high-risk AI systems?",
    "Qu'impose l'Article 50 en matière de transparence ?":
        "What does Article 50 require in terms of transparency?",
    "Pourquoi les obligations haut risque ont-elles été reportées à décembre 2027 ?":
        "Why were the high-risk obligations postponed to December 2027?",
    "La reconnaissance des émotions au travail ou à l'école est-elle permise ?":
        "Is emotion recognition at work or at school allowed?",
    "Qu'impose l'Article 50 à un outil qui génère du contenu visible par l'utilisateur ?":
        "What does Article 50 require of a tool that generates content visible to the user?",
    "Que se passe-t-il le 2 décembre 2026 pour les systèmes déjà sur le marché ?":
        "What happens on 2 December 2026 to systems already on the market?",
}


def pipeline_steps(lang: str = "en") -> list[str]:
    """The "how it works" steps in the requested UI language (English fallback)."""
    return list(PIPELINE_STEPS_FR if lang == "fr" else PIPELINE_STEPS)


def query_terms(question: str) -> list[str]:
    """Return the distinct, lowercased content words of a question, in order.

    Stopwords and very short tokens are dropped so highlighting marks the words
    that drove retrieval rather than every "the" or "de" in a passage.
    """
    seen: list[str] = []
    for w in _WORD_RE.findall(question.lower()):
        if len(w) >= _MIN_TERM_LEN and w not in _STOPWORDS and w not in seen:
            seen.append(w)
    return seen


def highlight_terms(text: str, terms: list[str]) -> str:
    """HTML-escape `text` and wrap whole-word, case-insensitive matches in <mark>.

    Escaping first means passage text can never inject markup into the page; the
    only HTML in the output is the <mark> tags added here.
    """
    escaped = html.escape(text)
    if not terms:
        return escaped
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(html.escape(t)) for t in terms) + r")\b",
        re.IGNORECASE,
    )
    return pattern.sub(r"<mark>\1</mark>", escaped)


def snippet(text: str, terms: list[str], max_chars: int = 420) -> str:
    """Return a window of at most `max_chars` around the first query-term match.

    Passages can be ~900 characters; showing the part that matched keeps the
    sources list scannable while the full passage stays one click away. Ellipses
    mark where the window was cut.
    """
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    lowered = text.lower()
    hits = [lowered.find(t) for t in terms if lowered.find(t) >= 0]
    first = min(hits) if hits else 0
    start = max(0, min(first - max_chars // 4, len(text) - max_chars))
    end = start + max_chars
    return ("..." if start > 0 else "") + text[start:end].strip() + ("..." if end < len(text) else "")


def format_location(source: str, meta: dict | None) -> str:
    """Label a passage as "file · p.N" (page only when the chunk carries one)."""
    page = (meta or {}).get("page")
    return f"{source} · p.{page}" if page else source


def style_citations(answer_text: str, n_sources: int) -> str:
    """Escape answer HTML and turn each [S#] into a styled inline badge.

    Valid markers get the `cite` class; markers pointing past the source list (a
    hallucinated reference) get `cite-bad`, so the problem is visible in the text
    itself and not only in the verification panel. Markdown is left intact.
    """
    escaped = html.escape(answer_text, quote=False)

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        cls = "cite" if 1 <= n <= n_sources else "cite-bad"
        return f'<span class="{cls}">S{n}</span>'

    return _CITATION_RE.sub(repl, escaped)


def example_questions(eval_path: Path, n: int = 4) -> list[str]:
    """Pick `n` example questions spread evenly through a corpus's eval file.

    The eval sets are grouped by section, so taking evenly spaced items rather than
    the first `n` shows the breadth of the corpus. A missing or unreadable file
    returns [] so the UI simply shows no examples.
    """
    try:
        lines = Path(eval_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    questions = []
    for line in lines:
        try:
            q = json.loads(line).get("question") if line.strip() else None
        except (json.JSONDecodeError, AttributeError):
            q = None
        if q:
            questions.append(q)
    if len(questions) <= n:
        return questions
    step = len(questions) / n
    return [questions[int(i * step)] for i in range(n)]


def localized_examples(eval_path: Path, ui_lang: str = "en", n: int = 4) -> list[str]:
    """Example questions for a corpus, shown in the requested UI language.

    The same questions as `example_questions` are picked, so both languages offer
    the same examples. "en" swaps each French question for its hand-written
    translation in EXAMPLE_TRANSLATIONS_EN; any other language returns the eval
    file's own wording. A question with no translation keeps its original text
    rather than disappearing (so an English corpus is unchanged in "fr").
    """
    questions = example_questions(eval_path, n)
    if ui_lang != "en":
        return questions
    return [EXAMPLE_TRANSLATIONS_EN.get(q, q) for q in questions]


def detect_language(text: str) -> str:
    """Guess "French" or "English" by counting each language's function words.

    The index stores no language metadata, so this is a deliberate heuristic for
    display only; it matters because answers and highlights are only reliable when
    the question is asked in the corpus's language.
    """
    words = _WORD_RE.findall(text.lower())
    fr = sum(w in _FR_STOPWORDS for w in words)
    en = sum(w in _EN_STOPWORDS for w in words)
    if fr == en == 0:
        return "unknown"
    return "French" if fr > en else "English"


def corpus_stats(chunks: list) -> dict:
    """Summarise an index: number of documents, chunks, PDF pages and languages.

    Language is detected per document (all its chunks joined) so one mixed passage
    does not mislabel a file.
    """
    by_source: dict[str, list[str]] = {}
    pages = set()
    for c in chunks:
        by_source.setdefault(c.source, []).append(c.text)
        if c.meta.get("page"):
            pages.add((c.source, c.meta["page"]))
    languages = sorted({detect_language(" ".join(t)) for t in by_source.values()} - {"unknown"})
    return {
        "documents": len(by_source),
        "chunks": len(chunks),
        "pages": len(pages),
        "languages": languages,
        "sources": sorted(by_source),
    }


def explain_refusal(best_score: float | None, threshold: float, lang: str = "en") -> str:
    """Explain a refusal in one sentence, quoting the best match against the gate.

    The gate is the dense cosine score, so naming both numbers tells the user
    whether the corpus is simply silent on the topic (far below) or the question
    was phrased differently from the documents (just below). `lang="fr"` returns
    the same explanation in French; anything else returns English.
    """
    if lang == "fr":
        return _explain_refusal_fr(best_score, threshold)
    if best_score is None:
        return "The index returned no passages at all, so there is nothing to ground an answer on."
    gap = threshold - best_score
    hint = (
        " It is close: try rephrasing with the documents' own wording or language."
        if gap < 0.05 else
        " The corpus most likely does not cover this topic."
    )
    return (
        f"The best matching passage scored {best_score:.2f} (cosine similarity), below the "
        f"{threshold:.2f} threshold required to answer.{hint}"
    )


def _explain_refusal_fr(best_score: float | None, threshold: float) -> str:
    """French version of `explain_refusal`, with the same near-miss rule (gap < 0.05)."""
    if best_score is None:
        return ("L'index n'a renvoyé aucun passage : il n'y a rien sur quoi fonder "
                "une réponse.")
    gap = threshold - best_score
    hint = (
        " C'est proche : reformulez avec les mots ou la langue des documents."
        if gap < 0.05 else
        " Le corpus ne couvre très probablement pas ce sujet."
    )
    return (
        f"Le passage le plus proche obtient {best_score:.2f} (similarité cosinus), sous le "
        f"seuil de {threshold:.2f} requis pour répondre.{hint}"
    )


def explain_grounding(score: float, min_grounding: float, ok: bool, lang: str = "en") -> str:
    """One line saying what the grounding score measures and whether it passed.

    `lang="fr"` returns it in French; anything else returns English.
    """
    if lang == "fr":
        verdict = "atteint" if ok else "n'atteint pas"
        return (
            f"{score:.0%} des mots porteurs de sens de la réponse figurent dans les passages "
            f"qu'elle cite ({verdict} le seuil de {min_grounding:.0%}, et chaque citation doit "
            "exister). C'est un contrôle lexical, pas une preuve que le sens est préservé."
        )
    verdict = "passes" if ok else "does not pass"
    return (
        f"{score:.0%} of the answer's content words appear in the passages it cites "
        f"({verdict} the {min_grounding:.0%} bar, and every citation must exist). "
        "It is a lexical check, not proof the meaning is preserved."
    )


def build_limitations(
    threshold: float, min_grounding: float, has_bm25: bool, lang: str = "en"
) -> list[str]:
    """Return the specific, honest limitations shown in the sidebar.

    Built from the live settings so the numbers quoted match what the app is
    actually running, and drawn from the known limitations of the engine.
    `lang="fr"` returns the same list in French; anything else returns English.
    """
    if lang == "fr":
        return _build_limitations_fr(threshold, min_grounding, has_bm25)
    items = [
        "Answers come only from the indexed corpus; anything outside it is refused, "
        "even if it is common knowledge.",
        "One retrieval pass and one LLM call per question: no multi-hop reasoning or "
        "query decomposition, so questions needing evidence from many places may be "
        "answered partially.",
        "Tables, figures and scanned (image-only) PDFs are poorly handled: text is "
        "extracted page by page with no OCR or table structure.",
        f"The verification is lexical (word overlap, bar {min_grounding:.0%}), not "
        "semantic entailment: it catches drift away from the sources, not a subtle "
        "misreading that reuses the same words.",
        "The LLM can still paraphrase a cited passage wrongly; always read the cited "
        "source before relying on a figure or obligation.",
        "No cross-encoder re-ranker: ordering relies on embeddings"
        + (", BM25 fusion" if has_bm25 else "")
        + " and MMR diversity.",
        f"The refusal threshold ({threshold:.2f} cosine) was tuned on small evaluation "
        "sets (about 25 questions each); it can refuse answerable questions or let a "
        "loosely related passage through.",
        "Questions work best in the language of the documents: cross-language matches "
        "score lower and are more likely to be refused.",
        "The answer language setting only changes the language the answer is written in: "
        "retrieval still matches the question's own wording, and a translated answer can "
        "share fewer words with its sources, which lowers the lexical grounding score.",
    ]
    return items


def _build_limitations_fr(threshold: float, min_grounding: float, has_bm25: bool) -> list[str]:
    """French version of `build_limitations`, item for item."""
    return [
        "Les réponses viennent uniquement du corpus indexé ; tout ce qui en sort est refusé, "
        "même s'il s'agit de connaissances courantes.",
        "Une seule recherche et un seul appel au LLM par question : pas de raisonnement en "
        "plusieurs étapes ni de décomposition de la question, donc une question qui demande "
        "des éléments dispersés peut recevoir une réponse partielle.",
        "Les tableaux, figures et PDF scannés (images seules) sont mal traités : le texte est "
        "extrait page par page, sans OCR ni structure de tableau.",
        f"La vérification est lexicale (recouvrement de mots, seuil {min_grounding:.0%}), pas "
        "une implication sémantique : elle détecte une réponse qui s'éloigne des sources, pas "
        "une erreur de lecture subtile qui reprend les mêmes mots.",
        "Le LLM peut encore mal paraphraser un passage cité ; lisez toujours la source citée "
        "avant de vous fier à un chiffre ou à une obligation.",
        "Pas de re-classement par cross-encoder : l'ordre repose sur les embeddings"
        + (", la fusion BM25" if has_bm25 else "")
        + " et la diversité MMR.",
        f"Le seuil de refus ({threshold:.2f} cosinus) a été réglé sur de petits jeux "
        "d'évaluation (environ 25 questions chacun) ; il peut refuser une question à laquelle "
        "on pouvait répondre ou laisser passer un passage vaguement lié.",
        "Les questions fonctionnent mieux dans la langue des documents : les correspondances "
        "entre langues obtiennent des scores plus bas et sont plus souvent refusées.",
        "Le choix de la langue de réponse ne change que la langue de rédaction : la recherche "
        "s'appuie toujours sur les mots de la question, et une réponse traduite partage moins "
        "de mots avec ses sources, ce qui fait baisser le score d'ancrage lexical.",
    ]


def llm_error_hint(error_message: str, provider: str) -> str:
    """Turn an LLM failure into an actionable message naming the setting to fix.

    `rag.llm.complete` reports why each provider failed ("groq: no API key",
    "ollama: ConnectionError..."); mapping those to the matching .env variable
    saves the user from reading a traceback.
    """
    msg = error_message.lower()
    steps = []
    if "groq: no api key" in msg or (provider == "groq" and "no api key" in msg):
        steps.append("set `GROQ_API_KEY` in `.env` (free key at console.groq.com)")
    if "openrouter: no api key" in msg:
        steps.append("or set `OPENROUTER_API_KEY`")
    if "ollama:" in msg:
        steps.append("or start Ollama locally (`OLLAMA_URL`) and pull `OLLAMA_LLM_MODEL`")
    if any(code in msg for code in ("401", "403", "unauthorized")):
        steps.insert(0, "check that your API key is valid (it was rejected)")
    if "429" in msg or "rate" in msg:
        steps.insert(0, "wait a minute or add `GROQ_API_KEY_2`..`_5` (rate limit hit)")
    if not steps:
        steps.append("check `LLM_PROVIDER` and the matching API key in `.env`")
    return (
        "No language model could be reached, so only the retrieved passages are shown. "
        "To get answers: " + "; ".join(steps) + ", then restart the app."
    )


def embed_error_hint(provider: str) -> str:
    """Actionable message when the question cannot be embedded (retrieval impossible)."""
    if provider == "ollama":
        return (
            "The embedding model could not be reached. Start Ollama (`ollama serve`, "
            "`OLLAMA_URL`) and pull `OLLAMA_EMBED_MODEL` (default `nomic-embed-text`)."
        )
    return (
        f"The embedding backend `{provider}` could not be reached. Check `EMBED_PROVIDER` "
        "and the matching `HOSTED_EMBED_*` settings in `.env`."
    )
