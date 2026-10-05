"""Deterministic, offline check of an answer against the sources it cites.

Turns "cite-or-refuse" from a claim into something checkable: it validates that
every [S#] marker points at a real source, flags claim-like sentences with no
citation, and computes a cheap lexical grounding score. No LLM call is made, so
it can run on every answer and in tests at zero cost.

Grounding compares SURFACE words: the answer's content words (accented letters
kept, English/French/Dutch function words dropped) are looked up in the cited
sources' text. An answer written in another language than its sources therefore
still scores lower than the same answer in the sources' language: numbers and
proper names match across languages, ordinary vocabulary does not. The French
and Dutch stopwords only stop penalising French/Dutch FUNCTION words, which used
to count as ungrounded content and could push a correct French answer to 0.00.
"""

import re
from dataclasses import dataclass, field

from rag.answer import REFUSAL_MESSAGE_SET, normalize_citations
from rag.types import Retrieved

_CITATION_RE = re.compile(r"\[S(\d+)\]")
# Runs of Unicode letters or digits (no underscore). Accented letters are words,
# and apostrophes split ("l'amende" -> "l", "amende"; "don't" -> "don", "t"), so
# French elisions are tokenised the same way whether the apostrophe is straight
# or typographic. Measurement change: tokens with accents used to be cut into
# fragments ("amende" from "l'amende" was fine, but "été" became nothing).
_WORD_RE = re.compile(r"[^\W_]+")
# Split after ., ! or ? followed by whitespace, or on newlines.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# A fragment ending with an abbreviation ("U.S.", "e.g.", "Inc.", "art.") is not
# a sentence end: it is glued back to the fragment that follows. Without this,
# "in the U.S. District Court" became two "sentences", and an uncited claim was
# counted twice. The list is small on purpose; an abbreviation missing from it
# only over-splits, as before. The letter run may follow Markdown emphasis.
_ABBREV_END_RE = re.compile(
    r"(?:(?:^|[\s(\[\"'“«*_])(?:[A-Za-z]\.)+"
    r"|\b(?:etc|inc|corp|ltd|no|nos|art|arts|vs|mr|mrs|ms|dr|st|fig|figs|p|pp|cf|al|approx"
    r"|env|ex|mme|mlle)\.)$",
    re.IGNORECASE,
)
_MIN_CLAIM_WORDS = 4  # a sentence needs MORE than this many content words to count as a claim

# Function words of the three answer languages (articles, prepositions, pronouns,
# auxiliaries) plus the one-letter leftovers of apostrophe splitting (l', d',
# qu', it's, don't). They carry no claim, so they count neither as answer content
# nor as source content when computing the grounding score.
_STOPWORDS = frozenset(
    # English
    "a an the and or but of to in on at by for with from as is are was were be been being "
    "it its this that these those which who whom what when where how not no do does did "
    "has have had can could will would should may might also than then there their they "
    "he she we you i his her our your into about over under so if "
    # French
    "le la les un une des du de d l et ou mais ne pas que qui quoi dont dans sur pour "
    "par avec sans est sont été être a ont ce cet cette ces il elle ils elles nous vous "
    "on se son sa ses leur leurs au aux y en lui me te toi moi même plus très ainsi "
    "donc car comme si lorsque c j m n qu s t "
    # Dutch
    "het een of maar van op voor met is zijn was waren wordt worden werd niet geen dat "
    "die deze dit er ook zij hij wij je u ik ze we men aan bij uit om tot als dan nog "
    "wel al naar door over onder hun haar zijn mijn jouw uw ons onze te heeft hebben had "
    "hadden kan kunnen zal zullen zou zouden moet moeten".split()
)


def extract_citations(text: str) -> list[int]:
    """Return distinct [S<n>] source indices in order of first appearance.

    Order and de-duplication keep the report stable and readable. Alternative
    markers such as "【S1】" are normalised first (see answer.normalize_citations),
    so a caller verifying raw model output counts the same citations as one
    verifying an answer from answer_question.
    """
    seen: list[int] = []
    for m in _CITATION_RE.finditer(normalize_citations(text)):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with a pragmatic heuristic.

    Breaks after '.', '!' or '?' followed by whitespace, and on newlines, then
    glues back a fragment that ended with a known abbreviation (_ABBREV_END_RE).
    It is not linguistically perfect (an unknown abbreviation still over-splits)
    but is dependency-free and deterministic, which is enough for flagging
    uncited claims. Measurement note: fewer false splits means fewer, longer
    sentences, so `uncited_sentences` counts can be lower than before.
    """
    out: list[str] = []
    for part in _SENTENCE_RE.split(text):
        if not part or not part.strip():
            continue
        part = part.strip()
        if out and _ABBREV_END_RE.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def _content_words(text: str) -> list[str]:
    """Lowercased words of `text` without citation markers or stopwords."""
    cleaned = _CITATION_RE.sub(" ", text).lower()
    return [w for w in _WORD_RE.findall(cleaned) if w not in _STOPWORDS]


@dataclass
class VerificationReport:
    """Outcome of verifying one answer, kept as plain data for CLI, logs and tests."""

    valid_citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    grounding_score: float = 0.0
    # ok: no hallucinated citation and grounding above the threshold. Unchanged
    # so callers that gate on it (the API, the UI) keep their behaviour.
    ok: bool = False
    # strict_ok: ok AND no claim-like sentence left without a citation. Stricter
    # reading of cite-or-refuse for evaluations; an uncited claim does not make an
    # answer ungrounded, but it is not fully "cited".
    strict_ok: bool = False


def verify_answer(
    answer_text: str, sources: list[Retrieved], min_grounding: float = 0.5
) -> VerificationReport:
    """Check an answer's citations and lexical grounding against its sources.

    grounding_score is the fraction of the answer's content words that also appear
    in the union of the CITED sources' text. It is a cheap faithfulness proxy, NOT an
    entailment check: it catches answers that drift away from their sources, not
    subtle misreadings. A hallucinated [S#] (out of range) makes the answer not ok.
    The exact refusal message, in any of the answer languages answer_question can
    return it in, is accepted as fully ok since it asserts nothing.
    Alternative citation markers ("【S1】") are normalised first, so uncited-sentence
    detection and grounding see the same citations as extract_citations.

    Measurement note: content words are Unicode-aware and French/Dutch function
    words are stopwords (see the module docstring), so French and Dutch answers
    score higher than they did when only English function words were dropped.
    `ok` keeps its definition; `strict_ok` additionally requires no uncited claim.
    """
    answer_text = normalize_citations(answer_text)
    if answer_text.strip() in REFUSAL_MESSAGE_SET:
        return VerificationReport(grounding_score=1.0, ok=True, strict_ok=True)

    cited = extract_citations(answer_text)
    valid = [i for i in cited if 1 <= i <= len(sources)]
    invalid = [i for i in cited if i not in valid]

    uncited = [
        s
        for s in split_sentences(answer_text)
        if not _CITATION_RE.search(s) and len(_content_words(s)) > _MIN_CLAIM_WORDS
    ]

    answer_words = _content_words(answer_text)
    source_words = {w for i in valid for w in _content_words(sources[i - 1].chunk.text)}
    if answer_words:
        score = sum(1 for w in answer_words if w in source_words) / len(answer_words)
    else:
        score = 0.0

    ok = not invalid and score >= min_grounding
    return VerificationReport(
        valid_citations=valid,
        invalid_citations=invalid,
        uncited_sentences=uncited,
        grounding_score=score,
        ok=ok,
        strict_ok=ok and not uncited,
    )


def format_report(report: VerificationReport) -> str:
    """Render a report as a short human-readable summary for CLI output and logs."""
    lines = [
        f"verification: {'OK' if report.ok else 'FAILED'} "
        f"(grounding {report.grounding_score:.2f})",
        f"  valid citations: {report.valid_citations or 'none'}",
    ]
    if report.invalid_citations:
        lines.append(f"  invalid citations (no such source): {report.invalid_citations}")
    for s in report.uncited_sentences:
        lines.append(f"  uncited claim: {s}")
    return "\n".join(lines)
