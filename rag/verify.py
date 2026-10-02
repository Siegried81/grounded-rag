"""Deterministic, offline check of an answer against the sources it cites.

Turns "cite-or-refuse" from a claim into something checkable: it validates that
every [S#] marker points at a real source, flags claim-like sentences with no
citation, and computes a cheap lexical grounding score. No LLM call is made, so
it can run on every answer and in tests at zero cost.
"""

import re
from dataclasses import dataclass, field

from rag.answer import REFUSAL_MESSAGE
from rag.types import Retrieved

_CITATION_RE = re.compile(r"\[S(\d+)\]")
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
# Split after ., ! or ? followed by whitespace, or on newlines.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_MIN_CLAIM_WORDS = 4  # a sentence needs MORE than this many content words to count as a claim

_STOPWORDS = frozenset(
    "a an the and or but of to in on at by for with from as is are was were be been being "
    "it its this that these those which who whom what when where how not no do does did "
    "has have had can could will would should may might also than then there their they "
    "he she we you i his her our your into about over under so if".split()
)


def extract_citations(text: str) -> list[int]:
    """Return distinct [S<n>] source indices in order of first appearance.

    Order and de-duplication keep the report stable and readable.
    """
    seen: list[int] = []
    for m in _CITATION_RE.finditer(text):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with a pragmatic heuristic.

    Breaks after '.', '!' or '?' followed by whitespace, and on newlines. It is not
    linguistically perfect (abbreviations may over-split) but is dependency-free and
    deterministic, which is enough for flagging uncited claims.
    """
    return [s.strip() for s in _SENTENCE_RE.split(text) if s and s.strip()]


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
    ok: bool = False


def verify_answer(
    answer_text: str, sources: list[Retrieved], min_grounding: float = 0.5
) -> VerificationReport:
    """Check an answer's citations and lexical grounding against its sources.

    grounding_score is the fraction of the answer's content words that also appear
    in the union of the CITED sources' text. It is a cheap faithfulness proxy, NOT an
    entailment check: it catches answers that drift away from their sources, not
    subtle misreadings. A hallucinated [S#] (out of range) makes the answer not ok.
    The exact refusal message is accepted as fully ok since it asserts nothing.
    """
    if answer_text.strip() == REFUSAL_MESSAGE:
        return VerificationReport(grounding_score=1.0, ok=True)

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

    return VerificationReport(
        valid_citations=valid,
        invalid_citations=invalid,
        uncited_sentences=uncited,
        grounding_score=score,
        ok=not invalid and score >= min_grounding,
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
