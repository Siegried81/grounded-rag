"""Optional NLI entailment check per cited sentence (`NLI_CHECK=true`), beside lexical grounding.

`verify.py` measures whether an answer reuses its cited passages' words. That
proves a claim is traceable, not that it follows: "X is not required [S1]"
against a passage saying "X is required" shares nearly every word. A natural
language inference (NLI) model reads a premise and a hypothesis together and
says whether the premise entails, contradicts or is neutral to it — the
question lexical overlap cannot ask.

How it is applied, and why:

- Only sentences carrying a valid citation are checked, each against the
  passages THAT SENTENCE cites, not the whole retrieved set: a claim is meant to
  follow from what it cites.
- Each cited passage is a separate premise and the best entailment wins. A
  passage fits the model's input window; several concatenated might not and
  would be silently truncated. The cost is that a claim needing two passages
  together reads as not entailed by either — a conservative error.
- It is REPORTED next to the lexical grounding, never folded into it: `ok`,
  `strict_ok` and `grounding_score` keep their definitions and every published
  number keeps its meaning. The model can be wrong too, and an unmeasured
  signal should not gate answers. Off by default.

`sentence-transformers` is imported lazily: the default install does not have
it, and this module must import without it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache

import config
from rag.answer import REFUSAL_MESSAGE_SET, normalize_citations
from rag.verify import extract_citations, split_sentences

LABELS = ("entailment", "neutral", "contradiction")
# A citation marker with the space before it, so "kept [S1]." becomes the
# plain sentence "kept." that the NLI model was trained on.
_MARKER_RE = re.compile(r"\s*\[S\d+\]")


@dataclass
class SentenceEntailment:
    """NLI verdict for one cited sentence against its best-supporting cited passage."""

    sentence: str
    cited: list[int]
    label: str            # entailment | neutral | contradiction (argmax at the best passage)
    entailment: float     # probability of entailment at the best passage
    contradiction: float  # highest contradiction probability over the cited passages
    best_source: int      # the [S#] whose passage entails it most


@dataclass
class NLIReport:
    """Per-sentence NLI verdicts plus two summary figures for display."""

    sentences: list[SentenceEntailment] = field(default_factory=list)

    @property
    def entailed_fraction(self) -> float | None:
        """Share of checked sentences labelled entailment; None when nothing was checked."""
        if not self.sentences:
            return None
        return sum(s.label == "entailment" for s in self.sentences) / len(self.sentences)

    @property
    def contradicted(self) -> list[str]:
        """Sentences the model reads as contradicted by their cited passage."""
        return [s.sentence for s in self.sentences if s.label == "contradiction"]


def _softmax(logits) -> list[float]:
    """Probabilities from one row of logits, computed stably."""
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


class NLIChecker:
    """Scores (premise, hypothesis) pairs with a sentence-transformers NLI CrossEncoder.

    Label order differs between NLI models, so it is read from the model's own
    `id2label` rather than assumed. The model loads on first use.
    """

    def __init__(self, model: str):
        self.model = model
        self._encoder = None
        self._order: list[str] = []

    def _load(self) -> None:
        """Load the model and the position of each of LABELS in its output."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "NLI_CHECK=true needs the `sentence-transformers` package "
                "(pip install -r requirements-optional.txt), or set NLI_CHECK=false."
            ) from exc
        self._encoder = CrossEncoder(self.model)
        id2label = self._encoder.model.config.id2label
        names = [str(id2label[i]).lower() for i in range(len(id2label))]
        if sorted(names) != sorted(LABELS):
            raise RuntimeError(f"{self.model} is not a 3-way NLI model (labels: {names})")
        self._order = names

    def probabilities(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """For each (premise, hypothesis), the probability of each of LABELS."""
        if not pairs:
            return []
        if self._encoder is None:
            self._load()
        logits = self._encoder.predict(pairs)
        return [dict(zip(self._order, _softmax([float(x) for x in row]))) for row in logits]


@lru_cache(maxsize=None)
def _cached(model: str) -> NLIChecker:
    """One checker per model name per process, so the model loads once."""
    return NLIChecker(model)


def get_checker():
    """The configured NLI checker, or None when `config.NLI_CHECK` is off (the default)."""
    return _cached(config.NLI_MODEL) if config.NLI_CHECK else None


def check_entailment(answer_text: str, sources: list, checker) -> NLIReport:
    """NLI verdict for every sentence of `answer_text` that cites a valid source.

    The sentence, without its [S#] markers, is the hypothesis; each passage it
    cites is a premise; the passage with the highest entailment probability
    gives the label. A refusal message asserts nothing and yields an empty
    report, as it verifies as ok in verify.py.
    """
    answer_text = normalize_citations(answer_text)
    if answer_text.strip() in REFUSAL_MESSAGE_SET:
        return NLIReport()
    todo = []
    for sentence in split_sentences(answer_text):
        cited = [n for n in extract_citations(sentence) if 1 <= n <= len(sources)]
        hypothesis = _MARKER_RE.sub("", sentence).strip()
        if cited and hypothesis:
            todo.append((sentence, cited, hypothesis))
    pairs = [(sources[n - 1].chunk.text, hyp) for _, cited, hyp in todo for n in cited]
    probs = iter(checker.probabilities(pairs))
    report = NLIReport()
    for sentence, cited, _ in todo:
        per_source = [(n, next(probs)) for n in cited]
        best_n, best = max(per_source, key=lambda item: item[1]["entailment"])
        report.sentences.append(SentenceEntailment(
            sentence=sentence,
            cited=cited,
            label=max(LABELS, key=lambda lab: best[lab]),
            entailment=best["entailment"],
            contradiction=max(p["contradiction"] for _, p in per_source),
            best_source=best_n,
        ))
    return report


def format_nli(report: NLIReport) -> str:
    """Short text summary for the CLI, printed under the verification report."""
    if report.entailed_fraction is None:
        return "NLI: no cited sentence to check"
    lines = [f"NLI: {report.entailed_fraction:.2f} of {len(report.sentences)} cited sentences "
             f"entailed by their cited passage"]
    for s in report.sentences:
        if s.label != "entailment":
            lines.append(f"  {s.label} (p_entail {s.entailment:.2f}): {s.sentence}")
    return "\n".join(lines)


def nli_summary(answer_text: str, sources: list) -> dict | None:
    """The NLI check as plain data for the API and the UIs; None when NLI_CHECK is off.

    A failure (package missing, model not found) is returned as `{"error": ...}`
    instead of raised: the check is an extra signal beside verification, and the
    answer, its sources and its lexical verification are still worth showing.
    """
    checker = get_checker()
    if checker is None:
        return None
    try:
        report = check_entailment(answer_text, sources, checker)
    except Exception as exc:  # noqa: BLE001 - any failure leaves the answer usable
        return {"model": config.NLI_MODEL, "error": str(exc)}
    fraction = report.entailed_fraction
    return {
        "model": config.NLI_MODEL,
        "error": None,
        "entailed_fraction": None if fraction is None else round(fraction, 4),
        "checked": len(report.sentences),
        "contradicted": report.contradicted,
        "sentences": [
            {"sentence": s.sentence, "cited": s.cited, "label": s.label,
             "entailment": round(s.entailment, 4), "contradiction": round(s.contradiction, 4),
             "best_source": s.best_source}
            for s in report.sentences
        ],
    }
