"""Tests for the optional NLI entailment check (rag/nli.py) and where it is reported.

No NLI model is loaded or downloaded: a fake checker returns preset
probabilities, and a stand-in `sentence_transformers` module covers the model
wrapper. What is pinned down is the contract: off by default, each cited
sentence is checked against the passages it cites, the result sits beside
the lexical verification and never changes `ok`, and failures are reported
rather than raised.
"""

import math
import sys

import pytest

import config
import rag.llm
import rag.nli as nli
from rag.verify import verify_answer
from tests.test_api import _r, client, setup  # noqa: F401 - shared API fixtures


class FakeChecker:
    """Returns preset probabilities per (premise, hypothesis); records the pairs asked."""

    def __init__(self, table, default=("neutral", 1.0)):
        self.table = table
        self.default = default
        self.pairs = []

    def probabilities(self, pairs):
        self.pairs.extend(pairs)
        out = []
        for pair in pairs:
            label, p = self.table.get(pair, self.default)
            rest = (1.0 - p) / 2
            out.append({lab: (p if lab == label else rest) for lab in nli.LABELS})
        return out


def _sources():
    return [_r(1, 0.9, text="Logs are kept for six months."),
            _r(2, 0.8, text="Providers must register the system.")]


def test_each_cited_sentence_is_checked_against_the_passages_it_cites():
    sources = _sources()
    answer = "Logs are kept for six months [S1]. Providers must register [S2]. Nothing here."
    checker = FakeChecker({
        ("Logs are kept for six months.", "Logs are kept for six months."): ("entailment", 0.9),
        ("Providers must register the system.", "Providers must register."): ("contradiction", 0.8),
    })
    report = nli.check_entailment(answer, sources, checker)
    assert [p[1] for p in checker.pairs] == ["Logs are kept for six months.", "Providers must register."]
    assert [s.label for s in report.sentences] == ["entailment", "contradiction"]
    assert report.entailed_fraction == 0.5
    assert report.contradicted == ["Providers must register [S2]."]


def test_the_best_entailing_passage_wins_and_contradiction_is_the_worst():
    sources = _sources()
    hyp = "Logs are kept and systems registered."
    checker = FakeChecker({
        (sources[0].chunk.text, hyp): ("contradiction", 0.6),
        (sources[1].chunk.text, hyp): ("entailment", 0.7),
    })
    report = nli.check_entailment("Logs are kept and systems registered [S1][S2].", sources, checker)
    (s,) = report.sentences
    assert s.label == "entailment" and s.best_source == 2
    assert s.entailment == pytest.approx(0.7) and s.contradiction == pytest.approx(0.6)


def test_invalid_citations_refusals_and_uncited_answers_check_nothing():
    checker = FakeChecker({})
    assert nli.check_entailment("Made up [S9].", _sources(), checker).entailed_fraction is None
    assert nli.check_entailment(rag.answer.REFUSAL_MESSAGE, [], checker).sentences == []
    assert nli.check_entailment("No citation at all here.", _sources(), checker).sentences == []
    assert checker.pairs == []


def test_off_by_default_and_never_imports_sentence_transformers(monkeypatch):
    monkeypatch.setattr(config, "NLI_CHECK", False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert nli.get_checker() is None
    assert nli.nli_summary("Logs [S1].", _sources()) is None


def test_lexical_verification_is_unchanged_by_the_check(monkeypatch):
    sources = _sources()
    answer = "Logs are kept for six months [S1]."
    before = verify_answer(answer, sources, min_grounding=0.3)
    monkeypatch.setattr(config, "NLI_CHECK", True)
    monkeypatch.setattr(nli, "get_checker", lambda: FakeChecker({}, default=("contradiction", 0.99)))
    summary = nli.nli_summary(answer, sources)
    assert summary["contradicted"] == [answer]
    assert verify_answer(answer, sources, min_grounding=0.3) == before


def test_summary_reports_a_failure_instead_of_raising(monkeypatch):
    class Broken:
        def probabilities(self, pairs):
            raise RuntimeError("model not found")

    monkeypatch.setattr(nli, "get_checker", lambda: Broken())
    monkeypatch.setattr(config, "NLI_MODEL", "some/nli")
    assert nli.nli_summary("Logs [S1].", _sources()) == {"model": "some/nli", "error": "model not found"}


def test_checker_reads_label_order_from_the_model_and_loads_lazily(monkeypatch):
    seen = {}

    class FakeCrossEncoder:
        def __init__(self, name):
            seen["name"] = name
            self.model = type("M", (), {"config": type("C", (), {
                "id2label": {0: "contradiction", 1: "entailment", 2: "neutral"}})()})()

        def predict(self, pairs):
            seen["pairs"] = pairs
            return [[0.0, math.log(3.0), 0.0] for _ in pairs]  # entailment 3/5

    module = type(sys)("sentence_transformers")
    module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    checker = nli.NLIChecker("m")
    assert seen == {}
    (probs,) = checker.probabilities([("premise", "hypothesis")])
    assert probs["entailment"] == pytest.approx(0.6)
    assert probs["contradiction"] == pytest.approx(0.2) and probs["neutral"] == pytest.approx(0.2)
    assert seen == {"name": "m", "pairs": [("premise", "hypothesis")]}


def test_a_model_that_is_not_three_way_nli_is_refused(monkeypatch):
    class TwoWay:
        def __init__(self, name):
            self.model = type("M", (), {"config": type("C", (), {"id2label": {0: "neg", 1: "pos"}})()})()

    module = type(sys)("sentence_transformers")
    module.CrossEncoder = TwoWay
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    with pytest.raises(RuntimeError, match="not a 3-way NLI model"):
        nli.NLIChecker("m").probabilities([("a", "b")])


def test_missing_sentence_transformers_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        nli.NLIChecker("m").probabilities([("a", "b")])


def test_api_reports_nli_beside_verification_without_changing_ok(setup, client, monkeypatch):
    monkeypatch.setattr(rag.llm, "complete",
                        lambda prompt, system=None: "High-risk systems must keep logs and documentation [S1].")
    req = {"corpus": "demo", "question": "What must high-risk systems keep?"}
    off = client.post("/api/ask", json=req).json()["verification"]
    assert off["nli"] is None

    monkeypatch.setattr(config, "NLI_CHECK", True)
    monkeypatch.setattr(nli, "get_checker", lambda: FakeChecker({}, default=("contradiction", 0.9)))
    on = client.post("/api/ask", json=req).json()["verification"]
    assert on["nli"]["checked"] == 1 and on["nli"]["entailed_fraction"] == 0.0
    assert on["nli"]["sentences"][0]["label"] == "contradiction"
    assert {k: v for k, v in on.items() if k != "nli"} == {k: v for k, v in off.items() if k != "nli"}
