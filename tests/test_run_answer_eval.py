"""Offline end-to-end tests for scripts/run_answer_eval.py.

Store, BM25 and embedder loaders and `retrieve` are monkeypatched, and the LLM is
replaced at `rag.llm.complete`, so the REAL `answer_question` and rag.verify run
with no index, network or API key.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

import rag.llm
from rag.answer import REFUSAL_MESSAGE
from rag.types import Chunk, Retrieved

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_answer_eval.py"

ITEMS = [
    {"id": "t-01", "question": "What payment services?", "answerable": True,
     "gold_sources": ["a.txt"], "key_facts": ["Apple Card", "Apple Pay"], "note": ""},
    {"id": "t-02", "question": "What was net income?", "answerable": False,
     "gold_sources": [], "key_facts": [], "note": ""},
    {"id": "t-03", "question": "Off-topic gate refusal?", "answerable": False,
     "gold_sources": [], "key_facts": [], "note": ""},
]
SOURCE_TEXT = "The Company offers Apple Card, a credit card, and Apple Pay, a payment service."


def _load_script():
    """Import scripts/run_answer_eval.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("run_answer_eval", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Script module with fake loaders, a fake retriever and a scripted fake LLM."""
    module = _load_script()
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text("".join(json.dumps(i) + "\n" for i in ITEMS), encoding="utf-8")
    retrieve_calls, llm_calls = [], []

    def fake_retrieve(question, store, embedder, top_k, bm25=None, mode=None):
        retrieve_calls.append({"bm25": bm25, "mode": mode, "top_k": top_k})
        if "gate" in question:
            return []  # below the dense threshold -> pipeline refuses without the LLM
        return [Retrieved(Chunk("c1", SOURCE_TEXT, "a.txt", 0, "c"), 0.8)]

    def fake_complete(prompt, *, system=None, order=None):
        llm_calls.append(system)
        if system == module.JUDGE_SYSTEM:
            return '```json\n{"score": 2, "reason": "supported"}\n```'
        if "net income" in prompt:
            return "The sources do not mention Apple's net income."
        return "Apple offers Apple Card and Apple Pay [S1]."

    monkeypatch.setattr(module.VectorStore, "load", staticmethod(lambda path: "store"))
    monkeypatch.setattr(module.BM25Index, "load", staticmethod(lambda path: "bm25"))
    monkeypatch.setattr(module, "get_embedder", lambda: "embedder")
    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag.llm, "complete", fake_complete)
    module.eval_file, module.out = eval_file, tmp_path / "out.jsonl"
    module.retrieve_calls, module.llm_calls = retrieve_calls, llm_calls
    return module


def _main(module, monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", [
        "run_answer_eval.py", "--corpus", "c", "--eval-file", str(module.eval_file),
        "--out", str(module.out), "--k", "3", "--mode", "hybrid", *extra,
    ])
    module.main()
    return [json.loads(line) for line in module.out.read_text(encoding="utf-8").splitlines()]


def test_end_to_end_scores_and_summary(harness, monkeypatch, capsys):
    rows = _main(harness, monkeypatch)
    assert [r["id"] for r in rows] == ["t-01", "t-02", "t-03"]
    assert harness.retrieve_calls[0] == {"bm25": "bm25", "mode": "hybrid", "top_k": 3}
    answered, worded_refusal, gate = rows
    assert not answered["refused"] and answered["key_fact_recall"] == 1.0
    assert answered["valid_citations"] == [1] and answered["verify_ok"] is True
    assert answered["retrieved_sources"] == ["a.txt"] and answered["corpus"] == "c"
    assert worded_refusal["refused"] and not worded_refusal["gate_refused"]
    assert gate["refused"] and gate["gate_refused"] and gate["answer"] == REFUSAL_MESSAGE
    # Two answer calls: the gate refusal never reached the LLM, and no judge ran.
    assert len(harness.llm_calls) == 2
    assert harness.JUDGE_SYSTEM not in harness.llm_calls
    out = capsys.readouterr().out
    assert "mode=hybrid" in out
    assert "refusal recall" in out and "1.000" in out
    assert "judge" not in out


def test_limit_and_sleep(harness, monkeypatch):
    sleeps = []
    monkeypatch.setattr(harness.time, "sleep", lambda s: sleeps.append(s))
    rows = _main(harness, monkeypatch, "--limit", "2", "--sleep", "1.5")
    assert len(rows) == 2
    assert sleeps == [1.5]  # between questions, not before the first


def test_judge_flag(harness, monkeypatch, capsys):
    rows = _main(harness, monkeypatch, "--judge")
    assert rows[0]["judge_score"] == 2
    assert "judge_score" not in rows[1] and "judge_score" not in rows[2]  # refusals not judged
    assert harness.llm_calls.count(harness.JUDGE_SYSTEM) == 1
    assert "judge faithfulness (0-2)" in capsys.readouterr().out


def test_llm_failure_is_recorded_and_excluded(harness, monkeypatch, capsys):
    def down(prompt, *, system=None, order=None):
        raise rag.llm.LLMError("All LLM providers failed: groq: 429")

    monkeypatch.setattr(rag.llm, "complete", down)
    rows = _main(harness, monkeypatch)
    assert rows[0]["error"].startswith("All LLM providers failed")
    assert rows[2]["refused"] and "error" not in rows[2]  # gate refusal needs no LLM
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "errors (excluded)" in out


def test_dense_fallback_without_bm25(harness, monkeypatch, capsys):
    def missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(harness.BM25Index, "load", staticmethod(missing))
    _main(harness, monkeypatch, "--limit", "1")
    assert harness.retrieve_calls[0]["bm25"] is None
    assert "mode=dense" in capsys.readouterr().out


def test_rows_record_provider_and_diagnosis(harness, monkeypatch, capsys):
    monkeypatch.setattr(rag.llm, "last_provider", lambda: "openrouter")
    answered, worded_refusal, gate = _main(harness, monkeypatch)
    assert answered["provider"] == "openrouter" and answered["diagnosis"] == "ok"
    assert answered["gold_sources"] == ["a.txt"]
    assert worded_refusal["provider"] == "openrouter" and worded_refusal["diagnosis"] == "ok"
    assert gate["provider"] is None and gate["diagnosis"] == "ok"  # no LLM call on the gate
    out = capsys.readouterr().out
    assert "diagnosis" in out and "generation_overanswer" in out
    # 3 rows: the per-row column plus the count table all say "ok".
    assert "ok" in out


def test_summary_separates_gate_refusals_from_model_refusals(harness, monkeypatch, capsys):
    """The two refusals have different origins, and the table reports them apart.

    t-03 is refused by the retrieval gate (no LLM call), t-02 by the model in its
    own words. The refusal scores merge both, so without this row a refusal F1 of
    1.000 would read as evidence about the cosine threshold.
    """
    _main(harness, monkeypatch)
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("refusals: gate / model"))
    assert line.split()[-1] == "1/1"


def test_summary_prints_wilson_intervals_for_binomial_metrics(harness, monkeypatch, capsys):
    _main(harness, monkeypatch)
    out = capsys.readouterr().out
    # 2 refusals, both right (tp=2, fp=0): precision 1.000 with Wilson [0.34-1.00].
    assert "refusal precision" in out and "1.000 [0.34-1.00]" in out
    # One verified answer that passed: 1/1.
    assert "verify ok rate" in out and "1.000 [0.21-1.00]" in out
    # Means of fractions carry no interval.
    line = next(l for l in out.splitlines() if l.startswith("key-fact recall "))
    assert "[" not in line


@pytest.mark.parametrize("row, expected", [
    ({"answerable": True, "refused": False, "key_fact_recall": 1.0,
      "gold_sources": ["a"], "retrieved_sources": ["a"]}, "ok"),
    ({"answerable": True, "refused": False, "key_fact_recall": None,
      "gold_sources": ["a"], "retrieved_sources": ["b"]}, "ok"),
    ({"answerable": True, "refused": True, "key_fact_recall": 0.0,
      "gold_sources": ["a"], "retrieved_sources": ["b", "c"]}, "retrieval"),
    ({"answerable": True, "refused": False, "key_fact_recall": 0.5,
      "gold_sources": ["a"], "retrieved_sources": ["b"]}, "retrieval"),
    ({"answerable": True, "refused": True, "key_fact_recall": 0.0,
      "gold_sources": ["a"], "retrieved_sources": ["b", "a"]}, "generation_refusal"),
    ({"answerable": True, "refused": False, "key_fact_recall": 0.5,
      "gold_sources": ["a"], "retrieved_sources": ["a"]}, "generation"),
    ({"answerable": False, "refused": True}, "ok"),
    ({"answerable": False, "refused": False}, "generation_overanswer"),
    ({"answerable": True, "refused": False, "error": "All LLM providers failed"}, "error"),
])
def test_diagnose(harness, row, expected):
    assert harness.diagnose(row) == expected


def test_diagnosis_counts_per_corpus_and_all(harness):
    rows = [{"corpus": "a", "diagnosis": "ok"}, {"corpus": "a", "diagnosis": "retrieval"},
            {"corpus": "b", "diagnosis": "ok"}]
    counts = harness.diagnosis_counts(rows)
    assert counts["a"] == {"ok": 1, "retrieval": 1}
    assert counts["b"] == {"ok": 1}
    assert counts["all"] == {"ok": 2, "retrieval": 1}
    assert "all" not in harness.diagnosis_counts(rows[:2])


def test_llm_error_row_is_diagnosed_as_error(harness, monkeypatch):
    def down(prompt, *, system=None, order=None):
        raise rag.llm.LLMError("All LLM providers failed: groq: 429")

    monkeypatch.setattr(rag.llm, "complete", down)
    rows = _main(harness, monkeypatch)
    assert rows[0]["diagnosis"] == "error" and rows[0]["provider"] is None
    assert rows[0]["retrieved_sources"] == ["a.txt"]


def test_eval_file_requires_single_corpus(harness, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "run_answer_eval.py", "--corpus", "a", "b", "--eval-file", str(harness.eval_file),
    ])
    with pytest.raises(SystemExit):
        harness.main()
