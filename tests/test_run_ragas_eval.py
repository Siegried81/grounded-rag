"""Tests for scripts/run_ragas_eval.py, with fake scorers in place of RAGAS.

Neither `ragas` nor `openai` is needed: `build_scorers` is the only function that
imports them, and every test that runs the scoring replaces it. What is checked:
which rows are scored and which are left out (and counted), the dry-run cost
guard, that a failed judge call is None and never 0, and the run-mean summary.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_ragas_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_ragas_eval", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROWS = [
    {"corpus": "a", "id": "q1", "question": "Q1", "answer": "A1 [S1]", "answerable": True,
     "refused": False, "contexts": ["ctx"], "provider": "azure_openai"},
    {"corpus": "a", "id": "q2", "question": "Q2", "answer": "A2 [S1]", "answerable": True,
     "refused": False, "contexts": ["ctx"], "provider": "azure_openai"},
    {"corpus": "a", "id": "q3", "answerable": False, "refused": True, "contexts": []},
    {"corpus": "a", "id": "q4", "answerable": True, "refused": True, "contexts": []},
    {"corpus": "a", "id": "q5", "answerable": True, "refused": False, "error": "429"},
    {"corpus": "a", "id": "q6", "question": "Q6", "answer": "A6", "answerable": True,
     "refused": False},
]


def test_imports_without_ragas(monkeypatch):
    monkeypatch.setitem(sys.modules, "ragas", None)
    _load()


def test_select_rows_keeps_answered_answerables_and_counts_the_rest():
    kept, excluded = _load().select_rows(ROWS)
    assert [r["id"] for r in kept] == ["q1", "q2"]
    assert excluded == {"unanswerable question": 1, "refused (answerable)": 1,
                        "llm error": 1, "no contexts recorded (older run)": 1}


def test_a_failed_call_is_none_not_zero():
    module = _load()

    async def ok(row):
        return 0.5

    async def boom(row):
        raise RuntimeError("429 Too Many Requests")

    results = asyncio.run(module.score_rows(ROWS[:1], {"faithfulness": ok,
                                                       "answer_relevancy": boom}, repeats=2))
    assert [r["repeat"] for r in results] == [0, 1]
    assert results[0]["faithfulness"] == 0.5
    assert results[0]["answer_relevancy"] is None
    assert "429" in results[0]["errors"]["answer_relevancy"]


def test_summary_averages_each_repeat_first():
    module = _load()
    results = [
        {"corpus": "a", "id": "q1", "repeat": 0, "faithfulness": 1.0},
        {"corpus": "a", "id": "q2", "repeat": 0, "faithfulness": 0.0},
        {"corpus": "a", "id": "q1", "repeat": 1, "faithfulness": 1.0},
        {"corpus": "a", "id": "q2", "repeat": 1, "faithfulness": None},
    ]
    s = module.summarize(results)["a"]["faithfulness"]
    assert s["min"] == 0.5 and s["max"] == 1.0 and s["mean"] == pytest.approx(0.75)
    assert s["repeats"] == 2 and s["failed_calls"] == 1
    assert module.summarize(results)["a"]["answer_relevancy"]["mean"] is None


def _run(module, monkeypatch, tmp_path, *extra):
    src = tmp_path / "answers.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in ROWS), encoding="utf-8")
    out = tmp_path / "ragas.jsonl"
    monkeypatch.setattr(sys, "argv", ["run_ragas_eval.py", "--results", str(src),
                                      "--out", str(out), *extra])
    module.main()
    return out


def test_without_yes_it_only_prints_the_cost(monkeypatch, tmp_path, capsys):
    module = _load()
    monkeypatch.setattr(module, "build_scorers", lambda *a: pytest.fail("judge built"))
    out = _run(module, monkeypatch, tmp_path, "--repeats", "3")
    printed = capsys.readouterr().out
    assert "would score 2 rows x 3 repeats x 3 metrics = 18 metric calls" in printed
    assert "dry run" in printed and not out.exists()


def test_with_yes_it_scores_writes_and_summarizes(monkeypatch, tmp_path, capsys):
    module = _load()
    built = []

    async def fixed(row):
        return 0.8

    def fake_build(judge, embed):
        built.append((judge, embed))
        return {name: fixed for name in module.METRICS}

    monkeypatch.setattr(module, "build_scorers", fake_build)
    out = _run(module, monkeypatch, tmp_path, "--yes", "--repeats", "2",
               "--judge-deployment", "judge-dep")
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4 and {r["judge"] for r in rows} == {"judge-dep"}
    assert built[0][0] == "judge-dep"
    printed = capsys.readouterr().out
    assert "faithfulness       0.800" in printed
    assert "left out, refused (answerable): 1" in printed


def test_repeats_below_one_is_refused(monkeypatch, tmp_path):
    module = _load()
    with pytest.raises(SystemExit):
        _run(module, monkeypatch, tmp_path, "--repeats", "0")
