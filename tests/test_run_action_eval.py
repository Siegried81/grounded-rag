"""Tests for scripts/run_action_eval.py and the labelled set in eval/actions/.

Two kinds of check. The labelled set itself: every labelled sentence is really
in its transcript, every owner id exists, every date parses, so a bad label
cannot pass for a model error. And the scoring: pairing by source sentence,
the counts, and a full run with an "oracle" model that returns the labels,
which must score perfectly, and a model that always fails, which must not.
"""

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pytest

from rag import llm
from rag.actions import ReviewedAction, load_directory
from rag.answer_metrics import normalize_text

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval" / "actions"


def _load():
    spec = importlib.util.spec_from_file_location("run_action_eval",
                                                  ROOT / "scripts" / "run_action_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _items():
    return [json.loads(line) for line in
            (EVAL / "actions_eval.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def _pred(sentence, owner_id=None, due=None, flags=()):
    return ReviewedAction("t", None, owner_id, due, sentence, "high", list(flags))


def test_labelled_set_is_consistent_with_its_transcripts():
    ids = {p.id for p in load_directory(EVAL / "directory.json")}
    items = _items()
    assert len(items) == 6 and sum(len(i["actions"]) for i in items) == 13
    for item in items:
        text = normalize_text((EVAL / "transcripts" / item["transcript"]).read_text(encoding="utf-8"))
        meeting = date.fromisoformat(item["meeting_date"])
        for a in item["actions"]:
            assert normalize_text(a["source_sentence"]) in text, (item["id"], a)
            assert a["owner_id"] is None or a["owner_id"] in ids
            assert a["due_date"] is None or date.fromisoformat(a["due_date"]) >= meeting


def test_pairing_uses_each_side_once_and_skips_weak_overlap():
    module = _load()
    gold = [{"source_sentence": "Sofie will update the incident register before Friday."},
            {"source_sentence": "Jan will send the audit findings by 15 March."}]
    predicted = [_pred("Jan will send the audit findings by 15 March"),
                 _pred("Sofie will update the incident register before Friday."),
                 _pred("Sofie will update the register"),           # second claim on gold 0
                 _pred("Something unrelated entirely")]
    assert module.pair_actions(predicted, gold) == [(0, 1), (1, 0)]


def test_score_meeting_counts_owner_date_and_review():
    module = _load()
    gold = [{"source_sentence": "a b c d", "owner_id": "p01", "due_date": "2026-03-06"},
            {"source_sentence": "e f g h", "owner_id": None, "due_date": None},
            {"source_sentence": "i j k l", "owner_id": "p02", "due_date": None}]
    predicted = [_pred("a b c d", "p01", "2026-03-06"),           # right
                 _pred("e f g h", "p05", None),                    # wrong owner, unflagged
                 _pred("i j k l", None, "2026-01-01", ["owner_missing"]),  # wrong, flagged
                 _pred("x y z w")]                                  # false positive
    c = module.score_meeting(predicted, gold)
    assert (c["predicted"], c["gold"], c["matched"]) == (4, 3, 3)
    assert c["owner_ok"] == 1 and c["date_ok"] == 2  # no date on either side is right
    assert c["wrong_owner_assigned"] == 1
    assert c["errors"] == 2 and c["errors_flagged"] == 1
    assert c["extra"] == 1 and c["extra_flagged"] == 0  # the false positive has no flag


def _oracle(items):
    """A fake model that returns the labelled actions of whichever meeting it is sent."""
    names = {p.id: p.name for p in load_directory(EVAL / "directory.json")}
    by_text = {}
    for item in items:
        text = (EVAL / "transcripts" / item["transcript"]).read_text(encoding="utf-8")
        by_text[text.splitlines()[0]] = {"actions": [
            {"title": "Do it", "owner": names.get(a["owner_id"]), "due_date": a["due_date"],
             "source_sentence": a["source_sentence"], "confidence": "high"}
            for a in item["actions"]]}

    def complete(prompt, system=None):
        first = next(line for line in prompt.splitlines()
                     if line and not line.startswith(("Meeting date", "<transcript>")))
        return json.dumps(by_text[first])

    return complete


def _main(module, monkeypatch, tmp_path):
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(sys, "argv", ["run_action_eval.py", "--out", str(out)])
    module.main()
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


def test_an_oracle_scores_perfectly(monkeypatch, tmp_path, capsys):
    module = _load()
    monkeypatch.setattr(llm, "complete", _oracle(_items()))
    rows = _main(module, monkeypatch, tmp_path)
    assert [r["error"] for r in rows] == [None] * 6
    printed = capsys.readouterr().out
    assert "precision (actions)          13/13" in printed
    assert "recall (actions)             13/13" in printed
    assert "owner accuracy (paired)      13/13" in printed
    assert "wrong owner assigned         0/13" in printed


def test_failed_meetings_count_as_nothing_found(monkeypatch, tmp_path, capsys):
    module = _load()

    def down(prompt, system=None):
        raise llm.LLMError("quota")

    monkeypatch.setattr(llm, "complete", down)
    rows = _main(module, monkeypatch, tmp_path)
    assert all(r["error"] for r in rows)
    printed = capsys.readouterr().out
    assert "meetings failed              6" in printed
    assert "recall (actions)             0/13" in printed


@pytest.mark.parametrize("text", ["", "   "])
def test_jaccard_of_empty_text_is_zero(text):
    assert _load()._jaccard(text, text) == 0.0
