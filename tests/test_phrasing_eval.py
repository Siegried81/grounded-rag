"""Offline tests for scripts/run_phrasing_eval.py and the phrasing dataset.

`load_corpus`, the embedder and `retrieve` are monkeypatched, so no index or
network is used. The dataset test ties every group to the retrieval eval set it
was taken from, so the gold sources cannot drift apart.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from rag.types import Chunk, Retrieved

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "run_phrasing_eval.py"
_DATASET = _ROOT / "eval" / "phrasings_eval.jsonl"


def _load_script():
    """Import scripts/run_phrasing_eval.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("run_phrasing_eval", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pe = _load_script()


def _item(group, style, question, corpus="c", sources=("a.txt",)):
    return {"corpus": corpus, "group": group, "style": style, "question": question,
            "relevant_sources": list(sources)}


ITEMS = [
    _item("g1", "original", "formal one"), _item("g1", "casual", "casual one"),
    _item("g1", "other_language", "translated one"),
    _item("g2", "original", "formal two"), _item("g2", "casual", "casual two MISS"),
    _item("g2", "other_language", "translated two"),
]


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Module with a fake retriever that misses any question containing "MISS"."""
    eval_file = tmp_path / "phrasings.jsonl"
    eval_file.write_text("".join(json.dumps(i) + "\n" for i in ITEMS), encoding="utf-8")
    calls = []

    def fake_retrieve(question, store, embedder, top_k, bm25=None, mode=None):
        calls.append({"question": question, "top_k": top_k, "mode": mode, "bm25": bm25})
        source = "other.txt" if "MISS" in question else "a.txt"
        return [Retrieved(Chunk("c1", "t", source, 0, "c"), 0.9)]

    monkeypatch.setattr(pe, "retrieve", fake_retrieve)
    monkeypatch.setattr(pe, "load_corpus", lambda corpus: ("store", "bm25"))
    monkeypatch.setattr(pe, "get_embedder", lambda: "embedder")
    pe.calls, pe.eval_file = calls, eval_file
    return pe


def test_per_style_hits_and_consistency(harness, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_phrasing_eval.py", "--eval-file", str(harness.eval_file),
                                      "--k", "3", "--mode", "hybrid"])
    harness.main()
    out = capsys.readouterr().out
    assert harness.calls[0] == {"question": "formal one", "top_k": 3, "mode": "hybrid",
                                "bm25": "bm25"}
    assert "hit@3 original         2/2 (100%, 95% CI 34%-100%)" in out
    assert "hit@3 casual           1/2 (50%, 95% CI 9%-91%)" in out
    assert "hit@3 other_language   2/2 (100%, 95% CI 34%-100%)" in out
    assert "phrasing consistency    1/2 (50%, 95% CI 9%-91%)" in out
    assert "groups=2 items=6" in out


def test_corpus_filter_and_corpus_cache(harness, monkeypatch, capsys):
    loads = []
    monkeypatch.setattr(harness, "load_corpus", lambda c: loads.append(c) or ("s", None))
    monkeypatch.setattr(sys, "argv", ["run_phrasing_eval.py", "--eval-file", str(harness.eval_file),
                                      "--corpus", "c", "nope"])
    harness.main()
    assert loads == ["c"]  # loaded once, not once per item
    assert "mode={'c': 'dense'}" in capsys.readouterr().out  # no BM25 -> dense, as the app


def test_consistency_counts_groups_not_items():
    rows = [{"group": "g", "hit": 1}, {"group": "g", "hit": 0}, {"group": "h", "hit": 0},
            {"group": "h", "hit": 0}]
    assert pe.consistency(rows) == (1, 2)
    assert pe.hits_by_style([{"style": "casual", "hit": 1}, {"style": "original", "hit": 0}]) == {
        "original": (0, 1), "casual": (1, 1),
    }


# --- dataset sanity ---------------------------------------------------------

def _originals(corpus: str) -> dict[str, list[str]]:
    items = pe.load_eval(_ROOT / "eval" / f"{corpus}_eval.jsonl")
    return {i["question"]: i["relevant_sources"] for i in items}


def test_dataset_groups_are_complete_and_anchored_to_the_eval_sets():
    items = pe.load_eval(_DATASET)
    groups = {}
    for i in items:
        assert i["style"] in pe.STYLES, i
        assert i["question"].strip() and i["relevant_sources"], i
        groups.setdefault(i["group"], []).append(i)
    assert len(groups) >= 6
    for name, members in groups.items():
        assert sorted(m["style"] for m in members) == sorted(pe.STYLES), name
        assert len({m["corpus"] for m in members}) == 1, name
        assert len({tuple(m["relevant_sources"]) for m in members}) == 1, name
        assert len({m["question"] for m in members}) == 3, name
        original = next(m for m in members if m["style"] == "original")
        known = _originals(original["corpus"])
        assert original["question"] in known, name
        assert known[original["question"]] == original["relevant_sources"], name
