"""Offline tests for scripts/run_eval.py: the eval must measure the app's retrieval.

Loaders and `retrieve` are monkeypatched, so no index, embedder or network is used.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from rag.types import Chunk, Retrieved

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_eval.py"


def _load_script():
    """Import scripts/run_eval.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("run_eval", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def run_eval(monkeypatch, tmp_path):
    """Script module with fake loaders and a recording `retrieve`, plus a 1-item eval file."""
    module = _load_script()
    eval_file = tmp_path / "eval.jsonl"
    eval_file.write_text(
        json.dumps({"question": "q?", "relevant_sources": ["a.txt"]}) + "\n", encoding="utf-8"
    )
    calls = []

    def fake_retrieve(question, store, embedder, top_k, bm25=None, mode=None):
        calls.append({"bm25": bm25, "mode": mode, "top_k": top_k})
        return [Retrieved(Chunk(id="c1", source="a.txt", ordinal=0, corpus="c", text="t"), 0.9)]

    monkeypatch.setattr(module.VectorStore, "load", staticmethod(lambda path: "store"))
    monkeypatch.setattr(module, "get_embedder", lambda: "embedder")
    monkeypatch.setattr(module, "retrieve", fake_retrieve)
    module.calls = calls
    module.eval_file = eval_file
    return module


def _main(module, monkeypatch, *extra):
    monkeypatch.setattr(
        sys, "argv", ["run_eval.py", "--eval-file", str(module.eval_file), "--k", "3", *extra]
    )
    module.main()


def test_eval_passes_bm25_and_configured_mode(run_eval, monkeypatch, capsys):
    monkeypatch.setattr(run_eval.BM25Index, "load", staticmethod(lambda path: "bm25"))
    monkeypatch.setattr(run_eval.config, "RETRIEVAL_MODE", "hybrid")
    _main(run_eval, monkeypatch)
    assert run_eval.calls == [{"bm25": "bm25", "mode": "hybrid", "top_k": 3}]
    assert "mode=hybrid" in capsys.readouterr().out


def test_eval_dense_mode_override(run_eval, monkeypatch, capsys):
    monkeypatch.setattr(run_eval.BM25Index, "load", staticmethod(lambda path: "bm25"))
    _main(run_eval, monkeypatch, "--mode", "dense")
    assert run_eval.calls[0]["mode"] == "dense"
    assert "mode=dense" in capsys.readouterr().out


def test_eval_falls_back_to_dense_without_bm25_index(run_eval, monkeypatch, capsys):
    def missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(run_eval.BM25Index, "load", staticmethod(missing))
    _main(run_eval, monkeypatch, "--mode", "hybrid")
    assert run_eval.calls[0]["bm25"] is None
    out = capsys.readouterr().out
    assert "mode=dense" in out
    assert "recall@3=1.000" in out


def test_hit_rate_comes_with_count_and_wilson_interval(run_eval, monkeypatch, capsys):
    monkeypatch.setattr(run_eval.BM25Index, "load", staticmethod(lambda path: "bm25"))
    _main(run_eval, monkeypatch)
    out = capsys.readouterr().out
    assert "questions=1" in out
    # 1/1 hit: Wilson gives [0.207-1.000], the normal approximation would give [1-1].
    assert "hit_rate@3=1.000 (95% CI [0.207-1.000], n=1)" in out


def test_defaults_point_at_the_sections_corpus_and_its_eval_file(run_eval, monkeypatch, capsys):
    monkeypatch.setattr(run_eval.BM25Index, "load", staticmethod(lambda path: "bm25"))
    assert run_eval.DEFAULT_CORPUS == "ai_act_sections"
    assert run_eval.default_eval_file("filings_sections") == (
        run_eval.config.ROOT / "eval" / "filings_sections_eval.jsonl"
    )
    asked = []

    def fake_default(corpus):
        asked.append(corpus)
        return run_eval.eval_file

    monkeypatch.setattr(run_eval, "default_eval_file", fake_default)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--corpus", "filings_sections"])
    run_eval.main()
    assert asked == ["filings_sections"]
    assert "corpus=filings_sections" in capsys.readouterr().out
