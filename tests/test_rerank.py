"""Tests for the optional cross-encoder re-ranker (rag/rerank.py) and its place in `retrieve`.

A fake re-ranker stands in for the model: no model is ever loaded or downloaded
here. What is pinned down is the contract: off by default, it re-orders only
the pool that passed the refusal gate, it never changes a passage's score, its
scores drive MMR's relevance, and a missing package is a clear error.
"""

import sys

import pytest

import config
import rag.rerank as rr
from rag.retrieve import retrieve
from tests.test_retrieve import FakeBM25, FakeEmbedder, FakeEmbedder2D, FakeStore, _r


class FakeReranker:
    """Scores each passage from a preset table keyed by its text; records its calls."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def score(self, question, passages):
        self.calls.append((question, list(passages)))
        return [self.table[p] for p in passages]


def _ids(results):
    return [r.chunk.id for r in results]


def test_off_by_default_and_never_imports_sentence_transformers(monkeypatch):
    monkeypatch.setattr(config, "RERANKER", "none")
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert rr.get_reranker() is None
    store = FakeStore([_r(1, 0.9), _r(2, 0.8)])
    assert _ids(retrieve("q", store, FakeEmbedder(), top_k=2, threshold=0.5, use_mmr=False)) == ["1", "2"]


def test_reorders_the_pool_and_keeps_every_score():
    store = FakeStore([_r(1, 0.9), _r(2, 0.8), _r(3, 0.7)])
    ce = FakeReranker({"t1": -2.0, "t2": 0.5, "t3": 3.0})
    out = retrieve("why?", store, FakeEmbedder(), top_k=2, threshold=0.5, use_mmr=False,
                   reranker=ce)
    assert _ids(out) == ["3", "2"]
    assert [r.score for r in out] == [0.7, 0.8]  # dense cosines, unchanged
    assert ce.calls == [("why?", ["t1", "t2", "t3"])]


def test_the_gate_is_untouched_by_the_reranker():
    store = FakeStore([_r(1, 0.4), _r(2, 0.3)])
    ce = FakeReranker({"t1": 9.0, "t2": 9.0})
    assert retrieve("q", store, FakeEmbedder(), top_k=2, threshold=0.5, reranker=ce) == []
    assert ce.calls == []  # a refusal never reaches the model

    store = FakeStore([_r(1, 0.9), _r(2, 0.3)])
    out = retrieve("q", store, FakeEmbedder(), top_k=2, threshold=0.5, use_mmr=False,
                   reranker=FakeReranker({"t1": 0.0, "t2": 9.0}))
    assert _ids(out) == ["1"]  # a passage below the gate cannot be promoted back in


def test_hybrid_pool_includes_keyword_hits_and_ties_keep_fused_order():
    store = FakeStore([_r(1, 0.9)])
    bm25 = FakeBM25([_r(9, 5.0), _r(1, 2.0)])
    ce = FakeReranker({"t1": 1.0, "t9": 1.0})
    fused = retrieve("q", store, FakeEmbedder(), top_k=2, threshold=0.5, bm25=bm25,
                     mode="hybrid", use_mmr=False, reranker=None)
    reranked = retrieve("q", store, FakeEmbedder(), top_k=2, threshold=0.5, bm25=bm25,
                        mode="hybrid", use_mmr=False, reranker=ce)
    assert sorted(ce.calls[0][1]) == ["t1", "t9"]
    assert _ids(reranked) == _ids(fused)


def test_reranker_scores_drive_mmr_relevance():
    # Dense cosine with the query puts 1 first; the cross-encoder prefers 3, and
    # MMR must diversify that order rather than fall back to dense cosine.
    store = FakeStore(
        [_r(1, 0.9), _r(2, 0.85), _r(3, 0.8)],
        vectors={"1": [1.0, 0.0], "2": [0.6, 0.8], "3": [0.6, -0.8]},
    )
    ce = FakeReranker({"t1": -1.0, "t2": 0.0, "t3": 4.0})
    out = retrieve("q", store, FakeEmbedder2D(), top_k=1, threshold=0.5, mode="dense",
                   use_mmr=True, reranker=ce)
    assert _ids(out) == ["3"]


def test_config_selects_one_cached_cross_encoder(monkeypatch):
    rr._cached.cache_clear()
    monkeypatch.setattr(config, "RERANKER", "cross_encoder")
    monkeypatch.setattr(config, "RERANK_MODEL", "some/cross-encoder")
    first, second = rr.get_reranker(), rr.get_reranker()
    assert first is second and first.model == "some/cross-encoder"
    rr._cached.cache_clear()


def test_cross_encoder_scores_pairs_with_a_lazily_loaded_model(monkeypatch):
    seen = {}

    class FakeCrossEncoder:
        def __init__(self, name):
            seen["name"] = name

        def predict(self, pairs):
            seen["pairs"] = pairs
            return [0.25 * i for i in range(len(pairs))]

    fake_module = type(sys)("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    ce = rr.CrossEncoderReranker("m")
    assert seen == {}  # nothing loaded at construction
    assert ce.score("q", ["a", "b"]) == [0.0, 0.25]
    assert seen == {"name": "m", "pairs": [("q", "a"), ("q", "b")]}
    assert ce.score("q", []) == []


def test_missing_sentence_transformers_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        rr.CrossEncoderReranker("m").score("q", ["a"])
