"""Offline tests for reciprocal rank fusion and MMR."""

import pytest

from rag.fusion import mmr, reciprocal_rank_fusion


def test_rrf_promotes_item_ranked_well_in_multiple_lists():
    """An item high in two lists beats one that is first in only one."""
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["x", "b", "y"]])
    assert fused[0][0] == "b"
    assert [s for _, s in fused] == sorted((s for _, s in fused), reverse=True)


def test_rrf_score_formula():
    """Scores are the sum of 1/(k+rank) with 1-based ranks."""
    fused = dict(reciprocal_rank_fusion([["a"], ["a"]], k=10))
    assert fused["a"] == pytest.approx(2 / 11)


def test_rrf_different_lengths_and_disjoint_ids():
    """Disjoint ids and uneven lists are all kept; empties are tolerated."""
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["x"], []])
    assert {i for i, _ in fused} == {"a", "b", "c", "x"}
    assert reciprocal_rank_fusion([]) == []


def test_mmr_relevance_diversity_and_top_k():
    """Most relevant first, then a diverse pick over a near-duplicate."""
    cands = [
        ("dup", [0.99, 0.14]),
        ("best", [1.0, 0.0]),
        ("diverse", [0.5, 0.87]),
    ]
    out = mmr([1.0, 0.0], cands, lambda_mult=0.3, top_k=2)
    assert out == ["best", "diverse"]
    assert len(mmr([1.0, 0.0], cands, top_k=10)) == 3


def test_mmr_pure_relevance_ignores_diversity():
    """lambda=1 orders purely by similarity to the query."""
    cands = [("dup", [0.99, 0.14]), ("best", [1.0, 0.0]), ("diverse", [0.5, 0.87])]
    assert mmr([1.0, 0.0], cands, lambda_mult=1.0, top_k=3) == ["best", "dup", "diverse"]


def test_mmr_empty_and_zero_vectors():
    """Empty input gives []; zero-norm vectors do not produce errors."""
    assert mmr([1.0, 0.0], []) == []
    out = mmr([0.0, 0.0], [("z", [0.0, 0.0]), ("a", [1.0, 0.0])], top_k=2)
    assert sorted(out) == ["a", "z"]


def test_mmr_relevance_overrides_query_cosine():
    # "a" matches the query vector, but the supplied relevance favours "b".
    cands = [("a", [1.0, 0.0]), ("b", [0.0, 1.0])]
    assert mmr([1.0, 0.0], cands, lambda_mult=1.0, top_k=2) == ["a", "b"]
    assert mmr([1.0, 0.0], cands, lambda_mult=1.0, top_k=2, relevance=[0.0, 1.0]) == ["b", "a"]
