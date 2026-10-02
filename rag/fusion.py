"""Rank fusion and diversity selection over generic ranked lists.

Pure functions on ids and vectors (no store or lexical imports) so they can
merge dense and BM25 results and de-duplicate near-identical passages.
"""

from __future__ import annotations

import numpy as np


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Merge ranked id lists (best first) by summing 1/(k+rank) per list.

    RRF uses ranks only, so dense cosine scores and BM25 scores, which live on
    incompatible scales, can be combined without normalisation. Ties keep
    first-seen order.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def _unit(v: np.ndarray) -> np.ndarray:
    """Return `v` scaled to unit length, or zeros if its norm is zero."""
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros_like(v)


def mmr(
    query_vec: list[float],
    candidates: list[tuple[str, list[float]]],
    lambda_mult: float = 0.5,
    top_k: int = 5,
    relevance: list[float] | None = None,
) -> list[str]:
    """Maximal Marginal Relevance selection of up to `top_k` candidate ids.

    Greedily picks the candidate maximising
    lambda*rel - (1-lambda)*max sim(already selected), so results stay
    relevant without being near-duplicates. lambda=1 is pure relevance, 0 pure
    diversity. Zero-norm vectors get similarity 0 instead of NaN.

    `rel` is the query cosine by default. Passing `relevance` (one score per
    candidate, ideally in [0, 1]) substitutes another relevance signal, such as
    a fused dense+BM25 score, so MMR diversifies that ranking instead of
    silently replacing it with dense similarity.
    """
    if not candidates or top_k <= 0:
        return []
    ids = [c[0] for c in candidates]
    mat = np.vstack([_unit(np.asarray(c[1], dtype=float)) for c in candidates])
    if relevance is None:
        rel = mat @ _unit(np.asarray(query_vec, dtype=float))
    else:
        rel = np.asarray(relevance, dtype=float)
    sim = mat @ mat.T
    selected: list[int] = []
    remaining = list(range(len(ids)))
    while remaining and len(selected) < top_k:

        def score(i: int) -> float:
            redundancy = max((sim[i, j] for j in selected), default=0.0)
            return lambda_mult * rel[i] - (1.0 - lambda_mult) * redundancy

        best = max(remaining, key=score)  # first max wins ties
        selected.append(best)
        remaining.remove(best)
    return [ids[i] for i in selected]
