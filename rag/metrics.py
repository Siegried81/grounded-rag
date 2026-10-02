"""Retrieval quality metrics over ranked lists of ids.

Pure functions with no I/O so retrieval quality can be measured and unit-tested
offline, instead of being asserted. Every function takes the ranked ids returned by
retrieval (best first) and the set of ids judged relevant.
"""

from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant ids found in the top-k results.

    Answers "did we find everything we should have?". Returns 0.0 when there are
    no relevant ids (nothing to recall) or k <= 0; k larger than the list is fine.
    """
    if not relevant_ids or k <= 0:
        return 0.0
    found = set(retrieved_ids[:k]) & relevant_ids
    return len(found) / len(relevant_ids)


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if any relevant id appears in the top-k results, else 0.0.

    A coarse "was the answer retrievable at all" signal; averaged over a question
    set it gives the hit rate. Empty inputs or k <= 0 give 0.0.
    """
    if k <= 0:
        return 0.0
    return 1.0 if any(i in relevant_ids for i in retrieved_ids[:k]) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant id (1/rank, rank starts at 1), else 0.0.

    Rewards putting a relevant chunk near the top; averaged over questions it is
    the Mean Reciprocal Rank.
    """
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k results that are relevant.

    The denominator is the number of results actually returned (at most k), so a
    short list is not penalised for missing slots. Empty list or k <= 0 gives 0.0.
    """
    if k <= 0:
        return 0.0
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    return sum(1 for i in top if i in relevant_ids) / len(top)
