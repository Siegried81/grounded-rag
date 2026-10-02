"""Retrieval step: find the chunks relevant enough to ground an answer.

Dense cosine search is the semantic backbone and the refusal gate; an optional
BM25 keyword channel is fused in (Reciprocal Rank Fusion) to catch exact terms
embeddings miss, and MMR reorders the survivors to drop near-duplicate passages.

The refusal rule is deliberately anchored to the DENSE cosine score: if nothing
clears `threshold` there, we return [] (the caller's signal to refuse) regardless
of retrieval mode. Hybrid fusion and MMR only reorder/diversify passages that are
already semantically relevant — they never lower the bar for answering.

Kept as a thin function over duck-typed `store`, `embedder` and `bm25` so it stays
testable with fakes and the backends remain swappable.
"""

from __future__ import annotations

import config
from rag.fusion import mmr, reciprocal_rank_fusion
from rag.types import Retrieved


def retrieve(
    question: str,
    store,
    embedder,
    top_k: int = config.TOP_K,
    threshold: float = config.SCORE_THRESHOLD,
    bm25=None,
    mode: str | None = None,
    use_mmr: bool | None = None,
    candidate_k: int | None = None,
) -> list[Retrieved]:
    """Return up to `top_k` chunks for `question`, or [] to signal a refusal.

    With `bm25` supplied and `mode="hybrid"`, dense and keyword rankings are fused;
    otherwise retrieval is dense-only. An empty result means nothing in the corpus
    is similar enough (by dense cosine) to ground an answer.
    """
    mode = mode or config.RETRIEVAL_MODE
    use_mmr = config.USE_MMR if use_mmr is None else use_mmr
    candidate_k = candidate_k or config.CANDIDATE_K

    query_vec = embedder.embed([question])[0]
    dense = store.search(query_vec, candidate_k)

    # Refusal gate: anchored to dense cosine so its meaning never changes.
    dense_pass = [r for r in dense if r.score >= threshold]
    if not dense_pass:
        return []

    if mode == "dense" or bm25 is None:
        ranked = dense_pass
    else:
        # Fuse the two channels by rank. Dense passages keep their cosine score
        # (useful for display and verification); keyword-only hits carry their
        # BM25 score as a fallback.
        sparse = bm25.search(question, candidate_k)
        by_id = {r.chunk.id: r for r in dense_pass}
        for r in sparse:
            by_id.setdefault(r.chunk.id, r)
        fused = reciprocal_rank_fusion(
            [[r.chunk.id for r in dense_pass], [r.chunk.id for r in sparse]]
        )
        ranked = [by_id[cid] for cid, _ in fused if cid in by_id]

    if use_mmr and len(ranked) > 1 and hasattr(store, "vectors_for_ids"):
        vecs = store.vectors_for_ids([r.chunk.id for r in ranked])
        candidates = [(r.chunk.id, vecs[r.chunk.id]) for r in ranked if r.chunk.id in vecs]
        if candidates:
            order = mmr(query_vec, candidates, lambda_mult=config.MMR_LAMBDA, top_k=top_k)
            by_id = {r.chunk.id: r for r in ranked}
            diversified = [by_id[cid] for cid in order]
            # Append any ranked items MMR could not place (no vector), keeping order.
            placed = set(order)
            diversified += [r for r in ranked if r.chunk.id not in placed]
            ranked = diversified

    return ranked[:top_k]
