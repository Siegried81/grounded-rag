"""Retrieval step: find the chunks relevant enough to ground an answer.

Dense cosine search is the semantic backbone and the refusal gate; an optional
BM25 keyword channel is fused in (Reciprocal Rank Fusion) to catch exact terms
embeddings miss, and MMR reorders the survivors to drop near-duplicate passages.

The refusal rule is deliberately anchored to the DENSE cosine score: if nothing
clears `threshold` there, we return [] (the caller's signal to refuse) regardless
of retrieval mode. The gate is per query, not per passage: once one dense hit
clears it, hybrid mode also lets in passages BM25 found on its own, whatever
their cosine. That's on purpose: BM25 is there to catch exact terms like an
article number that embeddings miss. Such a passage gets its dense cosine as
`score` and `score_source="bm25"`, so nobody mistakes it for a dense hit. The
optional cross-encoder re-ranker (rag/rerank.py, off by default) re-orders the
admitted pool and never changes a score.

Kept as a thin function over duck-typed `store`, `embedder` and `bm25` so it stays
testable with fakes and the backends remain swappable.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import config
from rag.fusion import mmr, reciprocal_rank_fusion
from rag.rerank import get_reranker, rerank
from rag.types import Retrieved

# Default of `retrieve(reranker=...)`: take the re-ranker from config.RERANKER.
# None cannot serve, because None is how a caller asks for no re-ranking.
_FROM_CONFIG = object()


class IndexMismatchError(ValueError):
    """The index was built with another embedding model than the one configured now."""


def check_same_embedder(store, embedder) -> None:
    """Raise IndexMismatchError if `store` was built with another model than `embedder`.

    Cosine scores are only comparable inside one embedding space. A query
    embedded by a different model than the index gives scores with no meaning,
    and the refusal threshold (SCORE_THRESHOLD) with them, without any error:
    the answers just get quietly worse. Duck-typed stores or embedders without a
    `model` (test fakes) are not checked.
    """
    built, current = getattr(store, "model", None), getattr(embedder, "model", None)
    if built and current and built != current:
        raise IndexMismatchError(
            f"index built with embedding model {built!r}, but the configured embedder is "
            f"{current!r}: re-run ingestion (python cli.py ingest --corpus ...) or restore the setting"
        )


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
    reranker=_FROM_CONFIG,
) -> list[Retrieved]:
    """Return up to `top_k` chunks for `question`, or [] to signal a refusal.

    With `bm25` supplied and `mode="hybrid"`, dense and keyword rankings are fused;
    otherwise retrieval is dense-only. An empty result means nothing in the corpus
    is similar enough (by dense cosine) to ground an answer.

    `reranker` (default: `config.RERANKER`, which is off) re-orders the pool that
    passed the gate before MMR and the `top_k` cut; None disables it. Anything
    with `score(question, passages) -> list[float]` will do.
    """
    if reranker is _FROM_CONFIG:
        reranker = get_reranker()
    mode = mode or config.RETRIEVAL_MODE
    use_mmr = config.USE_MMR if use_mmr is None else use_mmr
    candidate_k = candidate_k or config.CANDIDATE_K

    check_same_embedder(store, embedder)
    query_vec = embedder.embed([question])[0]
    dense = store.search(query_vec, candidate_k)

    # Refusal gate: anchored to dense cosine so its meaning never changes.
    dense_pass = [r for r in dense if r.score >= threshold]
    if not dense_pass:
        return []

    # MMR's relevance: the RRF rank in hybrid mode, the cross-encoder scores
    # after the re-ranker, else (dense mode) the query cosine.
    by_rank = False
    ce_scores: dict[str, float] | None = None
    if mode == "dense" or bm25 is None:
        ranked = dense_pass
    else:
        # Fuse the two channels by rank. Every passage shows its dense cosine as
        # `score`; `score_source` says which channel found it.
        sparse = bm25.search(question, candidate_k)
        by_id = _label_channels(dense_pass, dense, sparse, query_vec, store)
        fused = reciprocal_rank_fusion(
            [[r.chunk.id for r in dense_pass], [r.chunk.id for r in sparse]]
        )
        ranked = [by_id[cid] for cid, _ in fused if cid in by_id]
        by_rank = True

    if reranker is not None and len(ranked) > 1:
        # A sharper ORDER for the admitted pool; scores (and the gate) untouched.
        # Its scores, min-max rescaled, become MMR's relevance. They aren't
        # bunched like RRF's, and measured they beat the rank (decisions.md).
        ranked, ce_scores = rerank(question, ranked, reranker)

    if use_mmr and len(ranked) > 1 and hasattr(store, "vectors_for_ids"):
        vecs = store.vectors_for_ids([r.chunk.id for r in ranked])
        candidates = [(r.chunk.id, vecs[r.chunk.id]) for r in ranked if r.chunk.id in vecs]
        if candidates:
            relevance = None
            if ce_scores is not None:
                relevance = _min_max([ce_scores[cid] for cid, _ in candidates])
            elif by_rank:
                # In hybrid mode MMR must diversify the fused ranking: scoring
                # relevance by dense cosine would discard the BM25 contribution.
                # `candidates` is in fused order, so relevance is just the
                # rank, spread evenly over 0..1.
                relevance = _rank_relevance(len(candidates))
            order = mmr(
                query_vec,
                candidates,
                lambda_mult=config.MMR_LAMBDA,
                top_k=top_k,
                relevance=relevance,
            )
            by_id = {r.chunk.id: r for r in ranked}
            diversified = [by_id[cid] for cid in order]
            # Append any ranked items MMR could not place (no vector), keeping order.
            placed = set(order)
            diversified += [r for r in ranked if r.chunk.id not in placed]
            ranked = diversified

    return ranked[:top_k]


def _min_max(raw: list[float]) -> list[float]:
    """Rescale `raw` to 0..1 (all 1.0 when they're equal)."""
    lo, hi = min(raw), max(raw)
    return [(s - lo) / (hi - lo) if hi > lo else 1.0 for s in raw]


def _rank_relevance(n: int) -> list[float]:
    """MMR relevance for `n` ranked candidates: 1 - (r-1)/(n-1), so 1.0 down to 0.0.

    Min-max on the RRF scores doesn't work. With k=60 they're bunched (1/61 to
    1/80 per list), so after rescaling the passages both lists found sit at
    0.92 to 1.0 and the others at 0 to 0.18. For the last places the
    redundancy term then beat relevance and MMR picked mostly by diversity.
    The rank keeps the order and spaces it evenly. Measured in docs/decisions.md
    (2026-11-10).
    """
    if n <= 1:
        return [1.0] * n
    return [1.0 - i / (n - 1) for i in range(n)]


def _cosines(query_vec, vectors: dict[str, list[float]]) -> dict[str, float]:
    """Cosine of `query_vec` with each vector, by id. A zero vector gets 0.0."""
    q = np.asarray(query_vec, dtype=np.float32)
    q_norm = float(np.linalg.norm(q))
    out = {}
    for cid, vec in vectors.items():
        v = np.asarray(vec, dtype=np.float32)
        denom = q_norm * float(np.linalg.norm(v))
        out[cid] = float(v @ q) / denom if denom > 0 else 0.0
    return out


def _label_channels(dense_pass, dense, sparse, query_vec, store) -> dict[str, Retrieved]:
    """Every fused passage by id, with its dense cosine and the channel that found it.

    A passage in both lists is "both", one only dense hits found is "dense", one
    only BM25 found is "bm25". A BM25-only hit gets its cosine from the dense
    candidate list when it's there (below the threshold), else from its stored
    vector. No cosine gate on these: BM25 is there to rescue exact terms.
    """
    sparse_ids = {r.chunk.id for r in sparse}
    by_id = {
        r.chunk.id: replace(r, score_source="both" if r.chunk.id in sparse_ids else "dense")
        for r in dense_pass
    }
    known = {r.chunk.id: r.score for r in dense}
    missing = [r.chunk.id for r in sparse if r.chunk.id not in by_id and r.chunk.id not in known]
    if missing and hasattr(store, "vectors_for_ids"):
        known.update(_cosines(query_vec, store.vectors_for_ids(missing)))
    for r in sparse:
        if r.chunk.id not in by_id:
            # No vector (a fake store): the raw BM25 score stays, still labelled "bm25".
            by_id[r.chunk.id] = Retrieved(r.chunk, known.get(r.chunk.id, r.score), "bm25")
    return by_id


def top_dense_score(question: str, store, embedder) -> float | None:
    """Best dense cosine of `question` over the whole store, or None for an empty store.

    This is the number the refusal gate compares with the threshold. The first
    passage `retrieve` returns is not it: RRF and MMR re-order, so in hybrid
    mode that passage often has a lower cosine than the best match. Threshold
    analysis needs this value, and it is also what the UIs show on a refusal.
    Costs one extra query embedding.
    """
    best = store.search(embedder.embed([question])[0], 1)
    return best[0].score if best else None
