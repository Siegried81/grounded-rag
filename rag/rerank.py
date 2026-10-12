"""Optional cross-encoder re-ranker over the fused candidate pool (`RERANKER=cross_encoder`).

Dense retrieval embeds the question and each passage separately, then compares
the two vectors; BM25 counts shared terms. A cross-encoder reads the question
and the passage TOGETHER and outputs one relevance score, which orders a short
candidate list more sharply than either. It is too slow to run over a whole
corpus, so it only re-orders the pool retrieval already built — after the
refusal gate, before MMR and the `TOP_K` cut.

It never touches the gate: a passage's displayed `score` stays its dense cosine
(or BM25 score for a keyword-only hit), and the cross-encoder score only decides
the ORDER. So a refusal means exactly what it meant before; what changes is
which of the admitted passages reach the model. That changes answers, hence
off by default.

`sentence-transformers` is imported lazily: the default install does not have
it (it pulls in torch), and this module must import without it.
"""

from __future__ import annotations

from functools import lru_cache

import config


class CrossEncoderReranker:
    """Scores (question, passage) pairs with a sentence-transformers CrossEncoder.

    The model is loaded on first use, not at construction, so building one
    costs nothing until a question is actually re-ranked.
    """

    def __init__(self, model: str):
        self.model = model
        self._encoder = None

    def score(self, question: str, passages: list[str]) -> list[float]:
        """One relevance score per passage, higher is more relevant, in input order."""
        if not passages:
            return []
        if self._encoder is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError(
                    "RERANKER=cross_encoder needs the `sentence-transformers` package "
                    "(pip install -r requirements-optional.txt), or set RERANKER=none."
                ) from exc
            self._encoder = CrossEncoder(self.model)
        scores = self._encoder.predict([(question, p) for p in passages])
        return [float(s) for s in scores]


@lru_cache(maxsize=None)
def _cached(model: str) -> CrossEncoderReranker:
    """One re-ranker per model name per process, so the model loads once."""
    return CrossEncoderReranker(model)


def get_reranker():
    """The configured re-ranker, or None when `config.RERANKER` is "none" (the default)."""
    if config.RERANKER == "none":
        return None
    return _cached(config.RERANK_MODEL)


def rerank(question: str, ranked: list, reranker) -> tuple[list, dict[str, float]]:
    """Re-order `ranked` (Retrieved items) by cross-encoder score; also return the scores by id.

    Ties keep their incoming order (Python's sort is stable), so equal scores
    leave the fused ranking as it was. The returned scores feed MMR's
    relevance term in place of the fused or dense one.
    """
    scores = reranker.score(question, [r.chunk.text for r in ranked])
    by_id = {r.chunk.id: s for r, s in zip(ranked, scores)}
    return sorted(ranked, key=lambda r: -by_id[r.chunk.id]), by_id
