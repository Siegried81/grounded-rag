"""Optional approximate-nearest-neighbour index (FAISS HNSW) behind `VectorStore.search`.

The brute-force NumPy product in `store.py` stays the default and the reference:
at the shipped scale (under 1,000 chunks per corpus) it is exact and fast. An
HNSW graph only pays off from around 10^5 chunks, where a full matrix-vector
product per query starts to cost.

FAISS only PROPOSES rows here. The store then scores those rows with its own
cosine formula, on its own stored vectors, and orders them as the brute-force
path does, so a passage found through FAISS carries the score
`SCORE_THRESHOLD` was set against. What the ANN path can change is recall — a
passage the graph fails to propose is missing — never the score of a passage it
returns. That is why it is opt-in (`VECTOR_INDEX=faiss`).

`faiss` is imported lazily: the default install does not have it, and this
module must import (and the test suite run) without it.
"""

from __future__ import annotations

import numpy as np

import config

# HNSW graph degree: FAISS's usual range is 16-64; 32 is the common middle
# ground between build time, memory and recall.
HNSW_M = 32


def _load_faiss():
    """Import faiss, or explain how to install it."""
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "VECTOR_INDEX=faiss needs the `faiss-cpu` package: "
            "pip install -r requirements-optional.txt, or set VECTOR_INDEX=numpy."
        ) from exc
    return faiss


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length; a zero row stays zero (cosine 0, as in store.search)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


class FaissIndex:
    """HNSW inner-product graph over unit-length copies of a store's vectors.

    On unit vectors, inner product is cosine, so the graph searches the same
    similarity the brute-force path ranks by. It is built in memory when a
    store first searches with it: persisting it would be a second artifact
    that could drift from `vectors.npy`.
    """

    def __init__(self, vectors: np.ndarray, ef_search: int | None = None):
        """Build the graph over `vectors` (n x d)."""
        faiss = _load_faiss()
        unit = np.ascontiguousarray(_unit_rows(np.asarray(vectors, dtype=np.float32)))
        self._n = unit.shape[0]
        self._index = faiss.IndexHNSWFlat(unit.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efSearch = ef_search or config.FAISS_EF_SEARCH
        self._index.add(unit)

    def candidates(self, query_vec, top_k: int) -> list[int]:
        """Row numbers the graph proposes for `query_vec`, at least `top_k` of them when it can.

        Asks for max(top_k, efSearch) neighbours, so the list is never narrower
        than the graph's own search beam; the caller scores and cuts it.
        """
        if self._n == 0 or top_k <= 0:
            return []
        q = _unit_rows(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))
        _, ids = self._index.search(q, min(self._n, max(top_k, self._index.hnsw.efSearch)))
        return [int(i) for i in ids[0] if i >= 0]
