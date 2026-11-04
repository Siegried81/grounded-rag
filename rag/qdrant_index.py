"""Optional Qdrant-backed approximate-nearest-neighbour index behind `VectorStore.search`.

This mirrors `rag/ann.py` (the FAISS backend): the brute-force NumPy product in
`store.py` stays the default and the reference. Qdrant is a second opt-in ANN
option (`VECTOR_INDEX=qdrant`) for deployments that already run a Qdrant server
or need an index that lives outside the process.

Qdrant only PROPOSES rows here. The store then scores those rows with its own
cosine formula, on its own stored vectors, and orders them as the brute-force
path does, so a passage found through Qdrant carries the score
`SCORE_THRESHOLD` was set against. What the ANN path can change is recall — a
passage Qdrant fails to propose is missing — never the score of a passage it
returns. Same invariant as the FAISS backend.

The collection is built in memory's image: point id `i` is row `i` of the
store's vector matrix, the collection's vector size is the store's dimension,
and its distance is DOT (inner product) over unit-length copies of the vectors,
which equals cosine — the same trick `rag/ann.py` uses. Vectors are upserted
lazily on the first search, not in the constructor, so a store that is never
searched through Qdrant never touches the server.

`qdrant_client` is imported lazily: the default install does not have it, and
this module must import (and the test suite run) without it.
"""

from __future__ import annotations

import numpy as np

import config


def _load_qdrant():
    """Import qdrant_client, or explain how to install it."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client import models
    except ImportError as exc:
        raise RuntimeError(
            "VECTOR_INDEX=qdrant needs the `qdrant-client` package: "
            "pip install -r requirements-optional.txt, or set VECTOR_INDEX=numpy."
        ) from exc
    return QdrantClient, models


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length; a zero row stays zero (cosine 0, as in store.search)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


class QdrantIndex:
    """A Qdrant collection over unit-length copies of a store's vectors.

    On unit vectors, inner product (DOT) is cosine, so the collection searches
    the same similarity the brute-force path ranks by. Point id `i` is row `i`,
    so `candidates()` returns row numbers the caller can score directly. The
    collection is (re)created and filled on the first search: that keeps it in
    step with `vectors.npy` rather than being a second artifact that could drift.
    """

    def __init__(self, vectors: np.ndarray, url: str | None = None,
                 collection: str | None = None, ef_search: int | None = None):
        """Record `vectors` (n x d); defer the client and upsert to the first search."""
        self._unit = np.ascontiguousarray(_unit_rows(np.asarray(vectors, dtype=np.float32)))
        self._n = self._unit.shape[0]
        self._dim = int(self._unit.shape[1]) if self._unit.ndim == 2 else 0
        self._url = url or config.QDRANT_URL
        self._collection = collection or config.QDRANT_COLLECTION
        # Reuse the HNSW search beam knob so the two ANN backends widen the beam
        # the same way; the store rescores and cuts the result regardless.
        self._ef_search = ef_search or config.FAISS_EF_SEARCH
        self._client = None  # built on the first candidates() call

    def _ensure_ready(self) -> None:
        """Connect, (re)create the collection and upsert the vectors, once."""
        if self._client is not None:
            return
        QdrantClient, models = _load_qdrant()
        client = QdrantClient(url=self._url)
        client.recreate_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=self._dim, distance=models.Distance.DOT
            ),
        )
        if self._n:
            client.upsert(
                collection_name=self._collection,
                points=[
                    models.PointStruct(id=i, vector=self._unit[i].tolist())
                    for i in range(self._n)
                ],
            )
        self._client = client

    def candidates(self, query_vec, top_k: int) -> list[int]:
        """Row numbers Qdrant proposes for `query_vec`, at least `top_k` of them when it can.

        Asks for max(top_k, ef_search) neighbours, so the list is never narrower
        than the search beam; the caller scores and cuts it. Same contract as
        `FaissIndex.candidates()`.
        """
        if self._n == 0 or top_k <= 0:
            return []
        self._ensure_ready()
        q = _unit_rows(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))[0]
        limit = min(self._n, max(top_k, self._ef_search))
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=q.tolist(),
            limit=limit,
        )
        return [int(h.id) for h in hits if h.id is not None]
