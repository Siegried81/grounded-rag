"""Optional Qdrant-backed approximate-nearest-neighbour index behind `VectorStore.search`.

This mirrors `rag/ann.py` (the FAISS backend): the brute-force NumPy product in
`store.py` stays the default and the reference. Qdrant is a second opt-in ANN
option (`VECTOR_INDEX=qdrant`) for deployments that already run a Qdrant server
or need an index that lives outside the process.

Qdrant only PROPOSES rows here. The store then scores those rows with its own
cosine formula, on its own stored vectors, and orders them as the brute-force
path does, so a passage found through Qdrant carries the score
`SCORE_THRESHOLD` was set against. The ANN path can only lose recall: a passage
it doesn't propose is missing. It can't change the score of one it returns.
Same invariant as the FAISS backend.

The collection mirrors the store's matrix: point id `i` is row `i` of that
store's matrix; each store gets its own collection, named
`<QDRANT_COLLECTION>_<index dir>`, so two stores loaded in one process never
overwrite each other's points. The collection's vector size is the store's
dimension, and its distance is DOT (inner product) over unit-length copies of
the vectors, which is cosine, the same trick as `rag/ann.py`. Vectors are
upserted lazily on the first search, not in the constructor, so a store that is
never searched through Qdrant never touches the server.

`QDRANT_API_KEY` is sent only when it is set: Qdrant Cloud requires it, and a
local server started without a key accepts requests that carry none.

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
                 collection: str | None = None, ef_search: int | None = None,
                 api_key: str | None = None):
        """Record `vectors` (n x d); defer the client and upsert to the first search."""
        self._unit = np.ascontiguousarray(_unit_rows(np.asarray(vectors, dtype=np.float32)))
        self._n = self._unit.shape[0]
        self._dim = int(self._unit.shape[1]) if self._unit.ndim == 2 else 0
        self._url = url or config.QDRANT_URL
        self._api_key = api_key or config.QDRANT_API_KEY or None
        self._collection = collection or config.QDRANT_COLLECTION
        # Reuse the HNSW search beam knob so the two ANN backends widen the beam
        # the same way: passed as SearchParams.hnsw_ef in candidates().
        self._ef_search = ef_search or config.FAISS_EF_SEARCH
        self._client = None  # built on the first candidates() call

    def _ensure_ready(self) -> None:
        """Connect, (re)create the collection and upsert the vectors, once.

        recreate_collection is deprecated in current qdrant-client, so this
        does the same in two steps: delete if it exists, then create. The
        collection then stays in step with the store's vectors.

        The first server call is collection_exists; it is caught broadly
        because the client raises different exception types depending on the
        transport, and a user with no server running needs one clear message.
        """
        if self._client is not None:
            return
        QdrantClient, models = _load_qdrant()
        client = QdrantClient(url=self._url, api_key=self._api_key)
        try:
            exists = client.collection_exists(self._collection)
        except Exception as exc:
            raise RuntimeError(
                f"Qdrant server not reachable at {self._url}; "
                "start one or set VECTOR_INDEX=numpy"
            ) from exc
        if exists:
            client.delete_collection(self._collection)
        client.create_collection(
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

        The HNSW beam width goes to Qdrant as `SearchParams(hnsw_ef=ef_search)`,
        the knob that sets recall; `limit` is only how many results come back,
        so it is `top_k` (Qdrant widens the beam to `limit` itself when
        `limit > hnsw_ef`). The caller scores and cuts the list. Same contract
        as `FaissIndex.candidates()`.
        """
        if self._n == 0 or top_k <= 0:
            return []
        self._ensure_ready()
        _, models = _load_qdrant()
        q = _unit_rows(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))[0]
        response = self._client.query_points(
            collection_name=self._collection,
            query=q.tolist(),
            limit=min(self._n, top_k),
            search_params=models.SearchParams(hnsw_ef=self._ef_search),
        )
        return [int(p.id) for p in response.points if p.id is not None]
