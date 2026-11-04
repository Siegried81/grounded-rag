"""Tests for the optional Qdrant backend behind `VectorStore.search` (rag/qdrant_index.py).

These mirror `test_ann.py`: a small stand-in `qdrant_client` module stands in
for the real client, so the wiring is checked on every machine with no server
running and without `qdrant-client` installed. The checks are the same
invariants as the FAISS backend: the backend is only used when
VECTOR_INDEX=qdrant, Qdrant only proposes rows and the store scores them with
its own cosine, a missing package is a clear error, and the default path never
touches qdrant.

The stand-in's `search` is an exact inner-product search over the upserted unit
vectors, so when it proposes every row the result must equal the brute-force
path; a `propose` filter lets a test imitate a collection that misses rows.
"""

import sys
import types

import numpy as np
import pytest

import config
from rag.store import VectorStore
from rag.types import Chunk


def _store(n=200, dim=16, seed=0, zero_row=True):
    """A store of `n` random vectors (one of them all zeros) and the vectors themselves."""
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    if zero_row:
        vecs[3] = 0.0
    chunks = [Chunk(f"c{i}", f"text {i}", "a.txt", i, "demo") for i in range(n)]
    store = VectorStore("unused", "fake")
    store.add(chunks, vecs.tolist())
    return store, vecs


def _fake_qdrant(propose=None):
    """A stand-in qdrant_client module: an exact inner-product search that can drop rows.

    `propose(rows)` filters the rows a search returns, to imitate a collection
    that misses some neighbours. Records every created collection in
    `module.built` so a test can count how often the index is (re)built.
    """
    module = types.SimpleNamespace(built=[])

    class _Models:
        class Distance:
            DOT = "Dot"

        class VectorParams:
            def __init__(self, size, distance):
                self.size = size
                self.distance = distance

        class PointStruct:
            def __init__(self, id, vector):
                self.id = id
                self.vector = vector

    class _Hit:
        def __init__(self, id):
            self.id = id

    class QdrantClient:
        def __init__(self, url=None):
            self.url = url
            self._points = {}

        def recreate_collection(self, collection_name, vectors_config):
            self._points = {}
            module.built.append((collection_name, vectors_config.size))

        def upsert(self, collection_name, points):
            for p in points:
                self._points[p.id] = np.asarray(p.vector, dtype=np.float32)

        def search(self, collection_name, query_vector, limit):
            ids = sorted(self._points)
            data = np.stack([self._points[i] for i in ids])
            scores = data @ np.asarray(query_vector, dtype=np.float32)
            order = [ids[j] for j in np.argsort(-scores, kind="stable")]
            if propose:
                order = propose(order)
            return [_Hit(i) for i in order[:limit]]

    module.QdrantClient = QdrantClient
    module.models = _Models
    return module


def _install(monkeypatch, module):
    """Install the stand-in as `qdrant_client` and its `models` submodule."""
    monkeypatch.setitem(sys.modules, "qdrant_client", module)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", module.models)


def _ids(results):
    return [r.chunk.id for r in results]


def test_default_backend_never_touches_qdrant(monkeypatch):
    monkeypatch.setattr(config, "VECTOR_INDEX", "numpy")
    monkeypatch.setitem(sys.modules, "qdrant_client", None)  # any import would raise
    store, vecs = _store()
    assert len(store.search(vecs[0].tolist(), 5)) == 5


def test_qdrant_backend_returns_exactly_the_brute_force_results_when_it_finds_them(monkeypatch):
    store, vecs = _store()
    q = (vecs[7] + 0.1).tolist()
    monkeypatch.setattr(config, "VECTOR_INDEX", "numpy")
    exact = store.search(q, 10)
    _install(monkeypatch, _fake_qdrant())
    monkeypatch.setattr(config, "VECTOR_INDEX", "qdrant")
    approx = store.search(q, 10)
    assert _ids(approx) == _ids(exact)
    assert [r.score for r in approx] == pytest.approx([r.score for r in exact], abs=1e-6)


def test_a_missed_neighbour_is_missing_but_scores_do_not_move(monkeypatch):
    store, vecs = _store()
    q = vecs[11].tolist()
    monkeypatch.setattr(config, "VECTOR_INDEX", "numpy")
    exact = store.search(q, 6)
    missed = int(exact[0].chunk.id[1:])
    _install(monkeypatch, _fake_qdrant(lambda rows: [r for r in rows if r != missed]))
    monkeypatch.setattr(config, "VECTOR_INDEX", "qdrant")
    approx = store.search(q, 5)
    assert _ids(approx) == _ids(exact)[1:]
    assert [r.score for r in approx] == pytest.approx([r.score for r in exact[1:]], abs=1e-6)


def test_zero_vector_scores_zero_and_ties_keep_index_order(monkeypatch):
    store = VectorStore("unused", "fake")
    chunks = [Chunk(f"c{i}", "t", "a.txt", i, "demo") for i in range(4)]
    store.add(chunks, [[0, 0], [1, 0], [1, 0], [0, 1]])
    _install(monkeypatch, _fake_qdrant(lambda rows: list(reversed(rows))))
    monkeypatch.setattr(config, "VECTOR_INDEX", "qdrant")
    res = store.search([1, 0], 4)
    assert _ids(res) == ["c1", "c2", "c0", "c3"]
    assert res[2].score == 0.0


def test_the_collection_is_built_once_and_rebuilt_after_add(monkeypatch):
    fake = _fake_qdrant()
    _install(monkeypatch, fake)
    monkeypatch.setattr(config, "VECTOR_INDEX", "qdrant")
    store, vecs = _store(n=20, zero_row=False)
    store.search(vecs[0].tolist(), 3)
    store.search(vecs[1].tolist(), 3)
    assert len(fake.built) == 1
    store.add([Chunk("new", "t", "b.txt", 0, "demo")], [vecs[0].tolist()])
    assert "new" in _ids(store.search(vecs[0].tolist(), 2))
    assert len(fake.built) == 2


def test_collection_vector_size_matches_the_store_dimension(monkeypatch):
    fake = _fake_qdrant()
    _install(monkeypatch, fake)
    monkeypatch.setattr(config, "VECTOR_INDEX", "qdrant")
    store, vecs = _store(n=12, dim=24, zero_row=False)
    store.search(vecs[0].tolist(), 3)
    assert fake.built == [(config.QDRANT_COLLECTION, 24)]


def test_missing_qdrant_client_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    monkeypatch.setattr(config, "VECTOR_INDEX", "qdrant")
    store, vecs = _store(n=10)
    with pytest.raises(RuntimeError, match="qdrant-client"):
        store.search(vecs[0].tolist(), 3)
