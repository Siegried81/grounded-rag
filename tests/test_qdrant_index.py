"""Tests for the optional Qdrant backend behind `VectorStore.search` (rag/qdrant_index.py).

These mirror `test_ann.py`: a small stand-in `qdrant_client` module stands in
for the real client, so the wiring is checked on every machine with no server
running and without `qdrant-client` installed. The checks are the same
invariants as the FAISS backend: the backend is only used when
VECTOR_INDEX=qdrant, Qdrant only proposes rows and the store scores them with
its own cosine, a missing package is a clear error, and the default path never
touches qdrant.

The stand-in's `query_points` is an exact inner-product search over the upserted
unit vectors, so when it proposes every row the result must equal the brute-force
path; a `propose` filter lets a test imitate a collection that misses rows.

The stand-in keeps its points in one `server` dict keyed by collection name and
shared by every client it creates, as a real server would. That is what lets
`test_two_stores_in_one_process_do_not_share_a_collection` see two stores
overwrite each other's points when they use the same collection name.
"""

import sys
import types

import numpy as np
import pytest

import config
from rag.store import VectorStore
from rag.types import Chunk


def _store(n=200, dim=16, seed=0, zero_row=True, index_dir="unused"):
    """A store of `n` random vectors (one of them all zeros) and the vectors themselves."""
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    if zero_row:
        vecs[3] = 0.0
    chunks = [Chunk(f"c{i}", f"text {i}", "a.txt", i, "demo") for i in range(n)]
    store = VectorStore(index_dir, "fake")
    store.add(chunks, vecs.tolist())
    return store, vecs


def _fake_qdrant(propose=None, collection_exists=None):
    """A stand-in qdrant_client module: an exact inner-product search that can drop rows.

    `propose(rows)` filters the rows a search returns, to imitate a collection
    that misses some neighbours. `collection_exists`, when given, replaces the
    client's method, so a test can make the first server call raise. Keeps each
client it builds in `module.clients`, so a test can read the key it got. Records
    every created collection in `module.built` so a test can count how often the
    index is (re)built, and keeps all points in `module.server`, keyed by
    collection name and shared by every client, like one real server.
    """
    module = types.SimpleNamespace(built=[], server={}, clients=[], queries=[])

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

        class SearchParams:
            def __init__(self, hnsw_ef=None, exact=False):
                self.hnsw_ef = hnsw_ef
                self.exact = exact

    class _Hit:
        def __init__(self, id):
            self.id = id

    class QdrantClient:
        def __init__(self, url=None, api_key=None):
            self.url = url
            self.api_key = api_key
            module.clients.append(self)

        def collection_exists(self, collection_name):
            if collection_exists is not None:
                return collection_exists(collection_name)
            return collection_name in module.server

        def delete_collection(self, collection_name):
            module.server.pop(collection_name, None)

        def create_collection(self, collection_name, vectors_config):
            module.server[collection_name] = {}
            module.built.append((collection_name, vectors_config.size))

        def upsert(self, collection_name, points):
            for p in points:
                module.server[collection_name][p.id] = np.asarray(p.vector, dtype=np.float32)

        def query_points(self, collection_name, query, limit, search_params=None):
            module.queries.append({"limit": limit, "search_params": search_params})
            points = module.server[collection_name]
            ids = sorted(points)
            data = np.stack([points[i] for i in ids])
            scores = data @ np.asarray(query, dtype=np.float32)
            order = [ids[j] for j in np.argsort(-scores, kind="stable")]
            if propose:
                order = propose(order)
            return types.SimpleNamespace(points=[_Hit(i) for i in order[:limit]])

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
    store, vecs = _store(n=12, dim=24, zero_row=False, index_dir="ai_act")
    store.search(vecs[0].tolist(), 3)
    assert fake.built == [(f"{config.QDRANT_COLLECTION}_ai_act", 24)]


def test_two_stores_in_one_process_do_not_share_a_collection(monkeypatch):
    """Two corpora loaded side by side (api/main.py, app.py) must not overwrite each other.

    With one shared collection, store B's build replaces store A's points, and
    A's next search maps B's point ids onto A's rows: wrong passages with
    plausible scores. Each store needs its own collection.
    """
    fake = _fake_qdrant()
    _install(monkeypatch, fake)
    store_a, vecs_a = _store(n=40, seed=1, zero_row=False, index_dir="corpus_a")
    store_b, vecs_b = _store(n=5, dim=16, seed=2, zero_row=False, index_dir="corpus_b")
    q = (vecs_a[7] + 0.1).tolist()
    monkeypatch.setattr(config, "VECTOR_INDEX", "numpy")
    exact_a = store_a.search(q, 10)
    monkeypatch.setattr(config, "VECTOR_INDEX", "qdrant")
    assert _ids(store_a.search(q, 10)) == _ids(exact_a)
    assert len(store_b.search(vecs_b[0].tolist(), 3)) == 3
    again = store_a.search(q, 10)
    assert _ids(again) == _ids(exact_a)
    assert [r.score for r in again] == pytest.approx([r.score for r in exact_a], abs=1e-6)
    assert len({name for name, _ in fake.built}) == 2


def test_missing_qdrant_client_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    monkeypatch.setattr(config, "VECTOR_INDEX", "qdrant")
    store, vecs = _store(n=10)
    with pytest.raises(RuntimeError, match="qdrant-client"):
        store.search(vecs[0].tolist(), 3)


def test_unreachable_server_is_a_clear_error(monkeypatch):
    def refuse(collection_name):
        raise ConnectionError("connection refused")

    _install(monkeypatch, _fake_qdrant(collection_exists=refuse))
    monkeypatch.setattr(config, "VECTOR_INDEX", "qdrant")
    monkeypatch.setattr(config, "QDRANT_URL", "http://qdrant.test:6333")
    store, vecs = _store(n=10)
    with pytest.raises(RuntimeError, match="not reachable at http://qdrant.test:6333.*VECTOR_INDEX=numpy"):
        store.search(vecs[0].tolist(), 3)


def test_api_key_is_sent_when_set(monkeypatch):
    module = _fake_qdrant()
    _install(monkeypatch, module)
    monkeypatch.setattr(config, "VECTOR_INDEX", "qdrant")
    monkeypatch.setattr(config, "QDRANT_API_KEY", "secret-key")
    store, vecs = _store(n=10)
    store.search(vecs[0].tolist(), 3)
    assert [c.api_key for c in module.clients] == ["secret-key"]


def test_no_api_key_is_sent_to_a_local_server(monkeypatch):
    module = _fake_qdrant()
    _install(monkeypatch, module)
    monkeypatch.setattr(config, "VECTOR_INDEX", "qdrant")
    monkeypatch.setattr(config, "QDRANT_API_KEY", "")
    store, vecs = _store(n=10)
    store.search(vecs[0].tolist(), 3)
    assert [c.api_key for c in module.clients] == [None]


def test_ef_search_reaches_qdrant_as_hnsw_ef_and_limit_is_top_k(monkeypatch):
    """The beam width is Qdrant's hnsw_ef search parameter, not a larger limit."""
    module = _fake_qdrant()
    _install(monkeypatch, module)
    monkeypatch.setattr(config, "VECTOR_INDEX", "qdrant")
    monkeypatch.setattr(config, "FAISS_EF_SEARCH", 64)
    store, vecs = _store(n=200)
    store.search(vecs[0].tolist(), 5)
    (query,) = module.queries
    assert query["limit"] == 5
    assert query["search_params"].hnsw_ef == 64
