"""Tests for the optional FAISS backend behind `VectorStore.search` (rag/ann.py).

Most tests install a small stand-in `faiss` module, so the wiring is checked on
every machine: the backend is only used when VECTOR_INDEX=faiss, the graph only
proposes rows and the store scores them with its own cosine, a missing package
is a clear error, and the default path never imports faiss. One test runs the
real library when it is installed and is skipped otherwise.
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


def _fake_faiss(propose=None):
    """A stand-in faiss module: an exact inner-product 'graph' that can drop rows.

    `propose(rows)` filters the rows the index returns, to imitate a graph that
    misses some neighbours. Records every built index in `module.built`.
    """
    module = types.SimpleNamespace(METRIC_INNER_PRODUCT=0, built=[])

    class IndexHNSWFlat:
        def __init__(self, dim, m, metric):
            self.hnsw = types.SimpleNamespace(efSearch=16)
            self.data = np.empty((0, dim), dtype=np.float32)
            module.built.append(self)

        def add(self, x):
            self.data = np.vstack([self.data, x])

        def search(self, q, k):
            scores = self.data @ q[0]
            rows = list(np.argsort(-scores, kind="stable"))
            if propose:
                rows = propose(rows)
            rows = rows[:k] + [-1] * max(0, k - len(rows))
            return np.zeros((1, k)), np.asarray([rows])

    module.IndexHNSWFlat = IndexHNSWFlat
    return module


def _ids(results):
    return [r.chunk.id for r in results]


def test_default_backend_never_imports_faiss(monkeypatch):
    monkeypatch.setattr(config, "VECTOR_INDEX", "numpy")
    monkeypatch.setitem(sys.modules, "faiss", None)  # any import of faiss would raise
    store, vecs = _store()
    assert len(store.search(vecs[0].tolist(), 5)) == 5


def test_faiss_backend_returns_exactly_the_brute_force_results_when_it_finds_them(monkeypatch):
    store, vecs = _store()
    q = (vecs[7] + 0.1).tolist()
    monkeypatch.setattr(config, "VECTOR_INDEX", "numpy")
    exact = store.search(q, 10)
    monkeypatch.setitem(sys.modules, "faiss", _fake_faiss())
    monkeypatch.setattr(config, "VECTOR_INDEX", "faiss")
    approx = store.search(q, 10)
    assert _ids(approx) == _ids(exact)
    assert [r.score for r in approx] == pytest.approx([r.score for r in exact], abs=1e-6)


def test_a_missed_neighbour_is_missing_but_scores_do_not_move(monkeypatch):
    store, vecs = _store()
    q = vecs[11].tolist()
    monkeypatch.setattr(config, "VECTOR_INDEX", "numpy")
    exact = store.search(q, 6)
    missed = int(exact[0].chunk.id[1:])
    monkeypatch.setitem(sys.modules, "faiss", _fake_faiss(lambda rows: [r for r in rows if r != missed]))
    monkeypatch.setattr(config, "VECTOR_INDEX", "faiss")
    approx = store.search(q, 5)
    assert _ids(approx) == _ids(exact)[1:]
    assert [r.score for r in approx] == pytest.approx([r.score for r in exact[1:]], abs=1e-6)


def test_zero_vector_scores_zero_and_ties_keep_index_order(monkeypatch):
    store = VectorStore("unused", "fake")
    chunks = [Chunk(f"c{i}", "t", "a.txt", i, "demo") for i in range(4)]
    store.add(chunks, [[0, 0], [1, 0], [1, 0], [0, 1]])
    monkeypatch.setitem(sys.modules, "faiss", _fake_faiss(lambda rows: list(reversed(rows))))
    monkeypatch.setattr(config, "VECTOR_INDEX", "faiss")
    res = store.search([1, 0], 4)
    assert _ids(res) == ["c1", "c2", "c0", "c3"]
    assert res[2].score == 0.0


def test_the_graph_is_built_once_and_rebuilt_after_add(monkeypatch):
    fake = _fake_faiss()
    monkeypatch.setitem(sys.modules, "faiss", fake)
    monkeypatch.setattr(config, "VECTOR_INDEX", "faiss")
    store, vecs = _store(n=20, zero_row=False)
    store.search(vecs[0].tolist(), 3)
    store.search(vecs[1].tolist(), 3)
    assert len(fake.built) == 1
    store.add([Chunk("new", "t", "b.txt", 0, "demo")], [vecs[0].tolist()])
    assert "new" in _ids(store.search(vecs[0].tolist(), 2))
    assert len(fake.built) == 2


def test_missing_faiss_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", None)
    monkeypatch.setattr(config, "VECTOR_INDEX", "faiss")
    store, vecs = _store(n=10)
    with pytest.raises(RuntimeError, match="faiss-cpu"):
        store.search(vecs[0].tolist(), 3)


def test_real_faiss_recall_and_exact_scores(monkeypatch):
    pytest.importorskip("faiss")
    store, vecs = _store(n=3000, dim=32, seed=1)
    rng = np.random.default_rng(2)
    queries = rng.normal(size=(50, 32)).astype(np.float32)
    hits = 0
    for q in queries:
        monkeypatch.setattr(config, "VECTOR_INDEX", "numpy")
        exact = store.search(q.tolist(), 10)
        monkeypatch.setattr(config, "VECTOR_INDEX", "faiss")
        approx = store.search(q.tolist(), 10)
        exact_scores = {r.chunk.id: r.score for r in exact}
        hits += sum(r.chunk.id in exact_scores for r in approx)
        for r in approx:
            if r.chunk.id in exact_scores:
                assert r.score == pytest.approx(exact_scores[r.chunk.id], abs=1e-6)
    assert hits / (10 * len(queries)) >= 0.95
