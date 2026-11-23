"""Tests for the optional Azure AI Search backend behind `VectorStore.search`.

They mirror `test_qdrant_index.py`. `requests.request` is replaced by a small
in-memory stand-in for the service (`_FakeService`), so the wiring is checked
with no Azure account. The stand-in answers the REST calls the backend makes:
index create/delete/get, document upload, lookup by key, `$count`, and an exact
inner-product vector search. Its `visible_after` lets a test imitate the delay
before uploaded documents become searchable.

The invariants are the Qdrant ones (Azure only proposes rows, the store scores
them, the default path never calls the service) plus the two that are new here:
a current index is reused rather than re-uploaded, and an index holding other
vectors is rebuilt.
"""

import json
import re

import numpy as np
import pytest
import requests

import config
from rag import azure_search_index as azs
from rag.store import VectorStore
from rag.types import Chunk


class _R:
    def __init__(self, status, payload=None, text=None):
        self.status_code = status
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload or {})

    def json(self):
        return self._payload


class _FakeService:
    """One Azure AI Search service: indexes keyed by name, each a dict of documents."""

    def __init__(self, visible_after=0, reject=()):
        self.indexes: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []
        self.definitions: dict[str, dict] = {}
        self.visible_after = visible_after  # $count calls before uploads become visible
        self.reject = set(reject)           # document keys the upload refuses
        self._counts = 0

    def __call__(self, method, url, params=None, headers=None, json=None, timeout=None):
        assert params == {"api-version": azs.API_VERSION}
        assert headers["api-key"] == "admin"
        path = url.split(".net/", 1)[1]
        self.calls.append((method, path))
        m = re.fullmatch(r"indexes/([^/]+)(/docs(?:/(.+))?)?", path)
        name, docs, rest = m.group(1), m.group(2), m.group(3)
        index = self.indexes.get(name)
        if not docs:
            if method == "PUT":
                self.indexes[name] = {}
                self.definitions[name] = json
                return _R(201, json)
            if method == "DELETE":
                return _R(204) if self.indexes.pop(name, None) is not None else _R(404)
            return _R(200, self.definitions[name]) if index is not None else _R(404)
        if index is None:
            return _R(404)
        if rest == "$count":
            self._counts += 1
            n = len(index) if self._counts > self.visible_after else 0
            return _R(200, text=f"﻿{n}")
        if rest == "index":
            out = []
            for d in json["value"]:
                ok = d["id"] not in self.reject
                if ok:
                    index[d["id"]] = d
                out.append({"key": d["id"], "status": ok})
            return _R(207 if len(out) != sum(o["status"] for o in out) else 200, {"value": out})
        if rest == "search":
            q = np.asarray(json["vectorQueries"][0]["vector"], dtype=np.float32)
            ids = sorted(index, key=int)
            scores = np.stack([index[i]["vector"] for i in ids]) @ q
            order = [ids[j] for j in np.argsort(-scores, kind="stable")]
            order = [i for i in order if int(i) not in getattr(self, "hide", ())]
            return _R(200, {"value": [{"id": i} for i in order[: json["top"]]]})
        doc = index.get(rest)
        return _R(200, {"id": doc["id"], "fp": doc["fp"]}) if doc else _R(404)


@pytest.fixture
def service(monkeypatch):
    svc = _FakeService()
    monkeypatch.setattr(azs.requests, "request", svc)
    monkeypatch.setattr(azs.time, "sleep", lambda s: None)
    monkeypatch.setattr(config, "AZURE_SEARCH_ENDPOINT", "https://svc.search.windows.net")
    monkeypatch.setattr(config, "AZURE_SEARCH_API_KEY", "admin")
    monkeypatch.setattr(config, "VECTOR_INDEX", "azure_search")
    return svc


def _store(n=150, dim=16, seed=0, index_dir="ai_act_sections"):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    vecs[3] = 0.0
    chunks = [Chunk(f"c{i}", f"text {i}", "a.txt", i, "demo") for i in range(n)]
    store = VectorStore(index_dir, "fake")
    store.add(chunks, vecs.tolist())
    return store, vecs


def _ids(results):
    return [r.chunk.id for r in results]


def test_default_backend_never_calls_the_service(monkeypatch):
    monkeypatch.setattr(config, "VECTOR_INDEX", "numpy")
    monkeypatch.setattr(azs.requests, "request", lambda *a, **k: pytest.fail("service called"))
    store, vecs = _store()
    assert len(store.search(vecs[0].tolist(), 5)) == 5


def test_returns_exactly_the_brute_force_results_when_it_finds_them(monkeypatch, service):
    store, vecs = _store()
    q = (vecs[7] + 0.1).tolist()
    monkeypatch.setattr(config, "VECTOR_INDEX", "numpy")
    exact = store.search(q, 10)
    monkeypatch.setattr(config, "VECTOR_INDEX", "azure_search")
    approx = store.search(q, 10)
    assert _ids(approx) == _ids(exact)
    assert [r.score for r in approx] == pytest.approx([r.score for r in exact], abs=1e-6)


def test_a_missed_neighbour_is_missing_but_scores_do_not_move(monkeypatch, service):
    store, vecs = _store()
    q = vecs[11].tolist()
    monkeypatch.setattr(config, "VECTOR_INDEX", "numpy")
    exact = {r.chunk.id: r.score for r in store.search(q, 10)}
    monkeypatch.setattr(config, "VECTOR_INDEX", "azure_search")
    service.hide = {11}
    approx = store.search(q, 10)
    assert "c11" not in _ids(approx)
    for r in approx:
        if r.chunk.id in exact:
            assert r.score == pytest.approx(exact[r.chunk.id], abs=1e-6)


def test_first_search_builds_one_index_per_store_with_a_valid_name(service):
    store, vecs = _store(index_dir="ai_act_sections")
    store.search(vecs[0].tolist(), 3)
    assert list(service.indexes) == ["grounded-rag-ai-act-sections"]
    assert len(service.indexes["grounded-rag-ai-act-sections"]) == 150


def test_a_current_index_is_reused_without_uploading_again(monkeypatch, service):
    store, vecs = _store()
    store.search(vecs[0].tolist(), 3)
    service.calls.clear()
    again, _ = _store()  # same vectors, new process-like store
    again.search(vecs[0].tolist(), 3)
    methods = [m for m, _ in service.calls]
    assert "PUT" not in methods and "DELETE" not in methods
    assert not any(p.endswith("/docs/index") for _, p in service.calls)


def test_an_index_holding_other_vectors_is_rebuilt(service):
    first, vecs = _store(seed=0)
    first.search(vecs[0].tolist(), 3)
    other, other_vecs = _store(seed=1)  # same size, same name, different vectors
    other.search(other_vecs[0].tolist(), 3)
    assert ("DELETE", "indexes/grounded-rag-ai-act-sections") in service.calls
    assert other.search(other_vecs[5].tolist(), 1)[0].chunk.id == "c5"


def test_a_half_uploaded_index_is_rebuilt(service):
    store, vecs = _store()
    store.search(vecs[0].tolist(), 3)
    service.indexes["grounded-rag-ai-act-sections"].pop("42")
    service.calls.clear()
    _store()[0].search(vecs[0].tolist(), 3)
    assert ("PUT", "indexes/grounded-rag-ai-act-sections") in service.calls


def test_waits_until_uploaded_documents_are_searchable(monkeypatch, service):
    service.visible_after = 4  # the first $count calls see nothing yet
    store, vecs = _store()
    assert store.search(vecs[9].tolist(), 1)[0].chunk.id == "c9"


def test_gives_up_when_documents_never_become_searchable(monkeypatch, service):
    service.visible_after = 10_000
    store, vecs = _store()
    with pytest.raises(RuntimeError, match="did not reach 150 documents"):
        store.search(vecs[0].tolist(), 1)


def test_a_rejected_document_fails_the_build(service):
    service.reject = {"17"}
    store, vecs = _store()
    with pytest.raises(RuntimeError, match="rejected documents"):
        store.search(vecs[0].tolist(), 1)


def test_missing_settings_are_a_clear_error(monkeypatch, service):
    monkeypatch.setattr(config, "AZURE_SEARCH_API_KEY", "")
    store, vecs = _store()
    with pytest.raises(RuntimeError, match="AZURE_SEARCH_ENDPOINT and an admin"):
        store.search(vecs[0].tolist(), 1)


def test_unreachable_service_is_a_clear_error(monkeypatch, service):
    def down(*a, **k):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(azs.requests, "request", down)
    store, vecs = _store()
    with pytest.raises(RuntimeError, match="not reachable"):
        store.search(vecs[0].tolist(), 1)


def test_index_definition_is_cosine_hnsw_with_ef_search_in_azure_range(service):
    store, vecs = _store(dim=16)
    store.search(vecs[0].tolist(), 1)
    definition = service.definitions["grounded-rag-ai-act-sections"]
    vector_field = [f for f in definition["fields"] if f["name"] == "vector"][0]
    assert vector_field["dimensions"] == 16
    hnsw = definition["vectorSearch"]["algorithms"][0]["hnswParameters"]
    assert hnsw["metric"] == "cosine"
    assert hnsw["efSearch"] == 100  # FAISS_EF_SEARCH=64 clamped to Azure's minimum


@pytest.mark.parametrize("prefix,folder,expected", [
    ("grounded-rag", "ai_act_sections", "grounded-rag-ai-act-sections"),
    ("Grounded_RAG", "filings__sections", "grounded-rag-filings-sections"),
    ("-x-", "a" * 200, "x-" + "a" * 126),
])
def test_index_name_is_valid_for_azure(prefix, folder, expected):
    assert azs.index_name(prefix, folder) == expected


def test_adding_rows_drops_the_index_handle(service):
    store, vecs = _store()
    store.search(vecs[0].tolist(), 1)
    store.add([Chunk("new", "t", "b.txt", 0, "demo")], [vecs[1].tolist()])
    assert store._azure is None
