"""Optional Azure AI Search index behind `VectorStore.search` (`VECTOR_INDEX=azure_search`).

Same contract as `rag/ann.py` (FAISS) and `rag/qdrant_index.py` (Qdrant): the
brute-force NumPy product in `store.py` stays the default and the reference,
and Azure AI Search only PROPOSES rows. The store scores them with its own
cosine, on its own vectors, so a passage found through Azure carries the score
`SCORE_THRESHOLD` was set against and the refusal gate keeps its meaning. The
service can only cost recall, never move a score.

How it differs from the Qdrant backend, and why:

- It talks REST through `requests`, as `rag/llm.py` does, instead of the
  `azure-search-documents` SDK: one less package, and the tests stub the HTTP
  calls the same way the LLM tests do.
- It does not rebuild the index on every start. Qdrant usually runs next to the
  app; Azure is a remote paid service where re-uploading every vector on each
  cold start is slow. Each document carries a fingerprint of the store's
  vectors (`fp`), and the index is reused only when it holds exactly `n`
  documents and its first and last ones carry the current fingerprint.
  Anything else (missing index, other corpus build, half-finished upload) is
  deleted and rebuilt.
- Uploaded documents become searchable a moment later, not at once, so the
  build waits until the document count reaches `n` before the first query.
- HNSW `efSearch` must lie in [100, 1000] on Azure. `FAISS_EF_SEARCH` (64 by
  default) is reused like the Qdrant backend does, clamped into that range.

Document key `i` (a string, as Azure requires) is row `i` of the store's
matrix. The free tier allows three indexes, enough for the two shipped corpora:
one index per store, named `<AZURE_SEARCH_INDEX>-<index dir>`.
"""

from __future__ import annotations

import hashlib
import re
import time

import numpy as np
import requests

import config

API_VERSION = "2024-07-01"
TIMEOUT_S = 60
UPLOAD_BATCH = 100      # documents per upload request, far below Azure's 1000 / 16 MB caps
READY_POLLS = 60        # seconds to wait for uploaded documents to become searchable
EF_SEARCH_RANGE = (100, 1000)


def index_name(prefix: str, index_dir_name: str) -> str:
    """A valid Azure index name: lowercase letters, digits and single dashes, at most 128 chars.

    Corpus folders use underscores (`ai_act_sections`), which Azure rejects.
    """
    name = re.sub(r"[^a-z0-9]+", "-", f"{prefix}-{index_dir_name}".lower()).strip("-")
    return name[:128].rstrip("-")


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length; a zero row stays zero (cosine 0, as in store.search)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


class AzureSearchIndex:
    """An Azure AI Search vector index over unit-length copies of a store's vectors."""

    def __init__(self, vectors: np.ndarray, name: str, endpoint: str | None = None,
                 api_key: str | None = None, ef_search: int | None = None):
        """Record the vectors; the service is first contacted on the first search."""
        self._unit = np.ascontiguousarray(_unit_rows(np.asarray(vectors, dtype=np.float32)))
        self._n = self._unit.shape[0]
        self._dim = int(self._unit.shape[1]) if self._unit.ndim == 2 else 0
        self._name = name
        self._endpoint = (endpoint or config.AZURE_SEARCH_ENDPOINT).rstrip("/")
        self._api_key = api_key or config.AZURE_SEARCH_API_KEY
        low, high = EF_SEARCH_RANGE
        self._ef_search = min(max(ef_search or config.FAISS_EF_SEARCH, low), high)
        self._fingerprint = hashlib.sha256(self._unit.tobytes()).hexdigest()[:16]
        self._ready = False

    def _call(self, method: str, path: str, json_body=None, ok=(200,)) -> requests.Response:
        """One REST call to the service; a status outside `ok` raises with Azure's message."""
        resp = requests.request(
            method,
            f"{self._endpoint}/{path}",
            params={"api-version": API_VERSION},
            headers={"api-key": self._api_key, "Content-Type": "application/json"},
            json=json_body,
            timeout=TIMEOUT_S,
        )
        if resp.status_code not in ok:
            raise RuntimeError(
                f"Azure AI Search {method} {path} returned {resp.status_code}: {resp.text[:300]}"
            )
        return resp

    def _count(self) -> int:
        """Number of searchable documents in the index (the endpoint answers plain text)."""
        text = self._call("GET", f"indexes/{self._name}/docs/$count").text
        return int(text.strip().lstrip("﻿"))

    def _is_current(self) -> bool:
        """True when the index exists and holds exactly this store's vectors."""
        resp = self._call("GET", f"indexes/{self._name}", ok=(200, 404))
        if resp.status_code == 404 or self._count() != self._n:
            return False
        for key in {0, self._n - 1}:
            doc = self._call("GET", f"indexes/{self._name}/docs/{key}", ok=(200, 404))
            if doc.status_code == 404 or doc.json().get("fp") != self._fingerprint:
                return False
        return True

    def _definition(self) -> dict:
        """Index schema: string key, fingerprint, and the vector field under cosine HNSW."""
        return {
            "name": self._name,
            "fields": [
                {"name": "id", "type": "Edm.String", "key": True, "filterable": True},
                {"name": "fp", "type": "Edm.String", "filterable": True},
                {"name": "vector", "type": "Collection(Edm.Single)", "searchable": True,
                 "retrievable": False, "dimensions": self._dim,
                 "vectorSearchProfile": "default"},
            ],
            "vectorSearch": {
                "algorithms": [{"name": "hnsw", "kind": "hnsw",
                                "hnswParameters": {"metric": "cosine",
                                                   "efSearch": self._ef_search}}],
                "profiles": [{"name": "default", "algorithm": "hnsw"}],
            },
        }

    def _rebuild(self) -> None:
        """Delete the index, create it again, upload every row, wait until all are searchable."""
        self._call("DELETE", f"indexes/{self._name}", ok=(204, 404))
        self._call("PUT", f"indexes/{self._name}", self._definition(), ok=(200, 201))
        for start in range(0, self._n, UPLOAD_BATCH):
            docs = [{"@search.action": "upload", "id": str(i), "fp": self._fingerprint,
                     "vector": self._unit[i].tolist()}
                    for i in range(start, min(start + UPLOAD_BATCH, self._n))]
            # 207 means some documents failed: the per-document status says which.
            resp = self._call("POST", f"indexes/{self._name}/docs/index",
                              {"value": docs}, ok=(200, 207))
            failed = [d.get("key") for d in resp.json().get("value", []) if not d.get("status")]
            if failed:
                raise RuntimeError(f"Azure AI Search rejected documents {failed[:5]}")
        for _ in range(READY_POLLS):
            if self._count() >= self._n:
                return
            time.sleep(1)
        raise RuntimeError(
            f"Azure AI Search index {self._name} did not reach {self._n} documents "
            f"within {READY_POLLS} s"
        )

    def _ensure_ready(self) -> None:
        """Reuse the index when it is current, else rebuild it; once per process."""
        if self._ready:
            return
        if not (self._endpoint and self._api_key):
            raise RuntimeError(
                "VECTOR_INDEX=azure_search needs AZURE_SEARCH_ENDPOINT and an admin "
                "AZURE_SEARCH_API_KEY, or set VECTOR_INDEX=numpy"
            )
        try:
            current = self._is_current()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Azure AI Search not reachable at {self._endpoint}; "
                "check the endpoint or set VECTOR_INDEX=numpy"
            ) from exc
        if not current:
            self._rebuild()
        self._ready = True

    def candidates(self, query_vec, top_k: int) -> list[int]:
        """Row numbers the index proposes for `query_vec`; the caller scores and cuts them.

        Same contract as `FaissIndex.candidates()` and `QdrantIndex.candidates()`.
        """
        if self._n == 0 or top_k <= 0:
            return []
        self._ensure_ready()
        q = _unit_rows(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))[0]
        k = min(self._n, top_k)
        resp = self._call("POST", f"indexes/{self._name}/docs/search", {
            "select": "id",
            "top": k,
            "vectorQueries": [{"kind": "vector", "vector": q.tolist(),
                               "fields": "vector", "k": k}],
        })
        return [int(doc["id"]) for doc in resp.json().get("value", [])]
