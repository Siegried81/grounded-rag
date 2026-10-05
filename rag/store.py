"""A tiny persistent cosine-similarity vector store backed by NumPy.

At this project's scale (thousands of chunks) a brute-force, vectorised cosine
search is fast and far simpler than an external vector DB, and it keeps the index
as two inspectable files (`vectors.npy` + `store.json`) that can be rebuilt or
diffed without special tooling.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rag.types import Chunk, Retrieved

_VECTORS_FILE = "vectors.npy"
_META_FILE = "store.json"


class VectorStore:
    """In-memory chunks + vectors with cosine search and save/load to disk.

    The embedding `model` name is persisted next to the data so a caller can
    refuse to query an index built in a different embedding space.
    """

    def __init__(self, index_dir: Path, model: str):
        """Create an empty store that will persist to `index_dir`."""
        self.index_dir = Path(index_dir)
        self.model = model
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        """Return the number of stored chunks."""
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Append chunks with their vectors; lengths must match.

        Raising on a mismatch prevents a silent chunk/vector misalignment, which
        would return the wrong passage for a query with no visible error.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must have the same length "
                f"({len(chunks)} != {len(vectors)})"
            )
        if not chunks:
            return
        new = np.asarray(vectors, dtype=np.float32)
        if new.ndim != 2:
            raise ValueError("vectors must be a 2-D list of equal-length rows")
        if len(self._chunks) == 0:
            self._vectors = new
        else:
            self._vectors = np.vstack([self._vectors, new])
        self._chunks.extend(chunks)

    def search(self, query_vec: list[float], top_k: int) -> list[Retrieved]:
        """Return the `top_k` most cosine-similar chunks, best first.

        Vectorised as one matrix-vector product on L2-normalised rows. Zero-norm
        vectors get a score of 0 instead of NaN so they simply rank last.
        """
        if not self._chunks or top_k <= 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        norms = np.linalg.norm(self._vectors, axis=1)
        denom = norms * q_norm
        dots = self._vectors @ q
        scores = np.divide(dots, denom, out=np.zeros_like(dots), where=denom > 0)
        k = min(top_k, len(self._chunks))
        order = np.argsort(-scores, kind="stable")[:k]
        return [Retrieved(self._chunks[i], float(scores[i])) for i in order]

    def vectors_for_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Return the stored vector for each requested chunk id that exists.

        Exposed so a reranker (e.g. MMR) can measure passage-to-passage
        similarity without re-embedding text it already has vectors for.
        """
        index = {c.id: i for i, c in enumerate(self._chunks)}
        return {
            cid: self._vectors[index[cid]].tolist()
            for cid in ids
            if cid in index
        }

    def save(self) -> None:
        """Write vectors and chunk metadata to `index_dir`.

        Vectors go in a binary .npy (compact, exact); chunks in JSON (readable),
        in the same order so row i of the array matches chunk i.
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / _VECTORS_FILE, self._vectors)
        payload = {"model": self.model, "chunks": [asdict(c) for c in self._chunks]}
        (self.index_dir / _META_FILE).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        """Rebuild a store from a directory written by `save`."""
        index_dir = Path(index_dir)
        vec_path = index_dir / _VECTORS_FILE
        meta_path = index_dir / _META_FILE
        if not vec_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                f"No vector index found in {index_dir} "
                f"(expected {_VECTORS_FILE} and {_META_FILE}); run ingestion first."
            )
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        store = cls(index_dir, payload["model"])
        store._chunks = [Chunk(**c) for c in payload["chunks"]]
        store._vectors = np.load(vec_path).astype(np.float32, copy=False)
        # Search maps row i of the matrix to chunk i; a half-written index (one
        # file from an older ingest) would silently return the wrong passages.
        if len(store._vectors) != len(store._chunks):
            raise ValueError(
                f"Index in {index_dir} is inconsistent ({len(store._chunks)} chunks, "
                f"{len(store._vectors)} vectors); run ingestion again."
            )
        return store
