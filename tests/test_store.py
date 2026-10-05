"""Offline tests for the NumPy vector store: search, clamping, validation, persistence."""

import numpy as np
import pytest

from rag.store import VectorStore
from rag.types import Chunk


def _chunk(i: str) -> Chunk:
    """Build a minimal chunk with the given id."""
    return Chunk(id=i, text=f"text {i}", source="a.md", ordinal=0, corpus="c", meta={"page": 1})


def _store(tmp_path) -> VectorStore:
    """Build a 3-chunk store with axis-aligned vectors."""
    s = VectorStore(tmp_path / "idx", "fake-model")
    s.add([_chunk("x"), _chunk("y"), _chunk("z")], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    return s


def test_search_nearest_first(tmp_path):
    res = _store(tmp_path).search([1, 0, 0], 3)
    assert res[0].chunk.id == "x"
    assert res[0].score == pytest.approx(1.0)
    assert res[1].score == pytest.approx(0.0)
    assert [r.score for r in res] == sorted((r.score for r in res), reverse=True)


def test_search_empty_store(tmp_path):
    assert VectorStore(tmp_path, "m").search([1, 0], 3) == []


def test_top_k_clamped(tmp_path):
    assert len(_store(tmp_path).search([1, 0, 0], 50)) == 3


def test_zero_norm_vector_does_not_nan(tmp_path):
    s = VectorStore(tmp_path, "m")
    s.add([_chunk("a"), _chunk("b")], [[0, 0], [1, 0]])
    res = s.search([1, 0], 2)
    assert res[0].chunk.id == "b"
    assert res[1].score == 0.0


def test_add_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError):
        VectorStore(tmp_path, "m").add([_chunk("a")], [[1, 0], [0, 1]])


def test_save_load_roundtrip(tmp_path):
    s = _store(tmp_path)
    s.save()
    loaded = VectorStore.load(tmp_path / "idx")
    assert len(loaded) == 3
    assert loaded.model == "fake-model"
    assert loaded._chunks == s._chunks
    assert (loaded._vectors == s._vectors).all()
    assert loaded.search([0, 1, 0], 1)[0].chunk.id == "y"


def test_load_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path / "nope")


def test_load_rejects_chunk_vector_count_mismatch(tmp_path):
    # A half-written index (vectors from an older ingest) would otherwise map
    # row i to the wrong chunk and return the wrong passage without any error.
    s = _store(tmp_path)
    s.save()
    np.save(tmp_path / "idx" / "vectors.npy", np.eye(2, 3, dtype=np.float32))
    with pytest.raises(ValueError, match="3 chunks, 2 vectors"):
        VectorStore.load(tmp_path / "idx")
