"""Offline tests for the BM25 keyword index."""

import pytest

from rag.lexical import BM25Index, tokenize
from rag.types import Chunk


def _chunk(i: str, text: str) -> Chunk:
    """Build a minimal chunk for tests."""
    return Chunk(id=i, text=text, source="s.txt", ordinal=0, corpus="c", meta={})


CHUNKS = [
    _chunk("a", "The cat sat on the mat."),
    _chunk("b", "Quarterly revenue grew thanks to the ERP-2024 rollout."),
    _chunk("c", "Bananas are yellow fruit."),
    _chunk("d", "Totally different words here."),
]


def test_tokenize_lowercases_splits_and_drops_empties():
    """Punctuation and underscores split tokens; case is folded."""
    assert tokenize("Hello,  World_42!! ERP-2024") == ["hello", "world", "42", "erp", "2024"]
    assert tokenize("   ") == []


def test_search_ranks_matching_chunk_first():
    """The chunk containing the query terms outranks unrelated ones."""
    res = BM25Index.build(CHUNKS).search("revenue ERP rollout", top_k=3)
    assert res[0].chunk.id == "b"
    assert all(res[i].score >= res[i + 1].score for i in range(len(res) - 1))


def test_empty_query_and_empty_index():
    """Empty query or empty index returns no results."""
    assert BM25Index.build(CHUNKS).search("   ", top_k=3) == []
    empty = BM25Index.build([])
    assert len(empty) == 0
    assert empty.search("cat", top_k=3) == []


def test_len():
    """__len__ counts indexed chunks."""
    assert len(BM25Index.build(CHUNKS)) == 4


def test_save_load_round_trip(tmp_path):
    """A reloaded index returns identical results."""
    idx = BM25Index.build(CHUNKS)
    path = tmp_path / "sub" / "bm25.json"
    idx.save(path)
    loaded = BM25Index.load(path)
    assert len(loaded) == len(idx)
    for q in ["cat mat", "revenue", "fruit yellow"]:
        assert loaded.search(q, 3) == idx.search(q, 3)


def test_load_missing_file(tmp_path):
    """Loading a missing file raises a clear FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="BM25 index file not found"):
        BM25Index.load(tmp_path / "nope.json")
