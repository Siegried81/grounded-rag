"""Offline tests for rag.retrieve using tiny in-file fakes (no embedder/store imports).

Covers the refusal gate (anchored to dense cosine), the candidate-pool passthrough,
dense vs hybrid (BM25 fusion) modes, and MMR diversification.
"""

from rag.retrieve import retrieve
from rag.types import Chunk, Retrieved


class FakeEmbedder:
    """Returns a fixed vector and records every batch of texts it was asked to embed."""

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeStore:
    """Returns a preset result list and records the arguments of each search."""

    def __init__(self, results, vectors=None):
        self.results = results
        self.vectors = vectors or {}
        self.calls = []

    def search(self, query_vec, top_k):
        self.calls.append((query_vec, top_k))
        return self.results

    def vectors_for_ids(self, ids):
        return {i: self.vectors[i] for i in ids if i in self.vectors}


class FakeBM25:
    """Returns a preset keyword ranking for any query."""

    def __init__(self, results):
        self.results = results

    def search(self, query, top_k):
        return self.results


def _r(i, score):
    return Retrieved(Chunk(id=str(i), text=f"t{i}", source="a.md", ordinal=i, corpus="c"), score)


def test_keeps_at_or_above_threshold_and_drops_below():
    store = FakeStore([_r(1, 0.9), _r(2, 0.5), _r(3, 0.49)])
    out = retrieve("q", store, FakeEmbedder(), top_k=3, threshold=0.5, use_mmr=False)
    assert [r.chunk.id for r in out] == ["1", "2"]


def test_all_below_threshold_returns_empty_list():
    store = FakeStore([_r(1, 0.2), _r(2, 0.1)])
    assert retrieve("q", store, FakeEmbedder(), top_k=2, threshold=0.5) == []


def test_candidate_pool_size_is_passed_to_store():
    store = FakeStore([])
    retrieve("q", store, FakeEmbedder(), top_k=3, threshold=0.5, candidate_k=17)
    assert store.calls[0][1] == 17


def test_top_k_limits_the_output():
    store = FakeStore([_r(i, 0.9) for i in range(1, 6)])
    out = retrieve("q", store, FakeEmbedder(), top_k=2, threshold=0.5, use_mmr=False)
    assert len(out) == 2


def test_question_is_what_gets_embedded_and_searched():
    emb = FakeEmbedder()
    store = FakeStore([_r(1, 0.9)])
    retrieve("what is X?", store, emb, top_k=1, threshold=0.0, use_mmr=False)
    assert emb.calls == [["what is X?"]]
    assert store.calls[0][0] == [0.1, 0.2, 0.3]


def test_dense_mode_ignores_bm25():
    store = FakeStore([_r(1, 0.9), _r(2, 0.8)])
    bm25 = FakeBM25([_r(9, 5.0)])
    out = retrieve("q", store, FakeEmbedder(), top_k=5, threshold=0.5, bm25=bm25,
                   mode="dense", use_mmr=False)
    assert [r.chunk.id for r in out] == ["1", "2"]  # id 9 never enters in dense mode


def test_hybrid_fuses_bm25_keyword_only_hit():
    # id 1 passes the dense threshold; id 9 is a keyword-only hit from BM25.
    store = FakeStore([_r(1, 0.9)])
    bm25 = FakeBM25([_r(9, 5.0), _r(1, 2.0)])
    out = retrieve("q", store, FakeEmbedder(), top_k=5, threshold=0.5, bm25=bm25,
                   mode="hybrid", use_mmr=False)
    ids = {r.chunk.id for r in out}
    assert "1" in ids and "9" in ids  # fusion surfaces the keyword-only passage


class FakeEmbedder2D:
    """Embedder returning a fixed 2-D query vector, to match 2-D candidate vectors."""

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_mmr_runs_and_keeps_most_relevant_first():
    # MMR needs candidate vectors; here it reorders the survivors and respects top_k.
    # (MMR's diversity behaviour itself is unit-tested in test_fusion.py.)
    results = [_r(1, 0.9), _r(2, 0.89), _r(3, 0.88)]
    vectors = {"1": [1.0, 0.0], "2": [0.9, 0.1], "3": [0.0, 1.0]}
    store = FakeStore(results, vectors=vectors)
    out = retrieve("q", store, FakeEmbedder2D(), top_k=2, threshold=0.5,
                   mode="dense", use_mmr=True)
    assert len(out) == 2
    assert out[0].chunk.id == "1"  # the vector most similar to the query comes first


def test_hybrid_mmr_keeps_bm25_contribution():
    # Dense alone ranks 1 first (cosine 1.0 with the query); BM25 ranks 3 first.
    # Fused RRF puts 3 on top, and MMR must diversify that fused order instead of
    # reverting to dense cosine, which would silently discard the BM25 channel.
    store = FakeStore(
        [_r(1, 0.9), _r(2, 0.85), _r(3, 0.8)],
        vectors={"1": [1.0, 0.0], "2": [0.6, 0.8], "3": [0.6, -0.8]},
    )
    bm25 = FakeBM25([_r(3, 5.0), _r(2, 4.0)])
    hybrid = retrieve("q", store, FakeEmbedder2D(), top_k=1, threshold=0.5, bm25=bm25,
                      mode="hybrid", use_mmr=True)
    dense = retrieve("q", store, FakeEmbedder2D(), top_k=1, threshold=0.5, bm25=bm25,
                     mode="dense", use_mmr=True)
    assert hybrid[0].chunk.id == "3"
    assert dense[0].chunk.id == "1"
