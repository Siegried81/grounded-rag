"""Offline tests for rag.retrieve using tiny in-file fakes (no embedder/store imports).

Covers the refusal gate (anchored to dense cosine), the candidate-pool passthrough,
dense vs hybrid (BM25 fusion) modes, MMR diversification, the embedding-model
guard, and the RETRIEVAL_MODE validation in config.
"""

import importlib

import pytest

import config
from rag.retrieve import IndexMismatchError, check_same_embedder, retrieve
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


class StoreWithoutVectors:
    """A store that can search but can't hand back vectors."""

    def __init__(self, results):
        self.results = results

    def search(self, query_vec, top_k):
        return self.results


def _channels(out):
    return {r.chunk.id: (round(r.score, 4), r.score_source) for r in out}


def test_hybrid_bm25_only_hit_shows_its_dense_cosine_not_its_bm25_score():
    # Query vector is [0.1, 0.2, 0.3]; id 9's stored vector points the same way.
    store = FakeStore([_r(1, 0.9)], vectors={"1": [1.0, 0.0, 0.0], "9": [1.0, 2.0, 3.0]})
    bm25 = FakeBM25([_r(9, 7.5)])
    out = retrieve("q", store, FakeEmbedder(), top_k=5, threshold=0.5, bm25=bm25,
                   mode="hybrid", use_mmr=False)
    assert _channels(out) == {"1": (0.9, "dense"), "9": (1.0, "bm25")}


def test_hybrid_labels_a_passage_both_channels_found_as_both():
    store = FakeStore([_r(1, 0.9), _r(2, 0.8)])
    bm25 = FakeBM25([_r(2, 4.0)])
    out = retrieve("q", store, FakeEmbedder(), top_k=5, threshold=0.5, bm25=bm25,
                   mode="hybrid", use_mmr=False)
    assert _channels(out) == {"1": (0.9, "dense"), "2": (0.8, "both")}


def test_bm25_only_hit_below_the_threshold_still_gets_in_with_its_low_cosine():
    # No per-passage cosine gate for BM25 hits: they're there to catch exact
    # terms (an article number) that embeddings miss. The cosine shows it.
    store = FakeStore([_r(1, 0.9), _r(3, 0.1)])
    bm25 = FakeBM25([_r(3, 9.0)])
    out = retrieve("q", store, FakeEmbedder(), top_k=5, threshold=0.5, bm25=bm25,
                   mode="hybrid", use_mmr=False)
    assert _channels(out) == {"1": (0.9, "dense"), "3": (0.1, "bm25")}


def test_bm25_only_hit_keeps_its_raw_score_when_the_store_has_no_vectors():
    out = retrieve("q", StoreWithoutVectors([_r(1, 0.9)]), FakeEmbedder(), top_k=5,
                   threshold=0.5, bm25=FakeBM25([_r(9, 7.5)]), mode="hybrid", use_mmr=False)
    assert _channels(out)["9"] == (7.5, "bm25")


def test_dense_mode_labels_every_passage_dense():
    store = FakeStore([_r(1, 0.9), _r(2, 0.8)])
    out = retrieve("q", store, FakeEmbedder(), top_k=5, threshold=0.5,
                   bm25=FakeBM25([_r(2, 4.0)]), mode="dense", use_mmr=False)
    assert {r.score_source for r in out} == {"dense"}


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


# --- embedding-model guard ---------------------------------------------------

class _Named:
    """Store or embedder stand-in that only carries a `model` name."""

    def __init__(self, model):
        self.model = model


def test_check_same_embedder_raises_only_on_a_real_mismatch():
    with pytest.raises(IndexMismatchError, match="built with embedding model 'a'"):
        check_same_embedder(_Named("a"), _Named("b"))
    check_same_embedder(_Named("a"), _Named("a"))
    # Fakes without a model name (tests, duck-typed backends) are not checked.
    check_same_embedder(_Named(None), _Named("b"))
    check_same_embedder(FakeStore([]), FakeEmbedder())


def test_retrieve_refuses_to_query_an_index_from_another_embedder():
    store = FakeStore([_r(1, 0.9)])
    store.model = "nomic-embed-text"
    emb = FakeEmbedder()
    emb.model = "other-model"
    with pytest.raises(IndexMismatchError):
        retrieve("q", store, emb, top_k=1, threshold=0.0)
    assert emb.calls == []  # the question is never embedded


# --- config validation ---------------------------------------------------------

def test_invalid_retrieval_mode_fails_at_import(monkeypatch):
    # A typo would run hybrid (retrieve treats anything but "dense" as hybrid)
    # while the UIs displayed the typo, so config refuses to load instead.
    original = config.RETRIEVAL_MODE
    monkeypatch.setenv("RETRIEVAL_MODE", "sparse")
    try:
        with pytest.raises(ValueError, match="RETRIEVAL_MODE"):
            importlib.reload(config)
    finally:
        # Reload once more so every module sharing `config` sees the real settings again.
        monkeypatch.setenv("RETRIEVAL_MODE", original)
        importlib.reload(config)
    assert config.RETRIEVAL_MODE == original


def test_top_dense_score_is_the_cosine_the_gate_compares():
    from rag.retrieve import top_dense_score

    class Store:
        def search(self, vec, k):
            assert k == 1
            return [Retrieved(Chunk("x", "t", "s", 0, "c"), 0.91)]

    class Empty:
        def search(self, vec, k):
            return []

    class Emb:
        def embed(self, texts):
            return [[1.0, 0.0]]

    assert top_dense_score("q", Store(), Emb()) == 0.91
    assert top_dense_score("q", Empty(), Emb()) is None


# --- MMR relevance in hybrid mode ---------------------------------------------

def test_rank_relevance_is_linear_from_one_to_zero():
    from rag.retrieve import _rank_relevance

    assert _rank_relevance(5) == [1.0, 0.75, 0.5, 0.25, 0.0]
    assert _rank_relevance(1) == [1.0]
    assert _rank_relevance(0) == []


def test_hybrid_mmr_keeps_a_dense_hit_over_a_lower_ranked_bm25_hit(monkeypatch):
    # Fused order: 2, 3 (both lists), then 1 (dense only), then 4 (BM25 only).
    # Min-max on the RRF scores gave 1 and 4 almost the same relevance (0.03
    # and 0), so the third place went to 4, the one least like 2 and 3. With
    # rank relevance 1 keeps its lead (0.33 vs 0) and takes the place.
    monkeypatch.setattr(config, "MMR_LAMBDA", 0.6)
    store = FakeStore(
        [_r(1, 0.9), _r(2, 0.85), _r(3, 0.8)],
        vectors={"1": [0.8776, 0.4794], "2": [1.0, 0.0], "3": [0.9553, 0.2955],
                 "4": [0.9211, -0.3894]},
    )
    bm25 = FakeBM25([_r(2, 5.0), _r(3, 4.0), _r(4, 3.0)])
    out = retrieve("q", store, FakeEmbedder2D(), top_k=3, threshold=0.5, bm25=bm25,
                   mode="hybrid", use_mmr=True)
    assert [r.chunk.id for r in out] == ["2", "3", "1"]
