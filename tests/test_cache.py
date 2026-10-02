"""Offline tests for the persistent embedding cache."""

from rag.cache import EmbeddingCache


class FakeEmbedder:
    """Deterministic embedder that records how many texts it was asked to embed."""

    model = "fake"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]

    @property
    def n_embedded(self):
        return sum(len(c) for c in self.calls)


def test_first_call_embeds_all_second_is_free(tmp_path):
    cache, emb = EmbeddingCache(tmp_path / "c.json"), FakeEmbedder()
    first = cache.embed_with_cache(emb, ["a", "bb"])
    assert emb.n_embedded == 2
    assert cache.embed_with_cache(emb, ["a", "bb"]) == first
    assert emb.n_embedded == 2


def test_mixed_embeds_only_new_in_order(tmp_path):
    cache, emb = EmbeddingCache(tmp_path / "c.json"), FakeEmbedder()
    cache.embed_with_cache(emb, ["bb"])
    out = cache.embed_with_cache(emb, ["a", "bb", "cccc"])
    assert emb.calls[-1] == ["a", "cccc"]
    assert out == [[1.0, 1.0], [2.0, 1.0], [4.0, 1.0]]


def test_model_is_part_of_key(tmp_path):
    cache = EmbeddingCache(tmp_path / "c.json")
    cache.put("m1", "x", [1.0])
    assert cache.get("m2", "x") is None


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "sub" / "c.json"
    cache = EmbeddingCache(path)
    cache.put("fake", "hi", [0.5, 0.25])
    cache.save()
    assert EmbeddingCache.load(path).get("fake", "hi") == [0.5, 0.25]


def test_load_missing_file_is_empty(tmp_path):
    cache = EmbeddingCache.load(tmp_path / "nope.json")
    assert cache.get("fake", "x") is None
