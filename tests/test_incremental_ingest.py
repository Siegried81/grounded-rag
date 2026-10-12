"""Tests for incremental ingest: unchanged files are reused, and the index equals a full build.

The property that matters is the second one: an incremental build must write
exactly the index (chunks, ids, vectors, BM25) a full build of the same folder
writes, or retrieval would depend on the history of the folder. The embedder is
the deterministic fake, wrapped to count what it is asked to embed; the
embedding cache is off so a skipped file is skipped by the manifest, not served
by the cache.
"""

import json

import numpy as np
import pytest

import config
from rag.embed import FakeEmbedder
from rag.ingest import ingest_corpus, manifest_path
from rag.store import VectorStore

LONG = " ".join(f"Sentence number {i} says the provider keeps logs for {i} months." for i in range(40))


class CountingEmbedder(FakeEmbedder):
    """The fake embedder, recording every text it is asked to embed."""

    def __init__(self):
        self.seen: list[str] = []

    def embed(self, texts):
        self.seen.extend(texts)
        return super().embed(texts)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A three-file corpus under a temporary data/ and index/, cache off."""
    folder = tmp_path / "data" / "demo"
    folder.mkdir(parents=True)
    (folder / "a.txt").write_text(LONG, encoding="utf-8")
    (folder / "b.txt").write_text("Deployers keep the logs. " * 5, encoding="utf-8")
    (folder / "c.md").write_text("Annex III lists the high-risk areas.", encoding="utf-8")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "USE_EMBED_CACHE", False)
    return folder


def _snapshot(index_dir):
    """Everything a query reads from an index: chunk metadata, vectors, BM25 payload."""
    store = json.loads((index_dir / "store.json").read_text(encoding="utf-8"))
    return store, np.load(index_dir / "vectors.npy"), (index_dir / "bm25.json").read_text(encoding="utf-8")


def _full_build_elsewhere(monkeypatch, tmp_path):
    """Build the same folder from scratch into another index dir and snapshot it."""
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index_full")
    ingest_corpus("demo", embedder=FakeEmbedder(), incremental=False)
    return _snapshot(config.index_path("demo"))


def test_unchanged_corpus_embeds_nothing(corpus):
    ingest_corpus("demo", embedder=CountingEmbedder())
    emb, report = CountingEmbedder(), {}
    ingest_corpus("demo", embedder=emb, incremental=True, report=report)
    assert emb.seen == []
    assert report == {"reused": ["a.txt", "b.txt", "c.md"], "processed": [], "removed": [],
                      "new_chunks": 0}


def test_only_changed_and_new_files_are_embedded_and_removed_ones_dropped(corpus):
    ingest_corpus("demo", embedder=FakeEmbedder())
    (corpus / "b.txt").write_text("Deployers keep the logs for six months. " * 5, encoding="utf-8")
    (corpus / "d.txt").write_text("A new file about notified bodies.", encoding="utf-8")
    (corpus / "c.md").unlink()

    emb, report = CountingEmbedder(), {}
    store = ingest_corpus("demo", embedder=emb, incremental=True, report=report)
    assert report["reused"] == ["a.txt"]
    assert report["processed"] == ["b.txt", "d.txt"]
    assert report["removed"] == ["c.md"]
    assert emb.seen and not any("Sentence number" in t for t in emb.seen)
    assert report["new_chunks"] == len(emb.seen)
    assert {c.source for c, _ in store.rows()} == {"a.txt", "b.txt", "d.txt"}


def test_incremental_build_equals_a_full_build(corpus, monkeypatch, tmp_path):
    ingest_corpus("demo", embedder=FakeEmbedder())
    (corpus / "a.txt").write_text(LONG.replace("40", "forty"), encoding="utf-8")
    (corpus / "b.txt").unlink()
    (corpus / "0_first.txt").write_text("Sorted before every other file.", encoding="utf-8")
    ingest_corpus("demo", embedder=FakeEmbedder(), incremental=True)
    incremental = _snapshot(config.index_path("demo"))

    full = _full_build_elsewhere(monkeypatch, tmp_path)
    assert incremental[0] == full[0]
    assert np.array_equal(incremental[1], full[1])
    assert incremental[2] == full[2]


@pytest.mark.parametrize("change", ["chunk_size", "model", "no_manifest", "corrupt_store"])
def test_anything_that_invalidates_the_previous_build_rebuilds_everything(corpus, monkeypatch, change):
    ingest_corpus("demo", embedder=FakeEmbedder())
    emb = CountingEmbedder()
    if change == "chunk_size":
        monkeypatch.setattr(config, "CHUNK_SIZE", config.CHUNK_SIZE + 1)
    elif change == "model":
        monkeypatch.setattr(CountingEmbedder, "model", property(lambda self: "another-model"))
    elif change == "no_manifest":
        manifest_path("demo").unlink()
    else:
        (config.index_path("demo") / "vectors.npy").write_bytes(b"not an array")
    report = {}
    ingest_corpus("demo", embedder=emb, incremental=True, report=report)
    assert report["reused"] == [] and len(report["processed"]) == 3
    assert any("Sentence number" in t for t in emb.seen)


def test_a_file_whose_stored_chunks_disagree_with_the_manifest_is_rebuilt(corpus):
    ingest_corpus("demo", embedder=FakeEmbedder())
    manifest = json.loads(manifest_path("demo").read_text(encoding="utf-8"))
    manifest["files"]["b.txt"]["chunks"] += 1
    manifest_path("demo").write_text(json.dumps(manifest), encoding="utf-8")
    report = {}
    ingest_corpus("demo", embedder=FakeEmbedder(), incremental=True, report=report)
    assert report["processed"] == ["b.txt"]


def test_off_by_default_a_rebuild_embeds_everything(corpus, monkeypatch):
    monkeypatch.setattr(config, "INCREMENTAL_INGEST", False)
    ingest_corpus("demo", embedder=FakeEmbedder())
    emb = CountingEmbedder()
    ingest_corpus("demo", embedder=emb)
    assert len(emb.seen) == len(VectorStore.load(config.index_path("demo")))


def test_every_build_writes_the_manifest(corpus):
    ingest_corpus("demo", embedder=FakeEmbedder(), incremental=False)
    manifest = json.loads(manifest_path("demo").read_text(encoding="utf-8"))
    assert sorted(manifest["files"]) == ["a.txt", "b.txt", "c.md"]
    assert manifest["settings"]["model"] == "fake"
    assert manifest["files"]["c.md"]["chunks"] == 1
    assert len(manifest["files"]["a.txt"]["sha256"]) == 64
