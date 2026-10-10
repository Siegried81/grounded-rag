"""Tests for rag.ingest: document discovery and the build-if-missing entry point.

Document loading uses a temporary corpus folder; `ensure_index` runs the real
chunk/embed/store pipeline on it with the fake embedder, so no backend is needed.
"""

import pytest

import config
from rag.ingest import document_paths, ensure_index, load_documents


def test_load_documents_skips_readme_and_empty(tmp_path, monkeypatch):
    """README.md and blank files are skipped; real text is returned."""
    (tmp_path / "a.txt").write_text("Hello corpus.", encoding="utf-8")
    (tmp_path / "README.md").write_text("About this folder", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   \n", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("x,y", encoding="utf-8")
    monkeypatch.setattr(config, "corpus_dir", lambda name: tmp_path)

    assert load_documents("demo") == [("a.txt", "Hello corpus.")]


def test_document_paths_lists_indexed_files_only(tmp_path, monkeypatch):
    """Sorted, supported files directly in the folder; README, other suffixes
    and subfolders are left out. The API resolves a citation through this list,
    so anything it returns is a file a reader is allowed to open."""
    for name in ("b.txt", "a.md", "README.md", "notes.csv"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "dump.txt").write_text("not indexed", encoding="utf-8")
    monkeypatch.setattr(config, "corpus_dir", lambda name: tmp_path)

    assert [p.name for p in document_paths("demo")] == ["a.md", "b.txt"]


def test_document_paths_keeps_empty_files(tmp_path, monkeypatch):
    """Emptiness is judged by the caller (`load_documents`), not by the listing,
    so the two stay independent."""
    (tmp_path / "blank.txt").write_text("   \n", encoding="utf-8")
    monkeypatch.setattr(config, "corpus_dir", lambda name: tmp_path)

    assert [p.name for p in document_paths("demo")] == ["blank.txt"]
    assert load_documents("demo") == []


def _corpus_on_disk(tmp_path, monkeypatch, corpus: str = "demo"):
    """A one-file corpus under a temporary data/ and index/ pair, fake embedder."""
    (tmp_path / "data" / corpus).mkdir(parents=True)
    (tmp_path / "data" / corpus / "a.txt").write_text(
        "The provider must respond within 24 hours. Deployers keep the logs.", encoding="utf-8"
    )
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "EMBED_PROVIDER", "fake")


def test_ensure_index_builds_a_missing_index_once(tmp_path, monkeypatch):
    """First call builds (True) and leaves a loadable store; the second finds it
    and does nothing (False), so a reused disk costs no embedding call."""
    from rag.store import VectorStore

    _corpus_on_disk(tmp_path, monkeypatch)

    assert ensure_index("demo") is True
    assert VectorStore.exists(config.index_path("demo"))
    written = (config.index_path("demo") / "store.json").stat().st_mtime_ns

    assert ensure_index("demo") is False
    assert (config.index_path("demo") / "store.json").stat().st_mtime_ns == written


def test_ensure_index_rebuilds_a_half_written_index(tmp_path, monkeypatch):
    """One index file alone (an interrupted start) is not an index: it is built
    again rather than handed to `VectorStore.load`, which would refuse it."""
    _corpus_on_disk(tmp_path, monkeypatch)
    config.index_path("demo").mkdir(parents=True)
    (config.index_path("demo") / "store.json").write_text("{}", encoding="utf-8")

    assert ensure_index("demo") is True
    assert (config.index_path("demo") / "vectors.npy").is_file()


def test_ensure_index_fails_loudly_on_an_empty_corpus(tmp_path, monkeypatch):
    """An empty folder raises instead of writing an index that refuses everything."""
    _corpus_on_disk(tmp_path, monkeypatch)
    (tmp_path / "data" / "demo" / "a.txt").unlink()

    with pytest.raises(ValueError, match="No documents"):
        ensure_index("demo")
    assert not config.index_path("demo").exists()
