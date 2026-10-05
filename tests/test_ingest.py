"""Tests for rag.ingest document discovery, using a temporary corpus folder.

Only document loading is covered; the embed/store pipeline needs other modules
and real backends, so it is tested elsewhere.
"""

import config
from rag.ingest import document_paths, load_documents


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
