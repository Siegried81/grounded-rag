"""Tests for rag.ingest.load_documents, using a temporary corpus folder.

Only document loading is covered; the embed/store pipeline needs other modules
and real backends, so it is tested elsewhere.
"""

import config
from rag.ingest import load_documents


def test_load_documents_skips_readme_and_empty(tmp_path, monkeypatch):
    """README.md and blank files are skipped; real text is returned."""
    (tmp_path / "a.txt").write_text("Hello corpus.", encoding="utf-8")
    (tmp_path / "README.md").write_text("About this folder", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   \n", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("x,y", encoding="utf-8")
    monkeypatch.setattr(config, "corpus_dir", lambda name: tmp_path)

    assert load_documents("demo") == [("a.txt", "Hello corpus.")]
