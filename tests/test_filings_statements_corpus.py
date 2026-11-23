"""The shipped statements corpus and its eval files stay consistent with each other."""

import json

import config
from rag.ingest import _chunk_file, document_paths


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_every_statement_file_is_context_headed_and_chunks_within_the_size():
    paths = document_paths("filings_statements")
    assert len(paths) == 19
    for path in paths:
        chunks = _chunk_file(path, "filings_statements", "smart")
        assert chunks, path.name
        # No chunk is a bare header: each one carries content below the title.
        assert all(len(c.text.splitlines()) >= 3 for c in chunks), path.name
        # An unsplittable prose sentence may run over; a row never does.
        assert all(len(c.text) <= config.CHUNK_SIZE or c.meta["kind"] == "prose" for c in chunks)


def test_narrative_corpus_holds_only_the_six_items():
    names = [p.name for p in document_paths("filings_sections")]
    assert names == ["01_business.txt", "02_risk_factors.txt", "03_cybersecurity.txt",
                     "04_legal.txt", "05_mdna.txt", "06_market_risk.txt"]


def test_statements_eval_files_agree_and_point_at_existing_files():
    answers = _rows(config.ROOT / "eval" / "filings_statements_answers_eval.jsonl")
    retrieval = _rows(config.ROOT / "eval" / "filings_statements_eval.jsonl")
    answerable = [r for r in answers if r["answerable"]]
    assert [r["question"] for r in retrieval] == [r["question"] for r in answerable]
    assert [r["relevant_sources"] for r in retrieval] == [r["gold_sources"] for r in answerable]
    present = {p.name for p in document_paths("filings_statements")}
    for row in answerable:
        assert row["gold_sources"] and set(row["gold_sources"]) <= present
