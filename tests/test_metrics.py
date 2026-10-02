"""Offline tests for retrieval metrics and query logging."""

import pytest

from rag.logging_utils import log_query, read_log
from rag.metrics import hit_rate_at_k, mrr, precision_at_k, recall_at_k

IDS = ["a", "b", "c", "d"]


def test_recall_at_k():
    assert recall_at_k(IDS, {"a", "c"}, 2) == 0.5
    assert recall_at_k(IDS, {"a", "c"}, 3) == 1.0
    assert recall_at_k(IDS, {"a", "c"}, 100) == 1.0
    assert recall_at_k(IDS, {"z"}, 4) == 0.0
    assert recall_at_k([], {"a"}, 3) == 0.0
    assert recall_at_k(IDS, set(), 3) == 0.0


def test_hit_rate_at_k():
    assert hit_rate_at_k(IDS, {"c"}, 3) == 1.0
    assert hit_rate_at_k(IDS, {"c"}, 2) == 0.0
    assert hit_rate_at_k([], {"c"}, 2) == 0.0
    assert hit_rate_at_k(IDS, set(), 2) == 0.0
    assert hit_rate_at_k(IDS, {"a"}, 0) == 0.0


def test_mrr():
    assert mrr(IDS, {"a"}) == 1.0
    assert mrr(IDS, {"c", "d"}) == pytest.approx(1 / 3)
    assert mrr(IDS, {"z"}) == 0.0
    assert mrr([], {"a"}) == 0.0


def test_precision_at_k():
    assert precision_at_k(IDS, {"a", "c"}, 4) == 0.5
    assert precision_at_k(IDS, {"a"}, 2) == 0.5
    assert precision_at_k(["a", "b"], {"a", "b"}, 10) == 1.0  # short list
    assert precision_at_k(IDS, {"z"}, 3) == 0.0
    assert precision_at_k([], {"a"}, 3) == 0.0
    assert precision_at_k(IDS, {"a"}, 0) == 0.0


def test_log_roundtrip(tmp_path):
    p = tmp_path / "sub" / "q.jsonl"
    log_query(p, question="q1", corpus="c", n_retrieved=2, top_score=0.8,
              refused=False, extra={"k": 4})
    log_query(p, question="q2", corpus="c", n_retrieved=0, top_score=None, refused=True)
    rows = read_log(p)
    assert [r["question"] for r in rows] == ["q1", "q2"]
    assert rows[0]["k"] == 4 and rows[0]["top_score"] == 0.8
    assert rows[1]["refused"] is True and rows[1]["top_score"] is None
    assert "T" in rows[0]["timestamp"]


def test_read_log_missing(tmp_path):
    assert read_log(tmp_path / "nope.jsonl") == []


def test_log_query_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    # Parent is a regular file, so the directory cannot be created.
    log_query(blocker / "q.jsonl", question="q", corpus="c", n_retrieved=0,
              top_score=None, refused=True)
