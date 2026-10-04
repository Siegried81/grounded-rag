"""Shared test setup: no LLM response cache and no rate-limit waits.

The response cache writes to disk and would let one test's mocked reply answer
another test's identical request; the Groq rate-limit wait would make a test
sleep. Both are switched off here, and the tests that cover them switch them
back on explicitly.
"""

import pytest

import config


@pytest.fixture(autouse=True)
def _no_llm_cache_or_waits(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LLM_CACHE", False)
    monkeypatch.setattr(config, "LLM_CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(config, "GROQ_MAX_WAIT_S", 0.0)
