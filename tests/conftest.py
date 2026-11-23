"""Shared test setup: no LLM response cache, no rate-limit waits, settings pinned.

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


@pytest.fixture(autouse=True)
def _pinned_env_settings(monkeypatch):
    """Pin the two settings a local .env could change under the tests.

    config.py reads them at import. USER_TURN_GUARD is off, so a developer who
    turns it on locally does not fail every test that checks the default prompt;
    VERIFY_MIN_GROUNDING is the shipped 0.30, which is what verify_answer uses
    when no bar is passed. Tests that need another value set it themselves.
    """
    monkeypatch.setattr(config, "USER_TURN_GUARD", False)
    monkeypatch.setattr(config, "VERIFY_MIN_GROUNDING", 0.30)
