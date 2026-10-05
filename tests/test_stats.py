"""Reference-value tests for rag.stats (Wilson score interval)."""

import pytest

from rag.stats import format_rate, wilson


def test_wilson_matches_reference_values():
    low, high = wilson(15, 17)
    assert low == pytest.approx(0.657, abs=0.001)
    assert high == pytest.approx(0.967, abs=0.001)


def test_wilson_at_the_edges_stays_inside_unit_interval():
    low, high = wilson(2, 2)
    assert low == pytest.approx(0.342, abs=0.001) and high == 1.0
    low, high = wilson(0, 5)
    assert low == 0.0 and 0 < high < 0.5
    # The normal approximation would give zero width here; Wilson does not.
    assert high > 0.4


def test_wilson_is_undefined_without_observations():
    assert wilson(0, 0) is None


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson(3, 2)


def test_format_rate():
    assert format_rate(15, 17) == "15/17 (88%, 95% CI 66%-97%)"
    assert format_rate(0, 0) == "0/0 (n/a)"
