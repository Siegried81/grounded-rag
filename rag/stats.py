"""Confidence intervals for the binomial rates the evaluation scripts report.

The eval sets are small (10-40 questions), so a pass rate such as 15/17 is a
noisy estimate; printing it without an interval invites reading a 2-point
difference between runs as a real change. Wilson's score interval is used rather
than the normal approximation (p ± z·sqrt(p(1-p)/n)) because at these sizes the
normal interval breaks down exactly where our rates sit: at 0% or 100% it
collapses to zero width, and near the edges it runs outside [0, 1]. Wilson stays
inside [0, 1], keeps a sensible width at 0/n and n/n, and needs no library.
"""

from __future__ import annotations

import math


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval (low, high) for `successes` out of `n`; None when n == 0.

    `z` = 1.96 gives a 95% interval. The result is a two-sided interval on the
    true rate; it describes sampling noise only, not bias in the eval set.
    """
    if n <= 0:
        return None
    if successes < 0 or successes > n:
        raise ValueError(f"successes must be within 0..n, got {successes}/{n}")
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def format_rate(successes: int, n: int) -> str:
    """Render "k/n (pct, 95% CI lo-hi)" for the summary tables, or "0/0 (n/a)"."""
    ci = wilson(successes, n)
    if ci is None:
        return f"{successes}/{n} (n/a)"
    low, high = ci
    return f"{successes}/{n} ({successes / n:.0%}, 95% CI {low:.0%}-{high:.0%})"
