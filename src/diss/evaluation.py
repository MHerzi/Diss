"""Model-independent forecast evaluation."""

from __future__ import annotations

from math import erfc, log, sqrt
from typing import Any


def _weighted_log(count: int, probability: float) -> float:
    if count == 0:
        return 0.0
    if probability <= 0.0:
        return float("-inf")
    return count * log(probability)


def summarize(backtest: list[dict[str, Any]], probability: float) -> dict[str, Any]:
    """Compute violation frequency and the Kupiec unconditional-coverage test."""

    if not backtest:
        return {
            "observationCount": 0,
            "violationCount": 0,
            "violationRate": float("nan"),
            "expectedViolationRate": float("nan"),
            "kupiecStatistic": float("nan"),
            "kupiecPValue": float("nan"),
        }
    violations = [bool(row["violation"]) for row in backtest]
    observation_count = len(violations)
    violation_count = sum(violations)
    expected = 1.0 - probability
    observed = violation_count / observation_count
    null_likelihood = _weighted_log(violation_count, expected) + _weighted_log(
        observation_count - violation_count, 1.0 - expected
    )
    alternative = _weighted_log(violation_count, observed) + _weighted_log(
        observation_count - violation_count, 1.0 - observed
    )
    statistic = max(0.0, -2.0 * (null_likelihood - alternative))
    return {
        "observationCount": observation_count,
        "violationCount": violation_count,
        "violationRate": observed,
        "expectedViolationRate": expected,
        "kupiecStatistic": statistic,
        "kupiecPValue": erfc(sqrt(statistic / 2.0)),
    }
