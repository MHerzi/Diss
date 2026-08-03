"""Unconditional multivariate Gaussian adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..errors import DissError, require
from ..risk import gaussian_return_quantile
from .common import Forecast

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DeltaNormalModel:
    adapter_name: str
    training_observation_count: int
    asset_mean: FloatArray
    asset_covariance: FloatArray
    portfolio_weights: FloatArray
    portfolio_mean: float
    portfolio_variance: float
    portfolio_standard_deviation: float


def fit(
    returns: ArrayLike,
    config: dict[str, Any],
    state: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> tuple[DeltaNormalModel, dict[str, Any]]:
    values = np.asarray(returns, dtype=float)
    require(
        len(values) >= 2,
        "diss:deltaNormal:TooFewObservations",
        "At least two observations are required.",
    )
    asset_mean = np.mean(values, axis=0)
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    covariance = (covariance + covariance.T) / 2.0
    weights = np.asarray(config["risk"]["portfolioWeights"], dtype=float)
    portfolio_mean = float(asset_mean @ weights)
    portfolio_variance = float(weights @ covariance @ weights)
    tolerance = np.finfo(float).eps * max(1.0, np.linalg.norm(covariance))
    if portfolio_variance < -tolerance:
        raise DissError(
            "diss:deltaNormal:NegativePortfolioVariance", "Portfolio variance is negative."
        )
    portfolio_variance = max(portfolio_variance, 0.0)
    return (
        DeltaNormalModel(
            "deltaNormal",
            len(values),
            asset_mean,
            covariance,
            weights,
            portfolio_mean,
            portfolio_variance,
            float(np.sqrt(portfolio_variance)),
        ),
        {},
    )


def forecast(
    model: DeltaNormalModel,
    forecast_count: int,
    config: dict[str, Any],
    observed_updates: ArrayLike | None = None,
) -> Forecast:
    updates = None if observed_updates is None else np.asarray(observed_updates)
    if updates is not None and updates.size and len(updates) != forecast_count:
        raise DissError(
            "diss:deltaNormal:InvalidObservedUpdates", "Observed updates must match forecast count."
        )
    mean = np.full(forecast_count, model.portfolio_mean)
    standard_deviation = np.full(forecast_count, model.portfolio_standard_deviation)
    quantile = gaussian_return_quantile(mean, standard_deviation, config["risk"]["probability"])
    return Forecast(
        forecast_count, config["risk"]["probability"], mean, standard_deviation, quantile, -quantile
    )
