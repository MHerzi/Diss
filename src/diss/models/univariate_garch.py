"""Univariate AR-GARCH adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..errors import require
from ..marginal import MarginalFit, fit_marginals, forecast_path
from .common import Forecast


@dataclass(frozen=True, slots=True)
class UnivariateGarchModel:
    adapter_name: str
    training_observation_count: int
    training_returns: np.ndarray
    marginal: MarginalFit
    portfolio_weights: np.ndarray


def fit(
    returns: ArrayLike,
    config: dict[str, Any],
    state: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> tuple[UnivariateGarchModel, dict[str, Any]]:
    values = np.asarray(returns, dtype=float)
    require(
        values.shape[1] == 1,
        "diss:univariateGarch:SeriesCount",
        "Univariate GARCH requires one series.",
    )
    refit = True if context is None else bool(context.get("refit_marginal", True))
    marginal, next_state = fit_marginals(values, config, state, refit)
    return UnivariateGarchModel(
        "univariateGarch", len(values), values.copy(), marginal, np.ones(1)
    ), next_state


def forecast(
    model: UnivariateGarchModel,
    forecast_count: int,
    config: dict[str, Any],
    observed_updates: ArrayLike | None = None,
) -> Forecast:
    path = forecast_path(model.marginal, model.training_returns, forecast_count, observed_updates)
    distribution = model.marginal.models[0].distribution
    standard_quantile = float(distribution.ppf(1.0 - config["risk"]["probability"]))
    standard_deviation = np.sqrt(np.maximum(path.asset_variance[:, 0], 0.0))
    quantile = path.asset_mean[:, 0] + standard_quantile * standard_deviation
    correlation = np.ones((1, 1, forecast_count))
    return Forecast(
        forecast_count,
        config["risk"]["probability"],
        path.asset_mean[:, 0],
        standard_deviation,
        quantile,
        -quantile,
        path.asset_mean,
        path.asset_variance,
        correlation,
    )
