"""AR-GARCH-DCC adapter with recursive dependence forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..dependence import DccModel, fit_dcc, initialize_forecast_state, update_forecast_state
from ..marginal import MarginalFit, fit_marginals, forecast_path
from ..risk import gaussian_return_quantile
from .common import Forecast


@dataclass(frozen=True, slots=True)
class MultiGarchModel:
    adapter_name: str
    training_observation_count: int
    training_returns: np.ndarray
    marginal: MarginalFit
    dependence: DccModel
    portfolio_weights: np.ndarray


def fit(
    returns: ArrayLike,
    config: dict[str, Any],
    state: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> tuple[MultiGarchModel, dict[str, Any]]:
    values = np.asarray(returns, dtype=float)
    refit = True if context is None else bool(context.get("refit_marginal", True))
    marginal, next_state = fit_marginals(values, config, state, refit)
    dependence, next_state = fit_dcc(marginal.standardized_residuals, config, next_state)
    model = MultiGarchModel(
        "multiGarch",
        len(values),
        values.copy(),
        marginal,
        dependence,
        np.asarray(config["risk"]["portfolioWeights"], dtype=float),
    )
    return model, next_state


def forecast(
    model: MultiGarchModel,
    forecast_count: int,
    config: dict[str, Any],
    observed_updates: ArrayLike | None = None,
) -> Forecast:
    marginal_path = forecast_path(
        model.marginal, model.training_returns, forecast_count, observed_updates
    )
    series_count = model.training_returns.shape[1]
    correlations = np.zeros((series_count, series_count, forecast_count))
    portfolio_mean = np.zeros(forecast_count)
    portfolio_standard_deviation = np.zeros(forecast_count)
    state = initialize_forecast_state(model.dependence, model.marginal.standardized_residuals)
    for step in range(forecast_count):
        correlations[:, :, step] = state.correlation
        deviations = np.sqrt(np.maximum(marginal_path.asset_variance[step], 0.0))
        covariance = state.correlation * np.outer(deviations, deviations)
        portfolio_mean[step] = marginal_path.asset_mean[step] @ model.portfolio_weights
        portfolio_standard_deviation[step] = np.sqrt(
            max(float(model.portfolio_weights @ covariance @ model.portfolio_weights), 0.0)
        )
        if step < forecast_count - 1:
            state = update_forecast_state(state, marginal_path.standardized_updates[step])
    quantile = gaussian_return_quantile(
        portfolio_mean, portfolio_standard_deviation, config["risk"]["probability"]
    )
    return Forecast(
        forecast_count,
        config["risk"]["probability"],
        portfolio_mean,
        portfolio_standard_deviation,
        quantile,
        -quantile,
        marginal_path.asset_mean,
        marginal_path.asset_variance,
        correlations,
    )
