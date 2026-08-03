"""Static Gaussian-copula adapter with AR-GARCH margins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from ..dependence import GaussianCopula, fit_gaussian_copula
from ..marginal import MarginalFit, fit_marginals, forecast_path
from .common import Forecast


@dataclass(frozen=True, slots=True)
class MultiCopulaModel:
    adapter_name: str
    training_observation_count: int
    training_returns: np.ndarray
    marginal: MarginalFit
    dependence: GaussianCopula
    portfolio_weights: np.ndarray


def fit(
    returns: ArrayLike,
    config: dict[str, Any],
    state: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> tuple[MultiCopulaModel, dict[str, Any]]:
    values = np.asarray(returns, dtype=float)
    refit = True if context is None else bool(context.get("refit_marginal", True))
    marginal, next_state = fit_marginals(values, config, state, refit)
    pseudo = np.column_stack(
        [
            model.distribution.cdf(marginal.standardized_residuals[:, index])
            for index, model in enumerate(marginal.models)
        ]
    )
    boundary = 0.5 / (len(pseudo) + 1.0)
    copula = fit_gaussian_copula(np.clip(pseudo, boundary, 1.0 - boundary))
    next_state["copula_correlation"] = copula.correlation
    return (
        MultiCopulaModel(
            "multiCopula",
            len(values),
            values.copy(),
            marginal,
            copula,
            np.asarray(config["risk"]["portfolioWeights"], dtype=float),
        ),
        next_state,
    )


def forecast(
    model: MultiCopulaModel,
    forecast_count: int,
    config: dict[str, Any],
    observed_updates: ArrayLike | None = None,
) -> Forecast:
    marginal_path = forecast_path(
        model.marginal, model.training_returns, forecast_count, observed_updates
    )
    simulation_count = config["simulation"]["numPaths"]
    generator = np.random.default_rng(config["simulation"]["seed"])
    factor = np.linalg.cholesky(model.dependence.correlation)
    portfolio_mean = marginal_path.asset_mean @ model.portfolio_weights
    portfolio_standard_deviation = np.zeros(forecast_count)
    quantile = np.zeros(forecast_count)
    saved: list[np.ndarray] = []
    for step in range(forecast_count):
        scores = (
            generator.standard_normal((simulation_count, len(model.portfolio_weights))) @ factor.T
        )
        uniforms = np.clip(norm.cdf(scores), np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
        innovations = np.column_stack(
            [
                marginal_model.distribution.ppf(uniforms[:, series])
                for series, marginal_model in enumerate(model.marginal.models)
            ]
        )
        scenarios = marginal_path.asset_mean[step] + innovations * np.sqrt(
            marginal_path.asset_variance[step]
        )
        portfolio_scenarios = scenarios @ model.portfolio_weights
        portfolio_standard_deviation[step] = np.std(portfolio_scenarios, ddof=1)
        quantile[step] = np.quantile(portfolio_scenarios, 1.0 - config["risk"]["probability"])
        if config["output"]["saveSimulations"]:
            saved.append(scenarios)
    correlations = np.repeat(model.dependence.correlation[:, :, np.newaxis], forecast_count, axis=2)
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
        tuple(saved),
    )
