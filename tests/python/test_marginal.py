from __future__ import annotations

import numpy as np

from diss.config import defaults, validate_config
from diss.marginal import fit_marginals, forecast_path


def _simulate_garch(count: int, seed: int = 81) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = np.zeros(count)
    variance = np.full(count, 0.0004)
    for index in range(1, count):
        variance[index] = 0.00002 + 0.07 * returns[index - 1] ** 2 + 0.88 * variance[index - 1]
        returns[index] = np.sqrt(variance[index]) * rng.standard_normal()
    return returns[:, np.newaxis]


def _fixed_config(distribution: str = "Gaussian") -> dict:
    config = defaults()
    config["model"]["kind"] = "univariateGarch"
    config["marginal"].update(
        {
            "selectionPolicy": "fixed",
            "candidateFamilies": ["garch"],
            "distribution": distribution,
            "fixedSpecifications": [
                {"family": "garch", "arLag": 0, "archOrder": 1, "garchOrder": 1}
            ],
        }
    )
    config["optimization"].update({"maxIterations": 300, "maxFunctionEvaluations": 5_000})
    return config


def test_fitted_garch_has_positive_finite_variances() -> None:
    returns = _simulate_garch(350)
    config = validate_config(_fixed_config(), returns)
    marginal, state = fit_marginals(returns, config)
    assert np.all(np.isfinite(marginal.conditional_variances))
    assert np.all(marginal.conditional_variances > 0)
    assert np.isfinite(marginal.log_likelihood).all()
    assert len(state["marginal_models"]) == 1


def test_recursive_marginal_forecast_uses_observed_update() -> None:
    returns = _simulate_garch(300)
    config = validate_config(_fixed_config(), returns)
    marginal, _ = fit_marginals(returns, config)
    updates_a = np.array([[0.01], [0.0], [0.0]])
    updates_b = np.array([[0.05], [0.0], [0.0]])
    first = forecast_path(marginal, returns, 3, updates_a)
    second = forecast_path(marginal, returns, 3, updates_b)
    np.testing.assert_allclose(first.asset_variance[0], second.asset_variance[0], atol=0, rtol=0)
    assert abs(first.asset_variance[1, 0] - second.asset_variance[1, 0]) > 1e-12


def test_student_t_fit_estimates_valid_degrees_of_freedom() -> None:
    returns = _simulate_garch(400, seed=93)
    config = validate_config(_fixed_config("t"), returns)
    marginal, _ = fit_marginals(returns, config)
    assert marginal.models[0].distribution.degrees_of_freedom > 2
