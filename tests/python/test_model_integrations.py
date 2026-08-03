from __future__ import annotations

import numpy as np

from diss import defaults, run_experiment
from diss.experiment import compact_model


def _correlated_garch(count: int, series_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    correlation = np.full((series_count, series_count), 0.25)
    np.fill_diagonal(correlation, 1.0)
    shocks = rng.standard_normal((count, series_count)) @ np.linalg.cholesky(correlation).T
    returns = np.zeros_like(shocks)
    variances = np.full_like(shocks, 0.0004)
    for index in range(1, count):
        variances[index] = 0.00002 + 0.06 * returns[index - 1] ** 2 + 0.89 * variances[index - 1]
        returns[index] = np.sqrt(variances[index]) * shocks[index]
    return returns


def _fixed_margins(config: dict, count: int) -> None:
    config["marginal"].update(
        {
            "selectionPolicy": "fixed",
            "candidateFamilies": ["garch"],
            "maxARLag": 0,
            "fixedSpecifications": [
                {"family": "garch", "arLag": 0, "archOrder": 1, "garchOrder": 1}
                for _ in range(count)
            ],
        }
    )
    config["optimization"].update({"maxIterations": 300, "maxFunctionEvaluations": 8_000})


def test_univariate_garch_pipeline() -> None:
    values = _correlated_garch(320, 1, 202)
    config = defaults()
    config["model"]["kind"] = "univariateGarch"
    _fixed_margins(config, 1)
    result = run_experiment(values, config)
    assert np.isfinite(result.forecast.loss_var).all()
    assert result.forecast.asset_variance[0, 0] > 0
    compacted = compact_model(result.model)
    assert compacted.training_returns.shape == (0, 1)
    assert compacted.marginal.residuals.shape == (0, 1)


def test_static_gaussian_copula_pipeline_is_reproducible() -> None:
    values = _correlated_garch(300, 2, 221)
    config = defaults()
    config["model"]["kind"] = "multiCopula"
    config["dependence"].update({"kind": "copula", "dynamic": "none", "copulaType": "gaussian"})
    config["simulation"].update({"numPaths": 2_000, "seed": 15})
    _fixed_margins(config, 2)
    first = run_experiment(values, config)
    second = run_experiment(values, config)
    np.testing.assert_array_equal(first.forecast.loss_var, second.forecast.loss_var)
    assert np.linalg.eigvalsh(first.model.dependence.correlation).min() > 0


def test_multi_garch_pipeline_produces_positive_definite_correlation() -> None:
    values = _correlated_garch(280, 2, 241)
    config = defaults()
    config["model"]["kind"] = "multiGarch"
    config["dependence"].update({"kind": "dcc", "dynamic": "DCC"})
    _fixed_margins(config, 2)
    result = run_experiment(values, config)
    correlation = result.forecast.correlation[:, :, 0]
    assert np.linalg.eigvalsh(correlation).min() > 0
    assert np.isfinite(result.forecast.loss_var).all()
