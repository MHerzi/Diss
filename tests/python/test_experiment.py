from __future__ import annotations

import numpy as np

import diss
from diss import defaults, run_experiment, run_grid
from diss.io import load_run, save_run
from diss.registry import available_models, resolve_model


def _delta_config() -> dict:
    config = defaults()
    config["model"]["kind"] = "deltaNormal"
    return config


def test_registry_contains_all_canonical_models() -> None:
    assert available_models() == ("deltaNormal", "univariateGarch", "multiGarch", "multiCopula")
    assert resolve_model("MULTIGARCH").name == "multiGarch"


def test_compatibility_run_alias_matches_canonical_entry_point() -> None:
    values = np.random.default_rng(55).normal(size=(40, 2))
    compatibility = diss.run(values, _delta_config())
    canonical = run_experiment(values, _delta_config())
    np.testing.assert_array_equal(
        compatibility.forecast.loss_var,
        canonical.forecast.loss_var,
    )


def test_delta_normal_matches_manual_portfolio_calculation() -> None:
    rng = np.random.default_rng(72)
    values = rng.normal(size=(200, 3))
    config = _delta_config()
    config["risk"]["portfolioWeights"] = [0.2, 0.3, 0.5]
    result = run_experiment(values, config)
    weights = np.array([0.2, 0.3, 0.5])
    expected_mean = np.mean(values, axis=0) @ weights
    expected_variance = weights @ np.cov(values, rowvar=False, ddof=1) @ weights
    assert result.model.portfolio_mean == expected_mean
    assert result.model.portfolio_variance == expected_variance


def test_delta_normal_backtest_has_no_look_ahead() -> None:
    rng = np.random.default_rng(91)
    values = rng.normal(size=(50, 2))
    config = _delta_config()
    config["execution"]["mode"] = "backtest"
    config["backtest"].update({"initialWindow": 30, "forecastHorizon": 2, "stepSize": 2})
    result = run_experiment(values, config)
    first = result.backtest[0]
    expected_mean = np.mean(values[:30], axis=0).mean()
    assert first["forecastMean"] == expected_mean
    assert first["observationIndex"] == 31


def test_run_grid_matches_independent_runs() -> None:
    values = np.random.default_rng(101).normal(size=(100, 2))
    first = _delta_config()
    second = _delta_config()
    first["risk"]["probability"] = 0.95
    second["risk"]["probability"] = 0.99
    grid = run_grid(values, [first, second])
    independent = [run_experiment(values, first), run_experiment(values, second)]
    np.testing.assert_allclose(grid[0].forecast.loss_var, independent[0].forecast.loss_var)
    np.testing.assert_allclose(grid[1].forecast.loss_var, independent[1].forecast.loss_var)


def test_manifest_identity_is_deterministic_and_data_sensitive() -> None:
    values = np.random.default_rng(121).normal(size=(60, 2))
    first = run_experiment(values, _delta_config())
    second = run_experiment(values, _delta_config())
    changed = values.copy()
    changed[0, 0] += 1e-8
    third = run_experiment(changed, _delta_config())
    assert first.manifest["experimentId"] == second.manifest["experimentId"]
    assert first.manifest["dataHash"] != third.manifest["dataHash"]


def test_result_artifact_round_trip(tmp_path) -> None:
    values = np.random.default_rng(141).normal(size=(60, 2))
    result = run_experiment(values, _delta_config())
    directory = save_run(result, tmp_path)
    restored = load_run(directory)
    assert restored.manifest == result.manifest
    np.testing.assert_array_equal(restored.forecast.loss_var, result.forecast.loss_var)
    assert (directory / "manifest.json").is_file()
