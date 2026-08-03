from __future__ import annotations

import numpy as np
import pytest

from diss.backtest import create_schedule
from diss.config import defaults, from_legacy_spec, validate_config
from diss.data import prepare_data
from diss.errors import DissError


def test_defaults_are_independent() -> None:
    first = defaults()
    second = defaults()
    first["risk"]["probability"] = 0.5
    assert second["risk"]["probability"] == 0.99


def test_schema_one_dependence_fields_migrate() -> None:
    data = np.ones((20, 2))
    config = {
        "schemaVersion": 1,
        "model": {"kind": "multiGarch", "dynamic": "ADCC", "dccP": 1, "dccQ": 1, "dccG": 1},
    }
    migrated = validate_config(config, data)
    assert migrated["schemaVersion"] == 2
    assert migrated["dependence"]["dynamic"] == "ADCC"
    assert migrated["dependence"]["asymmetricOrder"] == 1


def test_mainfile_spec_maps_to_schema_two() -> None:
    config = from_legacy_spec(
        {
            "purpose": "backtest",
            "ModelType": "MultiGARCH",
            "DynamicType": "ADCC",
            "dccP": 1,
            "dccQ": 1,
            "dccG": 1,
            "univariate": "on",
            "ForecastStart": 100,
            "ForecastNumb": 5,
            "SimNumb": 2_000,
            "beta": 0.975,
        }
    )
    assert config["model"]["kind"] == "multiGarch"
    assert config["dependence"]["dynamic"] == "ADCC"
    assert config["backtest"]["forecastHorizon"] == 5
    assert config["simulation"]["numPaths"] == 2_000


def test_unsupported_model_fails_explicitly() -> None:
    with pytest.raises(DissError, match="no validated Python adapter") as captured:
        validate_config({"model": {"kind": "vineCopula"}}, np.ones((20, 3)))
    assert captured.value.identifier == "diss:config:UnsupportedModelAdapter"


def test_missing_rows_can_be_removed_explicitly() -> None:
    values = np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, 4.0]])
    dataset = prepare_data(values, {"missingAction": "omitRows"})
    np.testing.assert_array_equal(dataset.observation_index, [1, 3])
    np.testing.assert_array_equal(dataset.removed_rows, [2])


def test_missing_rows_error_by_default() -> None:
    with pytest.raises(DissError) as captured:
        prepare_data(np.array([[1.0], [np.inf]]), {"missingAction": "error"})
    assert captured.value.identifier == "diss:data:NonfiniteReturns"


def test_expanding_schedule_uses_half_open_windows() -> None:
    options = defaults()["backtest"]
    options.update({"initialWindow": 10, "forecastHorizon": 3, "stepSize": 3})
    schedule = create_schedule(20, options)
    assert schedule[0].estimation_start == 0
    assert schedule[0].estimation_end == 10
    assert schedule[0].forecast_start == 10
    assert schedule[0].forecast_end == 13
    assert schedule[-1].forecast_end <= 20


def test_rolling_schedule_respects_requested_history() -> None:
    options = defaults()["backtest"]
    options.update({"windowType": "rolling", "initialWindow": 10, "rollingWindow": 7})
    schedule = create_schedule(13, options)
    assert schedule[0].estimation_start == 3
    assert schedule[0].estimation_end - schedule[0].estimation_start == 7


def test_weights_are_normalized_when_omitted() -> None:
    config = defaults()
    config["model"]["kind"] = "deltaNormal"
    result = validate_config(config, np.ones((10, 4)))
    np.testing.assert_allclose(result["risk"]["portfolioWeights"], 0.25)
