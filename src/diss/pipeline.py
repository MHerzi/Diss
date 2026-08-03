"""Configuration-driven pipeline preparation and model dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from platform import python_version
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .backtest import ScheduleWindow, create_schedule
from .config import defaults, validate_config
from .data import Dataset, prepare_data
from .registry import resolve_model


@dataclass(frozen=True, slots=True)
class Plan:
    config: dict[str, Any]
    data: Dataset
    schedule: tuple[ScheduleWindow, ...]
    metadata: dict[str, Any]


def prepare(data: ArrayLike, config: dict[str, Any] | None = None) -> Plan:
    resolved = validate_config(config or defaults(), data)
    dataset = prepare_data(data, resolved["data"])
    resolved = validate_config(resolved, dataset.returns)
    schedule = (
        create_schedule(dataset.observation_count, resolved["backtest"])
        if resolved["execution"]["mode"] == "backtest"
        else ()
    )
    metadata = {
        "schemaVersion": resolved["schemaVersion"],
        "pythonVersion": python_version(),
        "created": datetime.now(UTC).isoformat(),
        "randomSeed": resolved["simulation"]["seed"],
    }
    return Plan(resolved, dataset, schedule, metadata)


def fit_window(
    returns: ArrayLike,
    config: dict[str, Any],
    state: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    adapter = resolve_model(config["model"]["kind"])
    return adapter.fit(
        np.asarray(returns, dtype=float), config, state or {}, context or {"refit_marginal": True}
    )


def forecast_window(
    model: Any,
    forecast_count: int,
    config: dict[str, Any],
    observed_updates: ArrayLike | None = None,
) -> Any:
    adapter = resolve_model(model.adapter_name)
    return adapter.forecast(model, forecast_count, config, observed_updates)
