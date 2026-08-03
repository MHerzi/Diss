"""Full-sample and walk-forward experiment execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .evaluation import summarize
from .io import create_manifest, hash_value
from .pipeline import Plan, fit_window, forecast_window, prepare
from .risk import aggregate_returns


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    mode: str
    model: Any
    forecast: Any
    backtest: list[dict[str, Any]]
    window_models: tuple[Any, ...]
    diagnostics: dict[str, Any]
    config: dict[str, Any]
    schedule: tuple[Any, ...]
    data_summary: dict[str, Any]
    metadata: dict[str, Any]
    schema_version: int
    evaluation: dict[str, Any]
    manifest: dict[str, Any]


def _run_full(plan: Plan, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    reuse = "marginal_models" in state and "marginal_specifications" in state
    model, state = fit_window(
        plan.data.returns,
        plan.config,
        state,
        {"refit_marginal": not reuse, "window": 1},
    )
    forecast = forecast_window(model, 1, plan.config)
    return {
        "model": model,
        "forecast": forecast,
        "backtest": [],
        "window_models": (),
        "diagnostics": {},
    }, state


def _run_backtest(plan: Plan, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    window_models: list[Any] = []
    config = plan.config
    for window in plan.schedule:
        training = plan.data.returns[window.estimation_start : window.estimation_end]
        observed = plan.data.returns[window.forecast_start : window.forecast_end]
        started = perf_counter()
        model, state = fit_window(
            training,
            config,
            state,
            {"refit_marginal": window.refit_marginal, "window": window.window},
        )
        fit_seconds = perf_counter() - started
        started = perf_counter()
        forecast = forecast_window(model, window.forecast_length, config, observed)
        forecast_seconds = perf_counter() - started
        actual = aggregate_returns(
            observed, config["risk"]["portfolioWeights"], config["risk"]["returnAggregation"]
        )
        for step in range(window.forecast_length):
            dataset_row = window.forecast_start + step
            row = {
                "window": window.window,
                "observationIndex": int(plan.data.observation_index[dataset_row]),
                "estimationStart": window.estimation_start + 1,
                "estimationEnd": window.estimation_end,
                "forecastMean": float(forecast.portfolio_mean[step]),
                "forecastStdDev": float(forecast.portfolio_standard_deviation[step]),
                "returnQuantile": float(forecast.return_quantile[step]),
                "lossVaR": float(forecast.loss_var[step]),
                "realizedReturn": float(actual[step]),
                "violation": bool(actual[step] < forecast.return_quantile[step]),
                "fitSeconds": fit_seconds,
                "forecastSeconds": forecast_seconds,
                "dependenceSuccess": bool(
                    getattr(getattr(model, "dependence", None), "success", True)
                ),
            }
            rows.append(row)
        if config["output"]["saveWindowModels"]:
            window_models.append(compact_model(model, config["output"]["saveTrainingData"]))
    violation_count = sum(row["violation"] for row in rows)
    diagnostics = {
        "violationCount": violation_count,
        "violationRate": violation_count / len(rows),
        "expectedViolationRate": 1.0 - config["risk"]["probability"],
    }
    return {
        "model": None,
        "forecast": None,
        "backtest": rows,
        "window_models": tuple(window_models),
        "diagnostics": diagnostics,
    }, state


def compact_model(model: Any, save_training_data: bool = False) -> Any:
    """Remove repeated training histories from a fitted window model."""

    if save_training_data or not hasattr(model, "training_returns"):
        return model
    series_count = model.training_returns.shape[1]
    changes: dict[str, Any] = {"training_returns": np.empty((0, series_count), dtype=float)}
    if hasattr(model, "marginal"):
        changes["marginal"] = replace(
            model.marginal,
            residuals=np.empty((0, series_count), dtype=float),
            conditional_variances=np.empty((0, series_count), dtype=float),
            standardized_residuals=np.empty((0, series_count), dtype=float),
        )
    return replace(model, **changes)


def _execute(
    data: ArrayLike,
    config: dict[str, Any] | None,
    initial_state: dict[str, Any] | None,
) -> tuple[ExperimentResult, dict[str, Any]]:
    started = perf_counter()
    plan = prepare(data, config)
    state = dict(initial_state or {})
    if plan.config["execution"]["mode"] == "full":
        payload, state = _run_full(plan, state)
    else:
        payload, state = _run_backtest(plan, state)
    metadata = dict(plan.metadata)
    metadata["elapsedSeconds"] = perf_counter() - started
    evaluation = summarize(payload["backtest"], plan.config["risk"]["probability"])
    manifest = create_manifest(plan)
    result = ExperimentResult(
        plan.config["execution"]["mode"],
        payload["model"],
        payload["forecast"],
        payload["backtest"],
        payload["window_models"],
        payload["diagnostics"],
        plan.config,
        plan.schedule,
        {
            "observationCount": plan.data.observation_count,
            "seriesCount": plan.data.series_count,
            "variableNames": plan.data.variable_names,
            "removedRows": plan.data.removed_rows,
            "originalType": plan.data.original_type,
        },
        metadata,
        plan.config["schemaVersion"],
        evaluation,
        manifest,
    )
    return result, state


def run_experiment(
    data: ArrayLike,
    config: dict[str, Any] | None = None,
    initial_state: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Execute one reproducible full-sample or backtest experiment."""

    result, _ = _execute(data, config, initial_state)
    return result


def run_grid(data: ArrayLike, configs: list[dict[str, Any]]) -> list[ExperimentResult]:
    """Run configurations while safely reusing identical full-sample marginals."""

    results: list[ExperimentResult] = []
    states: dict[str, dict[str, Any]] = {}
    for config in configs:
        marginal_key = hash_value(config.get("marginal", {}))
        mode = config.get("execution", {}).get("mode", "full")
        initial_state = states.get(marginal_key, {}) if mode == "full" else {}
        result, final_state = _execute(data, config, initial_state)
        results.append(result)
        if mode == "full":
            states[marginal_key] = final_state
    return results
