"""Python-native AR-GARCH marginal estimation and recursive forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, minimize

from .distributions import InnovationDistribution
from .errors import DissError, require

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MarginalSpecification:
    family: str
    ar_lag: int
    arch_order: int
    garch_order: int
    distribution: str


@dataclass(frozen=True, slots=True)
class MarginalModel:
    specification: MarginalSpecification
    include_constant: bool
    constant: float
    ar: FloatArray
    omega: float
    arch: FloatArray
    leverage: FloatArray
    garch: FloatArray
    distribution: InnovationDistribution
    log_likelihood: float
    bic: float
    success: bool
    optimizer_message: str

    def filter(self, returns: ArrayLike) -> tuple[FloatArray, FloatArray, float]:
        values = np.asarray(returns, dtype=float).reshape(-1)
        count = len(values)
        residuals = np.zeros(count, dtype=float)
        variances = np.zeros(count, dtype=float)
        presample_variance = max(float(np.var(values, ddof=1)), np.finfo(float).eps)
        expected_absolute_normal = np.sqrt(2.0 / np.pi)
        for observation in range(count):
            conditional_mean = self.constant
            for lag, coefficient in enumerate(self.ar, start=1):
                if observation - lag >= 0:
                    conditional_mean += coefficient * values[observation - lag]
            if self.specification.family == "egarch":
                log_variance = self.omega
                for lag, coefficient in enumerate(self.arch, start=1):
                    if observation - lag >= 0:
                        previous_variance = max(variances[observation - lag], np.finfo(float).eps)
                        standardized = residuals[observation - lag] / np.sqrt(previous_variance)
                        log_variance += coefficient * (abs(standardized) - expected_absolute_normal)
                        log_variance += self.leverage[lag - 1] * standardized
                for lag, coefficient in enumerate(self.garch, start=1):
                    previous = (
                        variances[observation - lag]
                        if observation - lag >= 0
                        else presample_variance
                    )
                    log_variance += coefficient * np.log(max(previous, np.finfo(float).eps))
                variance = float(np.exp(np.clip(log_variance, -50.0, 50.0)))
            else:
                variance = self.omega
                for lag, coefficient in enumerate(self.arch, start=1):
                    previous = residuals[observation - lag] if observation - lag >= 0 else 0.0
                    variance += coefficient * previous**2
                    if self.specification.family == "gjr" and previous < 0.0:
                        variance += self.leverage[lag - 1] * previous**2
                for lag, coefficient in enumerate(self.garch, start=1):
                    previous = (
                        variances[observation - lag]
                        if observation - lag >= 0
                        else presample_variance
                    )
                    variance += coefficient * previous
            variances[observation] = max(float(variance), np.finfo(float).eps)
            residuals[observation] = values[observation] - conditional_mean
        offset = max(
            self.specification.ar_lag, self.specification.arch_order, self.specification.garch_order
        )
        standardized = residuals[offset:] / np.sqrt(variances[offset:])
        likelihood = self.distribution.logpdf(standardized) - 0.5 * np.log(variances[offset:])
        return residuals, variances, float(np.sum(likelihood))


@dataclass(frozen=True, slots=True)
class MarginalFit:
    models: tuple[MarginalModel, ...]
    specifications: tuple[MarginalSpecification, ...]
    log_likelihood: FloatArray
    bic: FloatArray
    residuals: FloatArray
    conditional_variances: FloatArray
    standardized_residuals: FloatArray
    alignment_offset: int


@dataclass(frozen=True, slots=True)
class MarginalForecast:
    asset_mean: FloatArray
    asset_variance: FloatArray
    standardized_updates: FloatArray


def _order_for_series(value: int | list[int], series: int) -> int:
    return int(value if np.isscalar(value) else value[series])


def _unpack_parameters(
    parameters: FloatArray,
    specification: MarginalSpecification,
    include_constant: bool,
) -> tuple[float, FloatArray, float, FloatArray, FloatArray, FloatArray, InnovationDistribution]:
    position = 0
    constant = float(parameters[position]) if include_constant else 0.0
    position += int(include_constant)
    ar = np.asarray(parameters[position : position + specification.ar_lag], dtype=float)
    position += specification.ar_lag
    omega = float(parameters[position])
    position += 1
    arch = np.asarray(parameters[position : position + specification.arch_order], dtype=float)
    position += specification.arch_order
    if specification.family in {"egarch", "gjr"}:
        leverage = np.asarray(
            parameters[position : position + specification.arch_order], dtype=float
        )
        position += specification.arch_order
    else:
        leverage = np.zeros(specification.arch_order, dtype=float)
    garch = np.asarray(parameters[position : position + specification.garch_order], dtype=float)
    position += specification.garch_order
    degrees = float(parameters[position]) if specification.distribution.casefold() == "t" else None
    distribution = InnovationDistribution(specification.distribution, degrees)
    return constant, ar, omega, arch, leverage, garch, distribution


def _starting_values(
    values: FloatArray,
    specification: MarginalSpecification,
    include_constant: bool,
) -> tuple[FloatArray, Bounds]:
    variance = max(float(np.var(values, ddof=1)), 1e-8)
    parameters: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    scale = max(float(np.std(values, ddof=1)), 1e-6)
    if include_constant:
        parameters.append(float(np.mean(values)) * 0.05)
        lower.append(float(np.mean(values)) - 10 * scale)
        upper.append(float(np.mean(values)) + 10 * scale)
    parameters.extend([0.05] * specification.ar_lag)
    lower.extend([-0.98] * specification.ar_lag)
    upper.extend([0.98] * specification.ar_lag)
    if specification.family == "egarch":
        parameters.append(float(np.log(variance) * 0.05))
        lower.append(-30.0)
        upper.append(10.0)
    else:
        parameters.append(variance * 0.05)
        lower.append(max(variance * 1e-8, 1e-12))
        upper.append(max(variance * 20.0, 1e-6))
    parameters.extend([0.05 / specification.arch_order] * specification.arch_order)
    if specification.family == "egarch":
        lower.extend([-2.0] * specification.arch_order)
        upper.extend([2.0] * specification.arch_order)
    else:
        lower.extend([1e-8] * specification.arch_order)
        upper.extend([0.999] * specification.arch_order)
    if specification.family == "egarch":
        parameters.extend([-0.03] * specification.arch_order)
        lower.extend([-2.0] * specification.arch_order)
        upper.extend([2.0] * specification.arch_order)
    elif specification.family == "gjr":
        parameters.extend([0.03] * specification.arch_order)
        lower.extend([0.0] * specification.arch_order)
        upper.extend([0.999] * specification.arch_order)
    parameters.extend([0.90 / specification.garch_order] * specification.garch_order)
    lower.extend([1e-8] * specification.garch_order)
    upper.extend([0.999] * specification.garch_order)
    if specification.distribution.casefold() == "t":
        parameters.append(8.0)
        lower.append(2.05)
        upper.append(200.0)
    return np.asarray(parameters, dtype=float), Bounds(lower, upper)


def _stationarity(model: MarginalModel) -> bool:
    if model.specification.family == "egarch":
        return bool(np.sum(np.abs(model.garch)) < 0.999)
    persistence = float(np.sum(model.arch) + np.sum(model.garch))
    if model.specification.family == "gjr":
        persistence += 0.5 * float(np.sum(model.leverage))
    return persistence < 0.999


def fit_specification(
    series: ArrayLike,
    specification: MarginalSpecification,
    config: dict[str, Any],
) -> MarginalModel:
    """Estimate one AR-GARCH specification by maximum likelihood."""

    values = np.asarray(series, dtype=float).reshape(-1)
    include_constant = bool(config["marginal"]["includeConstant"])
    start, bounds = _starting_values(values, specification, include_constant)

    def build(
        parameters: FloatArray,
        *,
        likelihood: float = float("nan"),
        bic: float = float("nan"),
        success: bool = False,
        message: str = "",
    ) -> MarginalModel:
        constant, ar, omega, arch, leverage, garch, distribution = _unpack_parameters(
            parameters, specification, include_constant
        )
        return MarginalModel(
            specification,
            include_constant,
            constant,
            ar,
            omega,
            arch,
            leverage,
            garch,
            distribution,
            likelihood,
            bic,
            success,
            message,
        )

    def objective(parameters: FloatArray) -> float:
        model = build(parameters)
        if not _stationarity(model):
            return 1e12 + 1e8 * np.sum(np.asarray(parameters) ** 2)
        try:
            _, variances, likelihood = model.filter(values)
        except (FloatingPointError, DissError, ValueError):
            return 1e12
        if not np.isfinite(likelihood) or np.any(~np.isfinite(variances)):
            return 1e12
        return -likelihood

    output = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": config["optimization"]["maxIterations"],
            "maxfun": config["optimization"]["maxFunctionEvaluations"],
            "ftol": 1e-10,
        },
    )
    if not output.success or not np.isfinite(output.fun) or output.fun >= 1e11:
        raise DissError(
            "diss:marginal:OptimizationFailed", f"Marginal optimization failed: {output.message}"
        )
    parameter_count = len(output.x)
    bic = float(2.0 * output.fun + parameter_count * np.log(len(values)))
    return build(
        np.asarray(output.x),
        likelihood=float(-output.fun),
        bic=bic,
        success=True,
        message=str(output.message),
    )


def _fixed_specification(raw: dict[str, Any], config: dict[str, Any]) -> MarginalSpecification:
    def value(*names: str) -> Any:
        for name in names:
            if name in raw:
                return raw[name]
        raise DissError("diss:marginal:InvalidFixedSpecification", f"Missing field {names[0]}.")

    family = str(value("family", "Family")).lower()
    require(
        family in config["marginal"]["candidateFamilies"],
        "diss:marginal:InvalidFixedSpecification",
        f"Disabled family {family}.",
    )
    return MarginalSpecification(
        family,
        int(value("arLag", "ARLag")),
        int(value("archOrder", "ArchOrder")),
        int(value("garchOrder", "GarchOrder")),
        config["marginal"]["distribution"],
    )


def fit_marginals(
    returns: ArrayLike,
    config: dict[str, Any],
    state: dict[str, Any] | None = None,
    refit_marginals: bool = True,
) -> tuple[MarginalFit, dict[str, Any]]:
    """Select and fit one AR-GARCH model per return series."""

    values = np.asarray(returns, dtype=float)
    current_state = dict(state or {})
    settings = config["marginal"]
    series_count = values.shape[1]
    previous_models = current_state.get("marginal_models")
    previous_specifications = current_state.get("marginal_specifications")
    if settings["selectionPolicy"] == "eachWindow":
        refit_marginals = True
    if not refit_marginals and previous_models and len(previous_models) == series_count:
        models = tuple(previous_models)
    else:
        models_list: list[MarginalModel] = []
        fixed = settings["selectionPolicy"] == "fixed"
        for series_index in range(series_count):
            if fixed:
                candidates = [
                    _fixed_specification(settings["fixedSpecifications"][series_index], config)
                ]
            elif settings["selectionPolicy"] == "initialWindow" and previous_specifications:
                candidates = [previous_specifications[series_index]]
            else:
                candidates = [
                    MarginalSpecification(
                        family,
                        ar_lag,
                        _order_for_series(settings["archOrder"], series_index),
                        _order_for_series(settings["garchOrder"], series_index),
                        settings["distribution"],
                    )
                    for family in settings["candidateFamilies"]
                    for ar_lag in range(settings["maxARLag"] + 1)
                ]
            successful: list[MarginalModel] = []
            failures: list[str] = []
            for specification in candidates:
                try:
                    successful.append(
                        fit_specification(values[:, series_index], specification, config)
                    )
                except DissError as exception:
                    failures.append(str(exception))
            if not successful:
                raise DissError(
                    "diss:marginal:SelectionFailed",
                    f"All candidates failed for series {series_index + 1}: {'; '.join(failures)}",
                )
            models_list.append(min(successful, key=lambda model: model.bic))
        models = tuple(models_list)

    residuals = np.zeros_like(values)
    variances = np.zeros_like(values)
    likelihoods = np.zeros(series_count)
    for series_index, model in enumerate(models):
        residuals[:, series_index], variances[:, series_index], likelihoods[series_index] = (
            model.filter(values[:, series_index])
        )
    specifications = tuple(model.specification for model in models)
    offset = max(
        max(specification.ar_lag, specification.arch_order, specification.garch_order)
        for specification in specifications
    )
    require(
        offset < len(values) - 1,
        "diss:marginal:InsufficientAlignedData",
        "Too few aligned observations.",
    )
    standardized = residuals[offset:] / np.sqrt(variances[offset:])
    require(
        np.isfinite(standardized).all(),
        "diss:marginal:InvalidInference",
        "Marginal inference is nonfinite.",
    )
    fit = MarginalFit(
        models,
        specifications,
        likelihoods,
        np.asarray([model.bic for model in models]),
        residuals,
        variances,
        standardized,
        offset,
    )
    current_state.update({"marginal_models": models, "marginal_specifications": specifications})
    return fit, current_state


def _one_step(
    model: MarginalModel,
    returns: list[float],
    residuals: list[float],
    variances: list[float],
) -> tuple[float, float]:
    mean = model.constant
    for lag, coefficient in enumerate(model.ar, start=1):
        if len(returns) >= lag:
            mean += coefficient * returns[-lag]
    if model.specification.family == "egarch":
        log_variance = model.omega
        for lag, coefficient in enumerate(model.arch, start=1):
            if len(residuals) >= lag:
                standardized = residuals[-lag] / np.sqrt(max(variances[-lag], np.finfo(float).eps))
                log_variance += coefficient * (abs(standardized) - np.sqrt(2.0 / np.pi))
                log_variance += model.leverage[lag - 1] * standardized
        for lag, coefficient in enumerate(model.garch, start=1):
            if len(variances) >= lag:
                log_variance += coefficient * np.log(max(variances[-lag], np.finfo(float).eps))
        variance = float(np.exp(np.clip(log_variance, -50.0, 50.0)))
    else:
        variance = model.omega
        for lag, coefficient in enumerate(model.arch, start=1):
            if len(residuals) >= lag:
                residual = residuals[-lag]
                variance += coefficient * residual**2
                if model.specification.family == "gjr" and residual < 0:
                    variance += model.leverage[lag - 1] * residual**2
        for lag, coefficient in enumerate(model.garch, start=1):
            if len(variances) >= lag:
                variance += coefficient * variances[-lag]
    return float(mean), max(float(variance), np.finfo(float).eps)


def forecast_path(
    marginal: MarginalFit,
    training_returns: ArrayLike,
    forecast_count: int,
    observed_updates: ArrayLike | None = None,
) -> MarginalForecast:
    """Recursively forecast and update fitted marginal models."""

    training = np.asarray(training_returns, dtype=float)
    updates = None if observed_updates is None else np.asarray(observed_updates, dtype=float)
    if (
        updates is not None
        and updates.size
        and updates.shape != (forecast_count, training.shape[1])
    ):
        raise DissError(
            "diss:marginal:InvalidObservedUpdates", "Observed updates must match forecast shape."
        )
    means = np.zeros((forecast_count, training.shape[1]))
    variances_out = np.zeros_like(means)
    standardized_updates = np.zeros((max(forecast_count - 1, 0), training.shape[1]))
    return_history = [list(training[:, column]) for column in range(training.shape[1])]
    residual_history = [list(marginal.residuals[:, column]) for column in range(training.shape[1])]
    variance_history = [
        list(marginal.conditional_variances[:, column]) for column in range(training.shape[1])
    ]
    for step in range(forecast_count):
        for series, model in enumerate(marginal.models):
            means[step, series], variances_out[step, series] = _one_step(
                model, return_history[series], residual_history[series], variance_history[series]
            )
        if step < forecast_count - 1:
            next_return = means[step] if updates is None or not updates.size else updates[step]
            for series in range(training.shape[1]):
                residual = float(next_return[series] - means[step, series])
                return_history[series].append(float(next_return[series]))
                residual_history[series].append(residual)
                variance_history[series].append(float(variances_out[step, series]))
                standardized_updates[step, series] = residual / np.sqrt(variances_out[step, series])
    return MarginalForecast(means, variances_out, standardized_updates)
