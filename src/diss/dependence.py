"""DCC-family dependence and static Gaussian-copula kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, minimize
from scipy.stats import multivariate_t, t

from .distributions import InnovationDistribution
from .errors import DissError, require

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DccModel:
    dynamic_type: str
    parameters: FloatArray
    arch_order: int
    garch_order: int
    q_bar: FloatArray
    negative_q_bar: FloatArray
    negative_log_likelihood: float
    log_likelihood: float
    aic: float
    bic: float
    success: bool
    optimizer_message: str


@dataclass(slots=True)
class DccForecastState:
    intercept: FloatArray
    arch_matrix: FloatArray
    garch_matrix: FloatArray
    asymmetric_matrix: FloatArray
    current_q: FloatArray
    correlation: FloatArray


@dataclass(frozen=True, slots=True)
class GaussianCopula:
    correlation: FloatArray
    log_likelihood: float
    negative_log_likelihood: float
    success: bool = True


def gaussian_copula_pdf(pseudo_observations: ArrayLike, correlation: ArrayLike) -> FloatArray:
    """Evaluate an unrestricted Gaussian-copula density by Cholesky solve."""

    observations = np.asarray(pseudo_observations, dtype=float)
    correlation_array = np.asarray(correlation, dtype=float)
    require(
        observations.ndim == 2 and observations.shape[1] == len(correlation_array),
        "diss:copula:DimensionMismatch",
        "Pseudo-observations and correlation dimensions must agree.",
    )
    require(
        np.all((observations > 0.0) & (observations < 1.0)),
        "diss:copula:InvalidPseudoObservations",
        "Pseudo-observations must lie in (0, 1).",
    )
    scores = InnovationDistribution("Gaussian").ppf(observations)
    factor = np.linalg.cholesky(correlation_array)
    solved = np.linalg.solve(factor, scores.T)
    quadratic_difference = np.sum(solved**2, axis=0) - np.sum(scores**2, axis=1)
    log_determinant = 2.0 * np.log(np.diag(factor)).sum()
    return np.exp(-0.5 * (log_determinant + quadratic_difference))


def student_t_copula_pdf(
    pseudo_observations: ArrayLike,
    correlation: ArrayLike,
    degrees_of_freedom: float,
) -> FloatArray:
    """Evaluate a Student-t copula density."""

    observations = np.asarray(pseudo_observations, dtype=float)
    correlation_array = np.asarray(correlation, dtype=float)
    require(
        observations.ndim == 2 and observations.shape[1] == len(correlation_array),
        "diss:copula:DimensionMismatch",
        "Pseudo-observations and correlation dimensions must agree.",
    )
    require(
        np.all((observations > 0.0) & (observations < 1.0)),
        "diss:copula:InvalidPseudoObservations",
        "Pseudo-observations must lie in (0, 1).",
    )
    require(
        degrees_of_freedom > 2.0,
        "diss:distribution:InvalidDegreesOfFreedom",
        "Student-t degrees of freedom must exceed two.",
    )
    scores = t.ppf(observations, degrees_of_freedom)
    joint = multivariate_t.logpdf(
        scores,
        loc=np.zeros(observations.shape[1]),
        shape=correlation_array,
        df=degrees_of_freedom,
    )
    marginal = np.sum(t.logpdf(scores, degrees_of_freedom), axis=1)
    return np.exp(joint - marginal)


def _diagonal_sequence(values: FloatArray) -> FloatArray:
    series_count, order = values.shape
    matrices = np.zeros((order, series_count, series_count), dtype=float)
    for lag in range(order):
        matrices[lag] = np.diag(values[:, lag])
    return matrices


def parameter_matrices(
    parameters: ArrayLike,
    dynamic_type: str,
    arch_order: int,
    garch_order: int,
    series_count: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Convert the MATLAB-compatible DCC parameter vector to matrix sequences."""

    values = np.asarray(parameters, dtype=float).reshape(-1)
    if dynamic_type == "DCC":
        require(
            len(values) == arch_order + garch_order,
            "diss:dependence:InvalidParameterCount",
            "Invalid DCC parameter count.",
        )
        arch = np.tile(values[:arch_order], (series_count, 1))
        garch = np.tile(values[arch_order:], (series_count, 1))
        asymmetric = np.empty((series_count, 0))
    elif dynamic_type == "ADCC":
        require(
            len(values) == 2 * arch_order + garch_order,
            "diss:dependence:InvalidParameterCount",
            "Invalid ADCC parameter count.",
        )
        arch = np.tile(values[:arch_order], (series_count, 1))
        asymmetric = np.tile(values[arch_order : 2 * arch_order], (series_count, 1))
        garch = np.tile(values[2 * arch_order :], (series_count, 1))
    elif dynamic_type == "GDCC":
        count = arch_order * series_count
        require(
            len(values) == (arch_order + garch_order) * series_count,
            "diss:dependence:InvalidParameterCount",
            "Invalid GDCC parameter count.",
        )
        arch = values[:count].reshape((series_count, arch_order), order="F")
        garch = values[count:].reshape((series_count, garch_order), order="F")
        asymmetric = np.empty((series_count, 0))
    elif dynamic_type == "AGDCC":
        count = arch_order * series_count
        require(
            len(values) == (2 * arch_order + garch_order) * series_count,
            "diss:dependence:InvalidParameterCount",
            "Invalid AGDCC parameter count.",
        )
        arch = values[:count].reshape((series_count, arch_order), order="F")
        asymmetric = values[count : 2 * count].reshape((series_count, arch_order), order="F")
        garch = values[2 * count :].reshape((series_count, garch_order), order="F")
    else:
        raise DissError(
            "diss:dependence:UnsupportedDynamicType", f"Unsupported type {dynamic_type}."
        )
    return _diagonal_sequence(arch), _diagonal_sequence(garch), _diagonal_sequence(asymmetric)


def _intercept(
    q_bar: FloatArray,
    negative_q_bar: FloatArray,
    arch: FloatArray,
    garch: FloatArray,
    asymmetric: FloatArray,
) -> FloatArray:
    value = q_bar.copy()
    for matrix in arch:
        value -= matrix.T @ q_bar @ matrix
    for matrix in garch:
        value -= matrix.T @ q_bar @ matrix
    for matrix in asymmetric:
        value -= matrix.T @ negative_q_bar @ matrix
    return (value + value.T) / 2.0


def _correlation(q: FloatArray) -> FloatArray:
    diagonal = np.diag(q)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0):
        raise DissError(
            "diss:dependence:InvalidForecastCovariance", "DCC covariance has invalid diagonal."
        )
    scale = np.sqrt(diagonal)
    correlation = q / np.outer(scale, scale)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    np.linalg.cholesky(correlation)
    return correlation


def correlation_filter(
    standardized_residuals: ArrayLike,
    q_bar: ArrayLike,
    arch_matrices: FloatArray,
    garch_matrices: FloatArray,
    asymmetric_matrices: FloatArray,
    negative_q_bar: ArrayLike,
    *,
    return_path: bool = False,
) -> tuple[float, bool, FloatArray | None, FloatArray | None]:
    """Filter a DCC/ADCC/GDCC/AGDCC process using Cholesky solves."""

    residuals = np.asarray(standardized_residuals, dtype=float)
    q_mean = np.asarray(q_bar, dtype=float)
    negative_mean = np.asarray(negative_q_bar, dtype=float)
    intercept = _intercept(
        q_mean, negative_mean, arch_matrices, garch_matrices, asymmetric_matrices
    )
    q_history: list[FloatArray] = []
    correlations: list[FloatArray] = []
    negative = np.minimum(residuals, 0.0)
    nll = 0.0
    try:
        for observation in range(len(residuals)):
            current = intercept.copy()
            for lag, matrix in enumerate(arch_matrices, start=1):
                if observation - lag >= 0:
                    transformed = residuals[observation - lag] @ matrix
                    current += np.outer(transformed, transformed)
            for lag, matrix in enumerate(asymmetric_matrices, start=1):
                if observation - lag >= 0:
                    transformed = negative[observation - lag] @ matrix
                    current += np.outer(transformed, transformed)
            for lag, matrix in enumerate(garch_matrices, start=1):
                previous = q_history[observation - lag] if observation - lag >= 0 else q_mean
                current += matrix.T @ previous @ matrix
            current = (current + current.T) / 2.0
            correlation = _correlation(current)
            factor = np.linalg.cholesky(correlation)
            solved = np.linalg.solve(factor, residuals[observation])
            nll += 0.5 * (2.0 * np.log(np.diag(factor)).sum() + solved @ solved)
            q_history.append(current)
            if return_path:
                correlations.append(correlation)
    except (np.linalg.LinAlgError, DissError, FloatingPointError):
        return float("nan"), False, None, None
    return (
        float(nll),
        True,
        np.stack(correlations, axis=2) if return_path else None,
        np.stack(q_history, axis=2) if return_path else None,
    )


def _parameter_count(
    dynamic_type: str, arch_order: int, garch_order: int, series_count: int
) -> int:
    if dynamic_type == "DCC":
        return arch_order + garch_order
    if dynamic_type == "ADCC":
        return 2 * arch_order + garch_order
    if dynamic_type == "GDCC":
        return (arch_order + garch_order) * series_count
    if dynamic_type == "AGDCC":
        return (2 * arch_order + garch_order) * series_count
    raise DissError("diss:dependence:UnsupportedDynamicType", f"Unsupported type {dynamic_type}.")


def _starting_values(
    dynamic_type: str, arch_order: int, garch_order: int, series_count: int
) -> FloatArray:
    arch = np.full(
        arch_order * (series_count if "G" in dynamic_type else 1), np.sqrt(0.03 / arch_order)
    )
    garch = np.full(
        garch_order * (series_count if "G" in dynamic_type else 1), np.sqrt(0.90 / garch_order)
    )
    if dynamic_type in {"ADCC", "AGDCC"}:
        asymmetric = np.full(
            arch_order * (series_count if dynamic_type == "AGDCC" else 1),
            np.sqrt(0.01 / arch_order),
        )
        return np.concatenate((arch, asymmetric, garch))
    return np.concatenate((arch, garch))


def fit_dcc(
    standardized_residuals: ArrayLike, config: dict[str, Any], state: dict[str, Any]
) -> tuple[DccModel, dict[str, Any]]:
    """Estimate one DCC-family model using bounded SLSQP optimization."""

    residuals = np.asarray(standardized_residuals, dtype=float)
    observation_count, series_count = residuals.shape
    q_bar = np.cov(residuals, rowvar=False, ddof=1)
    q_bar = np.atleast_2d((q_bar + q_bar.T) / 2.0)
    try:
        np.linalg.cholesky(q_bar)
    except np.linalg.LinAlgError as exception:
        raise DissError(
            "diss:dependence:SingularResidualCovariance",
            "Residual covariance must be positive definite.",
        ) from exception
    negative_q_bar = np.atleast_2d(np.cov(np.minimum(residuals, 0.0), rowvar=False, ddof=1))
    settings = config["dependence"]
    dynamic_type = settings["dynamic"]
    arch_order = settings["archOrder"]
    garch_order = settings["garchOrder"]
    count = _parameter_count(dynamic_type, arch_order, garch_order, series_count)
    start = _starting_values(dynamic_type, arch_order, garch_order, series_count)
    if (
        state.get("dependence_type") == dynamic_type
        and np.size(state.get("dependence_parameters")) == count
    ):
        candidate = np.asarray(state["dependence_parameters"], dtype=float)
        if np.isfinite(candidate).all():
            start = candidate

    def objective(parameters: FloatArray) -> float:
        arch, garch, asymmetric = parameter_matrices(
            parameters, dynamic_type, arch_order, garch_order, series_count
        )
        value, valid, _, _ = correlation_filter(
            residuals, q_bar, arch, garch, asymmetric, negative_q_bar
        )
        return value if valid and np.isfinite(value) else 1e12

    def constraint(parameters: FloatArray) -> float:
        arch, garch, asymmetric = parameter_matrices(
            parameters, dynamic_type, arch_order, garch_order, series_count
        )
        return float(
            np.linalg.eigvalsh(_intercept(q_bar, negative_q_bar, arch, garch, asymmetric)).min()
            - 1e-7
        )

    output = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=Bounds(np.full(count, 1e-6), np.full(count, 1.0 - 1e-6)),
        constraints={"type": "ineq", "fun": constraint},
        options={
            "maxiter": config["optimization"]["maxIterations"],
            "ftol": 1e-8,
            "disp": config["optimization"]["display"] == "iter",
        },
    )
    if not output.success or not np.isfinite(output.fun):
        raise DissError(
            "diss:dependence:OptimizationFailed",
            f"{dynamic_type} estimation failed: {output.message}",
        )
    parameters = np.asarray(output.x, dtype=float)
    model = DccModel(
        dynamic_type,
        parameters,
        arch_order,
        garch_order,
        q_bar,
        negative_q_bar,
        float(output.fun),
        float(-output.fun),
        float(2 * count + 2 * output.fun),
        float(count * np.log(observation_count) + 2 * output.fun),
        True,
        str(output.message),
    )
    next_state = dict(state)
    next_state.update({"dependence_type": dynamic_type, "dependence_parameters": parameters})
    return model, next_state


def _forecast_q(model: DccModel, residuals: FloatArray) -> FloatArray:
    arch, garch, asymmetric = parameter_matrices(
        model.parameters,
        model.dynamic_type,
        model.arch_order,
        model.garch_order,
        residuals.shape[1],
    )
    intercept = _intercept(model.q_bar, model.negative_q_bar, arch, garch, asymmetric)
    q = intercept + garch[0].T @ model.q_bar @ garch[0]
    for residual in residuals:
        innovation = residual @ arch[0]
        negative = np.minimum(residual, 0.0) @ (
            asymmetric[0] if len(asymmetric) else np.zeros_like(arch[0])
        )
        q = (
            intercept
            + np.outer(innovation, innovation)
            + np.outer(negative, negative)
            + garch[0].T @ q @ garch[0]
        )
        q = (q + q.T) / 2.0
    return q


def initialize_forecast_state(
    model: DccModel, standardized_residuals: ArrayLike
) -> DccForecastState:
    require(
        model.arch_order == model.garch_order == 1,
        "diss:dependence:UnsupportedRecursiveOrder",
        "Recursive forecasts require DCC(1,1).",
    )
    residuals = np.asarray(standardized_residuals, dtype=float)
    arch, garch, asymmetric = parameter_matrices(
        model.parameters, model.dynamic_type, 1, 1, residuals.shape[1]
    )
    asymmetric_matrix = asymmetric[0] if len(asymmetric) else np.zeros_like(arch[0])
    intercept = _intercept(model.q_bar, model.negative_q_bar, arch, garch, asymmetric)
    q = _forecast_q(model, residuals)
    return DccForecastState(intercept, arch[0], garch[0], asymmetric_matrix, q, _correlation(q))


def update_forecast_state(
    state: DccForecastState, standardized_residual: ArrayLike
) -> DccForecastState:
    residual = np.asarray(standardized_residual, dtype=float).reshape(-1)
    innovation = residual @ state.arch_matrix
    negative = np.minimum(residual, 0.0) @ state.asymmetric_matrix
    q = (
        state.intercept
        + np.outer(innovation, innovation)
        + np.outer(negative, negative)
        + state.garch_matrix.T @ state.current_q @ state.garch_matrix
    )
    q = (q + q.T) / 2.0
    return DccForecastState(
        state.intercept,
        state.arch_matrix,
        state.garch_matrix,
        state.asymmetric_matrix,
        q,
        _correlation(q),
    )


def fit_gaussian_copula(pseudo_observations: ArrayLike) -> GaussianCopula:
    observations = np.asarray(pseudo_observations, dtype=float)
    require(
        observations.ndim == 2 and observations.shape[1] >= 2,
        "diss:copula:TooFewSeries",
        "A copula requires at least two series.",
    )
    require(
        np.all((observations > 0.0) & (observations < 1.0)),
        "diss:copula:InvalidPseudoObservations",
        "Pseudo-observations must lie in (0, 1).",
    )
    scores = InnovationDistribution("Gaussian").ppf(observations)
    correlation = np.corrcoef(scores, rowvar=False)
    correlation = (correlation + correlation.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    shrinkage = 1e-12
    while True:
        try:
            factor = np.linalg.cholesky(correlation)
            break
        except np.linalg.LinAlgError:
            if shrinkage > 1e-2:
                raise DissError(
                    "diss:copula:SingularCorrelation", "Copula correlation is singular."
                ) from None
            correlation = (1.0 - shrinkage) * correlation + shrinkage * np.eye(len(correlation))
            shrinkage *= 10.0
    solved = np.linalg.solve(factor, scores.T)
    difference = np.sum(solved**2, axis=0) - np.sum(scores**2, axis=1)
    log_likelihood = float(np.sum(-0.5 * (2.0 * np.log(np.diag(factor)).sum() + difference)))
    return GaussianCopula(correlation, log_likelihood, -log_likelihood)
