from __future__ import annotations

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.stats import norm

from diss.config import defaults
from diss.dependence import (
    DccModel,
    correlation_filter,
    fit_dcc,
    fit_gaussian_copula,
    gaussian_copula_pdf,
    initialize_forecast_state,
    parameter_matrices,
    student_t_copula_pdf,
    update_forecast_state,
)
from diss.distributions import InnovationDistribution


@pytest.mark.parametrize(
    "distribution",
    [InnovationDistribution("Gaussian"), InnovationDistribution("t", 7.5)],
)
def test_innovation_cdf_quantile_round_trip(distribution: InnovationDistribution) -> None:
    probabilities = np.linspace(0.001, 0.999, 101)
    recovered = distribution.cdf(distribution.ppf(probabilities))
    np.testing.assert_allclose(recovered, probabilities, atol=2e-12)


def test_student_t_is_standardized_to_unit_variance() -> None:
    rng = np.random.default_rng(31)
    distribution = InnovationDistribution("t", 6.0)
    samples = distribution.ppf(rng.uniform(0.0001, 0.9999, 200_000))
    assert np.var(samples) == pytest.approx(1.0, rel=0.025)


def test_gdcc_parameter_order_matches_matlab_column_major_order() -> None:
    arch, garch, asymmetric = parameter_matrices(np.arange(1, 13), "GDCC", 2, 2, 3)
    np.testing.assert_array_equal(np.diag(arch[0]), [1, 2, 3])
    np.testing.assert_array_equal(np.diag(arch[1]), [4, 5, 6])
    np.testing.assert_array_equal(np.diag(garch[0]), [7, 8, 9])
    assert asymmetric.shape == (0, 3, 3)


def _synthetic_model(kind: str, residuals: np.ndarray) -> DccModel:
    series_count = residuals.shape[1]
    if kind == "DCC":
        parameters = np.array([0.10, 0.80])
    elif kind == "ADCC":
        parameters = np.array([0.08, 0.04, 0.76])
    elif kind == "GDCC":
        parameters = np.r_[
            np.linspace(0.06, 0.10, series_count), np.linspace(0.72, 0.80, series_count)
        ]
    else:
        parameters = np.r_[
            np.linspace(0.06, 0.10, series_count),
            np.linspace(0.02, 0.04, series_count),
            np.linspace(0.72, 0.80, series_count),
        ]
    q_bar = np.cov(residuals, rowvar=False, ddof=1)
    negative = np.cov(np.minimum(residuals, 0.0), rowvar=False, ddof=1)
    return DccModel(
        kind, parameters, 1, 1, q_bar, negative, np.nan, np.nan, np.nan, np.nan, True, "test"
    )


@pytest.mark.parametrize("kind", ["DCC", "ADCC", "GDCC", "AGDCC"])
def test_recursive_state_matches_complete_refilter(kind: str) -> None:
    rng = np.random.default_rng(4601)
    history = rng.standard_normal((150, 3))
    updates = rng.standard_normal((8, 3))
    model = _synthetic_model(kind, history)
    state = initialize_forecast_state(model, history)
    growing = history.copy()
    for update in updates:
        state = update_forecast_state(state, update)
        growing = np.vstack((growing, update))
        reference = initialize_forecast_state(model, growing)
        np.testing.assert_allclose(state.current_q, reference.current_q, atol=1e-12, rtol=0)
        np.testing.assert_allclose(state.correlation, reference.correlation, atol=1e-12, rtol=0)


def test_correlation_filter_returns_positive_definite_path() -> None:
    rng = np.random.default_rng(71)
    residuals = rng.standard_normal((100, 3))
    model = _synthetic_model("ADCC", residuals)
    arch, garch, asymmetric = parameter_matrices(model.parameters, "ADCC", 1, 1, 3)
    likelihood, valid, correlations, covariance = correlation_filter(
        residuals,
        model.q_bar,
        arch,
        garch,
        asymmetric,
        model.negative_q_bar,
        return_path=True,
    )
    assert valid and np.isfinite(likelihood)
    assert correlations is not None and covariance is not None
    assert np.min([np.linalg.eigvalsh(correlations[:, :, step]).min() for step in range(100)]) > 0


@pytest.mark.parametrize("kind", ["DCC", "ADCC", "GDCC", "AGDCC"])
def test_each_dcc_family_can_be_fitted(kind: str) -> None:
    residuals = np.random.default_rng(771).standard_normal((180, 3))
    config = defaults()
    config["dependence"]["dynamic"] = kind
    config["optimization"]["maxIterations"] = 150
    model, state = fit_dcc(residuals, config, {})
    assert model.success
    assert np.isfinite(model.negative_log_likelihood)
    assert state["dependence_type"] == kind


def test_gaussian_copula_recovers_target_correlation() -> None:
    rng = np.random.default_rng(912)
    target = np.array([[1.0, 0.55, -0.20], [0.55, 1.0, 0.30], [-0.20, 0.30, 1.0]])
    scores = rng.standard_normal((50_000, 3)) @ np.linalg.cholesky(target).T
    copula = fit_gaussian_copula(norm.cdf(scores))
    np.testing.assert_allclose(copula.correlation, target, atol=0.015)
    assert np.linalg.eigvalsh(copula.correlation).min() > 0


def test_copula_densities_match_saved_matlab_reference() -> None:
    reference = loadmat("tests/legacyCopulaReference.mat")
    gaussian = gaussian_copula_pdf(reference["u"], reference["rho"])
    student = student_t_copula_pdf(reference["u"], reference["rho"], 7.0)
    np.testing.assert_allclose(
        gaussian,
        reference["gaussianLegacy"].reshape(-1),
        rtol=2e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        student,
        reference["tLegacy"].reshape(-1),
        rtol=2e-12,
        atol=1e-14,
    )
