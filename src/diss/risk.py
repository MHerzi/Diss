"""Portfolio-return aggregation and Gaussian risk helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from .errors import DissError


def aggregate_returns(
    returns: ArrayLike, weights: ArrayLike, method: str = "linear"
) -> NDArray[np.float64]:
    values = np.asarray(returns, dtype=float)
    weight_values = np.asarray(weights, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != len(weight_values):
        raise DissError("diss:risk:WeightDimensionMismatch", "Weights must match return columns.")
    if method != "linear":
        raise DissError("diss:risk:UnsupportedAggregation", f"Unsupported aggregation {method}.")
    return np.asarray(values @ weight_values, dtype=float)


def gaussian_return_quantile(
    mean: ArrayLike, standard_deviation: ArrayLike, confidence: float
) -> NDArray[np.float64]:
    return np.asarray(mean, dtype=float) + norm.ppf(1.0 - confidence) * np.asarray(
        standard_deviation, dtype=float
    )
