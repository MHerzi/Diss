"""Common model and forecast result types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Forecast:
    count: int
    confidence_level: float
    portfolio_mean: FloatArray
    portfolio_standard_deviation: FloatArray
    return_quantile: FloatArray
    loss_var: FloatArray
    asset_mean: FloatArray | None = None
    asset_variance: FloatArray | None = None
    correlation: FloatArray | None = None
    simulations: tuple[FloatArray, ...] = ()


def empty_asset_path() -> FloatArray:
    return np.empty((0, 0), dtype=float)
