"""Standardized innovation distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm, t

from .errors import DissError


@dataclass(frozen=True, slots=True)
class InnovationDistribution:
    name: str = "Gaussian"
    degrees_of_freedom: float | None = None

    def _student_scale(self) -> float:
        degrees = self.degrees_of_freedom
        if degrees is None or not np.isfinite(degrees) or degrees <= 2:
            raise DissError(
                "diss:distribution:InvalidDegreesOfFreedom",
                "Student-t degrees of freedom must exceed two.",
            )
        return float(np.sqrt((degrees - 2.0) / degrees))

    def cdf(self, values: ArrayLike) -> NDArray[np.float64]:
        values_array = np.asarray(values, dtype=float)
        if self.name.casefold() == "gaussian":
            return np.asarray(norm.cdf(values_array), dtype=float)
        if self.name.casefold() == "t":
            scale = self._student_scale()
            return np.asarray(t.cdf(values_array / scale, self.degrees_of_freedom), dtype=float)
        raise DissError(
            "diss:distribution:UnsupportedInnovation", f"Unsupported distribution {self.name}."
        )

    def ppf(self, probability: ArrayLike) -> NDArray[np.float64]:
        probability_array = np.asarray(probability, dtype=float)
        if np.any((probability_array <= 0.0) | (probability_array >= 1.0)):
            raise DissError(
                "diss:distribution:InvalidProbability", "Probabilities must lie in (0, 1)."
            )
        if self.name.casefold() == "gaussian":
            return np.asarray(norm.ppf(probability_array), dtype=float)
        if self.name.casefold() == "t":
            scale = self._student_scale()
            return np.asarray(
                t.ppf(probability_array, self.degrees_of_freedom) * scale, dtype=float
            )
        raise DissError(
            "diss:distribution:UnsupportedInnovation", f"Unsupported distribution {self.name}."
        )

    def logpdf(self, standardized_values: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(standardized_values, dtype=float)
        if self.name.casefold() == "gaussian":
            return np.asarray(norm.logpdf(values), dtype=float)
        if self.name.casefold() == "t":
            scale = self._student_scale()
            return np.asarray(
                t.logpdf(values / scale, self.degrees_of_freedom) - np.log(scale), dtype=float
            )
        raise DissError(
            "diss:distribution:UnsupportedInnovation", f"Unsupported distribution {self.name}."
        )


def innovation_cdf(values: ArrayLike, distribution: InnovationDistribution) -> NDArray[np.float64]:
    return distribution.cdf(values)


def innovation_quantile(
    probability: ArrayLike, distribution: InnovationDistribution
) -> NDArray[np.float64]:
    return distribution.ppf(probability)
