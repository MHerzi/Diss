"""Input-data normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import DissError, require


@dataclass(frozen=True, slots=True)
class Dataset:
    returns: NDArray[np.float64]
    observation_index: NDArray[np.int64]
    variable_names: tuple[str, ...]
    removed_rows: NDArray[np.int64]
    original_type: str
    time: tuple[Any, ...] = ()

    @property
    def observation_count(self) -> int:
        return int(self.returns.shape[0])

    @property
    def series_count(self) -> int:
        return int(self.returns.shape[1])


def prepare_data(data: Any, options: dict[str, Any]) -> Dataset:
    """Convert a numeric two-dimensional input into a validated dataset."""

    values = np.asarray(data)
    require(
        values.ndim == 2 and values.size > 0,
        "diss:data:UnsupportedType",
        "Input must be a nonempty 2-D numeric array.",
    )
    require(np.isrealobj(values), "diss:data:ComplexReturns", "Returns must be real-valued.")
    try:
        returns = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exception:
        raise DissError("diss:data:NonnumericVariable", "Returns must be numeric.") from exception
    original_count = returns.shape[0]
    index = np.arange(1, original_count + 1, dtype=np.int64)
    invalid = ~np.isfinite(returns).all(axis=1)
    removed = index[invalid]
    if invalid.any() and options["missingAction"] == "error":
        raise DissError("diss:data:NonfiniteReturns", f"Nonfinite return in row {int(removed[0])}.")
    if options["missingAction"] == "omitRows":
        returns = returns[~invalid]
        index = index[~invalid]
    require(
        len(returns) >= 2,
        "diss:data:TooFewObservations",
        "At least two complete observations are required.",
    )
    names = tuple(f"Series{number}" for number in range(1, returns.shape[1] + 1))
    return Dataset(returns, index, names, removed, type(data).__name__)
