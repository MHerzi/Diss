"""Compare recursive DCC forecasting against complete-history refiltering."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from diss.dependence import (
    DccModel,
    initialize_forecast_state,
    update_forecast_state,
)


def benchmark() -> dict[str, float]:
    rng = np.random.default_rng(3902)
    history = rng.standard_normal((2_000, 6))
    updates = rng.standard_normal((40, 6))
    q_bar = np.cov(history, rowvar=False, ddof=1)
    model = DccModel(
        "DCC",
        np.array([0.10, 0.80]),
        1,
        1,
        q_bar,
        np.cov(np.minimum(history, 0.0), rowvar=False, ddof=1),
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        True,
        "benchmark",
    )

    started = perf_counter()
    state = initialize_forecast_state(model, history)
    recursive = [state.correlation]
    for update in updates:
        state = update_forecast_state(state, update)
        recursive.append(state.correlation)
    recursive_seconds = perf_counter() - started

    started = perf_counter()
    complete = [initialize_forecast_state(model, history).correlation]
    growing = history.copy()
    for update in updates:
        growing = np.vstack((growing, update))
        complete.append(initialize_forecast_state(model, growing).correlation)
    complete_seconds = perf_counter() - started
    difference = float(np.max(np.abs(np.stack(recursive) - np.stack(complete))))
    return {
        "recursiveSeconds": recursive_seconds,
        "completeRefilterSeconds": complete_seconds,
        "speedup": complete_seconds / recursive_seconds,
        "maximumDifference": difference,
    }


if __name__ == "__main__":
    print(benchmark())
