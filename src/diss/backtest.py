"""Shared walk-forward schedule generation."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import DissError


@dataclass(frozen=True, slots=True)
class ScheduleWindow:
    window: int
    estimation_start: int
    estimation_end: int
    forecast_start: int
    forecast_end: int
    refit_marginal: bool

    @property
    def forecast_length(self) -> int:
        return self.forecast_end - self.forecast_start


def create_schedule(observation_count: int, options: dict) -> tuple[ScheduleWindow, ...]:
    """Create zero-based half-open estimation and forecast windows."""

    starts = range(options["initialWindow"], observation_count, options["stepSize"])
    windows: list[ScheduleWindow] = []
    last_refit_end = -(10**18)
    for forecast_start in starts:
        forecast_end = forecast_start + options["forecastHorizon"]
        if forecast_end > observation_count:
            if not options["includePartialFinalWindow"]:
                continue
            forecast_end = observation_count
        estimation_end = forecast_start
        estimation_start = 0
        if options["windowType"] == "rolling":
            estimation_start = estimation_end - options["rollingWindow"]
        refit = not windows or estimation_end - last_refit_end >= options["marginalRefitEvery"]
        if refit:
            last_refit_end = estimation_end
        windows.append(
            ScheduleWindow(
                len(windows) + 1,
                estimation_start,
                estimation_end,
                forecast_start,
                forecast_end,
                refit,
            )
        )
    if not windows:
        raise DissError("diss:backtest:NoForecastWindows", "Settings produce no forecast windows.")
    return tuple(windows)
