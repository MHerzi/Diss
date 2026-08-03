"""Python implementation of the dissertation risk-model pipeline."""

from .config import defaults, from_legacy_spec, validate_config
from .experiment import run_experiment, run_grid
from .pipeline import fit_window, forecast_window, prepare

__all__ = [
    "defaults",
    "fit_window",
    "forecast_window",
    "from_legacy_spec",
    "prepare",
    "run",
    "run_experiment",
    "run_grid",
    "validate_config",
]

# Compatibility alias for the MATLAB-facing ``diss.run`` entry point.
run = run_experiment
