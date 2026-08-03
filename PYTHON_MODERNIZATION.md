# Python modernization record

This document records the functional mapping from the MATLAB modernization to
the Python implementation on `codex/python-modernization`.

| MATLAB package | Python module | Responsibility |
| --- | --- | --- |
| `diss.config` | `diss.config` | Defaults, schema migration, validation, `MainFile` mapping |
| `diss.data` | `diss.data` | Numeric input normalization and missing-row policy |
| `diss.backtest` | `diss.backtest` | Shared expanding/rolling schedule |
| `diss.marginal` | `diss.marginal` | AR-GARCH/EGARCH/GJR estimation and recursive forecast |
| `diss.dependence` | `diss.dependence` | DCC/ADCC/GDCC/AGDCC, recursive state, Gaussian copula |
| `diss.registry` | `diss.registry` | Adapter lookup without central model switches |
| `diss.experiment` | `diss.experiment` | Full sample, walk-forward, grid execution, compact models |
| `diss.evaluation` | `diss.evaluation` | Violation rates and Kupiec coverage test |
| `diss.io` | `diss.io` | SHA-256 identities, manifest, result persistence |
| `diss.risk` | `diss.risk` | Linear portfolio aggregation and Gaussian quantiles |

## Verified numerical behavior

- DCC-family parameter vectors preserve MATLAB column-major parameter order.
- Recursive DCC, ADCC, GDCC, and AGDCC states equal complete-history
  refiltering to machine precision.
- Saved MATLAB Gaussian and Student-t copula-density fixtures match within
  `2e-12` relative tolerance.
- Sample covariance uses `ddof=1`, corresponding to MATLAB `cov`.
- Standardized Student-t CDF and quantile operations round-trip within
  floating-point tolerance and retain unit variance.
- Delta-Normal forecasts match a direct NumPy portfolio calculation.
- Static-copula simulation is exactly reproducible for a fixed seed.

## Performance snapshot

`benchmarks/benchmark_recursive_dcc_state.py` compares the state update with
rebuilding the entire filtered history for 2,000 observations, six series,
and 40 updates. The verified run produced zero numerical difference and an
approximately 39x speedup. Runtime varies by machine; equality is asserted
independently of timing.

## Validation snapshot

- Python 3.12.13
- NumPy 2.5.1
- SciPy 1.18.0
- 37 pytest tests passed
- 91% statement coverage
- Ruff lint and formatting checks passed
- `pip check` reported no broken requirements

The MATLAB runtime on the development machine remains unable to initialize,
so direct process-to-process execution was not possible. Cross-language
regression therefore uses saved MATLAB fixtures and exact translations of the
already regression-tested MATLAB numerical recursions.
