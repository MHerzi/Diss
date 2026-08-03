# Python software architecture

The primary Python source tree is `src/diss`. It follows the same statistical
boundaries as the MATLAB package while using Python-native modules,
dataclasses, type hints, NumPy arrays, SciPy optimizers, and pytest.

## Execution flow

1. `diss.config` migrates and validates schema-v2 configuration.
2. `diss.data` creates a validated immutable dataset.
3. `diss.backtest` creates zero-based, half-open walk-forward windows.
4. `diss.registry` resolves a model adapter without central model branching.
5. `diss.marginal` selects and estimates AR-GARCH margins.
6. `diss.dependence` estimates DCC-family or Gaussian-copula dependence.
7. `diss.experiment` runs full-sample or backtest experiments.
8. `diss.evaluation` computes common violation diagnostics.
9. `diss.io` records deterministic hashes and saves reproducible artifacts.

## Numerical conventions

- Sample covariances use `ddof=1`, matching MATLAB `cov` defaults.
- DCC parameter vectors and diagonal matrix ordering match the MATLAB kernels.
- DCC likelihoods use Cholesky factors rather than explicit inverses.
- Recursive marginal and DCC forecasts retain only state needed for the next
  update; forecasting does not refilter the complete history.
- Student-t innovations are standardized to unit variance.
- Random simulation uses NumPy's explicit `Generator` seeded by configuration.

## MATLAB comparison boundary

`src/+diss` and `tests/**/*.m` remain on this branch as the reference
implementation. Python tests live in `tests/python`. Frozen pre-modernization
MATLAB code remains isolated in `legacy` and is never added to Python's import
path.
