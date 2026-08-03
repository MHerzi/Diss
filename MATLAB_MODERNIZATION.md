# MATLAB modernization and performance verification

Baseline: repository commit `7f816038a2560c40b9cc2c8491ab461545ad6be1`.
The numerical checks and benchmarks were run with MATLAB R2021a on the
same machine. The rewritten code avoids deprecated constructs and follows
current MATLAB numerical practices while retaining R2021a compatibility.

## Main changes

- DCC, ADCC, GDCC, and AGDCC share a Cholesky-based correlation filter.
  Objective-only calls keep only the lagged Q states needed by the recursion.
- DCC forecast functions use a ring buffer instead of complete historical
  Q and R arrays.
- Monte Carlo VaR applies correlation to standardized innovations before
  marginal scaling and calculates portfolio quantiles from portfolio draws.
- Gaussian, t, Clayton, Gumbel, and Frank copula densities use numeric,
  vectorized formulas. Symbolic differentiation, `inline`, `vectorize`,
  `eval`, explicit inverses, and explicit determinants were removed.
- Empirical CDF and correlation-parameter conversion have correctly named,
  directly callable implementations; compatibility wrappers remain.
- The dynamic Patton copula path replaces one-cluster city-block `kmeans`
  calls with the equivalent cross-sectional median distance and `movsum`.
- Marginal-model candidate searches can skip Hessian, score, and robust
  covariance calculations. Full inference remains the default for final fits.
- `MainFile.m`, `setInputs.m`, and the multivariate backtest contain fixes
  for invalid script control flow, input parsing, AR-lag alignment, dimensions,
  misspelled variables, and incorrect function declarations.

## Regression tests

Run from the repository root:

```matlab
addpath(pwd)
addpath(fullfile(pwd, "tests"))
results = runtests(fullfile(pwd, "tests"));
assertSuccess(results)
```

The suite contains 23 tests covering numerical equivalence of all four DCC
variants, DCC forecasts, all supported copula families, Monte Carlo covariance
and VaR behavior, the Patton parameter path, empirical CDF behavior, and legacy
`rho2theta` ordering. Twenty-one tests completed successfully in MATLAB R2021a.
The two Patton reference tests were added after the local MATLAB launcher began
blocking during initialization; they pass the standalone MATLAB Code Analyzer
but still need one runtime execution when MATLAB can initialize normally.

The maximum relative discrepancy against saved legacy copula results is
`1.03e-14`; the DCC regression tolerance is `1e-10`.

## Benchmarks

The reproducible benchmark functions are in `benchmarks/`:

```matlab
addpath(fullfile(pwd, "benchmarks"))
benchmarkDccCorrelationFilter
benchmarkDccForecastPath
benchmarkVaRQuantiles
benchmarkEmpiricalCDF
benchmarkCopulaParameterPath
```

Measured results:

| Operation | Speedup | Relevant memory reduction |
|---|---:|---:|
| DCC objective filter | 1.05x | 1,025,024 B to 512 B for Q state |
| DCC forecast path | 1.29x | 1,538,048 B to 1,536 B for Q state |
| VaR quantile calculation | 8.02x | one batched quantile result; no sorted copy per probability |
| Empirical CDF transform | 5.08x | preallocated rank restoration |
| Clayton copula density | 17.94x | no dimension-specific expressions |
| Gumbel copula density | 81.45x | no run-time symbolic differentiation |
| Frank copula density | 20.32x | no run-time symbolic differentiation |
| Gaussian copula density | 1.85x | no explicit inverse or determinant |
| t copula density | 3.31x | no explicit inverse or determinant |

`benchmarkCopulaParameterPath` checks the new path against the former
one-cluster `kmeans` formulation before reporting its timing. No timing is
recorded here because MATLAB R2021a could not initialize for the final run.

## Verification boundary

The repository does not contain callable 64-bit implementations of
`garchcore`, `multigarchcore`, and `egarchcore` (only `maxcore.mexw32` is
present). Therefore the marginal-model fast path is syntax-checked but cannot
be benchmarked end-to-end from this checkout. Its optimization is isolated to
candidate selection and skips the numerical Hessian plus two likelihood
evaluations per parameter for score construction. Final selected models still
use the original inference path.
