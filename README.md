# GARCH and Copula dissertation code

The maintained implementation is the MATLAB namespace in `src/+diss`.
Historical functions and `MainFile.m` are isolated in `legacy` as a frozen
reference and are not required by the canonical pipeline.

## Start

Create the MATLAB Project once from the repository root:

```matlab
addpath("tools")
createProject
```

After opening the project, select a version-controlled chapter configuration
and run it with explicit input data:

```matlab
config = study.chapter03.dccGaussian();
results = diss.runExperiment(returns, config);
runDirectory = diss.io.saveRun(results, "results");
```

## Tests

```matlab
addpath("tools")
results = runAllTests();
```

Tests are separated into `unit`, `regression`, and `integration`. Numerical
legacy comparisons use frozen formulas or reference artifacts; new model
adapters also test the compatibility entry point and the canonical engine
against each other.

See `ARCHITECTURE.md` for supported adapters, extension points, and deliberate
validation failures for models that have not yet been migrated safely.
