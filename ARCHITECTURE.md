# Dissertation software architecture

The canonical MATLAB code lives in `src/+diss`. The `legacy` directory
contains the frozen historical implementation and is not part of the
production project path. Chapter-specific choices live in `experiments/+study`; they must only
construct configuration structs and call `diss.runExperiment`.

## Statistical pipeline

1. `diss.data` validates and aligns returns.
2. `diss.marginal` selects, fits, filters and forecasts AR-GARCH models.
3. A registered model adapter adds either no dependence, DCC dependence, or
   a copula dependence model.
4. `diss.experiment` executes full-sample or walk-forward experiments.
5. `diss.evaluation` computes model-independent forecast diagnostics.
6. `diss.io` writes a result artifact and a reproducibility manifest.

`legacy/MainFile.m` remains a numerical and historical reference only. New research
must not add branches to it.

## Supported canonical adapters

- `deltaNormal`
- `univariateGarch` with Gaussian or Student-t innovations
- `multiGarch` with DCC/ADCC/GDCC/AGDCC and Gaussian marginals
- `multiCopula` with a static Gaussian copula and Gaussian or Student-t
  marginals

Unsupported dynamic copulas and vine models fail during configuration
validation. Their legacy implementations remain available for regression
work until equivalent adapters and tests exist.

## Reproducibility

Every canonical result records the resolved configuration, data and
configuration hashes, Git commit, MATLAB release, installed products, random
seed and run time. Generated `results` and processed data are not committed.
