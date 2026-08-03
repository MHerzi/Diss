# Python dissertation pipeline

This branch provides a Python-native implementation of the package pipeline
from `codex/matlab-modernization`. The MATLAB implementation remains in
`src/+diss` as a numerical reference; production Python code lives in
`src/diss`.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
```

Python 3.11 or newer is required. NumPy provides the array layer and SciPy
provides probability distributions and constrained optimization.

## Quick start

```python
import numpy as np
from diss import defaults, run_experiment

rng = np.random.default_rng(42)
returns = rng.multivariate_normal(
    np.zeros(3),
    np.array([[0.0004, 0.0001, 0.00005],
              [0.0001, 0.0003, 0.00004],
              [0.00005, 0.00004, 0.0002]]),
    size=500,
)
config = defaults()
config["model"]["kind"] = "deltaNormal"
result = run_experiment(returns, config)
print(result.forecast.loss_var)
```

## Canonical adapters

- `deltaNormal`
- `univariateGarch` with Gaussian or standardized Student-t innovations
- `multiGarch` with DCC, ADCC, GDCC, or AGDCC dependence
- `multiCopula` with a static Gaussian copula

The Python API deliberately mirrors the schema-v2 MATLAB configuration. Old
schema-v1 dependence fields are migrated before validation. Unsupported vine,
mixture, dynamic-copula, and t-copula specifications fail explicitly rather
than silently selecting a different model.

## Validation

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check src/diss tests/python experiments/python
```

The test suite covers configuration migration, data handling, distribution
round trips, adapter dispatch, full-sample and walk-forward execution,
recursive DCC equivalence, copula recovery, deterministic manifests, result
round trips, and performance-sensitive state updates.

Current validation snapshot: 37 pytest tests pass with 91% statement coverage.
The recursive DCC benchmark returns the same state as complete refiltering and
is approximately 39 times faster on the current bundled benchmark run.

See [PYTHON_ARCHITECTURE.md](PYTHON_ARCHITECTURE.md) for the design and
migration boundary.
