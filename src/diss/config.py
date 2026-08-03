"""Configuration defaults, migration, and validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from .errors import DissError, require

SCHEMA_VERSION = 2
SUPPORTED_MODELS = (
    "deltaNormal",
    "univariateGarch",
    "multiGarch",
    "multiCopula",
)


def defaults() -> dict[str, Any]:
    """Return an independent default configuration dictionary."""

    return {
        "schemaVersion": SCHEMA_VERSION,
        "execution": {"mode": "full"},
        "model": {"kind": "multiGarch"},
        "dependence": {
            "kind": "dcc",
            "dynamic": "DCC",
            "archOrder": 1,
            "garchOrder": 1,
            "asymmetricOrder": 0,
            "copulaType": "gaussian",
            "families": [],
            "tails": "none",
            "options": {},
        },
        "marginal": {
            "enabled": True,
            "maxARLag": 1,
            "includeConstant": True,
            "archOrder": 1,
            "garchOrder": 1,
            "selectionPolicy": "initialWindow",
            "reestimateInBacktest": True,
            "engine": "scipy",
            "candidateFamilies": ["garch", "egarch", "gjr"],
            "distribution": "Gaussian",
            "fixedSpecifications": [],
        },
        "backtest": {
            "windowType": "expanding",
            "initialWindow": None,
            "rollingWindow": None,
            "forecastHorizon": 1,
            "stepSize": 1,
            "marginalRefitEvery": 1,
            "includePartialFinalWindow": False,
        },
        "simulation": {"numPaths": 10_000, "seed": 1},
        "risk": {
            "probability": 0.99,
            "portfolioWeights": None,
            "returnAggregation": "linear",
        },
        "inference": {"computeStandardErrors": False},
        "optimization": {
            "display": "off",
            "maxIterations": 500,
            "maxFunctionEvaluations": 10_000,
        },
        "data": {"missingAction": "error"},
        "output": {
            "saveWindowModels": False,
            "saveSimulations": False,
            "saveTrainingData": False,
        },
    }


def from_legacy_spec(specification: dict[str, Any]) -> dict[str, Any]:
    """Translate fields used by the historical ``MainFile.m`` Spec struct."""

    config = defaults()
    model_mapping = {
        "multigarch": "multiGarch",
        "multicopula": "multiCopula",
        "vinecopula": "vineCopula",
        "multimixcopula": "multiMixCopula",
        "dvine-mix": "dVineMix",
        "histsim_gauss": "histSimGaussian",
        "histsim_copula": "histSimCopula",
        "deltanormal": "deltaNormal",
    }
    dependence_mapping = {
        "multiGarch": "dcc",
        "multiCopula": "copula",
        "multiMixCopula": "copula",
        "histSimCopula": "copula",
        "vineCopula": "vine",
        "dVineMix": "vine",
    }

    if "purpose" in specification:
        config["execution"]["mode"] = str(specification["purpose"])
    if "ModelType" in specification:
        key = str(specification["ModelType"]).strip().casefold()
        if key not in model_mapping:
            raise DissError(
                "diss:config:UnknownLegacyModel",
                f"Unknown legacy ModelType: {specification['ModelType']}.",
            )
        kind = model_mapping[key]
        config["model"]["kind"] = kind
        config["dependence"]["kind"] = dependence_mapping.get(kind, "none")
    for old_name in ("DynamicType", "Dynamic"):
        if specification.get(old_name) not in (None, ""):
            config["dependence"]["dynamic"] = str(specification[old_name])
            break
    for old_name, new_name in (
        ("dccP", "archOrder"),
        ("dccQ", "garchOrder"),
        ("dccG", "asymmetricOrder"),
    ):
        if old_name in specification:
            config["dependence"][new_name] = specification[old_name]
    if "CopulaType" in specification:
        copula = str(specification["CopulaType"]).strip().casefold()
        if copula not in {"gauss", "gaussian", "t"}:
            raise DissError(
                "diss:config:UnknownLegacyCopula",
                f"Unknown legacy CopulaType: {specification['CopulaType']}.",
            )
        config["dependence"]["copulaType"] = "gaussian" if copula != "t" else "t"
    for old_name, new_name in (("family", "families"), ("tails", "tails")):
        if specification.get(old_name) not in (None, ""):
            config["dependence"][new_name] = specification[old_name]

    def on_off(name: str) -> bool:
        value = str(specification[name]).strip().casefold()
        if value not in {"on", "off"}:
            raise DissError(
                "diss:config:InvalidLegacySwitch",
                f"Legacy field {name} must be on or off.",
            )
        return value == "on"

    if "univariate" in specification:
        config["marginal"]["enabled"] = on_off("univariate")
    if "uniBacktest" in specification:
        config["marginal"]["reestimateInBacktest"] = on_off("uniBacktest")
    for old_name, new_name in (
        ("arlag", "maxARLag"),
        ("archP", "archOrder"),
        ("garchQ", "garchOrder"),
    ):
        if old_name in specification:
            config["marginal"][new_name] = specification[old_name]
    if "const" in specification:
        config["marginal"]["includeConstant"] = bool(specification["const"])
    if "ForecastStart" in specification:
        config["backtest"]["initialWindow"] = specification["ForecastStart"]
    if "ForecastNumb" in specification:
        config["backtest"]["forecastHorizon"] = specification["ForecastNumb"]
        config["backtest"]["stepSize"] = specification["ForecastNumb"]
    if "uniforecastP" in specification:
        config["backtest"]["marginalRefitEvery"] = specification["uniforecastP"]
    if "SimNumb" in specification:
        config["simulation"]["numPaths"] = specification["SimNumb"]
    if "beta" in specification:
        config["risk"]["probability"] = specification["beta"]
    if "stderrors" in specification:
        config["inference"]["computeStandardErrors"] = str(
            specification["stderrors"]
        ).strip().casefold() in {"an", "on"}
    return config


def migrate(config: dict[str, Any] | None) -> dict[str, Any]:
    """Migrate supported schema-v1 fields to the canonical schema."""

    supplied = deepcopy(config or {})
    version = int(supplied.get("schemaVersion", 1 if supplied else SCHEMA_VERSION))
    require(version <= SCHEMA_VERSION, "diss:config:FutureSchema", "Unsupported future schema.")
    if version < 2:
        model = supplied.setdefault("model", {})
        dependence = supplied.setdefault("dependence", {})
        for old, new in (
            ("dynamic", "dynamic"),
            ("dccP", "archOrder"),
            ("dccQ", "garchOrder"),
            ("dccG", "asymmetricOrder"),
        ):
            if old in model and new not in dependence:
                dependence[new] = model.pop(old)
        kind = str(model.get("kind", "multiGarch"))
        if kind.lower() == "multigarch":
            dependence.setdefault("kind", "dcc")
        elif kind.lower() in {"deltanormal", "univariategarch"}:
            dependence.update({"kind": "none", "dynamic": "none"})
        supplied["schemaVersion"] = 2
    return supplied


def _merge_known(base: dict[str, Any], supplied: dict[str, Any], path: str) -> dict[str, Any]:
    result = deepcopy(base)
    for name, value in supplied.items():
        child = f"{path}.{name}"
        if name not in base:
            raise DissError("diss:config:UnknownField", f"Unknown configuration field: {child}.")
        if isinstance(base[name], dict) and isinstance(value, dict) and base[name]:
            result[name] = _merge_known(base[name], value, child)
        else:
            result[name] = deepcopy(value)
    return result


def _choice(value: Any, candidates: tuple[str, ...], name: str) -> str:
    require(isinstance(value, str), "diss:config:InvalidText", f"{name} must be text.")
    matches = {candidate.casefold(): candidate for candidate in candidates}
    key = value.strip().casefold()
    require(key in matches, "diss:config:InvalidChoice", f"{name} must be one of {candidates}.")
    return matches[key]


def _positive_integer(value: Any, name: str, *, allow_zero: bool = False) -> int:
    require(
        isinstance(value, (int, np.integer)) and (value >= 0 if allow_zero else value > 0),
        "diss:config:InvalidInteger",
        f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer.",
    )
    return int(value)


def _series_count(data: Any) -> tuple[int, int]:
    values = np.asarray(data)
    require(values.ndim == 2 and values.size > 0, "diss:data:InvalidShape", "Data must be 2-D.")
    return int(values.shape[0]), int(values.shape[1])


def validate_config(config: dict[str, Any] | None, data: Any) -> dict[str, Any]:
    """Merge defaults, normalize values, and validate a pipeline configuration."""

    result = _merge_known(defaults(), migrate(config), "config")
    observation_count, series_count = _series_count(data)

    result["execution"]["mode"] = _choice(
        result["execution"]["mode"], ("full", "backtest"), "config.execution.mode"
    )
    result["model"]["kind"] = _choice(
        result["model"]["kind"],
        (
            *SUPPORTED_MODELS,
            "vineCopula",
            "multiMixCopula",
            "dVineMix",
            "histSimGaussian",
            "histSimCopula",
        ),
        "config.model.kind",
    )
    kind = result["model"]["kind"]
    require(
        kind in SUPPORTED_MODELS,
        "diss:config:UnsupportedModelAdapter",
        f"Model {kind} has no validated Python adapter.",
    )
    if kind in {"deltaNormal", "univariateGarch"}:
        result["dependence"].update({"kind": "none", "dynamic": "none"})

    dependence = result["dependence"]
    dependence["kind"] = _choice(
        dependence["kind"], ("none", "dcc", "copula", "vine"), "dependence.kind"
    )
    dependence["dynamic"] = _choice(
        dependence["dynamic"], ("none", "DCC", "ADCC", "GDCC", "AGDCC"), "dependence.dynamic"
    )
    dependence["copulaType"] = _choice(
        dependence["copulaType"], ("gaussian", "t"), "dependence.copulaType"
    )
    dependence["tails"] = _choice(
        dependence["tails"], ("none", "pareto", "empirical"), "dependence.tails"
    )
    for field in ("archOrder", "garchOrder"):
        dependence[field] = _positive_integer(dependence[field], f"dependence.{field}")
    dependence["asymmetricOrder"] = _positive_integer(
        dependence["asymmetricOrder"], "dependence.asymmetricOrder", allow_zero=True
    )

    marginal = result["marginal"]
    marginal["engine"] = _choice(marginal["engine"], ("scipy",), "marginal.engine")
    marginal["selectionPolicy"] = _choice(
        marginal["selectionPolicy"],
        ("initialWindow", "eachWindow", "fixed"),
        "marginal.selectionPolicy",
    )
    marginal["distribution"] = _choice(
        marginal["distribution"], ("Gaussian", "t"), "marginal.distribution"
    )
    marginal["maxARLag"] = _positive_integer(
        marginal["maxARLag"], "marginal.maxARLag", allow_zero=True
    )
    families = list(
        dict.fromkeys(str(item).strip().lower() for item in marginal["candidateFamilies"])
    )
    require(
        bool(families),
        "diss:config:MissingMarginalFamilies",
        "At least one marginal family is required.",
    )
    require(
        set(families) <= {"garch", "egarch", "gjr"},
        "diss:config:UnsupportedMarginalFamily",
        "Supported marginal families are garch, egarch, and gjr.",
    )
    marginal["candidateFamilies"] = families
    for field in ("archOrder", "garchOrder"):
        values = np.atleast_1d(marginal[field])
        require(
            len(values) in {1, series_count}
            and np.all(values >= 1)
            and np.all(values == np.floor(values)),
            "diss:config:InvalidOrder",
            f"marginal.{field} must be positive and scalar or per-series.",
        )
        marginal[field] = int(values[0]) if len(values) == 1 else [int(value) for value in values]
    if marginal["selectionPolicy"] == "fixed":
        require(
            len(marginal["fixedSpecifications"]) == series_count,
            "diss:config:InvalidFixedSpecifications",
            "Fixed selection requires one specification per series.",
        )

    backtest = result["backtest"]
    backtest["windowType"] = _choice(
        backtest["windowType"], ("expanding", "rolling"), "backtest.windowType"
    )
    for field in ("forecastHorizon", "stepSize", "marginalRefitEvery"):
        backtest[field] = _positive_integer(backtest[field], f"backtest.{field}")
    result["simulation"]["numPaths"] = _positive_integer(
        result["simulation"]["numPaths"], "simulation.numPaths"
    )
    result["simulation"]["seed"] = _positive_integer(
        result["simulation"]["seed"], "simulation.seed", allow_zero=True
    )
    probability = float(result["risk"]["probability"])
    require(
        0.0 < probability < 1.0,
        "diss:config:InvalidProbability",
        "Risk probability must lie in (0, 1).",
    )
    result["risk"]["probability"] = probability
    weights = result["risk"]["portfolioWeights"]
    if weights is None or np.size(weights) == 0:
        weights_array = np.full(series_count, 1.0 / series_count)
    else:
        weights_array = np.asarray(weights, dtype=float).reshape(-1)
        require(
            len(weights_array) == series_count,
            "diss:config:WeightDimension",
            "Weights must match series count.",
        )
        require(
            np.isclose(weights_array.sum(), 1.0, atol=1e-10),
            "diss:config:InvalidWeights",
            "Weights must sum to one.",
        )
    result["risk"]["portfolioWeights"] = weights_array
    result["risk"]["returnAggregation"] = _choice(
        result["risk"]["returnAggregation"], ("linear",), "risk.returnAggregation"
    )
    result["data"]["missingAction"] = _choice(
        result["data"]["missingAction"], ("error", "omitRows"), "data.missingAction"
    )

    require(
        series_count >= 2 or kind not in {"multiGarch", "multiCopula"},
        "diss:config:TooFewSeries",
        f"{kind} requires at least two series.",
    )
    require(
        series_count == 1 or kind != "univariateGarch",
        "diss:config:UnivariateSeriesCount",
        "Univariate GARCH requires exactly one series.",
    )
    if kind == "multiGarch":
        require(
            dependence["kind"] == "dcc" and dependence["dynamic"] != "none",
            "diss:config:MissingDependenceDynamics",
            "Multi-GARCH requires DCC dependence.",
        )
        require(
            dependence["archOrder"] == dependence["garchOrder"] == 1,
            "diss:config:UnsupportedDccOrder",
            "Recursive Multi-GARCH supports DCC(1,1).",
        )
        require(
            marginal["distribution"] == "Gaussian",
            "diss:config:UnsupportedMultiGarchDistribution",
            "Analytic Multi-GARCH currently requires Gaussian margins.",
        )
    if kind == "multiCopula":
        require(
            dependence["kind"] == "copula",
            "diss:config:MissingCopulaDependence",
            "Multi-copula requires copula dependence.",
        )
        require(
            dependence["copulaType"] == "gaussian" and dependence["dynamic"] == "none",
            "diss:config:UnsupportedCopulaSpecification",
            "Only a static Gaussian copula is implemented.",
        )

    if result["execution"]["mode"] == "backtest":
        initial = backtest["initialWindow"]
        require(
            initial is not None,
            "diss:config:MissingInitialWindow",
            "Backtests require an explicit initialWindow.",
        )
        backtest["initialWindow"] = _positive_integer(initial, "backtest.initialWindow")
        require(
            backtest["initialWindow"] < observation_count,
            "diss:config:InvalidInitialWindow",
            "initialWindow must be smaller than the sample.",
        )
        if backtest["windowType"] == "rolling":
            rolling = backtest["rollingWindow"] or backtest["initialWindow"]
            backtest["rollingWindow"] = _positive_integer(rolling, "backtest.rollingWindow")
            require(
                backtest["rollingWindow"] <= backtest["initialWindow"],
                "diss:config:InvalidRollingWindow",
                "rollingWindow cannot exceed initialWindow.",
            )
    return result
