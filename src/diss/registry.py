"""Model-adapter registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .errors import DissError
from .models import delta_normal, multi_copula, multi_garch, univariate_garch

FitCallable = Callable[..., tuple[Any, dict[str, Any]]]
ForecastCallable = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class ModelAdapter:
    name: str
    fit: FitCallable
    forecast: ForecastCallable


_ADAPTERS = {
    "deltanormal": ModelAdapter("deltaNormal", delta_normal.fit, delta_normal.forecast),
    "univariategarch": ModelAdapter(
        "univariateGarch", univariate_garch.fit, univariate_garch.forecast
    ),
    "multigarch": ModelAdapter("multiGarch", multi_garch.fit, multi_garch.forecast),
    "multicopula": ModelAdapter("multiCopula", multi_copula.fit, multi_copula.forecast),
}


def available_models() -> tuple[str, ...]:
    return tuple(adapter.name for adapter in _ADAPTERS.values())


def resolve_model(name: str) -> ModelAdapter:
    try:
        return _ADAPTERS[name.casefold()]
    except KeyError as exception:
        raise DissError(
            "diss:registry:UnknownModel", f"Unknown model adapter {name}."
        ) from exception
