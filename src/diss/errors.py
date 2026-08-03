"""Domain-specific exceptions with stable identifiers."""

from __future__ import annotations


class DissError(ValueError):
    """Error raised for invalid model configuration or numerical state."""

    def __init__(self, identifier: str, message: str) -> None:
        super().__init__(message)
        self.identifier = identifier


def require(condition: bool, identifier: str, message: str) -> None:
    """Raise :class:`DissError` when *condition* is false."""

    if not condition:
        raise DissError(identifier, message)
