"""Chapter 2: univariate standardized Student-t AR-GARCH configuration."""

from diss import defaults


def configuration() -> dict:
    config = defaults()
    config["model"]["kind"] = "univariateGarch"
    config["marginal"]["distribution"] = "t"
    return config
