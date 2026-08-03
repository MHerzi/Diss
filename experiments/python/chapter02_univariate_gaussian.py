"""Chapter 2: univariate Gaussian AR-GARCH configuration."""

from diss import defaults


def configuration() -> dict:
    config = defaults()
    config["model"]["kind"] = "univariateGarch"
    config["marginal"]["distribution"] = "Gaussian"
    return config
