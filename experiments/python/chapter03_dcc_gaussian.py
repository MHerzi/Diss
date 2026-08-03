"""Chapter 3: Gaussian-margin DCC configuration."""

from diss import defaults


def configuration() -> dict:
    config = defaults()
    config["model"]["kind"] = "multiGarch"
    config["dependence"].update({"kind": "dcc", "dynamic": "DCC"})
    config["marginal"]["distribution"] = "Gaussian"
    return config
