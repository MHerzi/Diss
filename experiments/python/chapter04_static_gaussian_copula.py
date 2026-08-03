"""Chapter 4: static Gaussian copula configuration."""

from diss import defaults


def configuration() -> dict:
    config = defaults()
    config["model"]["kind"] = "multiCopula"
    config["dependence"].update({"kind": "copula", "dynamic": "none", "copulaType": "gaussian"})
    return config
