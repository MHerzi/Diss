function config = staticGaussianCopula()
%STATICGAUSSIANCOPULA Static Gaussian copula with AR-GARCH marginals.

config = diss.config.defaults();
config.model.kind = "multiCopula";
config.dependence.kind = "copula";
config.dependence.dynamic = "none";
config.dependence.copulaType = "gaussian";
config.marginal.candidateFamilies = ["garch", "egarch", "gjr"];

end
