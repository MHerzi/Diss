function config = dccGaussian()
%DCCGAUSSIAN DCC Multi-GARCH experiment with Gaussian marginals.

config = diss.config.defaults();
config.model.kind = "multiGarch";
config.dependence.kind = "dcc";
config.dependence.dynamic = "DCC";
config.marginal.candidateFamilies = ["garch", "egarch", "gjr"];
config.marginal.distribution = "Gaussian";

end
