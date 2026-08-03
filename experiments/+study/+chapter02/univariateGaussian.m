function config = univariateGaussian()
%UNIVARIATEGAUSSIAN Baseline AR-GARCH experiment with Gaussian innovations.

config = diss.config.defaults();
config.model.kind = "univariateGarch";
config.marginal.candidateFamilies = ["garch", "egarch", "gjr"];
config.marginal.distribution = "Gaussian";

end
