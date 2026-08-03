function config = univariateStudentT()
%UNIVARIATESTUDENTT AR-GARCH experiment with Student-t innovations.

config = study.chapter02.univariateGaussian();
config.marginal.distribution = "t";

end
