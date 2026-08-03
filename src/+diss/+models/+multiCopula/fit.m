function [model, state] = fit(returns, config, state, context)
%FIT Fit AR-GARCH marginals followed by a static Gaussian copula.

arguments
    returns double {mustBeReal, mustBeFinite, mustBeNonempty}
    config (1, 1) struct
    state (1, 1) struct
    context (1, 1) struct = struct()
end

if isfield(context, 'RefitMarginal')
    refitMarginals = logical(context.RefitMarginal);
else
    refitMarginals = true;
end
[marginal, state] = diss.marginal.fit( ...
    returns, config, state, refitMarginals);
standardizedResiduals = marginal.StandardizedResiduals;
pseudoObservations = zeros(size(standardizedResiduals));
for series = 1:size(standardizedResiduals, 2)
    distribution = marginal.Models{series}.Distribution;
    pseudoObservations(:, series) = ...
        diss.distributions.innovationCdf( ...
        standardizedResiduals(:, series), distribution);
end
sampleCount = size(pseudoObservations, 1);
boundary = 0.5 / (sampleCount + 1);
pseudoObservations = min(max(pseudoObservations, boundary), 1 - boundary);
copula = diss.dependence.copula.fitGaussian(pseudoObservations);

model.Type = "multiCopula";
model.TrainingObservationCount = size(returns, 1);
model.TrainingReturns = returns;
model.Marginal = marginal;
model.Dependence = copula;
model.PortfolioWeights = config.risk.portfolioWeights;
state.CopulaCorrelation = copula.Correlation;

end
