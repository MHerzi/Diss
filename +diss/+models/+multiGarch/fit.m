function [model, state] = fit(returns, config, state, context)
%FIT Fit modern AR-GARCH marginals followed by a DCC-family model.

arguments
    returns double {mustBeReal, mustBeFinite, mustBeNonempty}
    config (1, 1) struct
    state (1, 1) struct
    context (1, 1) struct
end

if isfield(context, 'RefitMarginal')
    refitMarginals = logical(context.RefitMarginal);
else
    refitMarginals = true;
end

[marginal, state] = diss.models.multiGarch.fitMarginals( ...
    returns, config, state, refitMarginals);
[dependence, state] = diss.dependence.fitDcc( ...
    marginal.StandardizedResiduals, config, state);

model = struct();
model.Type = "multiGarch";
model.TrainingObservationCount = size(returns, 1);
model.TrainingReturns = returns;
model.Marginal = marginal;
model.Dependence = dependence;
model.PortfolioWeights = config.risk.portfolioWeights;

end
