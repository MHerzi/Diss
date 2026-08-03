function [model, state] = fit(returns, config, state, context)
%FIT Fit one AR-GARCH marginal model without a dependence layer.

arguments
    returns double {mustBeReal, mustBeFinite, mustBeNonempty}
    config (1, 1) struct
    state (1, 1) struct
    context (1, 1) struct = struct()
end

if size(returns, 2) ~= 1
    error('diss:univariateGarch:SeriesCount', ...
        'Univariate GARCH requires one return series.');
end
if isfield(context, 'RefitMarginal')
    refitMarginal = logical(context.RefitMarginal);
else
    refitMarginal = true;
end
[marginal, state] = diss.marginal.fit( ...
    returns, config, state, refitMarginal);

model.Type = "univariateGarch";
model.TrainingObservationCount = size(returns, 1);
model.TrainingReturns = returns;
model.Marginal = marginal;
model.PortfolioWeights = 1;

end
