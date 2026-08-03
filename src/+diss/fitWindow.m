function [model, state] = fitWindow(returns, config, state, context)
%FITWINDOW Fit one pipeline model using a standardized interface.

arguments
    returns double {mustBeReal, mustBeFinite, mustBeNonempty}
    config (1, 1) struct
    state (1, 1) struct = struct()
    context (1, 1) struct = struct('RefitMarginal', true)
end

adapter = diss.registry.resolveModel(config.model.kind);
[model, state] = adapter.Fit(returns, config, state, context);
model.AdapterName = adapter.Name;

end
