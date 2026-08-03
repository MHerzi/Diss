function [model, state] = fitWindow(returns, config, state, context)
%FITWINDOW Fit one pipeline model using a standardized interface.

arguments
    returns double {mustBeReal, mustBeFinite, mustBeNonempty}
    config (1, 1) struct
    state (1, 1) struct = struct()
    context (1, 1) struct = struct('RefitMarginal', true)
end

switch config.model.kind
    case "deltaNormal"
        [model, state] = diss.models.deltaNormal.fit( ...
            returns, config, state);
    case "multiGarch"
        [model, state] = diss.models.multiGarch.fit( ...
            returns, config, state, context);
    otherwise
        error('diss:run:ModelNotImplemented', ...
            ['The package pipeline does not yet implement model %s. ', ...
            'The legacy MainFile remains unchanged.'], config.model.kind);
end

end
