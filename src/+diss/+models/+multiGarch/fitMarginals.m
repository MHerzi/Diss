function [marginal, state] = fitMarginals( ...
    returns, config, state, refitMarginals)
%FITMARGINALS Compatibility wrapper for diss.marginal.fit.

[marginal, state] = diss.marginal.fit( ...
    returns, config, state, refitMarginals);

end
