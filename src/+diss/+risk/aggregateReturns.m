function portfolioReturns = aggregateReturns(returns, weights, method)
%AGGREGATERETURNS Aggregate asset returns using one explicit convention.

arguments
    returns double {mustBeReal, mustBeFinite}
    weights (1, :) double {mustBeReal, mustBeFinite}
    method (1, 1) string
end

if size(returns, 2) ~= numel(weights)
    error('diss:risk:WeightDimensionMismatch', ...
        'The number of portfolio weights must match the return series.');
end

switch method
    case "linear"
        portfolioReturns = returns * weights(:);
    otherwise
        error('diss:risk:UnsupportedAggregation', ...
            'Unsupported return aggregation: %s.', method);
end

end
