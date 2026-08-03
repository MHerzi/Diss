function probabilities = empiricalCDF(data, evaluationPoints)
%EMPIRICALCDF Empirical probability integral transform by column.
%   P = EMPIRICALCDF(DATA) returns ranks divided by T+1, preserving the
%   original convention that transformed observations are strictly below
%   one. P = EMPIRICALCDF(DATA, X) evaluates each column's empirical CDF at
%   the corresponding points in X.

    validateattributes(data, {'double'}, ...
        {'2d', 'real', 'finite', 'nonempty'}, mfilename, 'data');
    [observationCount, seriesCount] = size(data);

    if nargin < 2
        [~, order] = sort(data, 1);
        probabilities = zeros(size(data));
        ranks = (1:observationCount)' / (observationCount + 1);
        for series = 1:seriesCount
            probabilities(order(:, series), series) = ranks;
        end
        return
    end

    validateattributes(evaluationPoints, {'double'}, ...
        {'2d', 'real', 'finite', 'nonempty'}, mfilename, ...
        'evaluationPoints');
    if size(evaluationPoints, 2) == 1 && seriesCount > 1
        evaluationPoints = repmat(evaluationPoints, 1, seriesCount);
    elseif size(evaluationPoints, 2) ~= seriesCount
        error('Diss:EmpiricalCDF:InvalidEvaluationPointSize', ...
            ['evaluationPoints must have one column or one column per ', ...
             'data series.']);
    end

    queryCount = size(evaluationPoints, 1);
    probabilities = zeros(queryCount, seriesCount);
    for series = 1:seriesCount
        for query = 1:queryCount
            probabilities(query, series) = sum( ...
                data(:, series) <= evaluationPoints(query, series));
        end
    end
    probabilities = probabilities / observationCount;
end
