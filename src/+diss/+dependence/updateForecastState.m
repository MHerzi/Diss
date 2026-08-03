function state = updateForecastState(state, standardizedResidual)
%UPDATEFORECASTSTATE Advance a filtered DCC(1,1) state by one observation.

arguments
    state (1, 1) struct
    standardizedResidual (1, :) double {mustBeReal, mustBeFinite}
end

innovation = standardizedResidual * state.ArchMatrix;
negativeInnovation = min(standardizedResidual, 0) * ...
    state.AsymmetricMatrix;
currentQ = state.Intercept + innovation' * innovation + ...
    negativeInnovation' * negativeInnovation + ...
    state.GarchMatrix' * state.CurrentQ * state.GarchMatrix;
currentQ = (currentQ + currentQ') / 2;
diagonal = diag(currentQ);
if any(~isfinite(diagonal)) || any(diagonal <= 0)
    error('diss:dependence:InvalidForecastState', ...
        'The recursive DCC update produced an invalid covariance state.');
end
scale = sqrt(diagonal);
correlation = currentQ ./ (scale * scale');
correlation = (correlation + correlation') / 2;
correlation(1:size(correlation, 1) + 1:end) = 1;
[~, status] = chol(correlation, 'lower');
if status ~= 0
    error('diss:dependence:InvalidForecastCorrelation', ...
        'The recursive DCC update is not positive definite.');
end

state.CurrentQ = currentQ;
state.Correlation = correlation;

end
