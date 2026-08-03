function state = initializeForecastState(dependence, stdResiduals)
%INITIALIZEFORECASTSTATE Filter history once for recursive DCC forecasts.

arguments
    dependence (1, 1) struct
    stdResiduals double {mustBeReal, mustBeFinite, mustBeNonempty}
end

if dependence.ArchOrder ~= 1 || dependence.GarchOrder ~= 1
    error('diss:dependence:UnsupportedRecursiveOrder', ...
        'Recursive forecasting currently supports DCC(1,1) models.');
end

seriesCount = size(stdResiduals, 2);
[archMatrices, garchMatrices, asymmetricMatrices] = ...
    diss.dependence.parameterMatrices(dependence.Parameters, ...
    dependence.Type, dependence.ArchOrder, dependence.GarchOrder, ...
    seriesCount);
[correlation, covarianceState] = ...
    diss.dependence.correlation.forecastPath(stdResiduals, ...
    dependence.QBar, archMatrices, garchMatrices, ...
    asymmetricMatrices, dependence.NegativeQBar, 1);

intercept = dependence.QBar - ...
    archMatrices(:, :, 1)' * dependence.QBar * archMatrices(:, :, 1) - ...
    garchMatrices(:, :, 1)' * dependence.QBar * garchMatrices(:, :, 1);
if isempty(asymmetricMatrices)
    asymmetricMatrix = zeros(seriesCount);
else
    asymmetricMatrix = asymmetricMatrices(:, :, 1);
    intercept = intercept - asymmetricMatrix' * ...
        dependence.NegativeQBar * asymmetricMatrix;
end

state.Intercept = (intercept + intercept') / 2;
state.ArchMatrix = archMatrices(:, :, 1);
state.GarchMatrix = garchMatrices(:, :, 1);
state.AsymmetricMatrix = asymmetricMatrix;
state.CurrentQ = covarianceState(:, :, 1);
state.Correlation = correlation(:, :, 1);

end
