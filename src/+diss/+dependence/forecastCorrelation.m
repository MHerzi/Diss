function correlation = forecastCorrelation(dependence, stdResiduals)
%FORECASTCORRELATION Produce the next DCC correlation without future data.

arguments
    dependence (1, 1) struct
    stdResiduals double {mustBeReal, mustBeFinite, mustBeNonempty}
end

seriesCount = size(stdResiduals, 2);
[archMatrices, garchMatrices, asymmetricMatrices] = ...
    diss.dependence.parameterMatrices(dependence.Parameters, ...
    dependence.Type, dependence.ArchOrder, dependence.GarchOrder, ...
    seriesCount);
correlationPath = diss.dependence.correlation.forecastPath( ...
    stdResiduals, ...
    dependence.QBar, archMatrices, garchMatrices, ...
    asymmetricMatrices, dependence.NegativeQBar, 1);
correlation = correlationPath(:, :, 1);

end
