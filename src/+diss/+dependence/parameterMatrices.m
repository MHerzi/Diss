function [archMatrices, garchMatrices, asymmetricMatrices] = ...
    parameterMatrices(parameters, dynamicType, archOrder, ...
    garchOrder, seriesCount)
%PARAMETERMATRICES Convert DCC parameters into matrix sequences.

arguments
    parameters double {mustBeReal, mustBeFinite, mustBeNonnegative}
    dynamicType (1, 1) string
    archOrder (1, 1) double {mustBeInteger, mustBePositive}
    garchOrder (1, 1) double {mustBeInteger, mustBePositive}
    seriesCount (1, 1) double {mustBeInteger, mustBePositive}
end

parameters = parameters(:);
switch dynamicType
    case "DCC"
        requireParameterCount(parameters, archOrder + garchOrder);
        archValues = repmat(parameters(1:archOrder)', seriesCount, 1);
        garchValues = repmat(parameters(archOrder + ...
            (1:garchOrder))', seriesCount, 1);
        asymmetricValues = zeros(seriesCount, 0);
    case "ADCC"
        requireParameterCount(parameters, 2 * archOrder + garchOrder);
        archValues = repmat(parameters(1:archOrder)', seriesCount, 1);
        asymmetricValues = repmat(parameters(archOrder + ...
            (1:archOrder))', seriesCount, 1);
        garchValues = repmat(parameters(2 * archOrder + ...
            (1:garchOrder))', seriesCount, 1);
    case "GDCC"
        requireParameterCount(parameters, ...
            (archOrder + garchOrder) * seriesCount);
        archParameterCount = archOrder * seriesCount;
        archValues = reshape(parameters(1:archParameterCount), ...
            seriesCount, archOrder);
        garchValues = reshape(parameters(archParameterCount + 1:end), ...
            seriesCount, garchOrder);
        asymmetricValues = zeros(seriesCount, 0);
    case "AGDCC"
        requireParameterCount(parameters, ...
            (2 * archOrder + garchOrder) * seriesCount);
        archParameterCount = archOrder * seriesCount;
        archValues = reshape(parameters(1:archParameterCount), ...
            seriesCount, archOrder);
        asymmetricValues = reshape(parameters(archParameterCount + ...
            (1:archParameterCount)), seriesCount, archOrder);
        garchValues = reshape(parameters(2 * archParameterCount + 1:end), ...
            seriesCount, garchOrder);
    otherwise
        error('diss:dependence:UnsupportedDynamicType', ...
            'Unsupported dynamic dependence type: %s.', dynamicType);
end

archMatrices = diagonalMatrixSequence(archValues);
garchMatrices = diagonalMatrixSequence(garchValues);
asymmetricMatrices = diagonalMatrixSequence(asymmetricValues);

end

function matrices = diagonalMatrixSequence(values)
[seriesCount, order] = size(values);
matrices = zeros(seriesCount, seriesCount, order, 'like', values);
for lag = 1:order
    matrices(:, :, lag) = diag(values(:, lag));
end
end

function requireParameterCount(parameters, expectedCount)
if numel(parameters) ~= expectedCount
    error('diss:dependence:InvalidParameterCount', ...
        'Expected %d dependence parameters but received %d.', ...
        expectedCount, numel(parameters));
end
end
