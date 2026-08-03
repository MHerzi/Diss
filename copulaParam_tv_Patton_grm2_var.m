function [copulaParameterPath, generalizedDistance] = ...
    copulaParam_tv_Patton_grm2_var(family, processParameters, ~, ...
    data, operation, horizon)
%COPULAPARAM_TV_PATTON_GRM2_VAR Dynamic Archimedean copula parameter.
%   The latent parameter follows the Patton-style ARMA(1,1) recursion
%
%       psi(t) = K + AR*psi(t-1) + MA*GD(t).
%
%   GD is the mean absolute distance from the component-wise median over a
%   ten-observation window. The former implementation obtained this value
%   by repeatedly running one-cluster city-block k-means. For one cluster,
%   the median calculation below is mathematically equivalent and avoids
%   the clustering setup and temporary output arrays.

    movingAverageLags = 10;
    family = validatestring(family, ...
        {'clayton', 'gumbel', 'rotclayton'}, mfilename, 'family');
    operation = validatestring(operation, ...
        {'kalibrieren', 'vorhersage'}, mfilename, 'operation');
    if nargin < 6 || isempty(horizon)
        if strcmp(operation, 'kalibrieren')
            horizon = 0;
        else
            error('Diss:Copula:MissingForecastHorizon', ...
                'A forecast horizon is required for operation vorhersage.');
        end
    end
    validateattributes(processParameters, {'double'}, ...
        {'vector', 'real', 'finite', 'numel', 3}, ...
        mfilename, 'processParameters');
    validateattributes(data, {'double'}, ...
        {'2d', 'real', 'finite', 'nonempty'}, mfilename, 'data');
    validateattributes(horizon, {'double'}, ...
        {'scalar', 'integer', 'nonnegative', 'finite'}, ...
        mfilename, 'horizon');

    [observationCount, dimension] = size(data);
    if observationCount < movingAverageLags
        error('Diss:Copula:InsufficientPattonHistory', ...
            'At least ten observations are required for the Patton path.');
    end
    if dimension < 2 || dimension > 14
        error('Diss:Copula:InvalidDimension', ...
            'Dynamic copulas require between 2 and 14 series.');
    end
    if any(data(:) < 0 | data(:) > 1)
        error('Diss:Copula:DataOutsideUnitInterval', ...
            'Copula observations must lie in the closed unit interval.');
    end
    if strcmp(operation, 'kalibrieren')
        horizon = 0;
    end

    pathLength = observationCount + horizon;
    latentParameter = zeros(pathLength, 1);
    absoluteDistance = zeros(pathLength, 1);
    generalizedDistance = zeros(pathLength, 1);
    latentParameter(1) = processParameters(1);

    constant = processParameters(1);
    autoregressive = processParameters(2);
    movingAverage = processParameters(3);

    crossSectionalMedian = median(data, 2);
    crossSectionalDistance = sum( ...
        abs(data - crossSectionalMedian), 2);
    trailingDistance = movsum(crossSectionalDistance, ...
        [movingAverageLags - 1, 0]);
    initialDistance = trailingDistance(movingAverageLags);
    absoluteDistance(1:movingAverageLags) = initialDistance;

    for observation = 2:pathLength
        if observation <= movingAverageLags
            currentDistance = initialDistance;
        elseif observation <= observationCount + 1
            currentDistance = trailingDistance(observation - 1);
        else
            currentDistance = mean(absoluteDistance( ...
                observation-movingAverageLags:observation-1));
        end
        absoluteDistance(observation) = currentDistance;
        generalizedDistance(observation) = ...
            currentDistance / movingAverageLags;
        latentParameter(observation) = constant + ...
            autoregressive * latentParameter(observation - 1) + ...
            movingAverage * generalizedDistance(observation);
    end

    switch family
        case {'clayton', 'rotclayton'}
            copulaParameterPath = exp(latentParameter) + 1e-6;
        case 'gumbel'
            copulaParameterPath = exp(latentParameter) + 1 + 1e-6;
    end
    copulaParameterPath = min(copulaParameterPath, 20);

    if strcmp(operation, 'vorhersage')
        copulaParameterPath = copulaParameterPath( ...
            observationCount + (1:horizon));
    end
end
