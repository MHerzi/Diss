function varargout = copulapdfmultivariat_grm(family, data, varargin)
%COPULAPDFMULTIVARIAT_GRM Multivariate copula probability density.
%   DENSITY = COPULAPDFMULTIVARIAT_GRM(FAMILY, U, PARAMETER) evaluates a
%   Gaussian, t, Clayton, rotated Clayton, Gumbel, or Frank copula. U must
%   contain observations in rows and variables in columns.
%
%   Gaussian: PARAMETER is a correlation matrix.
%   t:         the inputs are correlation matrix and degrees of freedom.
%   Others:    PARAMETER is a scalar or one value per observation.
%
%   This implementation is entirely numeric. It replaces the previous
%   run-time symbolic differentiation and avoids explicit inverses.

    if nargout > 2
        error('Diss:Copula:TooManyOutputs', ...
            'At most density and the dependence parameter are returned.');
    end
    family = validatestring(family, ...
        {'gaussian', 't', 'clayton', 'frank', 'gumbel', 'rotclayton'}, ...
        mfilename, 'family');
    validateattributes(data, {'double'}, ...
        {'2d', 'real', 'finite', 'nonempty'}, mfilename, 'data');
    if any(data(:) <= 0 | data(:) >= 1)
        error('Diss:Copula:DataOutsideOpenUnitInterval', ...
            'Copula observations must be strictly between zero and one.');
    end

    [observationCount, dimension] = size(data);
    if dimension < 2 || dimension > 14
        error('Diss:Copula:InvalidDimension', ...
            'Copula dimension must be between 2 and 14.');
    end

    switch family
        case 'gaussian'
            requireParameterCount(varargin, 1, family);
            correlationMatrix = validateCorrelationMatrix( ...
                varargin{1}, dimension, family);
            density = gaussianCopulaDensity(data, correlationMatrix);
            dependenceParameter = correlationMatrix;

        case 't'
            requireParameterCount(varargin, 2, family);
            correlationMatrix = validateCorrelationMatrix( ...
                varargin{1}, dimension, family);
            degreesOfFreedom = varargin{2};
            validateattributes(degreesOfFreedom, {'double'}, ...
                {'scalar', 'real', 'finite', 'positive'}, ...
                mfilename, 'degreesOfFreedom');
            density = tCopulaDensity( ...
                data, correlationMatrix, degreesOfFreedom);
            dependenceParameter = correlationMatrix;

        case {'clayton', 'rotclayton'}
            requireParameterCount(varargin, 1, family);
            theta = expandObservationParameter( ...
                varargin{1}, observationCount, 'theta');
            theta = max(theta, 1e-3);
            if strcmp(family, 'rotclayton')
                data = 1 - data;
            end
            density = claytonCopulaDensity(data, theta);
            dependenceParameter = theta;

        case 'gumbel'
            requireParameterCount(varargin, 1, family);
            theta = expandObservationParameter( ...
                varargin{1}, observationCount, 'theta');
            if any(theta < 1)
                error('Diss:Copula:InvalidGumbelParameter', ...
                    'Gumbel parameters must be greater than or equal to one.');
            end
            density = gumbelCopulaDensity(data, theta);
            dependenceParameter = theta;

        case 'frank'
            requireParameterCount(varargin, 1, family);
            theta = expandObservationParameter( ...
                varargin{1}, observationCount, 'theta');
            if dimension > 2 && any(theta <= 0)
                error('Diss:Copula:InvalidFrankParameter', ...
                    ['Frank parameters must be positive for dimensions ', ...
                     'greater than two.']);
            end
            density = frankCopulaDensity(data, theta);
            dependenceParameter = theta;
    end

    varargout{1} = density;
    if nargout >= 2
        varargout{2} = dependenceParameter;
    end
end

function density = gaussianCopulaDensity(data, correlationMatrix)
    transformedData = norminv(data);
    lowerFactor = chol(correlationMatrix, 'lower');
    correlatedQuadraticForm = sum( ...
        (lowerFactor \ transformedData').^2, 1)';
    independentQuadraticForm = sum(transformedData.^2, 2);
    logDensity = -0.5 * ...
        (correlatedQuadraticForm - independentQuadraticForm) - ...
        sum(log(diag(lowerFactor)));
    density = exp(logDensity);
end

function density = tCopulaDensity(data, correlationMatrix, degreesOfFreedom)
    [~, dimension] = size(data);
    transformedData = tinv(data, degreesOfFreedom);
    lowerFactor = chol(correlationMatrix, 'lower');
    quadraticForm = sum((lowerFactor \ transformedData').^2, 1)';

    logNormalizingConstant = gammaln((degreesOfFreedom + dimension) / 2) + ...
        (dimension - 1) * gammaln(degreesOfFreedom / 2) - ...
        dimension * gammaln((degreesOfFreedom + 1) / 2) - ...
        sum(log(diag(lowerFactor)));
    logDensity = logNormalizingConstant - ...
        (degreesOfFreedom + dimension) / 2 .* ...
        log1p(quadraticForm ./ degreesOfFreedom) + ...
        (degreesOfFreedom + 1) / 2 .* ...
        sum(log1p(transformedData.^2 ./ degreesOfFreedom), 2);
    density = exp(logDensity);
end

function density = claytonCopulaDensity(data, theta)
    dimension = size(data, 2);
    coefficient = prod(1 + theta .* (1:dimension-1), 2);
    generatorSum = sum(data .^ (-theta), 2) - dimension + 1;
    density = coefficient .* prod(data .^ (-theta - 1), 2) .* ...
        generatorSum .^ (-dimension - 1 ./ theta);
end

function density = gumbelCopulaDensity(data, theta)
    dimension = size(data, 2);
    negativeLogData = -log(data);
    alpha = 1 ./ theta;
    generatorSum = sum(negativeLogData .^ theta, 2);

    derivativeCoefficients = zeros(numel(theta), dimension + 1);
    derivativeCoefficients(:, 1) = 1;
    for derivativeOrder = 0:dimension-1
        nextCoefficients = zeros(size(derivativeCoefficients));
        for powerIndex = 0:derivativeOrder
            currentCoefficient = ...
                derivativeCoefficients(:, powerIndex + 1);
            nextCoefficients(:, powerIndex + 1) = ...
                nextCoefficients(:, powerIndex + 1) + ...
                currentCoefficient .* ...
                (powerIndex .* alpha - derivativeOrder);
            nextCoefficients(:, powerIndex + 2) = ...
                nextCoefficients(:, powerIndex + 2) - ...
                alpha .* currentCoefficient;
        end
        derivativeCoefficients = nextCoefficients;
    end

    derivativePolynomial = zeros(size(theta));
    for powerIndex = 0:dimension
        derivativePolynomial = derivativePolynomial + ...
            derivativeCoefficients(:, powerIndex + 1) .* ...
            generatorSum .^ (powerIndex .* alpha - dimension);
    end
    positiveGeneratorDerivative = (-1)^dimension .* ...
        exp(-generatorSum .^ alpha) .* derivativePolynomial;
    if any(positiveGeneratorDerivative <= 0 | ...
            ~isfinite(positiveGeneratorDerivative))
        error('Diss:Copula:GumbelNumericalFailure', ...
            'The Gumbel generator derivative is not positive and finite.');
    end

    logMarginalDerivativeProduct = dimension .* log(theta) + ...
        (theta - 1) .* sum(log(negativeLogData), 2) - ...
        sum(log(data), 2);
    density = exp(log(positiveGeneratorDerivative) + ...
        logMarginalDerivativeProduct);
end

function density = frankCopulaDensity(data, theta)
    [observationCount, dimension] = size(data);
    density = ones(observationCount, 1);
    nonIndependent = abs(theta) >= 1e-7;
    if ~any(nonIndependent)
        return
    end

    activeTheta = theta(nonIndependent);
    activeData = data(nonIndependent, :);
    generatorArguments = -log( ...
        expm1(-activeTheta .* activeData) ./ expm1(-activeTheta));
    generatorSum = sum(generatorArguments, 2);
    q = -expm1(-activeTheta) .* exp(-generatorSum);

    polynomialCoefficients = eulerianPolynomial(dimension - 1);
    polynomial = zeros(size(q));
    for powerIndex = 0:numel(polynomialCoefficients)-1
        polynomial = polynomial + ...
            polynomialCoefficients(powerIndex + 1) .* q.^powerIndex;
    end
    polylogarithm = q .* polynomial ./ (1 - q).^dimension;
    positiveGeneratorDerivative = polylogarithm ./ activeTheta;
    marginalDerivativeMagnitudes = activeTheta ./ ...
        expm1(activeTheta .* activeData);
    if any(positiveGeneratorDerivative <= 0) || ...
            any(marginalDerivativeMagnitudes(:) <= 0)
        error('Diss:Copula:FrankNumericalFailure', ...
            'The Frank density could not be evaluated as a positive value.');
    end
    logDensity = log(positiveGeneratorDerivative) + ...
        sum(log(marginalDerivativeMagnitudes), 2);
    density(nonIndependent) = exp(logDensity);
end

function coefficients = eulerianPolynomial(order)
    coefficients = 1;
    for currentOrder = 2:order
        previous = coefficients;
        coefficients = zeros(1, currentOrder);
        for index = 0:currentOrder-1
            if index <= currentOrder - 2
                coefficients(index + 1) = coefficients(index + 1) + ...
                    (index + 1) * previous(index + 1);
            end
            if index >= 1
                coefficients(index + 1) = coefficients(index + 1) + ...
                    (currentOrder - index) * previous(index);
            end
        end
    end
end

function parameter = expandObservationParameter( ...
    parameter, observationCount, argumentName)
    validateattributes(parameter, {'double'}, ...
        {'vector', 'real', 'finite', 'nonempty'}, mfilename, argumentName);
    if isscalar(parameter)
        parameter = repmat(parameter, observationCount, 1);
    elseif numel(parameter) == observationCount
        parameter = parameter(:);
    else
        error('Diss:Copula:InvalidParameterLength', ...
            '%s must be scalar or contain one value per observation.', ...
            argumentName);
    end
end

function correlationMatrix = validateCorrelationMatrix( ...
    correlationMatrix, dimension, family)
    validateattributes(correlationMatrix, {'double'}, ...
        {'2d', 'real', 'finite', 'square'}, mfilename, ...
        'correlationMatrix');
    if ~isequal(size(correlationMatrix), [dimension, dimension])
        error('Diss:Copula:InvalidCorrelationSize', ...
            '%s copula requires a dimension-by-dimension matrix.', family);
    end
    if norm(correlationMatrix - correlationMatrix', 'fro') > 1e-10 || ...
            any(abs(diag(correlationMatrix) - 1) > 1e-10)
        error('Diss:Copula:InvalidCorrelationMatrix', ...
            'The correlation matrix must be symmetric with unit diagonal.');
    end
    [~, cholStatus] = chol(correlationMatrix, 'lower');
    if cholStatus ~= 0
        error('Diss:Copula:NonPositiveDefiniteCorrelation', ...
            'The correlation matrix must be positive definite.');
    end
end

function requireParameterCount(parameters, expectedCount, family)
    if numel(parameters) ~= expectedCount
        error('Diss:Copula:InvalidParameterCount', ...
            '%s copula expects %d dependence parameter input(s).', ...
            family, expectedCount);
    end
end
