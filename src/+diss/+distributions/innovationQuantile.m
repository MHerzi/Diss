function quantileValue = innovationQuantile(probability, distribution)
%INNOVATIONQUANTILE Quantile of a standardized innovation distribution.

arguments
    probability double {mustBeGreaterThan(probability, 0), ...
        mustBeLessThan(probability, 1)}
    distribution (1, 1) struct
end

name = lower(string(distribution.Name));
switch name
    case "gaussian"
        quantileValue = -sqrt(2) * erfcinv(2 * probability);
    case "t"
        degreesOfFreedom = distribution.DoF;
        if ~isfinite(degreesOfFreedom) || degreesOfFreedom <= 2
            error('diss:distribution:InvalidDegreesOfFreedom', ...
                'Student-t degrees of freedom must exceed two.');
        end
        quantileValue = standardizedStudentTQuantile( ...
            probability, degreesOfFreedom);
    otherwise
        error('diss:distribution:UnsupportedInnovation', ...
            'Unsupported innovation distribution: %s.', name);
end

end

function value = standardizedStudentTQuantile(probability, degreesOfFreedom)
value = zeros(size(probability));
nonmedian = probability ~= 0.5;
if ~any(nonmedian(:))
    return
end
selectedProbability = probability(nonmedian);
tailProbability = min(selectedProbability, 1 - selectedProbability);
betaQuantile = betaincinv(2 * tailProbability, ...
    degreesOfFreedom / 2, 0.5);
unstandardized = sqrt(degreesOfFreedom * ...
    (1 ./ betaQuantile - 1));
unstandardized(selectedProbability < 0.5) = ...
    -unstandardized(selectedProbability < 0.5);
value(nonmedian) = unstandardized * sqrt( ...
    (degreesOfFreedom - 2) / degreesOfFreedom);
end
