function probability = innovationCdf(values, distribution)
%INNOVATIONCDF CDF of a standardized innovation distribution.

arguments
    values double {mustBeReal, mustBeFinite}
    distribution (1, 1) struct
end

name = lower(string(distribution.Name));
switch name
    case "gaussian"
        probability = 0.5 * erfc(-values / sqrt(2));
    case "t"
        degreesOfFreedom = distribution.DoF;
        if ~isfinite(degreesOfFreedom) || degreesOfFreedom <= 2
            error('diss:distribution:InvalidDegreesOfFreedom', ...
                'Student-t degrees of freedom must exceed two.');
        end
        rawValues = values / sqrt( ...
            (degreesOfFreedom - 2) / degreesOfFreedom);
        betaArgument = degreesOfFreedom ./ ...
            (degreesOfFreedom + rawValues .^ 2);
        lowerTail = 0.5 * betainc(betaArgument, ...
            degreesOfFreedom / 2, 0.5);
        probability = lowerTail;
        nonnegative = rawValues >= 0;
        probability(nonnegative) = 1 - lowerTail(nonnegative);
    otherwise
        error('diss:distribution:UnsupportedInnovation', ...
            'Unsupported innovation distribution: %s.', name);
end

end
