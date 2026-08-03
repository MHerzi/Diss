function copula = fitGaussian(pseudoObservations)
%FITGAUSSIAN Estimate an unrestricted static Gaussian copula.

arguments
    pseudoObservations double {mustBeReal, mustBeFinite, mustBeNonempty}
end

if size(pseudoObservations, 2) < 2
    error('diss:copula:TooFewSeries', ...
        'A multivariate copula requires at least two series.');
end
if any(pseudoObservations(:) <= 0 | pseudoObservations(:) >= 1)
    error('diss:copula:InvalidPseudoObservations', ...
        'Pseudo-observations must lie strictly between zero and one.');
end

normalScores = diss.distributions.innovationQuantile( ...
    pseudoObservations, struct('Name', "Gaussian"));
correlation = corrcoef(normalScores);
correlation = stabilizeCorrelation(correlation);
logLikelihood = gaussianLogLikelihood(normalScores, correlation);

copula = struct( ...
    'Type', "gaussianCopula", ...
    'Correlation', correlation, ...
    'LogLikelihood', logLikelihood, ...
    'NegativeLogLikelihood', -logLikelihood, ...
    'ExitFlag', 1);

end

function correlation = stabilizeCorrelation(correlation)
correlation = (correlation + correlation') / 2;
correlation(1:size(correlation, 1) + 1:end) = 1;
[~, status] = chol(correlation, 'lower');
shrinkage = 1e-12;
while status ~= 0 && shrinkage <= 1e-2
    correlation = (1 - shrinkage) * correlation + ...
        shrinkage * eye(size(correlation));
    correlation(1:size(correlation, 1) + 1:end) = 1;
    [~, status] = chol(correlation, 'lower');
    shrinkage = shrinkage * 10;
end
if status ~= 0
    error('diss:copula:SingularCorrelation', ...
        'Gaussian copula correlation is not positive definite.');
end
end

function logLikelihood = gaussianLogLikelihood(scores, correlation)
factor = chol(correlation, 'lower');
solved = factor \ scores';
quadraticDifference = sum(solved .^ 2, 1)' - sum(scores .^ 2, 2);
logDeterminant = 2 * sum(log(diag(factor)));
logLikelihood = sum(-0.5 * ...
    (logDeterminant + quadraticDifference));
end
