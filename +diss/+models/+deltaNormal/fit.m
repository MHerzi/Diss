function [model, state] = fit(returns, config, ~)
%FIT Estimate an unconditional multivariate Gaussian return model.

arguments
    returns double {mustBeReal, mustBeFinite, mustBeNonempty}
    config (1, 1) struct
    ~
end

if size(returns, 1) < 2
    error('diss:deltaNormal:TooFewObservations', ...
        'Delta-Normal estimation requires at least two observations.');
end

assetMean = mean(returns, 1);
assetCovariance = cov(returns, 0);
assetCovariance = (assetCovariance + assetCovariance') / 2;
weights = config.risk.portfolioWeights;
portfolioMean = assetMean * weights(:);
portfolioVariance = weights * assetCovariance * weights(:);

% Roundoff can produce a tiny negative value for a semidefinite matrix.
varianceTolerance = eps(max(1, norm(assetCovariance, 'fro')));
if portfolioVariance < -varianceTolerance
    error('diss:deltaNormal:NegativePortfolioVariance', ...
        'The estimated portfolio variance is negative.');
end
portfolioVariance = max(portfolioVariance, 0);

model = struct();
model.Type = "deltaNormal";
model.TrainingObservationCount = size(returns, 1);
model.AssetMean = assetMean;
model.AssetCovariance = assetCovariance;
model.PortfolioWeights = weights;
model.PortfolioMean = portfolioMean;
model.PortfolioVariance = portfolioVariance;
model.PortfolioStdDev = sqrt(portfolioVariance);

state = struct();

end
