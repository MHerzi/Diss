% Diss package
%
% Pipeline preparation
%   diss.prepare                 - Validate data/config and create a run plan.
%   diss.run                     - Execute an implemented model pipeline.
%   diss.runExperiment           - Canonical dissertation experiment API.
%   diss.fitWindow               - Fit a model through the common interface.
%   diss.forecastWindow          - Forecast through the common interface.
%
% Configuration
%   diss.config.defaults         - Return a reproducible default config.
%   diss.config.validate         - Normalize and validate a config.
%   diss.config.fromLegacySpec   - Translate the legacy Spec struct.
%   diss.config.migrate          - Upgrade older configuration schemas.
%
% Backtesting
%   diss.backtest.createSchedule - Create the shared backtest schedule.
%
% Dependence models
%   diss.dependence.fitDcc              - Fit DCC-family parameters.
%   diss.dependence.forecastCorrelation - Forecast the next correlation.
%   diss.dependence.parameterMatrices   - Expand DCC parameters.
%   diss.dependence.copula.fitGaussian  - Fit a static Gaussian copula.
%
% Marginal models
%   diss.marginal.fit             - Select and fit AR-GARCH marginals.
%   diss.marginal.forecastPath    - Recursively forecast fitted marginals.
%
% Implemented model adapters
%   diss.models.deltaNormal.fit      - Fit an unconditional Gaussian model.
%   diss.models.univariateGarch.fit  - Fit a univariate AR-GARCH model.
%   diss.models.multiGarch.fit       - Fit AR-GARCH plus DCC dependence.
%   diss.models.multiCopula.fit      - Fit marginals plus Gaussian copula.
%
% Results
%   diss.evaluation.summarize     - Produce common forecast diagnostics.
%   diss.io.saveRun               - Save results and a run manifest.
%   diss.io.loadRun               - Load a saved result artifact.
