% Diss package
%
% Pipeline preparation
%   diss.prepare                 - Validate data/config and create a run plan.
%   diss.run                     - Execute an implemented model pipeline.
%   diss.fitWindow               - Fit a model through the common interface.
%   diss.forecastWindow          - Forecast through the common interface.
%
% Configuration
%   diss.config.defaults         - Return a reproducible default config.
%   diss.config.validate         - Normalize and validate a config.
%   diss.config.fromLegacySpec   - Translate the legacy Spec struct.
%
% Backtesting
%   diss.backtest.createSchedule - Create the shared backtest schedule.
%
% Dependence models
%   diss.dependence.fitDcc              - Fit DCC-family parameters.
%   diss.dependence.forecastCorrelation - Forecast the next correlation.
%   diss.dependence.parameterMatrices   - Expand DCC parameters.
%
% Implemented model adapters
%   diss.models.deltaNormal.fit      - Fit an unconditional Gaussian model.
%   diss.models.multiGarch.fit       - Fit AR-GARCH plus DCC dependence.
