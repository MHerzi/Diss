function [meanForecast, meanSquareError, varianceForecast] = ...
    forecastOneStep(model, history, residuals, variances)
%FORECASTONESTEP Call the Econometrics Toolbox ARIMA forecast method.

[meanForecast, meanSquareError, varianceForecast] = forecast( ...
    model, 1, history, 'E0', residuals, 'V0', variances);

end
