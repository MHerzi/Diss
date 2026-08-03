function [meanForecast, meanSquareError, varianceForecast] = ...
    forecastMarginal(model, history, residuals, variances)
%FORECASTMARGINAL Call the Econometrics Toolbox ARIMA forecast method.

[meanForecast, meanSquareError, varianceForecast] = forecast( ...
    model, 1, history, 'E0', residuals, 'V0', variances);

end
