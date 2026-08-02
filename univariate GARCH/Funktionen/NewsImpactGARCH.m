function out = NewsImpactGARCH(startval,dist,endval,GARCHOutput,plotfigure)

% News Impact curves for univariate GARCH models (GARCH, TGARCH, GJRGARCH,
%    EGARCH); uses simulated innovations to compare different GARCH models
% 
% USAGE:
%       out = NewsImpactGARCH(startval,dist,endval,GARCHOutput)
% 
% INPUTS:
%         startval:    number where simulated Innovations should start
%             dist:    distance bewteen simulated Innovations
%           endval:    last value of simulated Innovations
%       plotfigure:    if 'on' plot news impact curves
% 
% OUTPUTS: 
%         t x k array of simulated news impact curves
%        
% References: Zivot, E., Wang, J.(2006), Modeling Financial Timer Series
% with S-PLUS"
% 
% Author: Martin Grziska, 05/15/2010
k = size(GARCHOutput,1);
params = cell(k,1);
A = cell(k,1);
longVariance = cell(k,1);
out = cell(k,1);
epsilon = startval:dist:endval;
% formulas, p.245-246
for i=1:k
    params{i} = GARCHOutput{i}.ParamsGARCH;
    if strcmp(GARCHOutput{i}.GARCH,'GARCH')
        longVariance{i} = params{i}(1)./(1-params{i}(2)-params{i}(3));
        A{i} = params{i}(1) + params{i}(2)*longVariance{i};
        out{i} = A{i} + params{i}(2)*epsilon.^2;
    elseif strcmp(GARCHOutput{i}.GARCH,'TGARCH')
        longVariance{i} = params{i}(1)./(1-params{i}(2)+(params{i}(3)/2)-params{i}(4));
        S = epsilon<0;
        A{i} = params{i}(1) + params{i}(4)*longVariance{i};
        out{i} = A{i} + (params{i}(2)+params{i}(3)*S)'.*epsilon';
    elseif strcmp(GARCHOutput{i}.GARCH,'GJRGARCH')
        longVariance{i} = params{i}(1)./(1-params{i}(2)+(params{i}(3)/2)-params{i}(4));
        S = epsilon<0;
        A{i} = params{i}(1) + params{i}(4)*longVariance{i};
        out{i} = A{i} + (params{i}(2)+params{i}(3)*S)'.*epsilon'.^2;
    elseif strcmp(GARCHOutput{i}.GARCH,'EGARCH')
        if strcmp(GARCHOutput{i}.dist,'Gauss')
            longVariance{i} = exp((params{i}(1) + params{i}(3)*sqrt(2/pi))/(1-params{i}(4)));
        elseif strcmp(GARCHOutput{i}.dist,'STUDENTST') || strcmp(GARCHOutput{i}.dist,'GED') || strcmp(GARCHOutput{i}.dist,'SKEWT')
            termGamma = 2*sqrt(params{i}(5)-2)*gamma((params{i}(5)+1)/2)/((params{i}(5)-1)*gamma(params{i}(5)/2)*sqrt(pi));
            longVariance{i} = exp((params{i}(1)+params{i}(3)*termGamma)/(1-params{i}(4)));
        end
        A{i} = sqrt(longVariance{i})^(2*params{i}(4))*exp(params{i}(1));
            out{i} = A{i}*exp(params{i}(3)*(abs(epsilon)+params{i}(2))/sqrt(longVariance{i}));
    end
end

if strcmp(plotfigure,'on')
    figure;
    for i=1:k
        plot(out{i})
        hold on
    end
end