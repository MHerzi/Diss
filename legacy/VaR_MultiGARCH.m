function [Output,rnd_ret2] = VaR_MultiGARCH(MeanForecast,VarianceForecast,GARCHOutput,Rt,n,daten,PrtFig,~)

% Berechnung des VaR für AGDCC-Modelle (auch copula mit AGDCC-Struktur)
%
% USAGE: [Output] = VaR_MultiGARCH(MeanForecast,VarianceForecast,GARCHOutput,Rt,n,daten,start)
%
% INPUTS:
%          MeanForecast : t x k Zeitreihe mit conditional mean forecasts
%         VarianceForecast : t x k Zeitreihe mit varianz-forecasts
%           GARCHOutput : Structure-Variable der univariate GARCH-Modell
%                        (AR_MarignalModel.m)
%                    Rt : foregecastete Korrelationsmatrix
%                     n : # der simulierten Variablen für MC-VaR
%                PrtFig : string: 'on' or 'off'
%
%
% OUTPUTS:
%         Output:      : structure mit allen notwendigen Informationen
%
% Author: Martin Grziska, 04/28/2010



% %%%%%%%%%  Berechnung des VaR  %%%%%%%%%%
% Der VaR wird hier mit der Monte-Carlo Methode berechnet; in diesem Fall
% werden 5000 Werte generiert und je nach Konfidenzniveau dann der
% entsprechende Wert ausgewählt; dabei werden die Variablen unabhängig
% voneinadner erzeugt; dann wird die geschätzte Korrelatiuonsmatrix, bzw
% deren Vorhersage mit der Cholesky-Zerlegung zerlegt; die Cholesky-Matrix
% wird dann benutzt, um den unkorrelierten Variablen die entsprechende
% Abhängigkeitsstruktur zu geben. Aus den MC-generierten Variabeln mit der
% durch Cholesky erzeugten Abhängigkeitsstruktur wird dann ein Portfolio
% gebildet und der entsprechende VaR abgelesen. Dieser Wert wird dann
% letztendlich mit dem tatsächlich realisiertem Gewinn oder Verlsut an
% diesem Tag verglichen
% 1.
% generiere standardisierte Zufallsvariablen mit der Verteilung die oben
% zur Schätzung der univariaten GARCH-Modelle verwendet wurde
% 5000 i.i.d. (0,1)-Variablen mit Student T-Verteilung für den HFRXEMN und
% GED-Verteilung für den DJW1
% Formel: y_t+1 = mu + sigma_t+1*z
% z: standardisierte Zufallsvariable mit oben festgelegter Verteilung
% sigma_t: Standardabweichung aus dem GARCH-Modell
% mu: Mittelwert aus dem ARMA-Modell


% generiere Returns aus den Zufallsvaribalen (für den Zeitpunkt t) und
% wähle den jeweiligen VaR aus - je nach Konfidenzniveau
% n: Anzahl der MC-Simulaionen

[t,k] = size(MeanForecast);
validateattributes(MeanForecast, {'double'}, ...
    {'2d', 'real', 'finite', 'nonempty'}, mfilename, 'MeanForecast');
validateattributes(VarianceForecast, {'double'}, ...
    {'2d', 'real', 'finite', 'nonnegative', ...
     'size', size(MeanForecast)}, mfilename, 'VarianceForecast');
validateattributes(n, {'double'}, ...
    {'scalar', 'integer', 'positive', 'finite'}, mfilename, 'n');
if ~isequal(size(Rt), [k, k, t])
    error('Diss:VaR:InvalidCorrelationPathSize', ...
        'Rt must be a k-by-k-by-t correlation path.');
end
if size(daten, 1) < t || size(daten, 2) ~= k
    error('Diss:VaR:InvalidRealizedDataSize', ...
        'daten must contain at least t rows and exactly k series.');
end
if numel(GARCHOutput) ~= k
    error('Diss:VaR:InvalidGarchOutputSize', ...
        'GARCHOutput must contain one model per series.');
end
for series = 1:k
    if ~isstruct(GARCHOutput{series}) || ...
            ~isfield(GARCHOutput{series}, 'dist') || ...
            ~strcmpi(GARCHOutput{series}.dist, 'GAUSS')
        error('Diss:VaR:UnsupportedDistribution', ...
            ['VaR_MultiGARCH currently supports Gaussian marginal ', ...
             'innovations only.']);
    end
end

probabilities = [0.001, 0.01, 0.05, 0.10, 0.90, 0.95, 0.99, 0.999];
marginalQuantiles = zeros(t, k, numel(probabilities));
portfolioQuantiles = zeros(t, numel(probabilities));
if nargout >= 2
    rnd_ret2 = cell(1, t);
else
    rnd_ret2 = [];
end

for forecastIndex = 1:t
    simulatedReturns = simulateCorrelatedGaussianReturns( ...
        MeanForecast(forecastIndex, :), ...
        VarianceForecast(forecastIndex, :), ...
        Rt(:, :, forecastIndex), n);
    currentMarginalQuantiles = quantile( ...
        simulatedReturns, probabilities, 1);
    marginalQuantiles(forecastIndex, :, :) = ...
        reshape(currentMarginalQuantiles', 1, k, []);
    portfolioQuantiles(forecastIndex, :) = quantile( ...
        mean(simulatedReturns, 2), probabilities, 1);
    if nargout >= 2
        rnd_ret2{forecastIndex} = simulatedReturns;
    end
end

VaR9991 = num2cell(marginalQuantiles(:, :, 1), 1);
VaR991  = num2cell(marginalQuantiles(:, :, 2), 1);
VaR951  = num2cell(marginalQuantiles(:, :, 3), 1);
VaR901  = num2cell(marginalQuantiles(:, :, 4), 1);
Port9991 = portfolioQuantiles(:, 1);
Port991  = portfolioQuantiles(:, 2);
Port951  = portfolioQuantiles(:, 3);
Port901  = portfolioQuantiles(:, 4);
Port101  = portfolioQuantiles(:, 5);
Port51   = portfolioQuantiles(:, 6);
Port11   = portfolioQuantiles(:, 7);
Port0011 = portfolioQuantiles(:, 8);

Output.VaRPortfolio9991 = Port9991;
Output.VaRPortfolio991 = Port991;
Output.VaRPortfolio951 = Port951;
Output.VaRPortfolio901 = Port901;
Output.VaRPortfolio0011 = Port0011;
Output.VaRPortfoli0011 = Port0011; % Backward-compatible misspelled field.
Output.VaRPortfolio11 = Port11;
Output.VaRPortfolio51 = Port51;
Output.VaRPortfolio101 = Port101;

% Realized equally weighted portfolio profit/loss.
true_PuL = log(mean(exp(daten(end-t+1:end, :)), 2));
Output.truePuL = true_PuL;

% mache den Backtest nur auf den out-of-sample Ergebnissen
% VaR (95/1)
if strcmp(PrtFig,'on')
    figure
    a=area(Port951(:,1));
    set(a,'FaceColor',[1 1 0])
    hold on
    b=area(Port51(:,1));
    set(b,'FaceColor',[1 1 0])
    hold on
    plot(true_PuL(:,:),'.')
    title('VaR (95/1)')
    % VaR (99/1)
    figure
    a=area(Port991(:,1));
    set(a,'FaceColor',[1 1 0])
    hold on
    b=area(Port11(:,1));
    set(b,'FaceColor',[1 1 0])
    hold on
    plot(true_PuL(:,1),'.')
    title('VaR (99/1)')
    % VaR (90/1)
    figure
    a=area(Port901(:,1));
    set(a,'FaceColor',[1 1 0])
    hold on
    b=area(Port101(:,1));
    set(b,'FaceColor',[1 1 0])
    hold on
    plot(true_PuL(:,:),'.')
    title('VaR (90/1)')
    % VaR (999/1)
    figure
    a=area(Port9991(:,1));
    set(a,'FaceColor',[1 1 0])
    hold on
    b=area(Port0011(:,1));
    set(b,'FaceColor',[1 1 0])
    hold on
    plot(true_PuL(:,:),'.')
    title('VaR (999/1)')
end





% Zähle VaR-Verletzungen im linken Verteilungsschwanz.
VaRviol991 = double(Port991 > true_PuL);
SumVaRviol991 = sum(VaRviol991);
Output.SumVaRviol991 = SumVaRviol991;

VaRviol951 = double(Port951 > true_PuL);
SumVaRviol951 = sum(VaRviol951);
Output.SumVaRviol951 = SumVaRviol951;

VaRviol901 = double(Port901 > true_PuL);
SumVaRviol901 = sum(VaRviol901);
Output.SumVaRviol901 = SumVaRviol901;

VaRviol9991 = double(Port9991 > true_PuL);
SumVaRviol9991 = sum(VaRviol9991);
Output.SumVaRviol9991 = SumVaRviol9991;

% Referenzen Wilmott, P. (2006). Wilmott on Quantitative Finance 2nd ed.

% ------------------------------------------------------------------------
% -------------------------------------------------------------------------
% VaR-Tests
% Tests nach Christoffersen ("Backtesting"), S.4,
%
% %%%%% unconditional Coverage Test %%%%%%%%%%%
% Simple means (unconditional coverage) test H0 : E(It) = p (im Mittel treten p-Verletzungen auf)
% Formel: MT sqrt(T)*(pi - p)/sqrt(VaR(I_t) ist Normalverteilt (0,1)
% I_t: Sequenz von Nullen und Einsen - die hit sequence (Eins: der Verlust der PuL ist größer
% als der VaR)
% pi : sample mean der hit-sequence
% Bilde den Sample Average der hit-sequence
T1 = SumVaRviol991;
T = size(Port991,1);
mean_hit = T1/T;
p = 0.01; %Konfidenzniveau
var_It = var(VaRviol991);
MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% kritischer Wert der Normalverteilung
norm_krit = norminv(0.90,0,1);
% Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% Teststatistik
% pval = 1-normcdf(abs(MT),0,1);
pval = 2*min(normcdf(MT),1-normcdf(MT)); %p-Wert für zweiseitigen Test
Output.UncondCoverage991_MT = [MT norm_krit pval];

% für VaR95-1
T1 = SumVaRviol951;
T = size(Port951,1);
mean_hit = T1/T;
p = 0.05; %Konfidenzniveau
var_It = var(VaRviol951);
MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% kritischer Wert der Normalverteilung
norm_krit = norminv(0.90,0,1);
% Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% Teststatistik
pval = 2*min(normcdf(MT),1-normcdf(MT));
Output.UncondCoverage951_MT = [MT norm_krit pval];

% für VaR90-1
T1 = SumVaRviol901;
T = size(Port901,1);
mean_hit = T1/T;
p = 0.1; %Konfidenzniveau
var_It = var(VaRviol901);
MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% kritischer Wert der Normalverteilung
norm_krit = norminv(0.9,0,1);
% Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% Teststatistik
pval = 2*min(normcdf(MT),1-normcdf(MT));
Output.UncondCoverage901_MT = [MT norm_krit pval];

% für VaR999-1
T1 = SumVaRviol9991;
T = size(Port9991,1);
mean_hit = T1/T;
p = 0.001; %Konfidenzniveau
var_It = var(VaRviol9991);
MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% kritischer Wert der Normalverteilung
norm_krit = norminv(0.90,0,1);
% Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% Teststatistik
pval = 2*min(normcdf(MT),1-normcdf(MT));
Output.UncondCoverage9991_MT = [MT norm_krit pval];

% -----------------------------------------------------------------------
% Likelihood ratio test
% Nullhypothese pi = p, also erwartete Wert der hit-Sequenz (pi) soll dem
% coverage ratio (p) entsprechen
T1 = SumVaRviol991;
T = size(Port991,1);
T0 = T-T1;
p=0.01;
% optimierte Likelihood
Like1_991 = (1-T1/T)^T0*(T1/T)^T1;
Like2_991 = (1-p)^T0*p^T1;
% Likelihood Ratio Teststatistik
LR991 = -2*log(Like2_991/Like1_991);
% kritischer Wert der Chi^2-Verteilung
krit_chi991 = chi2inv(0.90,1);
% Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
pval = 1-chi2cdf(LR991,1); %für p-value siehe Christoffersen (2003), S.185
Output.Independence991_LR = [LR991 krit_chi991 pval];

% für VaR 951
T1 = SumVaRviol951;
T = size(Port951,1);
T0 = T-T1;
p=0.05;
% optimierte Likelihood
Like1_951 = (1-T1/T)^T0*(T1/T)^T1;
Like2_951 = (1-p)^T0*p^T1;
% Likelihood Ratio Teststatistik
LR951 = -2*log(Like2_951/Like1_951);
% kritischer Wert der Chi^2-Verteilung
krit_chi951 = chi2inv(0.90,1);
% Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
pval = 1-chi2cdf(LR951,1);
Output.Independence951_LR = [LR951 krit_chi951 pval];


% für VaR 901
T1 = SumVaRviol901;
T = size(Port901,1);
T0 = T-T1;
p=0.001;
% optimierte Likelihood
Like1_901 = (1-T1/T)^T0*(T1/T)^T1;
Like2_901 = (1-p)^T0*p^T1;
% Likelihood Ratio Teststatistik
LR901 = -2*log(Like2_901/Like1_901);
% kritischer Wert der Chi^2-Verteilung
krit_chi901 = chi2inv(0.90,1);
% Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
pval = 1-chi2cdf(LR901,1);
Output.Independence901_LR = [LR901 krit_chi901 pval];

% für VaR 9991
T1 = SumVaRviol9991;
T = size(Port9991,1);
T0 = T-T1;
p=0.1;
% optimierte Likelihood
Like1_9991 = (1-T1/T)^T0*(T1/T)^T1;
Like2_9991 = (1-p)^T0*p^T1;
% Likelihood Ratio Teststatistik
LR9991 = -2*log(Like2_9991/Like1_9991);
% kritischer Wert der Chi^2-Verteilung
krit_chi9991 = chi2inv(0.90,1);
% Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
pval = 1-chi2cdf(LR9991,1);
Output.Independence9991_LR = [LR9991 krit_chi9991 pval];

% -----------------------------------------------------------------------
% Ljung-Box (independence testing)
% siehe(Jin (2009), "Large Portfolio Risk Management with Dynamic Copulas)
% H0: model fit is adequate
[~,pValue_991,Qstat_991,CriticalValue_991] = lbqtest( ...
    VaRviol991, 'Lags', 5, 'Alpha', .10, 'DoF', 5);
Output.LB991(1) = Qstat_991;
Output.LB991(2) = CriticalValue_991;
Output.LB991(3) = pValue_991;
% für VaR 951
[~,pValue_951,Qstat_951,CriticalValue_951] = lbqtest( ...
    VaRviol951, 'Lags', 5, 'Alpha', .10, 'DoF', 5);
Output.LB951(1) = Qstat_951;
Output.LB951(2) = CriticalValue_951;
Output.LB951(3) = pValue_951;
% für VaR 901
[~,pValue_901,Qstat_901,CriticalValue_901] = lbqtest( ...
    VaRviol901, 'Lags', 5, 'Alpha', .10, 'DoF', 5);
Output.LB901(1) = Qstat_901;
Output.LB901(2) = CriticalValue_901;
Output.LB901(3) = pValue_901;
% für VaR 9991
[~,pValue_9991,Qstat_9991,CriticalValue_9991] = lbqtest( ...
    VaRviol9991, 'Lags', 5, 'Alpha', .10, 'DoF', 5);
Output.LB9991(1) = Qstat_9991;
Output.LB9991(2) = CriticalValue_9991;
Output.LB9991(3) = pValue_9991;


% lower partial moment
ELPM991 = cell(k,1);
for j=1:k
    for i=1:size(VaR991{k},1)
        ELPM991{j}(i,:) = -elpm(MeanForecast(i,j),sqrt(VarianceForecast(i,j)),VaR991{j}(i),[0 1 2 3]);
    end
end
Output.ELPM991 = ELPM991;

ELPM951 = cell(k,1);
for j=1:k
    for i=1:size(VaR951{k},1)
        ELPM951{j}(i,:) = -elpm(MeanForecast(i,j),sqrt(VarianceForecast(i,j)),VaR951{j}(i),[0 1 2 3]);
    end
end
Output.ELPM951 = ELPM951;

ELPM901 = cell(k,1);
for j=1:k
    for i=1:size(VaR901{k},1)
        ELPM901{j}(i,:) = -elpm(MeanForecast(i,j),sqrt(VarianceForecast(i,j)),VaR901{j}(i),[0 1 2 3]);
    end
end
Output.ELPM901 = ELPM901;

ELPM9991 = cell(k,1);
for j=1:k
    for i=1:size(VaR9991{k},1)
        ELPM9991{j}(i,:) = -elpm(MeanForecast(i,j),sqrt(VarianceForecast(i,j)),VaR9991{j}(i),[0 1 2 3]);
    end
end
Output.ELPM9991 = ELPM9991;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% alter Code nur zu Dokuzwecken
% % Die Cholesky-Zerlegung wird benutzt um den unkorrelierten
% % Zufallsvariablen die gewünschte Korrelation zu geben; (siehe z.B. Wilmott(2006), S.1275
% % Rt ist der Korrelationsforecast von t auf t+1
% RChol=zeros(k,k,t);
% for i = 1:t
%     RChol(:,:,i) = chol(Rt(:,:,i));
% end

%
% % gebe den unkorrelierten Variablen die gewünschte Abhängigkeitsstruktur
% rndRt991 = zeros(t,k);
% VaR991trans = zeros(t,k);
% for i=1:k
%     VaR991trans(:,i)=VaR991{i};
% end
% for i=1:t
%     rndRt991(i,:) = (RChol(:,:,i)'*VaR991trans(i,:)')';
% end
% clear VaR991trans
% % Bilde Portfolio VaR
% Port991 = zeros(t,1);
% for i=1:k
%     Port991 = Port991+(1/k)*rndRt991(:,i);
% end
% clear rndRt991
% Output.VaRPortfolio991 = Port991;
%
% rndRt951 = zeros(t,k);
% VaR951trans = zeros(t,k);
% for i=1:k
%     VaR951trans(:,i)=VaR951{i};
% end
% for i=1:t
%     rndRt951(i,:) = (RChol(:,:,i)'*VaR951trans(i,:)')';
% end
% clear VaR951trans
% % Bilde Portfolio VaR
% Port951 = zeros(t,1);
% for i=1:k
%     Port951 = Port951+(1/k)*rndRt951(:,i);
% end
% clear rndRt951
% Output.VaRPortfolio951 = Port951;
%
% rndRt901 = zeros(t,k);
% VaR901trans = zeros(t,k);
% for i=1:k
%     VaR901trans(:,i)=VaR901{i};
% end
% for i=1:t
%     rndRt901(i,:) = (RChol(:,:,i)'*VaR901trans(i,:)')';
% end
% clear VaR991trans
% % Bilde Portfolio VaR
% Port901 = zeros(t,1);
% for i=1:k
%     Port901 = Port901+(1/k)*rndRt901(:,i);
% end
% clear rndRt901
% Output.VaRPortfolio901 = Port901;
%
% rndRt9991 = zeros(t,k);
% VaR9991trans = zeros(t,k);
% for i=1:k
%     VaR9991trans(:,i)=VaR9991{i};
% end
% for i=1:t
%     rndRt9991(i,:) = (RChol(:,:,i)'*VaR9991trans(i,:)')';
% end
% clear VaR9991trans
% % Bilde Portfolio VaR
% Port9991 = zeros(t,1);
% for i=1:k
%     Port9991 = Port9991+(1/k)*rndRt9991(:,i);
% end
% clear rndRt9991
% Output.VaRPortfolio991 = Port991;
%
% rndRt_pred1 = zeros(t,k);
% VaR11trans = zeros(t,k);
% for i=1:k
%     VaR11trans(:,i)=VaR11{i};
% end
% for i=1:t
%     rndRt_pred1(i,:) = (RChol(:,:,i)'*VaR11trans(i,:)')';
% end
% clear VaR11trans
% % Bilde Portfolio VaR
% Port11 = zeros(t,1);
% for i=1:k
%     Port11 = Port11+(1/k)*rndRt_pred1(:,i);
% end
% clear rndRt_pred1
% Output.VaRPortfolio11 = Port11;
%
% rndRt51 = zeros(t,k);
% VaR51trans = zeros(t,k);
% for i=1:k
%     VaR51trans(:,i)=VaR51{i};
% end
% for i=1:t
%     rndRt51(i,:) = (RChol(:,:,i)'*VaR51trans(i,:)')';
% end
% clear VaR51trans
% % Bilde Portfolio VaR
% Port51 = zeros(t,1);
% for i=1:k
%     Port51 = Port51+(1/k)*rndRt51(:,i);
% end
% clear rndRt51
% Output.VaRPortfolio51 = Port51;
%
% rndRt101 = zeros(t,k);
% VaR101trans = zeros(t,k);
% for i=1:k
%     VaR101trans(:,i)=VaR101{i};
% end
% for i=1:t
%     rndRt101(i,:) = (RChol(:,:,i)'*VaR101trans(i,:)')';
% end
% clear VaR101trans
% % Bilde Portfolio VaR
% Port101 = zeros(t,1);
% for i=1:k
%     Port101 = Port101+(1/k)*rndRt101(:,i);
% end
% clear rndRt101
% Output.VaRPortfolio101 = Port101;
%
% rndRt0011 = zeros(t,k);
% VaR0011trans = zeros(t,k);
% for i=1:k
%     VaR0011trans(:,i)=VaR0011{i};
% end
% for i=1:t
%     rndRt0011(i,:) = (RChol(:,:,i)'*VaR0011trans(i,:)')';
% end
% clear VaR0011trans
% % Bilde Portfolio VaR
% Port0011 = zeros(t,1);
% for i=1:k
%     Port0011 = Port0011+(1/k)*rndRt0011(:,i);
% end
% clear rndRt51
% Output.VaRPortfolio101 = Port101;
%
% % Berechne die wahre PuL
% % Es gehen Beobachtungen durch die AR-Schätzung verloren (in diesem Fall
% % eine)
% true_PuL=zeros(size(Port991,1),1);
% for i=1:k
%     true_PuL = true_PuL + 1/k*exp(daten(end-t2+1:end,i)); %bei Verwendung von Originaldaten muss für die einperiodige Vorhersage
% end
% true_PuL = log(true_PuL);
% Output.truePuL = true_PuL;
%
% % mache den Backtest nur auf den out-of-sample Ergebnissen
% % VaR (95/1)
% if strcmp(PrtFig,'on')
%     figure
%     a=area(Port951(:,1));
%     set(a,'FaceColor',[1 1 0])
%     hold on
%     b=area(Port51(:,1));
%     set(b,'FaceColor',[1 1 0])
%     hold on
%     plot(true_PuL(:,:),'.')
%     title('VaR (95/1)')
%     % VaR (99/1)
%     figure
%     a=area(Port991(:,1));
%     set(a,'FaceColor',[1 1 0])
%     hold on
%     b=area(Port11(:,1));
%     set(b,'FaceColor',[1 1 0])
%     hold on
%     plot(true_PuL(:,1),'.')
%     title('VaR (99/1)')
%     % VaR (90/1)
%     figure
%     a=area(Port901(:,1));
%     set(a,'FaceColor',[1 1 0])
%     hold on
%     b=area(Port101(:,1));
%     set(b,'FaceColor',[1 1 0])
%     hold on
%     plot(true_PuL(:,:),'.')
%     title('VaR (90/1)')
%     % VaR (999/1)
%     figure
%     a=area(Port9991(:,1));
%     set(a,'FaceColor',[1 1 0])
%     hold on
%     b=area(Port0011(:,1));
%     set(b,'FaceColor',[1 1 0])
%     hold on
%     plot(true_PuL(:,:),'.')
%     title('VaR (999/1)')
% end
%
%
%
% % Zähel die VaR-Verletzungen (nur left tail), um die Modellgüte zu validieren
% VaRviol991=zeros(size(Port991,1),1);
% for i = 1:size(Port991,1)
%     if Port991(i) > true_PuL(i)
%         VaRviol991(i) =  1;
%     else VaRviol991(i) =  0;
%     end
% end
% SumVaRviol991 = sum(VaRviol991);
% Output.SumVaRviol991 = sum(VaRviol991);
%
% VaRviol951=zeros(size(Port951,1),1);
% for i = 1:size(Port951,1)
%     if Port951(i) > true_PuL(i)
%         VaRviol951(i) =  1;
%     else VaRviol951(i) =  0;
%     end
% end
% SumVaRviol951 = sum(VaRviol951);
% Output.SumVaRviol951 = sum(VaRviol951);
%
% VaRviol901=zeros(size(Port901,1),1);
% for i = 1:size(Port901,1)
%     if Port901(i) > true_PuL(i)
%         VaRviol901(i) =  1;
%     else VaRviol901(i) =  0;
%     end
% end
% SumVaRviol901 = sum(VaRviol901);
% Output.SumVaRviol901 = sum(VaRviol901);
%
% VaRviol9991=zeros(size(Port9991,1),1);
% for i =  1:size(Port9991,1)
%     if Port9991(i) > true_PuL(i)
%         VaRviol9991(i) =  1;
%     else VaRviol9991(i) =  0;
%     end
% end
% SumVaRviol9991 = sum(VaRviol9991);
% Output.SumVaRviol9991 = sum(VaRviol9991);
%
% % Referenzen Wilmott, P. (2006). Wilmott on Quantitative Finance 2nd ed.
%
% % ------------------------------------------------------------------------
% % -------------------------------------------------------------------------
% % VaR-Tests
% % Tests nach Christoffersen ("Backtesting"), S.4,
% %
% % %%%%% unconditional Coverage Test %%%%%%%%%%%
% % Simple means (unconditional coverage) test H0 : E(It) = p (im Mittel treten p-Verletzungen auf)
% % Formel: MT sqrt(T)*(pi - p)/sqrt(VaR(I_t) ist Normalverteilt (0,1)
% % I_t: Sequenz von Nullen und Einsen - die hit sequence (Eins: der Verlust der PuL ist größer
% % als der VaR)
% % pi : sample mean der hit-sequence
% % Bilde den Sample Average der hit-sequence
% T1 = SumVaRviol991;
% T = size(Port991,1);
% mean_hit = T1/T;
% p = 0.01; %Konfidenzniveau
% var_It = var(VaRviol991)/T;
% MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% % kritischer Wert der Normalverteilung
% norm_krit = norminv(0.90,0,1);
% % Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% % Teststatistik
% pval = 1-normcdf(MT,0,1);
% Output.UncondCoverage991_MT = [MT norm_krit pval];
%
% % für VaR95-1
% T1 = SumVaRviol951;
% T = size(Port951,1);
% mean_hit = T1/T;
% p = 0.05; %Konfidenzniveau
% var_It = var(VaRviol951)/T;
% MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% % kritischer Wert der Normalverteilung
% norm_krit = norminv(0.90,0,1);
% % Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% % Teststatistik
% pval = 1-normcdf(MT,0,1);
% Output.UncondCoverage951_MT = [MT norm_krit pval];
%
% % für VaR90-1
% T1 = SumVaRviol901;
% T = size(Port901,1);
% mean_hit = T1/T;
% p = 0.1; %Konfidenzniveau
% var_It = var(VaRviol901)/T;
% MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% % kritischer Wert der Normalverteilung
% norm_krit = norminv(0.90,0,1);
% % Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% % Teststatistik
% pval = 1-normcdf(MT,0,1);
% Output.UncondCoverage901_MT = [MT norm_krit pval];
%
% % für VaR999-1
% T1 = SumVaRviol9991;
% T = size(Port9991,1);
% mean_hit = T1/T;
% p = 0.001; %Konfidenzniveau
% var_It = var(VaRviol9991)/T;
% MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% % kritischer Wert der Normalverteilung
% norm_krit = norminv(0.90,0,1);
% % Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% % Teststatistik
% pval = 1-normcdf(MT,0,1);
% Output.UncondCoverage9991_MT = [MT norm_krit pval];
%
% % -----------------------------------------------------------------------
% % Likelihood ratio test
% % Nullhypothese pi = p, also erwartete Wert der hit-Sequenz (pi) soll dem
% % coverage ratio (p) entsprechen
% T1 = SumVaRviol991;
% T = size(Port991,1);
% T0 = T-T1;
% p=0.01;
% % optimierte Likelihood
% Like1_991 = (1-T1/T)^T0*(T1/T)^T1;
% Like2_991 = (1-p)^T0*p^T1;
% % Likelihood Ratio Teststatistik
% LR991 = -2*log(Like2_991/Like1_991);
% % kritischer Wert der Chi^2-Verteilung
% krit_chi991 = chi2inv(0.90,1);
% % Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% % Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
% pval = 1-chi2cdf(LR991,1);
% Output.Independence991_LR = [LR991 krit_chi991 pval];
%
% % für VaR 951
% T1 = SumVaRviol951;
% T = size(Port951,1);
% T0 = T-T1;
% p=0.05;
% % optimierte Likelihood
% Like1_951 = (1-T1/T)^T0*(T1/T)^T1;
% Like2_951 = (1-p)^T0*p^T1;
% % Likelihood Ratio Teststatistik
% LR951 = -2*log(Like2_951/Like1_951);
% % kritischer Wert der Chi^2-Verteilung
% krit_chi951 = chi2inv(0.90,1);
% % Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% % Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
% pval = 1-chi2cdf(LR951,1);
% Output.Independence951_LR = [LR951 krit_chi951 pval];
%
%
% % für VaR 901
% T1 = SumVaRviol901;
% T = size(Port901,1);
% T0 = T-T1;
% p=0.1;
% % optimierte Likelihood
% Like1_901 = (1-T1/T)^T0*(T1/T)^T1;
% Like2_901 = (1-p)^T0*p^T1;
% % Likelihood Ratio Teststatistik
% LR901 = -2*log(Like2_901/Like1_901);
% % kritischer Wert der Chi^2-Verteilung
% krit_chi901 = chi2inv(0.90,1);
% % Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% % Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
% pval = 1-chi2cdf(LR901,1);
% Output.Independence901_LR = [LR901 krit_chi901 pval];
%
% % für VaR 9991
% T1 = SumVaRviol9991;
% T = size(Port9991,1);
% T0 = T-T1;
% p=0.1;
% % optimierte Likelihood
% Like1_9991 = (1-T1/T)^T0*(T1/T)^T1;
% Like2_9991 = (1-p)^T0*p^T1;
% % Likelihood Ratio Teststatistik
% LR9991 = -2*log(Like2_9991/Like1_9991);
% % kritischer Wert der Chi^2-Verteilung
% krit_chi9991 = chi2inv(0.90,1);
% % Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% % Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
% pval = 1-chi2cdf(LR9991,1);
% Output.Independence9991_LR = [LR9991 krit_chi9991 pval];
%
% % -----------------------------------------------------------------------
% % Ljung-Box (independence testing)
% % siehe(Jin (2009), "Large Portfolio Risk Management with Dynamic Copulas)
% % H0: model fit is adequate
% [H_991,pValue_991,Qstat_991,CriticalValue_991] = lbqtest(VaRviol991,5,.10,5);
% Output.LB991(1) = Qstat_991;
% Output.LB991(2) = CriticalValue_991;
% Output.LB991(3) = pValue_991;
% % für VaR 951
% [H_951,pValue_951,Qstat_951,CriticalValue_951] = lbqtest(VaRviol951,5,.10,5);
% Output.LB951(1) = Qstat_951;
% Output.LB951(2) = CriticalValue_951;
% Output.LB951(3) = pValue_951;
% % für VaR 901
% [H_901,pValue_901,Qstat_901,CriticalValue_901] = lbqtest(VaRviol901,5,.10,5);
% Output.LB901(1) = Qstat_901;
% Output.LB901(2) = CriticalValue_901;
% Output.LB901(3) = pValue_901;
% % für VaR 9991
% [H_9991,pValue_9991,Qstat_9991,CriticalValue_9991] = lbqtest(VaRviol9991,5,.10,5);
% Output.LB951(1) = Qstat_9991;
% Output.LB951(2) = CriticalValue_9991;
% Output.LB9991(3) = pValue_9991;
%
%
% % lower partial moment
% ELPM991 = cell(k,1);
% for j=1:k
%     for i=1:size(VaR991{k},1)
%         ELPM991{j}(i,:) = -elpm(MeanForecast(i,j),sqrt(VarianceForecast(i,j)),VaR991{j}(i),[0 1 2 3]);
%     end
% end
% Output.ELPM991 = ELPM991;
%
% ELPM951 = cell(k,1);
% for j=1:k
%     for i=1:size(VaR951{k},1)
%         ELPM991{j}(i,:) = -elpm(MeanForecast(i,j),sqrt(VarianceForecast(i,j)),VaR951{j}(i),[0 1 2 3]);
%     end
% end
% Output.ELPM951 = ELPM951;
%
% ELPM901 = cell(k,1);
% for j=1:k
%     for i=1:size(VaR901{k},1)
%         ELPM901{j}(i,:) = -elpm(MeanForecast(i,j),sqrt(VarianceForecast(i,j)),VaR901{j}(i),[0 1 2 3]);
%     end
% end
% Output.ELPM901 = ELPM901;
%
% ELPM9991 = cell(k,1);
% for j=1:k
%     for i=1:size(VaR9991{k},1)
%         ELPM9991{j}(i,:) = -elpm(MeanForecast(i,j),sqrt(VarianceForecast(i,j)),VaR9991{j}(i),[0 1 2 3]);
%     end
% end
% Output.ELPM9991 = ELPM9991;
