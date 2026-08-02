function [Stat,zt] = GOF_MVGARCH(stdresid,Rt)

% Goodness-of-fit Tests für MVGARCH(Anderson-Darling
% und Kolmogorov)
%
% INPUTS:
%        stdresid: stdresid t x k
%              Rt: Korrelationsmatrix k x k x t
%              ht: Varianzen t x k
%
% OUTPUTS:
%         Stat: Structure mit AD,ADD,KS und AKS Statistiken
% Author: Martin Grziska, 31.03.2011

[t,k] = size(stdresid);

if sum(any(stdresid>1))<k || sum(any(stdresid<0))<k
    error('Daten-Input müssen standardisierte Residuen sein')
end


%             % Matrix wird folgendermaßen partitioniert:
%             f(x2|x1):
%             x_1 = d1x1 vector; x_2 = d2x1 vector; S = [S11
%             S12; S21 S22]; S11 = d1xd1; S12 = d2xd1;
%             S21=d2xd1; S22 = d2xd2;
% Formel ist dieselbe wie für die Gaussian Copula (aus Chen, Fan, Patton (2004)"Simple Tests...), da die Gaußsche Copula
% mit normalen Randverteilungen die multivariate Normalverteilung ist
for i=1:t
    for jj=2:k
        zt(i,jj) = normcdf((stdresid(i,jj) - stdresid(i,1:jj-1)*Rt(1:jj-1,1:jj-1,i)^(-1)*Rt(1:jj-1,jj,i))./sqrt(1-Rt(jj,1:jj-1,i)*Rt(1:jj-1,1:jj-1,i)^(-1)*Rt(1:jj-1,jj,i)));
    end
end
%             in der ersten Spalte stehen empirische Margins
%             der transformierten stdresid
zt(:,1) = normcdf(stdresid(:,1));

% zu den Tests siehe auch Köck(2008) Multivariate Copula-Modelle für
% Finanzmarktdaten, S.120 f.
% Test von Breymann(2003)"Dependence structure for multivariate
% high-frequancy data",Quantitative Finance;
% wandle die pseudavariablen zt mit inverser Normalverteilung um; quadriere
% die transformtierten variablen und summiere sie. Die Variable S hat dann
% eine chi2-Verteilung mit d-Freiheitsgraden,
for i=1:t
    for j=1:k
        invzt2(i,j) = (norminv(zt(i,j)))^2;
    end
    S(i,1) = sum(invzt2(i,:));
end

% Für die Formeln siehe Köck (2008)"Multivatriate Copula-Modelle für
% Finanzmarktdaten, S.121

% nehme an, dass S ch2 verteilt ist: wende ch2 cdf auf S an
Sempiricalchi2=chi2cdf(S,k);
% wie ist S wirklich verteilt: wende empirische cdf an
Sempirical = empiricalCDF(S);

% Anderson-Darling
% Daten müssen geordnet werden
Sempiricalchi2order = sort(Sempiricalchi2);
Sempiricalorder = sort(Sempirical);
% AD-test gibt mehr Gewicht auf die Tails als KS-test
for i=1:t
    AD(i) = abs(Sempiricalchi2order(i)-Sempiricalorder(i))/(sqrt(Sempiricalorder(i)*(1-Sempiricalorder(i))));
end
AD = max(AD)*sqrt(t);
Stat.AD = AD;

% Average Anderson-Darling
for i=1:t
    AAD(i) = abs(Sempiricalchi2order(i)-Sempiricalorder(i))/(sqrt(Sempiricalorder(i)*(1-Sempiricalorder(i))));
end
AAD = mean(AAD);
Stat.AAD = AAD;

% Kolomogorv-Smirnov
for i=1:t
    KS(i) = abs(Sempiricalchi2order(i)-Sempiricalorder(i));
end
KS =max(KS)*sqrt(t);
Stat.KS = KS;

% Average Kolomogorv-Smirnov
for i=1:t
    AKS(i) = abs(Sempiricalchi2order(i)-Sempiricalorder(i));
end
AKS = mean(AKS);
Stat.AKS = AKS;

% Kolmogorov-Smirnov
% Nullhypothses: Datensample kommt von angegebener Verteilung (H=0);
% [Kol.h,Kol.pvalue,Kol.ks] = kstest2(S,randomchi2);
% KS test mit den unif(0,1) Variablen empirische transformierte ch2 (S)
% gegen unif(0,1) random data
[Stat.Kol.h,Stat.Kol.pvalue,Stat.Kol.ksstat] = kstest2(Sempiricalchi2,Sempirical);

%

% % Tests nur für univariate Zeitreihen (ergeben sich aus bivariaten Copulas
% % Berechne Anderson-Darling test stat nach Rahman et al (2006)" A modified
% % Anderson-Darling Test for Uniformity"
% % ordne Daten von klein nach Groß
% Sempiricalorder = sort(Sempirical,'ascend');
% for i=1:t
%     A2(i) = (2*i-1)*log(Sempiricalorder(i))+(2*t+1-2*i)*log(1-Sempiricalorder(i));
% end
% A2 = sum(A2);
% AD = -t-1/t*A2;

