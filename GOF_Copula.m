function [Stat,zt] = GOF_Copula(stdresid,Weights_tv,CopParam_tv,copparameters,Spec,upareto)

% Goodness-of-fit Tests für Clayton und rotated Clayton Copulas (Anderson-Darling
% und Kolmogorov)
% INPUTS:
%        stdresid: stdresid t x k
%        weights: Vector mit Gewichten aus der Copula (1x2 cell bei
%        mixture)
%       CopParam_tv : zeitvariierende Copula-Parameter
% Weights_tv: zeitvariierende Gewichte
%
% OUTPUTS:
%         Stat: Structure mit AD,ADD,KS und AKS Statistiken
% 
% Author: Martin Grziska, last modification: 04/22/2011

[t,k] = size(stdresid);

if sum(any(stdresid>1))<k || sum(any(stdresid<0))<k
    error('Daten-Input müssen standardisierte Residuen sein')
end

% Bei archimedischen Copulas und Mixture überprüfe Dimension
n_Cop=size(Spec.family,2);
% wenn v nicht existiert kreiere Variable damit die Schleife unten bei v1
% anfängt
clear v
if ~exist('v')
    v=0;
end

% Zur Sicherheit:lösche alle v's:
clear v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22 v23 v24

% lese die Daten als v-Variablen
for i = 1:k
    % wandle die standardisierten Residuen in empirische Margins um
    if strcmp(Spec.tails,'pareto')
        u(:,i) = upareto{i}.cdf(stdresid(:,i));
    else
        u(:,i) = empiricalCDF(stdresid(:,i));
    end
    var = genvarname('v', who);
    eval([var ' = u(:, i);'])
end

for ii=1:n_Cop
    archCop_Param = CopParam_tv{ii};
    switch Spec.family{ii}
        case {'clayton' 'rotclayton' 'gumbel'}
            for i = 1:t
                % Dichte der entsprechenden archimedischen Copula berechnen
                %             zt1 ist der Zähler; zt2 der Nenner; zt =
                %             zt1./zt2;
                for jj=2:k
                    % Erzeuge die conditional Funkitionen: Zur Erzeugung dieser Funktionen
                    % siehe im Detail Cherubini et al (2004). Copula Methods in Finance Seite
                    % 182
                    % Beispiel 3 Variablen:
                    % Im Zähler steht delta^2 C(u1,u2,u3)/(delta u1 * delta u2) also die
                    % Copula-Funktion abgeleitet nach u1 und u2
                    AD_pdffunc_zaehler = AD_copulapdffunc_zaehler(Spec.family{ii}, jj);
                    % Im Nenner steht delta^2 C(u1,u2)/(delta u1 * delta u2) also die
                    % Copula-Funktion eine Dimension geringer als die im Zähler abgeleitet nach
                    % u1 und u2
                    AD_pdffunc_nenner = AD_copulapdffunc_nenner(Spec.family{ii}, jj);
                    switch jj
                        case 2
                            %                                 Formel siehe McNeil et al (2005), S197;
                            %                                 allgemein:
                            %                                 C(u2|u1) =  (\partial / \partial u1) C(u1,u2);
                            if strcmp(Spec.family{ii},'clayton')
                                zt{ii}(i,jj) = AD_pdffunc_zaehler(archCop_Param(i), v1(i),v2(i)); %für bivariate kein division
                            elseif strcmp(Spec.family{ii},'rotclayton')
                                zt{ii}(i,jj) = AD_pdffunc_zaehler(archCop_Param(i), v1(i),v2(i)); %für bivariate kein division
                            elseif strcmp(Spec.family{ii},'gumbel')
                                zt{ii}(i,jj) = AD_pdffunc_zaehler(archCop_Param(i), v1(i),v2(i)); %für bivariate kein division
                            end
                        case 3
                            zt1 = AD_pdffunc_zaehler(archCop_Param(i), v1(i),v2(i),v3(i));
                            zt2 = AD_pdffunc_nenner(archCop_Param(i), v1(i),v2(i));
                            zt{ii}(i,jj) = zt1./zt2;
                        case 4
                            zt1 = AD_pdffunc_zaehler(archCop_Param(i), v1(i),v2(i),v3(i),v4(i));
                            zt2 = AD_pdffunc_nenner(archCop_Param(i), v1(i),v2(i),v3(i));
                            zt{ii}(i,jj) = zt1./zt2;
                        case 5
                            zt1 = AD_pdffunc_zaehler(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i));
                            zt2 = AD_pdffunc_nenner(archCop_Param(i), v1(i),v2(i),v3(i),v4(i));
                            zt{ii}(i,jj) = zt1./zt2;
                        case 6
                            zt1 = AD_pdffunc_zaehler(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i));
                            zt2 = AD_pdffunc_nenner(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i));
                            zt{ii}(i,jj) = zt1./zt2;
                        case 7
                            zt1 = AD_pdffunc_zaehler(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i));
                            zt2 = AD_pdffunc_nenner(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i));
                            zt{ii}(i,jj) = zt1./zt2;
                        case 8
                            zt1 = AD_pdffunc_zaehler(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i));
                            zt2 = AD_pdffunc_nenner(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i));
                            zt{ii}(i,jj) = zt1./zt2;
                        case 9
                            zt1 = AD_pdffunc_zaehler(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));
                            zt2 = AD_pdffunc_nenner(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i));
                            zt{ii}(i,jj) = zt1./zt2;
                        case 10
                            zt1 = AD_pdffunc_zaehler(archCop_Param(i), v1(i), v10(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));
                            zt2 = AD_pdffunc_nenner(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));
                            zt{ii}(i,jj) = zt1./zt2;
                        case 11
                            zt1 = AD_pdffunc_zaehler(archCop_Param(i), v1(i),v10(i),v11(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));
                            zt2 = AD_pdffunc_nenner(archCop_Param(i), v1(i),v10(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));
                            zt{ii}(i,jj) = zt1./zt2;
                        case 12
                            zt1 = AD_pdffunc_zaehler(archCop_Param(i), v1(i),v10(i),v11(i),v12(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));
                            zt2 = AD_pdffunc_nenner(archCop_Param(i), v1(i),v10(i),v11(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));
                            zt{ii}(i,jj) = zt1./zt2;
                        case 13
                            zt1 = AD_pdffunc_zaehler(archCop_Param(i), v1(i),v10(i),v11(i),v12(i),v13(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));
                            zt2 = AD_pdffunc_nenner(archCop_Param(i), v1(i),v10(i),v11(i),v12(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));
                            zt{ii}(i,jj) = zt1./zt2;
                        case 14
                            zt1 = AD_pdffunc_zaehler(archCop_Param(i), v1(i),v10(i),v11(i),v12(i),v13(i),v14(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));
                            zt2 = AD_pdffunc_nenner(archCop_Param(i), v1(i),v10(i),v11(i),v12(i),v13(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));
                            zt{ii}(i,jj) = zt1./zt2;
                    end
                end
            end
            %             empirische margins in der ersten Spalte
            zt{ii}(:,1) = u(:,1);
            
        case('gaussian')
            %             siehe Chen,Fan,Patton (2004)
            GaussData = norminv(u);
            theta = CopParam_tv{ii};
            for i=1:t
                for jj=2:k
                    zt{ii}(i,jj) = normcdf((GaussData(i,jj) - GaussData(i,1:jj-1)*theta(1:jj-1,1:jj-1,i)^(-1)*theta(1:jj-1,jj,i))./sqrt(1-theta(jj,1:jj-1,i)*theta(1:jj-1,1:jj-1,i)^(-1)*theta(1:jj-1,jj,i)));
                end
            end
            %             in der ersten Spalte stehen empirische Margins
            %             der transformierten stdresid
            zt{ii}(:,1) = u(:,1);
            
            
        case('t')
            %             bestimme wieviele dynamische parameter die t-copula hat; der
            %             d.o.f.Parameter steht nach den dynamischen (DCC-Parametern)
            if strcmp(Spec.DynamicType,'DCC')
                DynPara=2;
            elseif strcmp(Spec.DynamicType,'ADCC')
                DynPara=3;
            elseif strcmp(Spec.DynamicType,'GDCC')
                DynPara = k*(Spec.archP+Spec.garchQ);
            elseif strcmp(Spec.DynamicType,'AGDCC')
                DynPara = k*(Spec.archP+Spec.garchQ+1);
            end
            nu = copparameters(DynPara+1); %extrahiere den Freiheitsgrad
            %
            %             %             Ansazt von Chen,Fan,Patton (2004)
            %             %             zum Verfahren für die t-Copula siehe Chen,Fan, Patton(2004),
            %             %             "Simple Test for Models of Dependence Between Multiple
            %             %             Financial Time Series, with Applicatons to U.S. Equity
            %             %             Returns and Exchange Rates", Formel
            %             %             siehe Seite 20.
            %             berechne Scale Matrix aus Korrelationsmatrix
            %             (siehe Chen,Fan,Patton, S.20)
            Scale1 = CopParam_tv{ii}*((nu-2)/nu);
            Tdata = tinv(u,nu);
            zt{ii}(:,1) = u(:,1); %empirische cdf in der ersten Spalte
            %             % Matrix wird folgendermaßen partitioniert:
            %             x_1 = d1x1 vector; x_2 = d2x1 vector; S = [S11
            %             S12; S21 S22]; S11 = d1xd1; S12 = d2xd1;
            %             S21=d2xd1; S22 = d2xd2;
            for i=1:t
                for jj=2:k
                    Scale2 = Scale1(1:jj,1:jj,i); %Scale.Matrix muss für jede Dimension jj bei z_jj,t neu gebildet werden
                    S11 = Scale2(1:jj-1,1:jj-1); %extrahiere S11 aus der gsamten Scale matrix für Dimension jj;
                    S12 = Scale2(1:jj-1,jj);
                    S21 = S12';
                    S22 = Scale2(jj,jj);
                    mu(i,jj) =  S21*Scale2(1:jj-1,1:jj-1)^(-1)*Tdata(i,1:jj-1)'; %für die Formel des mean siehe Box/Jenkins, S.264; mu_2 + beta_{2.1}*(X_2-mu_1), wobei beta_2.1 = S21*S11^{-1} und hier mu_1=mu_2=0 (!!! entspricht nicht ganz genau Chen et al !!!)
                    ScaleT(i,jj) = (((nu+Tdata(i,1:jj-1)*S11^(-1)*Tdata(i,1:jj-1)')./(nu+jj-1))*(S22-S21*S11^(-1)*S12));
                    zt{ii}(i,jj) = tcdf((Tdata(i,jj) - mu(i,jj))./sqrt(ScaleT(i,jj)), nu+jj-1); %im Gegensatz zu Fan et al standardisiere, so dass die tcdf(X,nu,0,1) von Matlab benutzt werden kann
                end
            end
    end
end



% wenn mixture vorliegt mixe die erzeugten zt der jeweiligen copula
% zusammen
if n_Cop>1
    weight = Weights_tv;
    %     gehee Geichte durch,ob kleiner als 0 oder größer als 1
    for i=1:2
        for j=1:t
            if weight(j,i)<0
                weight(j,i) = 0;
            elseif weight(j,i)>1
                weight(j,i) = 1;
            end
        end
    end
    %     Überlegung zur Erzeugung von conditional Zahlen aus mixture Copula (CM):
    %     CM(u1,u2) = w1*C1(u1,u2)+w2*C2(u1,u2)
    %     allgemein conditional copula: C_2|1(u2|u1) = \partial/\partial u1 C(u1,u2)
    %     conditional mixture: CM2|1 = (\partial/\partial u1) (w1*C1(u1,u2) +  w2*C2(u,1u2))
    %                                = w1*C1_1|2(u1,u2) + w2*C2_1|2(u1,u2)
    %     Zusammenfassung: es werden also die gewichteten partiellen
    %     Ableitungen summiert
    for i=1:t
        zt_mix(i,:) = weight(i,1).*zt{1}(i,:) + weight(i,2).*zt{2}(i,:);
    end
    clear zt
    zt{1}=zt_mix;
end

% zu den Tests siehe auch Köck(2008) Multivariate Copula-Modelle für
% Finanzmarktdaten, S.120 f.
% Test von Breymann(2003)"Dependence structure for multivariate
% high-frequancy data",Quantitative Finance;
% wandle die pseudavariablen zt mit inverser Normalverteilung um; quadriere
% die transformtierten variablen und summiere sie. Die Variable S hat dann
% eine chi2-Verteilung mit d-Freiheitsgraden,
for i=1:t
    for j=1:k
        invzt2(i,j) = (norminv(zt{1}(i,j)))^2;
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

