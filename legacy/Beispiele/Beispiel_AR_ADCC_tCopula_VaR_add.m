% Beispiel-ode für eine Schätzung a) univariater GARCH-Modelle b)
% AGDCC-Modelle c) VAR und d)Christoffersen HIT-ratios
% -----------------------------------------------------------------------
% Für den Forecast und somit den VaR wird die Mehtode des data-snooping angewendet. Das
% sample wird in zwei Abschnitte aufgeteilt. Mit dem ersten Abschnitt
% werden die Koeffizienten geschätzt. Dann wird mit den geschätzen
% Koeffizienten über den zweiten Abschnitt die Vorhersagekraft getestet;


% clear all;

% % Lese den  Daten ein
% % z.B.
% cd G:\Daten
% y = xlsread('only_Hedge_HFRX_20090611',1);
% daten=price2ret(y(:,2:5));

[t,k]=size(daten);

archP=1;
garchQ=1;
dccP=1;
dccQ=1;
dccG=1;
asymmG=1;
const=1;
arlag=3;
forecastP = 100;
snoopP = 250;
start=3000;

if isempty(archP)
    archP=ones(1,k);
elseif length(archP)==1
    archP=ones(1,k)*archP;
end

if isempty(garchQ)
    garchQ=ones(1,k);
elseif length(garchQ)==1
    garchQ=ones(1,k)*garchQ;
end

if isempty(asymmG)
    asymmG=ones(1,k);
elseif length(garchQ)==1
    asymmG=ones(1,k)*garchQ;
end

% Marginalmodelle (für DCC Modell und alle Derivate selbigens muss die
% Normalverteilung für die Schätzung der Marginalverteilung angenommen
% werden
% Für die BEstimmung des optimalen Modells schätze Modelle über komplette
% Datenreihe
[GARCHOutput] = AR_MarginalModel_normal(daten,archP(1),garchQ(1),const,arlag);

% Bringe die Innovations und die Varianzen auf die gleiche Länge, durch
% unterschiedliche AR-terme gehen unterschiedlich viel Beobachtungen
% verloren
for i=1:k
    maxarlag(i)=GARCHOutput{i}.arlag;
    arlag(i) = GARCHOutput{i}.arlag;
end
maxarlag=max(maxarlag);

for i=1:k
    if GARCHOutput{i}.arlag<maxarlag
        ardiff(i) = maxarlag-GARCHOutput{i}.arlag;
    else ardiff(i) = 0;
    end
    ht_new(:,i) = GARCHOutput{i}.ht(ardiff(i)+1:end);
    resid_new(:,i) = GARCHOutput{i}.Innovations(ardiff(i)+1:end);
    stdresid(:,i) = resid_new(:,i)./sqrt(ht_new(:,i));
    scores_new{i} = GARCHOutput{i}.Scores(ardiff(i)+1:end,:);
    daten_new(:,i) = daten(maxarlag+1:end,i);
    U_new(:,i) = GARCHOutput{i}.U(ardiff(i)+1:end);
end

[t,k] = size(U_new);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Schätze das ADCC-t-copula-Modell
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Festlegen der Optimierungseinstellungen
options = optimset('Display','iter','TolCon',10^-12,'TolFun',10^-4,'TolX',10^-6,'Algorithm','interior-point','Hessian','bfgs','FunValCheck','off','MaxFunEvals',10000);

% Hier wird die t-Verteilung der Innovations angenommen
% 8a tri-StudentT
[kappa_static_t,nuhat_static_t,nuci_static_t] = copulafit('t',U_new,'Options',options);

% Constraints nach Hamilton GJR-constraints
A = [-1 -1 0 0; 0 0 -1 0; 0 0 0 -1];
b = [zeros(2,1)-2*options.TolCon; -2.1];

% Constraints nach Cappiello et al (2006): ADCC-Parameter müssen positiv
% sein; t-copula: nu>2
% A = eye(k+1);
% A(:,k+1)=zeros(k+1,1); 
% A=[A;-eye(k+1);];
% b = [ones(3,1)-2*options.TolCon; 0; zeros(3,1)-2*options.TolCon;  -2.1];

epsilon=10e-3;
theta0 = [.01; .001; .8; nuhat_static_t];
[ kappa_ADCC LL_ADCC] = fmincon('multi_tCopula_tvp1_ADCCRt',theta0,A,b,[],[],[],[],'multi_tCopula_ADCC_nonlincon_grm',options,U_new(1:end-snoopP+1,:),dccP,dccQ,epsilon);
[LL_ADCC Rt_ADCC] = multi_tCopula_tvp1_ADCCRt(kappa_ADCC,U_new,archP,garchQ);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----------------------------------------------------------------------

% -----------------------------------------------------------------------
% Bestimme GARCH-Modelle über Zeitfenster, wobei bei einem bestimmten
% Zeitpunkt angefangen wird und dann sukzessive Perioden hinzugefügt werden
k=1;
for i=start:forecastP:t
    [GARCHOutput_add{k}] = AR_MarginalModel_add(daten(1:i,:),const,archP(1),garchQ(1),GARCHOutput);
    k=k+1;
end
    
% Bestimme über die zweite Hälfte des samples den forecast für die
% univariaten GARCH-Varianzen und Mean, sowie die Residuen, die sich
% ergeben durch daten-AR_pred: daten_t = omega + PHI*daten_t-1
% daten:kompletter datensatz
% Auch hier werden jeweils wie oben die Modelle mit sukzessiv zunhemnden
% % Datensätzen geschätzt
% der forecast betragt sowohl für mean als auch variance 1 Periode;
k=1;
for i=start:forecastP:t-forecastP
    [AR_pred_all{k},ht_pred_all{k}, resid_pred_all{k}] = ar_univariate_snoop_forecast1P(daten(1:i+forecastP+maxarlag-1,:),GARCHOutput_add{k},const);
    k=k+1;
end
% durch (eventuell) unterschiedliche AR-lags gehen (eventuell)
% Beobachtungen verloren; bringe alle auf gleich Länge
for i=1:k-1
    for j=1:size(daten,2)
        AR_pred_all{i}{j} = AR_pred_all{i}{j}(ardiff(j)+1:end);
        ht_pred_all{i}{j} = ht_pred_all{i}{j}(ardiff(j)+1:end);
        resid_pred_all{i}{j} = resid_pred_all{i}{j}(ardiff(j)+1:end);
    end
end

    
% Bringe die Daten der jeweiligen Forecast-Periode in eine Zeitreihe - im
% Beispiel steht also von 1:start + Länge des forecast (Variable: forecastP - also z.B. 20 Perioden) die Daten, die mit dem ersten Satz
% Koeffizienten geschätzt wurden, danach folgen 20 Daten, die mit dem
% zweiten Satz Koeffizienten geschätzt wurden usw.; beachte die 20 Perioden
% sind 20 1-periodige forecasts
k=size(daten,2);
for i=1:k
    AR_pred(:,i)=AR_pred_all{1}{i}';
    ht_pred(:,i)=ht_pred_all{1}{i};
    resid_pred(:,i)=resid_pred_all{1}{i};
end

% füge die unterscheidlichen Zeiträume (mit den unterschiedlich geschätzten
% Parametern) in eine Zeitreihe (pro Datenzeitreihe) zusammen
g=2;
for j=start+forecastP:forecastP:t-forecastP
    for h=1:k
        AR_pred(j:j+forecastP-1,h) = AR_pred_all{g}{h}(end-forecastP+1:end)';
        ht_pred(j:j+forecastP-1,h) = ht_pred_all{g}{h}(end-forecastP+1:end)';
        resid_pred(j:j+forecastP-1,h) = resid_pred_all{g}{h}(end-forecastP+1:end)';
    end
    g=g+1;
end

% ----------------------------------------------------------------------
% Berechne den multivariaten Forecast mit den Koeffizienten aus dem ersten
% Sample
% Eingesetz werden die Residuen aus univariate_ar_snoop (zusammengesetzte
% Zeitreihe aus allen möglichen forecasts

% Füge die für jeden Zeitraum geschätzen Parameters und die agdccparameter
% zusammen
for j=1:size(GARCHOutput_add,2)
    h=1;
    for i=1:k
        parameters_add{j}(h:h+size(GARCHOutput_add{j}{i}.ParamsGARCH,1)-1,1) = GARCHOutput_add{j}{i}.ParamsGARCH;
        h = h + size(GARCHOutput_add{j}{i}.ParamsGARCH,1);
    end
    parameters_add{j}=[parameters_add{j}; kappa_ADCC];
    h=1;
    g=1;
end

for i=1:k
    H(:,i) = ht_pred(:,i);
    errortype(i) = GARCHOutput{i}.errortype;
    garchtype(i) = GARCHOutput{i}.garchtype;
    asymmG(i) = GARCHOutput{i}.leverage;
end

% Forecast mit der zusammengesetzten residuen-zeitreihe und den jeweiligen
% univariaten GARCH-Parameters; die ADCC-Parameter sind die Parameter die
% mit dem ersten Teil des samples geschätzt wurden
for i=1:size(GARCHOutput_add,2)
 [Rt_pred_new{i},Qt_pred_new{i}] = multi_tCopula_ADCC_snoop(parameters_add{i}, resid_pred, archP, garchQ, asymmG, garchtype, errortype, dccP, dccQ, GARCHOutput);
end

Rt_pred(:,:,:) = Rt_pred_new{1}(:,:,1:start+forecastP-1);
Qt_pred(:,:,:) = Qt_pred_new{1}(:,:,1:start+forecastP-1);

l=2;
for i=start+forecastP:forecastP:t-forecastP
    Rt_pred(:,:,i:i+forecastP-1) = Rt_pred_new{l}(:,:,i:i+forecastP-1);
    Qt_pred(:,:,i:i+forecastP-1) = Qt_pred_new{l}(:,:,i:i+forecastP-1);
    l=l+1;
end

Rt_pred_DCC = Rt_pred;
Ht=zeros(k,k,t);
stdresid=zeros(t,k);
Hstd=H.^(0.5);
for i=1:size(Rt_pred,3);
    Ht(:,:,i)=diag(Hstd(i,:))*Rt_pred(:,:,i)*diag(Hstd(i,:));
    stdresid(i,:)=resid_pred(i,:)*Ht(:,:,i)^(-0.5);
end

save tempHt Ht
clear Ht



% %%%%%%%%%  Berechnung des VaR  %%%%%%%%%%
% Der VaR wird hier mit der Monte-Carlo Methode berechnet; in diesem Fall
% werden 10000 Werte generiert und je nach Konfidenzniveau dann der
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
% 10000 i.i.d. (0,1)-Variablen mit Student T-Verteilung für den HFRXEMN und
% GED-Verteilung für den DJW1
% Formel: y_t+1 = mu + sigma_t+1*z
% z: standardisierte Zufallsvariable mit oben festgelegter Verteilung
% sigma_t: Standardabweichung aus dem GARCH-Modell
% mu: Mittelwert aus dem ARMA-Modell


% generiere Returns aus den Zufallsvaribalen (für den Zeitpunkt t) und
% wähle den jeweiligen VaR aus - je nach Konfidenzniveau
% n: Anzahl der MC-Simulaionen

MeanForecast = AR_pred;
SigmaForecast = sqrt(ht_pred);

n=1000;
t=length(MeanForecast);

rnd_ret = cell(1,k);
for i=1:k
    rnd_ret{1,i} = zeros(n,t);
end
VaR991 = cell(1,k);
for i=1:k
    VaR991{1,i} = zeros(t,1);
end
VaR951 = cell(1,k);
for i=1:k
    VaR951{1,i} = zeros(t,1);
end
VaR11 = cell(1,k);
for i=1:k
    VaR11{1,i} = zeros(t,1);
end
VaR51 = cell(1,k);
for i=1:k
    VaR51{1,i} = zeros(t,1);
end

for i=1:k
    for j=1:t
        rnd_ret{1,i}(:,j) = MeanForecast(j,i) + SigmaForecast(j,i)*normrnd(0,1,n,1);
    end
    rnd_ret{1,i}=sort(rnd_ret{1,i});
end

for i=1:k
    for j=1:t
        VaR991{i}(j) = rnd_ret{i}(n*0.01,j);
        VaR951{i}(j) = rnd_ret{i}(n*0.05,j);
        VaR11{i}(j)  = rnd_ret{i}(n*0.99,j);
        VaR51{i}(j)  = rnd_ret{i}(n*0.95,j);
    end
end


% Die Cholesky-Zerlegung wird benutzt um den unkorrelierten
% Zufallsvariablen die gewünschte Korrelation zu geben; (siehe z.B. Wilmott(2006), S.1275
% Rt_pred ist der Korrelationsforecast von t auf t+1
RChol=zeros(k,k,t);
for i = 1:t
    RChol(:,:,i) = chol(Rt_pred(:,:,i));
end


% gebe den unkorrelierten Variablen die gewünschte Abhängigkeitsstruktur
rndRt991 = zeros(t,k);
VaR991trans = zeros(t,k);
for i=1:k
    VaR991trans(:,i)=VaR991{i};
end
for i=1:t
    rndRt991(i,:) = (RChol(:,:,i)*VaR991trans(i,:)')';
end
clear VaR991trans
% Bilde Portfolio VaR
Port991 = zeros(t,1);
for i=1:k
    Port991 = Port991+(1/k)*rndRt991(:,i);
end
clear rndRt991

rndRt951 = zeros(t,k);
VaR951trans = zeros(t,k);
for i=1:k
    VaR951trans(:,i)=VaR951{i};
end
for i=1:t
    rndRt951(i,:) = (RChol(:,:,i)*VaR951trans(i,:)')';
end
clear VaR951trans
% Bilde Portfolio VaR
Port951 = zeros(t,1);
for i=1:k
    Port951 = Port951+(1/k)*rndRt951(:,i);
end
clear rndRt951

rndRt_pred1 = zeros(t,k);
VaR11trans = zeros(t,k);
for i=1:k
    VaR11trans(:,i)=VaR11{i};
end
for i=1:t
    rndRt_pred1(i,:) = (RChol(:,:,i)*VaR11trans(i,:)')';
end
clear VaR11trans
% Bilde Portfolio VaR
Port11 = zeros(t,1);
for i=1:k
    Port11 = Port11+(1/k)*rndRt_pred1(:,i);
end
clear rndRt_pred1

rndRt51 = zeros(t,k);
VaR51trans = zeros(t,k);
for i=1:k
    VaR51trans(:,i)=VaR51{i};
end
for i=1:t
    rndRt51(i,:) = (RChol(:,:,i)*VaR51trans(i,:)')';
end
clear VaR51trans
% Bilde Portfolio VaR
Port51 = zeros(t,1);
for i=1:k
    Port51 = Port51+(1/k)*rndRt51(:,i);
end
clear rndRt51

% Berechne die wahre PuL
% Es gehen Beobachtungen durch die AR-Schätzung verloren (in diesem Fall
% eine)
true_PuL=zeros(size(Port991,1),1);
for i=1:k
    true_PuL = true_PuL + 1/k*daten_new(1:size(Port991,1),i);
end

% VaR (95/1)
figure
a=area(Port951(:,1));
set(a,'FaceColor',[1 1 0])
hold on
b=area(Port51(:,1));
set(b,'FaceColor',[1 1 0])
hold on
plot(true_PuL,'.')
title('AGDCC-VaR (95/1)')

% VaR (99/1)
figure
a=area(Port991(:,1));
set(a,'FaceColor',[1 1 0])
hold on
b=area(Port11(:,1));
set(b,'FaceColor',[1 1 0])
hold on
plot(true_PuL,'.')
title('AGDCC-VaR (99/1)')

% Zähel die VaR-Verletzungen (nur left tail), um die Modellgüte zu validieren
VaRviol991=zeros(t,1);
for i = 1:t-1
    if Port991(i) > true_PuL(i)
        VaRviol991(i) =  1;
    else VaRviol991(i) =  0;
    end
end
SumVaRviol991 = sum(VaRviol991);
opt_Verletzung = round((t-1)/100);
% Bei 1 hätte man das optimale VaR-Modell
VaRviol991per = SumVaRviol991./opt_Verletzung;

VaRviol951=zeros(t,1);
for i = 1:t-1
    if Port951(i) > true_PuL(i)
        VaRviol951(i) =  1;
    else VaRviol951(i) =  0;
    end
end
SumVaRviol951 = sum(VaRviol951);
% Bei 1 hätte man das optimale VaR-Modell
VaRviol951per = SumVaRviol951./opt_Verletzung;

% Referenzen Wilmott, P. (2006). Wilmott on Quantitative Finance 2nd ed.

% Tests nach Christoffersen ("Backtesting"), S.4,
%
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
var_It = var(Port991)/T;
MT = sqrt(T)*(mean_hit - p)/sqrt(var_It);
% kritischer Wert der Normalverteilung
norm_krit = norminv(0.99,0,1);
% Vergleich des kritischen Wertes der Normalverteilung und der errechnete
% Teststatistik
Vergleich_test1 = [MT norm_krit];

% Likelihood ratio test
% Nullhypothese pi = p, also erwartete Wert der hit-Sequenz (pi) soll dem
% coverage ratio (p) entsprechen
T0 = T-T1;
% optimierte Likelihood
Like1 = (1-T1/T)^T0*(T1/T)^T1;
Like2 = (1-p)^T0*p^T1;
% Likelihood Ratio Teststatistik
LR = -2*log(Like2/Like1);
% kritischer Wert der Chi^2-Verteilung
krit_chi = chi2inv(0.99,1);
% Vergleich des kritischen Wertes der Chi^2-Verteilung und der errechneten
% Teststatistik; asymptotisch ist der Test chi^2 mit d.o.f. = 1, verteilt
Vergleich_test2 = [LR krit_chi];

