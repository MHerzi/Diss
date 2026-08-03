function Spec = setInputs(daten)

% Spezifiziert die Aufgabe welches Modell geschätzt werden soll, welchen
% Zeitraum (komplett oder backtest), wieviele Simulationen für den MC-VaR
% verwendet werden sollen etc.
% !!! Achtung: als directory darf nicht C:\Windows\system32\... stehen,
% gibt ansonsten einen Konflikt mit der Matlab-Funktion input !!!

[t,k]=size(daten);
a = menu('Schätzung von:', 'Multivariaten GARCH', 'Multivariaten Copulas', 'Multivariaten Vine-Copulas', 'Multivariaten Mixture-Copulas','Multivariate (single) archimedische Copulas','Multivariate D-Vine Mixture','Histroische Simulation VaR','Delta-Normal VaR','Soll eine CVaR-Portfoliooptimierung durchgeführt werden?');
if a == 1
    Spec.ModelType = 'MultiGARCH';
    e = menu('Spezifizieren der Dynamik','DCC','ADCC','GDCC','AGDCC');
    if e == 1
        Spec.DynamicType = 'DCC';
    elseif e == 2
        Spec.DynamicType = 'ADCC';
    elseif e == 3
        Spec.DynamicType = 'GDCC';
    elseif e == 4
        Spec.DynamicType = 'AGDCC';
    end
    f = menu('ARCH-Term des DCC','1','2','3');
    Spec.dccP = f;
    g = menu('GARCH-Term des DCC','1','2','3');
    Spec.dccQ = g;
    if e == 2 || e == 4
        h = menu('Asymmetric-Term des DCC','1');
        if h == 1
            Spec.dccG=1;
        end
    else Spec.dccG=0;
    end

elseif a == 2
    Spec.ModelType = 'MultiCopula';
    aa = menu('Marginalverteilung?','Pareto','Empirisch','bekannt');
    if aa == 1
        Spec.tails = 'pareto';
    elseif aa == 2
        Spec.tails = 'empirical';
    else
        Spec.tails = [];
    end

    b = menu('Spezifizieren der Verteilung','Gauss','t');
    if b == 1
        Spec.CopulaType = 'Gauss';
        Spec.family{1}='gaussian';
        e = menu('Spezifizieren der Dynamik','DCC','ADCC','GDCC','AGDCC');
        if e == 1
            Spec.DynamicType = 'DCC';
            f = menu('ARCH-Term des DCC','1','2','3');
            if f == 1
                Spec.dccP = 1;
            elseif f==2
                Spec.dccP = 1;
            elseif f == 3
                Spec.dccP = 1;
            end
            g = menu('GARCH-Term des DCC','1','2','3');
            if g == 1
                Spec.dccQ = 1;
            elseif g==2
                Spec.dccQ = 1;
            elseif g == 3
                Spec.dccQ = 1;
            end
            Spec.dccG=0;
        elseif e == 2
            Spec.DynamicType = 'ADCC';
            f = menu('ARCH-Term des ADCC','1','2','3');
            if f == 1
                Spec.dccP = 1;
            elseif f==2
                Spec.dccP = 1;
            elseif f == 3
                Spec.dccP = 1;
            end
            g = menu('GARCH-Term des ADCC','1','2','3');
            if g == 1
                Spec.dccQ = 1;
            elseif g==2
                Spec.dccQ = 1;
            elseif g == 3
                Spec.dccQ = 1;
            end
            h = menu('Asymmetric-Term des ADCC','1');
            if h == 1
                Spec.dccG=1;
            end
        elseif e == 3
            Spec.DynamicType = 'GDCC';
            f = menu('ARCH-Term des GDCC','1','2','3');
            if f == 1
                Spec.dccP = 1;
            elseif f==2
                Spec.dccP = 1;
            elseif f == 3
                Spec.dccP = 1;
            end
            g = menu('GARCH-Term des GDCC','1','2','3');
            if g == 1
                Spec.dccQ = 1;
            elseif g==2
                Spec.dccQ = 1;
            elseif g == 3
                Spec.dccQ = 1;
            end
            Spec.dccG=0;
        elseif e == 4
            Spec.DynamicType = 'AGDCC';
            f = menu('ARCH-Term des AGDCC','1','2','3');
            if f == 1
                Spec.dccP = 1;
            elseif f==2
                Spec.dccP = 1;
            elseif f == 3
                Spec.dccP = 1;
            end
            g = menu('GARCH-Term des AGDCC','1','2','3');
            if g == 1
                Spec.dccQ = 1;
            elseif g==2
                Spec.dccQ = 1;
            elseif g == 3
                Spec.dccQ = 1;
            end
            h = menu('Asymmetric-Term des AGDCC','1');
            if h == 1
                Spec.dccG=1;
            end
        end
    elseif b == 2
        Spec.CopulaType = 't';
        Spec.family{1}='t';
        e = menu('Spezifizieren der Dynamik','DCC','ADCC','GDCC','AGDCC');
        if e == 1
            Spec.DynamicType = 'DCC';
            f = menu('ARCH-Term des DCC','1','2','3');
            if f == 1
                Spec.dccP = 1;
            elseif f==2
                Spec.dccP = 1;
            elseif f == 3
                Spec.dccP = 1;
            end
            g = menu('GARCH-Term des DCC','1','2','3');
            if g == 1
                Spec.dccQ = 1;
            elseif g==2
                Spec.dccQ = 1;
            elseif g == 3
                Spec.dccQ = 1;
            end
            Spec.dccG = 0;
        elseif e == 2
            Spec.DynamicType = 'ADCC';
            f = menu('ARCH-Term des DCC','1','2','3');
            if f == 1
                Spec.dccP = 1;
            elseif f==2
                Spec.dccP = 1;
            elseif f == 3
                Spec.dccP = 1;
            end
            g = menu('GARCH-Term des DCC','1','2','3');
            if g == 1
                Spec.dccQ = 1;
            elseif g==2
                Spec.dccQ = 1;
            elseif g == 3
                Spec.dccQ = 1;
            end
            h = menu('Asymmetric-Term des AGDCC','1');
            if h == 1
                Spec.dccG=1;
            end
        elseif e == 3
            Spec.DynamicType = 'GDCC';
            f = menu('ARCH-Term des GDCC','1','2','3');
            if f == 1
                Spec.dccP = 1;
            elseif f==2
                Spec.dccP = 1;
            elseif f == 3
                Spec.dccP = 1;
            end
            g = menu('GARCH-Term des GDCC','1','2','3');
            if g == 1
                Spec.dccQ = 1;
            elseif g==2
                Spec.dccQ = 1;
            elseif g == 3
                Spec.dccQ = 1;
            end
            Spec.dccG=0;
        elseif e == 4
            Spec.DynamicType = 'AGDCC';
            f = menu('ARCH-Term des AGDCC','1','2','3');
            if f == 1
                Spec.dccP = 1;
            elseif f==2
                Spec.dccP = 1;
            elseif f == 3
                Spec.dccP = 1;
            end
            g = menu('GARCH-Term des AGDCC','1','2','3');
            if g == 1
                Spec.dccQ = 1;
            elseif g==2
                Spec.dccQ = 1;
            elseif g == 3
                Spec.dccQ = 1;
            end
            h = menu('Asymmetric-Term des AGDCC','1');
            if h == 1
                Spec.dccG=1;
            end
        end
    end
elseif a == 3
    Spec.ModelType = 'VineCopula';
    aa = menu('Paretotails/empirische Margins?','Pareto','Empirisch','nein');
    if aa == 1
        Spec.tails = 'pareto';
    elseif aa == 2
        Spec.tails = 'empirical';
    else
        Spec.tails = [];
    end
elseif a == 4
    Spec.ModelType = 'MultiMixCopula';
    aa = menu('Marginalverteilung','Pareto','Empirisch','bekannt');
    if aa == 1
        Spec.tails = 'pareto';
    elseif aa == 2
        Spec.tails = 'empirical';
    else
        Spec.tails = [];
    end
    b = menu('Wähle 1. Copula-Familie für die Mixture','Gaussian','t','Clayton','Gumbel');
    if b == 1
        Spec.family{1}='gaussian';
        c = menu('Wähle 2. Copula-Familie für die Mixture','Clayton','Gumbel','Rotated Clayton');
        if c == 1
            Spec.family{2} = 'clayton';
        elseif c == 2
            Spec.family{2} = 'gumbel';
        elseif c ==3
            Spec.family{2} = 'rotclayton';
        end
        d = menu('Wähle die dynamische Spezifikation der Gauß-Copula','DCC','ADCC','GDCC','AGDCC');
        if d == 1
            Spec.Dynamic='DCC';
        elseif d == 2
            Spec.Dynamic = 'ADCC';
        elseif d == 3
            Spec.Dynamic = 'GDCC';
        elseif d == 4
            Spec.Dynamic = 'AGDCC';
        end
    elseif b == 2
        Spec.family{1}='t';
        c = menu('Wähle 2. Copula-Familie für die Mixture','Clayton','Gumbel','Rotated Clayton');
        if c == 1
            Spec.family{2} = 'clayton';
        elseif c == 2
            Spec.family{2} = 'gumbel';
        elseif c ==3
            Spec.family{2} = 'rotclayton';
        end
        d = menu('Wähle die dynamische Spezifikation der t-Copula','DCC','ADCC','GDCC','AGDCC');
        if d == 1
            Spec.Dynamic='DCC';
        elseif d == 2
            Spec.Dynamic = 'ADCC';
        elseif d == 3
            Spec.Dynamic = 'GDCC';
        elseif d == 4
            Spec.Dynamic = 'AGDCC';
        end
    elseif b == 3
        Spec.family{1}='Clayton';
        menu('Es wird automatisch die Gumbel als 2. Copula-Familie gewählt','Gumbel')
        Spec.family{2}='Gumbel';
    elseif b == 4
        Spec.family{1}='Gumbel';
        menu('Es wird automatisch die Clayton-Copula als 2. Copula-Familie gewählt','Clayton')
        Spec.family{2}='Clayton';
    end
elseif a==5
    Spec.ModelType = 'MultiMixCopula';
    aa = menu('Paretotails/empirische Margins?','Pareto','Empirisch','nein');
    if aa == 1
        Spec.tails = 'pareto';
    elseif aa == 2
        Spec.tails = 'empirical';
    else
        Spec.tails = [];
    end
    b = menu('Wähle multivariate archimedische Copula aus','Clayton','Gumbel','Rotated Clayton');
    if b==1
        Spec.family{1}='clayton';
    elseif b==2
        Spec.family{1}='gumbel';
    elseif b==3
        Spec.family{1}='rotclayton';
    end
    Spec.Dynamic=[]; %multivariate archimedische Copulas werden generell nach AR(1) geschätzt
elseif a==6
    Spec.ModelType = 'DVine-Mix';
    aa=menu('Schätzung von Standardfehlern','ja','nein');
    if aa==1
        Spec.CopStat='an';
    elseif aa==2
        Spec.CopStat='aus';
    end
elseif a==7
    aaa = menu('Window oder Pearson','W','P');
    if aaa == 1
        Spec.Sim='Window';
    elseif aaa == 2
        Spec.Sim = 'Pearson';
    end
    aa = menu('Fensterlänge?','100','200','300');
    if aa == 1
        Spec.window = 100;
    elseif aa == 2
        Spec.window = 200;
    elseif aa == 3
        Spec.window = 300;
    end
    bb = menu('Gauss oder Copula Marginalmodelle','Gauss','Copula');
    if bb == 1
        Spec.ModelType='HistSim_Gauss';
    else
        Spec.ModelType='HistSim_Copula';
    end
elseif a==8
    Spec.ModelType='DeltaNormal';
    Spec.ForecastNumb = input('Anzahl der Vorhersageperioden:','s');
    Spec.ForecastNumb = str2double(Spec.ForecastNumb);
elseif a==9
    Spec.PortOpt = 'on';
    Spec.ModelType=[];
    fff = menu('Konfidenzniveau','.99','.95','.90');
    if fff == 1
        Spec.beta = .99;
    elseif fff == 2
        Spec.beta = .95;
    elseif fff == 3
        Spec. beta = .90;
    end
    UB = input('Upper Bound der einzelnen Gewichte:','s');
    Spec.UB = str2double(UB);
    LB = input('Lower Bound der einzelnen Gewichte:','s');
    Spec.LB = str2double(LB);
    Spec.ForecastNumb = input('Anzahl der Vorhersageperioden:','s');
    Spec.ForecastNumb = str2double(Spec.ForecastNumb);
    zt = input('In wieviele Zeiträume soll das Backtest-sample eingeteilt werden:','s');
    zt = str2double(zt);
    arp = input('Maximale AR-Laglänge der univariten GARCH-Modelle:','s');
    arp = str2double(arp);
    %disp('Beachte: Der Forecast darf nur bis zum Zeitpunkt t-1 gehen')
    %Spec.ForecastStart = input('Startzeitpunkt der Vorhersage:','s');
    disp('Der Startzeitpunkt für den Backtest wird so berechnet, dass genau bis zum bis zum Zeitpunkt t-1 vorhersagen gemachte werden')
    Spec.ForecastStart = t-arp-zt*Spec.ForecastNumb;
    return
end

c = menu('Über welchen Zeitraum soll Modell geschätzt werden?','Gesamt','Backtest');
if c == 1
    Spec.purpose = 'full';
elseif c == 2
    Spec.purpose = 'backtest';
    d = menu('Anzahl der Simulationen für Monte-Carlo VaR','1000','5000','10000');
    if d==1
        Spec.SimNumb = 1000;
    elseif d==2
        Spec.SimNumb = 5000;
    elseif d==3
        Spec.SimNumb = 10000;
    end
    fprintf('Länge des gesamten Datensatzes: %d\n', t);
    Spec.ForecastNumb = input('Anzahl der Vorhersageperioden:','s');
    Spec.ForecastNumb = str2double(Spec.ForecastNumb);
    zt = input('In wieviele Zeiträume soll das Backtest-sample eingeteilt werden:','s');
    zt = str2double(zt);
    arp = input('Maximale AR-Laglänge der univariten GARCH-Modelle:','s');
    arp = str2double(arp);
    %disp('Beachte: Der Forecast darf nur bis zum Zeitpunkt t-1 gehen')
    %Spec.ForecastStart = input('Startzeitpunkt der Vorhersage:','s');
    disp('Der Startzeitpunkt für den Backtest wird so berechnet, dass genau bis zum bis zum Zeitpunkt t-1 vorhersagen gemachte werden')
    Spec.ForecastStart = t-arp-zt*Spec.ForecastNumb;
end

d = menu('Schätzung von univariaten GARCH-Modellen?','ja','nein');
if d == 1
    Spec.univariate = 'on';
    e = menu('Spezifizieren der AR-laglänge','1','2','3','4','5');
    if e ==1
        Spec.arlag = 1;
    elseif e == 2
        Spec.arlag = 2;
    elseif e == 3
        Spec.arlag = 3;
    elseif e == 4
        Spec.arlag = 4;
    elseif e == 5
        Spec.arlag = 5;
    end
    f = menu('Schätzen mit Konstante','ja','nein');
    if f == 1
        Spec.const=1;
    elseif f == 2
        Spec.const=0;
    end
elseif d == 2
    Spec.univariate = 'off';
    % alle Modelle wurden mit Konstante geschätzt
    Spec.const=1;
end

if a<6 %D-Vine Mix Std werden saparat bestimmt
    j = menu('Schätzen von Standardfehlern','ja','nein');
    if j==1
        Spec.stderrors='an';
    else
        Spec.stderrors='aus';
    end
end

% alle GARCH-Modelle werden nur (1,1) geschätzt
Spec.archP=1;
Spec.garchQ=1;
% alle Modelle werden mit Konstante geschätzt
Spec.const=1;

% univariaten BAcktest-Modelle wurden mit 25 Tagen vorhersage geschätzt;
% univariate und multivariate Vorhersgaeperioden weichen voneindader abe
ee = menu('Sollen univariate GARCH für den Backtest geschätzt werden?','ja','nein');
if ee ==1
    Spec.uniBacktest='on';
    eee = menu('Wie lang soll der jeweilige Backtest-Zeitraum sein?','750','250','150','125','25');
    if eee == 1
        Spec.uniforecastP = 750;
    elseif eee == 2
        Spec.uniforecastP = 250;
    elseif eee == 3
        Spec.uniforecastP = 150;
    elseif eee == 4
        Spec.uniforecastP = 125;
    elseif eee == 5
        Spec.uniforecastP = 25;
    end
else
    Spec.uniBacktest='off';
    eee = menu('Wie lang ist der jeweilige Backtest-Zeitraum der univariaten GARCH-Modelle?','750','250','150','125','25');
    if eee == 1
        Spec.uniforecastP = 750;
    elseif eee == 2
        Spec.uniforecastP = 250;
    elseif eee == 3
        Spec.uniforecastP = 150;
    elseif eee == 4
        Spec.uniforecastP = 125;
    elseif eee == 5
        Spec.uniforecastP = 25;
    end
end



