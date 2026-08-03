% Die Funktion simuliert Mixture Copulas. Hierzu werden die bedingte
% Inversionsmethode für die Gauss und t-Copula angewandt. Die archimedische
% Copulas werden mit dem Verfahren von Marshall und Olkin simuliert.
%
% INPUT:
% family: die Familien der Mixture Copula werden als Cell-Struktur übergeben
% Coeff: die Koeffizienten der einzelnen Copulafamilien werden übergeben.
% Bei der t-Copula wird in Coeff{i}{1} die Korrelationsmatrix und in
% Coeff{i}{2} der Freiheitsgrad übergeben.
% Die Parameter müssen als Cell-Struktur übergeben werden.
% weight: Die Gewichte der Mixture Copula werden als Double-Struktur
% übergeben
% dim: die Dimension der Simulation; dim1 = Länge, dim2 = Breite
%
%   Author: Valentin Braun, Martin Grziska
%   Phd Student in finance Goethe Universität,
%   Ludwig-Maximilians-Universit�t M�nchen
function sim = copulamixrnd(family, Coeff, weight, dim2, dim1)

family = lower(family);
weight = weight(:)';
nc = numel([str2double(family)]); % Anzahl der Copulas in der Mixture Struktur

% Prüfen dass Familien Struktur als Cell-Struktur übergeben wird
if ~iscell(family)
    family = {family};
end

% Prüfen ob zumindest 2 Zeitreihen modelliert werden sollen
if dim2<2
    error('Copulas müssen min. 2 Zeitreihen modellieren');
end

% Prüfen dass sich die Gewichte zu genau 1 aufaddieren
if round(10000.*sum(weight, 2)) ~= 10000
    %     error('die Gewichte der Mixture Copula müssen sich zu genau 1 aufaddieren');
    weight = 1./sum(weight).*weight;
end

% gleichverteilte unabhängige Zufallszahlen simulieren. Platzhalter
% schaffen
u = zeros(dim1,dim2,nc);

% % Das simulierte
% % Datenset bildet die Grundlage für alle Copula Simulationen. Hierdurch
% % wird sichergestellt dass innerhalb der Mixture Copula auf dieselbe
% % Grundlage zurückgegriffen wird.
% v = rand(dim1, dim2);

for i = 1:nc
    switch family{i}
        
        case 'gaussian'
            % Coeff beinhaltet die Korrelationsmatrix
            % Prüfen dass die Copulaparameter zur Dimension passen
            [s1 s2] = size(Coeff{i});
            if dim2 ~= s1 || dim2 ~= s2
                error('Koeffizienten passen nicht zur Dimension');
            end
            % Prüfen dass Coeff eine Korrelationsmatrix ist
            if max(max(Coeff{i}))>1 || min(min(Coeff{i}))<-1
                error('Die implementierte Matrix ist keine Korrelationsmatrix');
            end
            % Simulieren einer Gauss Copula
            u(:,:,i) = copularnd_grm('gaussian', Coeff{i}, dim1);
            
        case 't'
            % Coeff beinhaltet die Korrelationsmatrix und den Freiheitsgrad
            % Prüfen dass die Copulaparameter zur Dimension passen
            [s1 s2] = size(Coeff{i}{1});
            if dim2 ~= s1 || dim2 ~= s2
                error('Koeffizienten passen nicht zur Dimension');
            end
            % Prüfen dass Coeff eine Korrelationsmatrix ist
            if max(max(Coeff{i}{1}))>1 || min(min(Coeff{i}{1}))<-1
                error('Die implementierte Matrix ist keine Korrelationsmatrix');
            end
            % Prüfen dass genau ein Freiheitsgrad pbergeben wird
            if size(Coeff{i}{2})*ones(2,1) ~= 2
                error('Es muss bei der t-Copula genau 1 Freiheitsgrad übergeben werden');
            end
            % Simulieren einer t-Copula
            u(:,:,i) = copularnd_grm('t', Coeff{i}{1}, Coeff{i}{2}, dim1);
            
        case {'clayton' 'gumbel' 'frank' 'rotclayton'}
            % Prüfen dass der Copula Parameter im multidimensionalen Fall pos. ist
            if dim2>2
                switch family{i}
                    case {'clayton' 'frank' 'rotclayton'}
                        if Coeff{i} < 1e-3
                            %                             error('Im multidimensionalen Fall muss der archimedische Frank, Clayton Copula Parameter > 1e-3 sein');
                            Coeff{i} = 1e-3;
                        end
                    case 'gumbel'
                        if Coeff{i} <= 1
                            %                             error('Im multidimensionalen Fall muss der archimedische Gumbel Copula Parameter > 1 sein');
                            Coeff{i} = 1+1e-3;
                        end
                end
            end
            % Prüfen ob die Dimensionsgrenzen für archimedische Copulas eingehalten werden
            if dim2<2 || dim2 > 14
                error('archimedische Copulas können min. 2 Zeitreihen und max. 14 Zeitreihen modellieren');
            end
            % Coeff beinhaltet den CopulaParameter
            [s1 s2] = size(Coeff{i});
            % Prüfen dass nur genau 1 Copula Parameter für die
            % archimedische Copula übergeben wird
            if s1~=1 || s2~=1
                error('archimedische Copulas besitzen immer nur genau 1 Copula Parameter');
            end
            %             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Simulieren einer archimedischen Copula
            %             u(:,:,i) = copularndarchimedianmultivariat(family{i}, dim2, dim1, Coeff{i});
            u(:,:,i) = copularndMultiArchimed(family{i},Coeff{i},dim1,dim2);
            % % % % % % %
            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
    end
end

% Die Mixture Copula wird entsprechend dem Verfahren von Hong, Tu, Zhou
% (2007) konstruiert. Hierzu verwende ich eine Multinominalverteilung für
% die Gewichte der Mixture Copula konstruiert. Hierdurch wird
% sichergestellt dass die Copulas ihren Gewichten entsprechend oft gezogen
% werden. Wenn eine Copula gezogen wird, modelliert diese die
% Abhängigkeitsstruktur der simulierten Daten. Die resultierenden Daten
% sind Mixture Copula Simulationen.
weight = 1./sum(weight).*weight; % Es wird sichergestellt dass die Gewichte sich genau zu 1 aufaddieren
drawCopula = mnrnd(1, weight, dim1);
% Gewichte werden in Dimensionen eingeteilt. Jede Dimension repräsentiert
% eine Copula der Mixture Struktur.
simCopula = permute(drawCopula, [1 3 2]);
simCopula =  repmat(simCopula, 1, dim2); % Copula Ziehung für Hadamard Multiplikation vorbereiten
% Entsprechend der Copula Gewichte werden die Daten von den jeweiligen
% Copulas simuliert. Bsp.: 30% Clayton, 70% Gauss -> 30% der Daten stammen
% aus Simulationen mit der Clayton Copula und 70% wurden mit der Gauss
% Copula modelliert.
uTotal = simCopula.* u;
% Aufaddieren der gezogenen Simulation über die Dimensionen
% erzeugt Mixture Copula Simulationen
sim = sum(uTotal, 3);




