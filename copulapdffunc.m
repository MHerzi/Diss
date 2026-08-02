% Copula Density Funktionsterm
%
%   Author: Valentin Braun
%   Phd Student in finance Goethe Universität

function varargout = copulapdffunc(family, DatasetSize); 

% Familie der Archimedian Copulas definieren und bei falschen Input
% Parametern Fehlermeldung ausgeben
if ischar(family)
    families = {'gaussian', 't', 'clayton','frank','gumbel'};

    i = strmatch(lower(family), families);
    if numel(i) > 1
        error('stats:copulafit:InvalidFamily', ...
              'Ambiguous copula family: ''%s''.',family);
    elseif numel(i) == 1
        family = families{i};
    else
        error('stats:copulafit:InvalidFamily', ...
              'Unrecognized copula family: ''%s''',family);
    end
else
    error('stats:copulafit:InvalidFamily', ...
          'FAMILY must be a copula family name.');
end

% Inputargumente aus copulafitarchimedianmultivariat Funktion verarbeiten
if nargin ~= 2
    error('Wrong Number of Input Arguments')
end

% Überprüfen ob Anzahl der Indices korrekt ist; Beschränkung auf 10 Indices
if DatasetSize <2 || DatasetSize > 10
    error('Matlab: Wrong Number of indices')
end

% Auswahl der übergebenen Copula Familie und der korrekten Optimierungsfunktion 
switch family   
    case {'gaussian', 't'}
        gdiff = str2func('copulapdf');
        
    case 'clayton'
    % Auswahl der Dichtefunktion unter Berücksichtigung der
        % Anzahl der Indices
        
        % Variablen als numerische Variablen characterisieren
        clear v1 v2 v3 v4 v5 v6 v7 v8 v9 v10  t
        syms v1 v2 v3 v4 v5 v6 v7 v8 v9 v10  t
        switch DatasetSize
            case 2
                g = 1/(v1^(-t)+v2^(-t) - (DatasetSize-1))^(1/t); % Aufpassen auf n-1 term innerhalb der Klammer
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
            case 3
                g = 1/(v1^(-t)+v2^(-t)+v3^(-t) - (DatasetSize-1))^(1/t); % Aufpassen auf n-1 term innerhalb der Klammer
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
            case 4
                g = 1/(v1^(-t)+v2^(-t)+v3^(-t)+v4^(-t) - (DatasetSize-1))^(1/t); % Aufpassen auf n-1 term innerhalb der Klammer
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
            case 5
                g = 1/(v1^(-t)+v2^(-t)+v3^(-t)+v4^(-t)+v5^(-t) - (DatasetSize-1))^(1/t); % Aufpassen auf n-1 term innerhalb der Klammer
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
            case 6
                g = 1/(v1^(-t)+v2^(-t)+v3^(-t)+v4^(-t)+v5^(-t)+v6^(-t) - (DatasetSize-1))^(1/t); % Aufpassen auf n-1 term innerhalb der Klammer
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
            case 7
                g = 1/(v1^(-t)+v2^(-t)+v3^(-t)+v4^(-t)+v5^(-t)+v6^(-t)+v7^(-t) - (DatasetSize-1))^(1/t); % Aufpassen auf n-1 term innerhalb der Klammer
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
                gdiff = diff(gdiff, v7); % 7 Variablen
            case 8
                g = 1/(v1^(-t)+v2^(-t)+v3^(-t)+v4^(-t)+v5^(-t)+v6^(-t)+v7^(-t)+v8^(-t) - (DatasetSize-1))^(1/t); % Aufpassen auf n-1 term innerhalb der Klammer
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
                gdiff = diff(gdiff, v7); % 7 Variablen
                gdiff = diff(gdiff, v8); % 8 Variablen
            case 9
                g = 1/(v1^(-t)+v2^(-t)+v3^(-t)+v4^(-t)+v5^(-t)+v6^(-t)+v7^(-t)+v8^(-t)+v9^(-t) - (DatasetSize-1))^(1/t); % Aufpassen auf n-1 term innerhalb der Klammer
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
                gdiff = diff(gdiff, v7); % 7 Variablen
                gdiff = diff(gdiff, v8); % 8 Variablen
                gdiff = diff(gdiff, v9); % 9 Variablen
            case 10
                g = 1/(v1^(-t)+v2^(-t)+v3^(-t)+v4^(-t)+v5^(-t)+v6^(-t)+v7^(-t)+v8^(-t)+v9^(-t)+v10^(-t) - (DatasetSize-1))^(1/t); % Aufpassen auf n-1 term innerhalb der Klammer
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
                gdiff = diff(gdiff, v7); % 7 Variablen
                gdiff = diff(gdiff, v8); % 8 Variablen
                gdiff = diff(gdiff, v9); % 9 Variablen
                gdiff = diff(gdiff, v10); % 10 Variablen
        end    
                
    case 'gumbel'
    % Auswahl der Dichtefunktion unter Berücksichtigung der
        % Anzahl der Indices
        
        % Variablen als numerische Variablen characterisieren
        clear v1 v2 v3 v4 v5 v6 v7 v8 v9 v10  t
        syms v1 v2 v3 v4 v5 v6 v7 v8 v9 v10  t
        switch DatasetSize
            case 2
                g = exp(  -1*(  (  (-log(v1))^t + (-log(v2))^t   )^(1/t)  )  );   % Aufpassen auf Klammersetzen
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
            case 3
                g = exp(  -1*(  (  (-log(v1))^t + (-log(v2))^t + (-log(v3))^t    )^(1/t)  )  ) ;  % Aufpassen auf Klammersetzen
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
            case 4
                g = exp(  -1*(  (  (-log(v1))^t + (-log(v2))^t + (-log(v3))^t + (-log(v4))^t   )^(1/t)  )  );   % Aufpassen auf Klammersetzen
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
            case 5
                g = exp(  -1*(  (  (-log(v1))^t + (-log(v2))^t + (-log(v3))^t + (-log(v4))^t + (-log(v5))^t    )^(1/t)  )  );   % Aufpassen auf Klammersetzen
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
            case 6
                g = exp(  -1*(  (  (-log(v1))^t + (-log(v2))^t + (-log(v3))^t + (-log(v4))^t + (-log(v5))^t + (-log(v6))^t    )^(1/t)  )  );   % Aufpassen auf Klammersetzen
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
            case 7
                g = exp(  -1*(  (  (-log(v1))^t + (-log(v2))^t + (-log(v3))^t + (-log(v4))^t + (-log(v5))^t + (-log(v6))^t + (-log(v7))^t    )^(1/t)  )  );   % Aufpassen auf Klammersetzen
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
                gdiff = diff(gdiff, v7); % 7 Variablen
            case 8
                g = exp(  -1*(  (  (-log(v1))^t + (-log(v2))^t + (-log(v3))^t + (-log(v4))^t + (-log(v5))^t + (-log(v6))^t + (-log(v7))^t + (-log(v8))^t    )^(1/t)  )  );   % Aufpassen auf Klammersetzen
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
                gdiff = diff(gdiff, v7); % 7 Variablen
                gdiff = diff(gdiff, v8); % 8 Variablen
            case 9
                g = exp(  -1*(  (  (-log(v1))^t + (-log(v2))^t + (-log(v3))^t + (-log(v4))^t + (-log(v5))^t + (-log(v6))^t + (-log(v7))^t + (-log(v8))^t + (-log(v9))^t    )^(1/t)  )  );   % Aufpassen auf Klammersetzen
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
                gdiff = diff(gdiff, v7); % 7 Variablen
                gdiff = diff(gdiff, v8); % 8 Variablen
                gdiff = diff(gdiff, v9); % 9 Variablen
            case 10
                g = exp(  -1*(  (  (-log(v1))^t + (-log(v2))^t + (-log(v3))^t + (-log(v4))^t + (-log(v5))^t + (-log(v6))^t + (-log(v7))^t + (-log(v8))^t + (-log(v9))^t + (-log(v10))^t   )^(1/t)  )  );   % Aufpassen auf Klammersetzen
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
                gdiff = diff(gdiff, v7); % 7 Variablen
                gdiff = diff(gdiff, v8); % 8 Variablen
                gdiff = diff(gdiff, v9); % 9 Variablen
                gdiff = diff(gdiff, v10); % 10 Variablen
        end    
        
    case 'frank'
    % Auswahl der Dichtefunktion unter Berücksichtigung der
        % Anzahl der Indices
        
        % Variablen als numerische Variablen characterisieren
        clear v1 v2 v3 v4 v5 v6 v7 v8 v9 v10  t
        syms v1 v2 v3 v4 v5 v6 v7 v8 v9 v10  t
        switch DatasetSize
            case 2
                g = -1/t * log( 1 + (   (exp(-v1*t)-1) * (exp(-v2*t)-1)    /((exp(-t)-1)^(DatasetSize-1))  )  );  
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
            case 3
                g = -1/t * log( 1 + (   (exp(-v1*t)-1) * (exp(-v2*t)-1) * (exp(-v3*t)-1)    /((exp(-t)-1)^(DatasetSize-1))  )  );  
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
            case 4
                g = -1/t * log( 1 + (   (exp(-v1*t)-1) * (exp(-v2*t)-1) * (exp(-v3*t)-1) * (exp(-v4*t)-1)    /((exp(-t)-1)^(DatasetSize-1))  )  );  
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
            case 5
                g = -1/t * log( 1 + (   (exp(-v1*t)-1) * (exp(-v2*t)-1) * (exp(-v3*t)-1) * (exp(-v4*t)-1) * (exp(-v5*t)-1)    /((exp(-t)-1)^(DatasetSize-1))  )  );  
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
            case 6
                g = -1/t * log( 1 + (   (exp(-v1*t)-1) * (exp(-v2*t)-1) * (exp(-v3*t)-1) * (exp(-v4*t)-1) * (exp(-v5*t)-1) * (exp(-v6*t)-1)    /((exp(-t)-1)^(DatasetSize-1))  )  );  
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
            case 7
                g = -1/t * log( 1 + (   (exp(-v1*t)-1) * (exp(-v2*t)-1) * (exp(-v3*t)-1) * (exp(-v4*t)-1) * (exp(-v5*t)-1) * (exp(-v6*t)-1) * (exp(-v7*t)-1)    /((exp(-t)-1)^(DatasetSize-1))  )  );  
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
                gdiff = diff(gdiff, v7); % 7 Variablen
            case 8
                g = -1/t * log( 1 + (   (exp(-v1*t)-1) * (exp(-v2*t)-1) * (exp(-v3*t)-1) * (exp(-v4*t)-1) * (exp(-v5*t)-1) * (exp(-v6*t)-1) * (exp(-v7*t)-1) * (exp(-v8*t)-1)    /((exp(-t)-1)^(DatasetSize-1))  )  );  
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
                gdiff = diff(gdiff, v7); % 7 Variablen
                gdiff = diff(gdiff, v8); % 8 Variablen
            case 9
                g = -1/t * log( 1 + (   (exp(-v1*t)-1) * (exp(-v2*t)-1) * (exp(-v3*t)-1) * (exp(-v4*t)-1) * (exp(-v5*t)-1) * (exp(-v6*t)-1) * (exp(-v7*t)-1) * (exp(-v8*t)-1) * (exp(-v9*t)-1)    /((exp(-t)-1)^(DatasetSize-1))  )  );  
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
                gdiff = diff(gdiff, v7); % 7 Variablen
                gdiff = diff(gdiff, v8); % 8 Variablen
                gdiff = diff(gdiff, v9); % 9 Variablen
            case 10
                g = -1/t * log( 1 + (   (exp(-v1*t)-1) * (exp(-v2*t)-1) * (exp(-v3*t)-1) * (exp(-v4*t)-1) * (exp(-v5*t)-1) * (exp(-v6*t)-1) * (exp(-v7*t)-1) * (exp(-v8*t)-1) * (exp(-v9*t)-1) * (exp(-v10*t)-1)    /((exp(-t)-1)^(DatasetSize-1))  )  );  
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
                gdiff = diff(gdiff, v7); % 7 Variablen
                gdiff = diff(gdiff, v8); % 8 Variablen
                gdiff = diff(gdiff, v9); % 9 Variablen
                gdiff = diff(gdiff, v10); % 10 Variablen
        end
end

% Einsetzen der Variablenvektoren in die bestimmten Dichtefunktionen (nur
% für multivariate Gumbel und Frank Copula)
switch lower(family)
    case {'clayton', 'gumbel', 'frank'}

        % gdiff in Vektorenform umwandeln um mit Daten rechnen zu können
        gdiff = vectorize(gdiff);
        % Löschen der numerischen Variablen
        clear v1 v2 v3 v4 v5 v6 v7 v8 v9 v10  t g
        % Konvertieren des Formel Strings in nutzbare Formel
        gdiff = inline(gdiff);
end

% Dichtefunktionsterm ausgeben
varargout{1} = gdiff;
        
