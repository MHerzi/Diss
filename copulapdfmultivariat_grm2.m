% bei archimedischen Copulas wird nur der Copulaparameter in varargin
% übergeben 
% bei der Gauss Copula wird die Korrelationsstruktur via varargin
% übergeben
% bei der t-Copula wird die Korrelationsstruktur via varargin{1} übergeben 
% und der Freiheitsgrad mit varargin{2}
% !!!!!!!!!!!!!!! Achtung !!!!!!!!!!!!!!!!!!!!!
% Im bivariaten Fall muss für die Gauss und t-Copula eine
% Korrelationsmatrix eingegeben werden. Ein einfache r
% Korrelationsparameter führt zu falschen Dichten!
%
%   Author: Valentin Braun
%   Phd Student in finance Goethe Universität
function varargout = copulapdfmultivariat_grm2(family, data, varargin) 

% Familie der Archimedian Copulas definieren und bei falschen Input
% Parametern Fehlermeldung ausgeben
if ischar(family)
    families = {'gaussian', 't', 'clayton','frank','gumbel', 'rotclayton'};

    i = strmatch(lower(family), families);
    if numel(i) > 1
        error('Multivariate Copulas bestehen nur aus einer Copulafamilie');
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

% Aufbereiten der Informationen aus varargin 
z = data; %varargin{1}(:,1:end);
DatasetSize = size(z, 2);
[T,N]=size(z);

% Überprüfen ob Anzahl der Indices korrekt ist; Beschränkung auf 10 Indices
if DatasetSize <2 || DatasetSize > 14
    error('Falsche Indexanzahl')
end

% Auswahl der übergebenen Copula Familie und der korrekten Optimierungsfunktion 
switch family   
    case 'gaussian'
        % Transformieren in normalverteilte Variablen via inverse
        % Normalverteilungsfunktion und rho bestimmen
        trdata = norminv(z);
        rho = varargin{1};
        % Überprüfen dass immer eine Korrelationsmatrix und nicht nur ein
        % Korrelationskoeffizient angegeben wird
        if numel(rho) < 4
            error('Gauss Copula benötigt eine Korrelationsmatrix. Korrelationskoeffizient ist ist nicht ausreichend');
        end
        % Bestimmen des oberen Cholesky Faktors
        [R,err] = cholcov(rho,0);
        if (err ~= 0) || any(diag(rho) ~= 1)
%             error('Rho must be symmetric and positive definite.');
            LL = NaN(T,1); % Bei Fehlermeldung wird NaN als Loglikelihood Wert übergeben
        else
            % Berechnen der Logarithmierten Dichtefunktion der Gauss Copula
            logSqrtDetRho = sum(log(diag(R)));
            LL = trdata/R;
            LL = -0.5 .* sum(LL.^2 - trdata.^2,2) - logSqrtDetRho;   
        end
        % Bestimmen der Dichte
        density.Gauss = exp(LL);
        % ausgeben der Dichte
        varargout{1} = density.Gauss;
        % ausgeben der korrelationsmatrix
        varargout{2} = rho;
        
    case 't'
        % Übergeben des Copula Parameters (DoF)
        DoF = varargin{2};
        % Übergeben des Copula Parameters (Rho)
        rho = varargin{1};
        % Überprüfen dass immer eine Korrelationsmatrix und nicht nur ein
        % Korrelationskoeffizient angegeben wird
        if numel(rho) < 4
            error('t-Copula benötigt eine Korrelationsmatrix. Korrelationskoeffizient ist ist nicht ausreichend');
        end
        % Inverse der t-Verteilung für Daten~[0,1]
        trdata = tinv(z, DoF);        
        Rt=repmat(rho,[1 1 T]);
%         % standardisierte t-Copula Loglikelihood Funktion; Chen, Fan,
%         % Patton (2004)
%         LL=zeros(T,1); 
%         for i=1:T
%             LL(i) = gammaln((DoF+N)/2) + (N-1)*gammaln(DoF/2) - N*gammaln((DoF+1)/2) - 0.5*log(det(Rt(:,:,i)));
%             LL(i) = LL(i) - (DoF+N)/2*log(1+trdata(i,:)*inv(Rt(:,:,i))*trdata(i,:)'./(DoF-2)); 
%             LL(i) = LL(i) + (DoF+1)/2*sum(log(1+(trdata(i,:).^2/(DoF-2))));
%         end
        % NICHT standardisierte t-Copula Loglikelihood Funktion
        LL=zeros(T,1); 
        for i=1:T
            LL(i) = gammaln((DoF+N)/2) + (N-1)*gammaln(DoF/2) - N*gammaln((DoF+1)/2) - 0.5*log(det(Rt(:,:,i)));
            LL(i) = LL(i) - (DoF+N)/2*log(1+trdata(i,:)*inv(Rt(:,:,i))*trdata(i,:)'./DoF); 
            LL(i) = LL(i) + (DoF+1)/2*sum(log(1+(trdata(i,:).^2/DoF)));
        end
        % t-Copula Dichte
        density.t = exp(LL);
        % ausgeben der Dichte
        varargout{1} = density.t;
        % ausgeben der Korrelationsmatrix
        varargout{2} = rho;
        
        
    case {'clayton', 'rotclayton'}
        % Übergeben des Copula Parameters
        cp = varargin{1};
        cp(cp<1e-3) = 1e-3; % Sicherstellen dass der Clayton CopParameter immer min 1e-3 beträgt
        % Auswahl der Dichtefunktion unter Berücksichtigung der
        % Anzahl der Indices
        if DatasetSize == 2
            [density.Clayton] = (cp.*(1./cp + 1))./(z(:,1).^(cp + 1).*z(:,2).^(cp + 1).*(1./z(:,1).^cp + 1./z(:,2).^cp - 1).^(1./cp + 2));
        elseif DatasetSize == 3
            [density.Clayton] = (cp.^2.*(1./cp + 1).*(1./cp + 2))./(z(:,1).^(cp + 1).*z(:,2).^(cp + 1).*z(:,3).^(cp + 1).*(1./z(:,1).^cp + 1./z(:,2).^cp + 1./z(:,3).^cp - 2).^(1./cp + 3));
        elseif DatasetSize == 4
            [density.Clayton] = (cp.^3.*(1./cp + 1).*(1./cp + 2).*(1./cp + 3))./(z(:,1).^(cp + 1).*z(:,2).^(cp + 1).*z(:,3).^(cp + 1).*z(:,4).^(cp + 1).*(1./z(:,1).^cp + 1./z(:,2).^cp + 1./z(:,3).^cp + 1./z(:,4).^cp - 3).^(1./cp + 4));
        elseif DatasetSize == 5
            [density.Clayton] = (cp.^4.*(1./cp + 1).*(1./cp + 2).*(1./cp + 3).*(1./cp + 4))./(z(:,1).^(cp + 1).*z(:,2).^(cp + 1).*z(:,3).^(cp + 1).*z(:,4).^(cp + 1).*z(:,5).^(cp + 1).*(1./z(:,1).^cp + 1./z(:,2).^cp + 1./z(:,3).^cp + 1./z(:,4).^cp + 1./z(:,5).^cp - 4).^(1./cp + 5));
        elseif DatasetSize == 6
            [density.Clayton] = (cp.^5.*(1./cp + 1).*(1./cp + 2).*(1./cp + 3).*(1./cp + 4).*(1./cp + 5))./(z(:,1).^(cp + 1).*z(:,2).^(cp + 1).*z(:,3).^(cp + 1).*z(:,4).^(cp + 1).*z(:,5).^(cp + 1).*z(:,6).^(cp + 1).*(1./z(:,1).^cp + 1./z(:,2).^cp + 1./z(:,3).^cp + 1./z(:,4).^cp + 1./z(:,5).^cp + 1./z(:,6).^cp - 5).^(1./cp + 6));
        elseif DatasetSize == 7
            [density.Clayton] = (cp.^6.*(1./cp + 1).*(1./cp + 2).*(1./cp + 3).*(1./cp + 4).*(1./cp + 5).*(1./cp + 6))./(z(:,1).^(cp + 1).*z(:,2).^(cp + 1).*z(:,3).^(cp + 1).*z(:,4).^(cp + 1).*z(:,5).^(cp + 1).*z(:,6).^(cp + 1).*z(:,7).^(cp + 1).*(1./z(:,1).^cp + 1./z(:,2).^cp + 1./z(:,3).^cp + 1./z(:,4).^cp + 1./z(:,5).^cp + 1./z(:,6).^cp + 1./z(:,7).^cp - 6).^(1./cp + 7));
        elseif DatasetSize == 8
            [density.Clayton] = (cp.^7.*(1./cp + 1).*(1./cp + 2).*(1./cp + 3).*(1./cp + 4).*(1./cp + 5).*(1./cp + 6).*(1./cp + 7))./(z(:,1).^(cp + 1).*z(:,2).^(cp + 1).*z(:,3).^(cp + 1).*z(:,4).^(cp + 1).*z(:,5).^(cp + 1).*z(:,6).^(cp + 1).*z(:,7).^(cp + 1).*z(:,8).^(cp + 1).*(1./z(:,1).^cp + 1./z(:,2).^cp + 1./z(:,3).^cp + 1./z(:,4).^cp + 1./z(:,5).^cp + 1./z(:,6).^cp + 1./z(:,7).^cp + 1./z(:,8).^cp - 7).^(1./cp + 8)) ;
        elseif DatasetSize == 9
            [density.Clayton] = (cp.^8.*(1./cp + 1).*(1./cp + 2).*(1./cp + 3).*(1./cp + 4).*(1./cp + 5).*(1./cp + 6).*(1./cp + 7).*(1./cp + 8))./(z(:,1).^(cp + 1).*z(:,2).^(cp + 1).*z(:,3).^(cp + 1).*z(:,4).^(cp + 1).*z(:,5).^(cp + 1).*z(:,6).^(cp + 1).*z(:,7).^(cp + 1).*z(:,8).^(cp + 1).*z(:,9).^(cp + 1).*(1./z(:,1).^cp + 1./z(:,2).^cp + 1./z(:,3).^cp + 1./z(:,4).^cp + 1./z(:,5).^cp + 1./z(:,6).^cp + 1./z(:,7).^cp + 1./z(:,8).^cp + 1./z(:,9).^cp - 8).^(1./cp + 9));
        elseif DatasetSize == 10
            [density.Clayton] = (cp.^9.*(1./cp + 1).*(1./cp + 2).*(1./cp + 3).*(1./cp + 4).*(1./cp + 5).*(1./cp + 6).*(1./cp + 7).*(1./cp + 8).*(1./cp + 9))./(z(:,1).^(cp + 1).*z(:,2).^(cp + 1).*z(:,3).^(cp + 1).*z(:,4).^(cp + 1).*z(:,5).^(cp + 1).*z(:,6).^(cp + 1).*z(:,7).^(cp + 1).*z(:,8).^(cp + 1).*z(:,9).^(cp + 1).*z(:,10).^(cp + 1).*(1./z(:,1).^cp + 1./z(:,2).^cp + 1./z(:,3).^cp + 1./z(:,4).^cp + 1./z(:,5).^cp + 1./z(:,6).^cp + 1./z(:,7).^cp + 1./z(:,8).^cp + 1./z(:,9).^cp + 1./z(:,10).^cp - 9).^(1./cp + 10));
        elseif DatasetSize == 11
            [density.Clayton] = (cp.^10.*(1./cp + 1).*(1./cp + 2).*(1./cp + 3).*(1./cp + 4).*(1./cp + 5).*(1./cp + 6).*(1./cp + 7).*(1./cp + 8).*(1./cp + 9).*(1./cp + 10))./(z(:,1).^(cp + 1).*z(:,10).^(cp + 1).*z(:,11).^(cp + 1).*z(:,2).^(cp + 1).*z(:,3).^(cp + 1).*z(:,4).^(cp + 1).*z(:,5).^(cp + 1).*z(:,6).^(cp + 1).*z(:,7).^(cp + 1).*z(:,8).^(cp + 1).*z(:,9).^(cp + 1).*(1./z(:,1).^cp + 1./z(:,10).^cp + 1./z(:,11).^cp + 1./z(:,2).^cp + 1./z(:,3).^cp + 1./z(:,4).^cp + 1./z(:,5).^cp + 1./z(:,6).^cp + 1./z(:,7).^cp + 1./z(:,8).^cp + 1./z(:,9).^cp - 9).^(1./cp + 11));
        elseif DatasetSize == 12
            [density.Clayton] = (cp.^11.*(1./cp + 1).*(1./cp + 2).*(1./cp + 3).*(1./cp + 4).*(1./cp + 5).*(1./cp + 6).*(1./cp + 7).*(1./cp + 8).*(1./cp + 9).*(1./cp + 10).*(1./cp + 11))./(z(:,1).^(cp + 1).*z(:,10).^(cp + 1).*z(:,11).^(cp + 1).*z(:,12).^(cp + 1).*z(:,2).^(cp + 1).*z(:,3).^(cp + 1).*z(:,4).^(cp + 1).*z(:,5).^(cp + 1).*z(:,6).^(cp + 1).*z(:,7).^(cp + 1).*z(:,8).^(cp + 1).*z(:,9).^(cp + 1).*(1./z(:,1).^cp + 1./z(:,10).^cp + 1./z(:,11).^cp + 1./z(:,12).^cp + 1./z(:,2).^cp + 1./z(:,3).^cp + 1./z(:,4).^cp + 1./z(:,5).^cp + 1./z(:,6).^cp + 1./z(:,7).^cp + 1./z(:,8).^cp + 1./z(:,9).^cp - 11).^(1./cp + 12));
        elseif DatasetSize == 13
            [density.Clayton] = (cp.^12.*(1./cp + 1).*(1./cp + 2).*(1./cp + 3).*(1./cp + 4).*(1./cp + 5).*(1./cp + 6).*(1./cp + 7).*(1./cp + 8).*(1./cp + 9).*(1./cp + 10).*(1./cp + 11).*(1./cp + 12))./(z(:,1).^(cp + 1).*z(:,10).^(cp + 1).*z(:,11).^(cp + 1).*z(:,12).^(cp + 1).*z(:,13).^(cp + 1).*z(:,2).^(cp + 1).*z(:,3).^(cp + 1).*z(:,4).^(cp + 1).*z(:,5).^(cp + 1).*z(:,6).^(cp + 1).*z(:,7).^(cp + 1).*z(:,8).^(cp + 1).*z(:,9).^(cp + 1).*(1./z(:,1).^cp + 1./z(:,10).^cp + 1./z(:,11).^cp + 1./z(:,12).^cp + 1./z(:,13).^cp + 1./z(:,2).^cp + 1./z(:,3).^cp + 1./z(:,4).^cp + 1./z(:,5).^cp + 1./z(:,6).^cp + 1./z(:,7).^cp + 1./z(:,8).^cp + 1./z(:,9).^cp - 12).^(1./cp + 13));
        elseif DatasetSize == 14
            [density.Clayton] = (cp.^13.*(1./cp + 1).*(1./cp + 2).*(1./cp + 3).*(1./cp + 4).*(1./cp + 5).*(1./cp + 6).*(1./cp + 7).*(1./cp + 8).*(1./cp + 9).*(1./cp + 10).*(1./cp + 11).*(1./cp + 12).*(1./cp + 13))./(z(:,1).^(cp + 1).*z(:,10).^cp + 1).*z(:,11).^(cp + 1).*z(:,12).^(cp + 1).*z(:,13).^(cp + 1).*z(:,14).^(cp + 1).*z(:,2).^(cp + 1).*z(:,3).^(cp + 1).*z(:,4).^(cp + 1).*z(:,5).^(cp + 1).*z(:,6).^(cp + 1).*z(:,7).^(cp + 1).*z(:,8).^(cp + 1).*z(:,9).^(cp + 1).*(1./z(:,1).^cp + 1./z(:,10).^cp + 1./z(:,11).^cp + 1./z(:,12).^cp + 1./z(:,13).^cp + 1./z(:,14).^cp + 1./z(:,2).^cp + 1./z(:,3).^cp + 1./z(:,4).^cp + 1./z(:,5).^cp + 1./z(:,6).^cp + 1./z(:,7).^cp + 1./z(:,8).^cp + 1./z(:,9).^cp - 13).^(1./cp + 14);
        end
        varargout{1} = density.Clayton;    
       
        
    case 'gumbel'
 % Auswahl der Anzahl der Dichtefunktion unter Berücksichtigung der
        % Anzahl der Indices
        cp = varargin{1};
        cp(cp<1+1e-3) = 1+ 1e-3;
        switch DatasetSize
            case 2
                [density.gumbel] = ((-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*((-log(v1)).^cp+(-log(v2)).^cp).^(2./cp-2))./(v1.*v2.*exp(((-log(v1)).^cp+(-log(v2)).^cp).^(1./cp)))-(cp.*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp).^(1./cp-2))./(v1.*v2.*exp(((-log(v1)).^cp+(-log(v2)).^cp).^(1./cp)));
                g = exp(  -1*(  (  (-log(v1))^cp + (-log(v2))^cp   )^(1/cp)  )  );   % Aufpassen auf Klammersetzen
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
            case 3
                [density.gumbel] = ((-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp).^(2./cp-2))./(v1.*v2.*v3.*exp(((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp).^(1./cp)))-(cp.*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp).^(2./cp-3))./(v1.*v2.*v3.*exp(((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp).^(1./cp)))+(cp.^2.*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(1./cp-1).*(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp).^(1./cp-3))./(v1.*v2.*v3.*exp(((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp).^(1./cp)))-(cp.*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp).^(1./cp-2))./(v1.*v2.*v3.*exp(((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp).^(1./cp)));              
            case 4
               [density.gumbel] = 1./v1./v2./v3./v4.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(4./cp-4)+cp.^2./v1./v2./v3./v4.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(1./cp-1).^2.*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(2./cp-4)-2.*cp./v1./v2./v3./v4.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(2./cp-2)+cp.^2./v1./v2./v3./v4.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(2./cp-2).*(2./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(2./cp-4)-2.*cp./v1./v2./v3./v4.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(2./cp-3)+2.*cp.^2./v1./v2./v3./v4.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(1./cp-1).*(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(1./cp-3)-cp.^3./v1./v2./v3./v4.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(1./cp-1).*(1./cp-2).*(1./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp).^(1./cp-4);
            case 5
                [density.gumbel] = 1./v1./v2./v3./v4./v5.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(4./cp-4)-cp./v1./v2./v3./v4./v5.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(4./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(4./cp-5)-cp.^3./v1./v2./v3./v4./v5.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(1./cp-1).^2.*(2./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(2./cp-5)-2.*cp./v1./v2./v3./v4./v5.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(2./cp-3)+cp.^2./v1./v2./v3./v4./v5.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(1./cp-1).^2.*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(2./cp-4)+4.*cp.^2./v1./v2./v3./v4./v5.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(1./cp-1).*(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(2./cp-2)-2.*cp.^3./v1./v2./v3./v4./v5.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(1./cp-1).^2.*(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp-3)-cp.^3./v1./v2./v3./v4./v5.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(2./cp-2).*(2./cp-3).*(2./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(2./cp-5)+4.*cp.^2./v1./v2./v3./v4./v5.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(1./cp-1).*(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(2./cp-3)-2.*cp./v1./v2./v3./v4./v5.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(2./cp-2)+3.*cp.^2./v1./v2./v3./v4./v5.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(2./cp-2).*(2./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(2./cp-4)+cp.^4./v1./v2./v3./v4./v5.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(1./cp-1).*(1./cp-2).*(1./cp-3).*(1./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp-5)-3.*cp.^3./v1./v2./v3./v4./v5.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(1./cp-1).*(1./cp-2).*(1./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp).^(1./cp-4);
            case 6
                [density.gumbel] = 1./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(4./cp-4)+2.*cp.^2./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(2./cp-2).^2.*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(4./cp-6)-3.*cp./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(4./cp-4)+3.*cp.^2./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(1./cp-1).^2.*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-4)+cp.^2./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(4./cp-4).*(4./cp-5).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(4./cp-6)+2.*cp.^4./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(1./cp-1).^2.*(1./cp-2).^2.*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-6)-2.*cp./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(4./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(4./cp-5)-cp.^3./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(1./cp-1).^3.*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-4)-2.*cp.^3./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(1./cp-1).^2.*(2./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-5)+5.*cp.^2./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(2./cp-2).*(2./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-4)-2.*cp./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-3)+cp.^4./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(1./cp-1).^2.*(2./cp-4).*(2./cp-5).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-6)-8.*cp.^3./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(1./cp-1).*(1./cp-2).*(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-3)+cp.^4./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(2./cp-2).*(2./cp-3).*(2./cp-4).*(2./cp-5).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-6)-7.*cp.^3./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(1./cp-1).*(2./cp-2).*(2./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-4)+6.*cp.^2./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(1./cp-1).*(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-2)-2.*cp.^3./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(1./cp-1).^2.*(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-3)-7.*cp.^3./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(1./cp-1).*(1./cp-2).*(1./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-2)+5.*cp.^4./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(1./cp-1).^2.*(1./cp-2).*(1./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-4)-4.*cp.^3./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(2./cp-2).*(2./cp-3).*(2./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-5)+6.*cp.^2./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(1./cp-1).*(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(2./cp-3)+4.*cp.^4./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(1./cp-1).*(1./cp-2).*(1./cp-3).*(1./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-5)-cp.^5./v1./v2./v3./v4./v5./v6.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(1./cp-1).*(1./cp-2).*(1./cp-3).*(1./cp-4).*(1./cp-5).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp).^(1./cp-6);
            case 7
               [density.gumbel] = 1./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(4./cp-4)-3.*cp./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(4./cp-4)-3.*cp./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(4./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(4./cp-5)+4.*cp.^2./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(2./cp-2).^2.*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(4./cp-6)-2.*cp.^3./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(2./cp-2).^2.*(4./cp-6).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(4./cp-7)-cp.^3./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(4./cp-4).*(4./cp-5).*(4./cp-6).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(4./cp-7)-2.*cp.^5./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).^2.*(1./cp-2).^2.*(2./cp-6).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-7)-cp.^3./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).^3.*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-4)+5.*cp.^2./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).*(4./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(4./cp-5)+3.*cp.^4./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).^3.*(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-4)-5.*cp.^3./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(2./cp-2).^2.*(2./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-4)-3.*cp./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(4./cp-4)+3.*cp.^2./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).^2.*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-4)+3.*cp.^2./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(4./cp-4).*(4./cp-5).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(4./cp-6)+3.*cp.^4./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).^3.*(2./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-5)+4.*cp.^4./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).^2.*(1./cp-2).^2.*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-6)+9.*cp.^2./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).*(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(4./cp-4)-9.*cp.^3./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).^2.*(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-4)-5.*cp.^3./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).^2.*(2./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-5)+3.*cp.^4./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).^2.*(2./cp-4).*(2./cp-5).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-6)-8.*cp.^3./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).^2.*(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-2)-5.*cp.^5./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).^2.*(1./cp-2).^2.*(1./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-4)-9.*cp.^3./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(2./cp-2).*(2./cp-3).*(2./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-5)-cp.^5./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).^2.*(2./cp-4).*(2./cp-5).*(2./cp-6).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-7)+8.*cp.^2./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).*(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-3)+7.*cp.^2./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(2./cp-2).*(2./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-4)-cp.^5./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(2./cp-2).*(2./cp-3).*(2./cp-4).*(2./cp-5).*(2./cp-6).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-7)+11.*cp.^4./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).*(1./cp-2).*(1./cp-3).*(1./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-5).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-2)-9.*cp.^5./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).^2.*(1./cp-2).*(1./cp-3).*(1./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-5)+11.*cp.^4./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).*(2./cp-2).*(2./cp-3).*(2./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-5)-20.*cp.^3./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).*(1./cp-2).*(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-3)+15.*cp.^4./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).*(1./cp-2).*(1./cp-3).*(2./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-3)+5.*cp.^4./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(2./cp-2).*(2./cp-3).*(2./cp-4).*(2./cp-5).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-6)-13.*cp.^3./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).*(2./cp-2).*(2./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-4)+15.*cp.^4./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).*(1./cp-2).*(2./cp-2).*(2./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-4)-13.*cp.^3./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).*(1./cp-2).*(1./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-4).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(2./cp-2)+7.*cp.^4./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).^2.*(1./cp-2).*(1./cp-3).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-2).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-4)-5.*cp.^5./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).*(1./cp-2).*(1./cp-3).*(1./cp-4).*(1./cp-5).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-1).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-6)+cp.^6./v1./v2./v3./v4./v5./v6./v7.*exp(-((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp)).*(-log(v1)).^(cp-1).*(-log(v2)).^(cp-1).*(-log(v3)).^(cp-1).*(-log(v4)).^(cp-1).*(-log(v5)).^(cp-1).*(-log(v6)).^(cp-1).*(-log(v7)).^(cp-1).*(1./cp-1).*(1./cp-2).*(1./cp-3).*(1./cp-4).*(1./cp-5).*(1./cp-6).*((-log(v1)).^cp+(-log(v2)).^cp+(-log(v3)).^cp+(-log(v4)).^cp+(-log(v5)).^cp+(-log(v6)).^cp+(-log(v7)).^cp).^(1./cp-7);
            case 8
                g = exp(  -1*(  (  (-log(v1))^cp + (-log(v2))^cp + (-log(v3))^cp + (-log(v4))^cp + (-log(v5))^cp + (-log(v6))^cp + (-log(v7))^cp + (-log(v8))^cp    )^(1/cp)  )  );   % Aufpassen auf Klammersetzen
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
                g = exp(  -1*(  (  (-log(v1))^cp + (-log(v2))^cp + (-log(v3))^cp + (-log(v4))^cp + (-log(v5))^cp + (-log(v6))^cp + (-log(v7))^cp + (-log(v8))^cp + (-log(v9))^cp    )^(1/cp)  )  );   % Aufpassen auf Klammersetzen
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
                g = exp(  -1*(  (  (-log(v1))^cp + (-log(v2))^cp + (-log(v3))^cp + (-log(v4))^cp + (-log(v5))^cp + (-log(v6))^cp + (-log(v7))^cp + (-log(v8))^cp + (-log(v9))^cp + (-log(v10))^cp   )^(1/cp)  )  );   % Aufpassen auf Klammersetzen
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
            case 11
                g = exp(  -1*(  (  (-log(v1))^cp + (-log(v2))^cp + (-log(v3))^cp + (-log(v4))^cp + (-log(v5))^cp + (-log(v6))^cp + (-log(v7))^cp + (-log(v8))^cp + (-log(v9))^cp + (-log(v10))^cp + (-log(v11))^cp    )^(1/cp)  )  );   % Aufpassen auf Klammersetzen
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
                gdiff = diff(gdiff, v11); % 11 Variablen
            case 12
                g = exp(  -1*(  (  (-log(v1))^cp + (-log(v2))^cp + (-log(v3))^cp + (-log(v4))^cp + (-log(v5))^cp + (-log(v6))^cp + (-log(v7))^cp + (-log(v8))^cp + (-log(v9))^cp + (-log(v10))^cp + (-log(v11))^cp + (-log(v12))^cp      )^(1/cp)  )  );   % Aufpassen auf Klammersetzen
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
                gdiff = diff(gdiff, v11); % 11 Variablen
                gdiff = diff(gdiff, v12); %12 Variablen
            case 13
                 g = exp(  -1*(  (  (-log(v1))^cp + (-log(v2))^cp + (-log(v3))^cp + (-log(v4))^cp + (-log(v5))^cp + (-log(v6))^cp + (-log(v7))^cp + (-log(v8))^cp + (-log(v9))^cp + (-log(v10))^cp + (-log(v11))^cp + (-log(v12))^cp + (-log(v13))^cp      )^(1/cp)  )  );   % Aufpassen auf Klammersetzen
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
                gdiff = diff(gdiff, v11); % 11 Variablen
                gdiff = diff(gdiff, v12); %12 Variablen
                gdiff = diff(gdiff, v13); % 13 Variablen
            case 14
                 g = exp(  -1*(  (  (-log(v1))^cp + (-log(v2))^cp + (-log(v3))^cp + (-log(v4))^cp + (-log(v5))^cp + (-log(v6))^cp + (-log(v7))^cp + (-log(v8))^cp + (-log(v9))^cp + (-log(v10))^cp + (-log(v11))^cp + (-log(v12))^cp + (-log(v13))^cp + (-log(v14))^cp       )^(1/cp)  )  );   % Aufpassen auf Klammersetzen
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
                gdiff = diff(gdiff, v11); % 11 Variablen
                gdiff = diff(gdiff, v12); %12 Variablen
                gdiff = diff(gdiff, v13); % 13 Variablen     
                gdiff = diff(gdiff, v14); % 14 Variablen
        end    
        
    case 'frank'
    % Auswahl der Anzahl der Dichtefunktion unter Berücksichtigung der
        % Anzahl der Indices
        
        % Variablen als numerische Variablen characterisieren
        clear v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14  cp
        syms v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14  cp
        switch DatasetSize
            case 2
                g = -1/cp * log( 1 + (   (exp(-v1*cp)-1) * (exp(-v2*cp)-1)    /((exp(-cp)-1)^(DatasetSize-1))  )  );  
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
            case 3
                g = -1/cp * log( 1 + (   (exp(-v1*cp)-1) * (exp(-v2*cp)-1) * (exp(-v3*cp)-1)    /((exp(-cp)-1)^(DatasetSize-1))  )  );  
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
            case 4
                g = -1/cp * log( 1 + (   (exp(-v1*cp)-1) * (exp(-v2*cp)-1) * (exp(-v3*cp)-1) * (exp(-v4*cp)-1)    /((exp(-cp)-1)^(DatasetSize-1))  )  );  
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
            case 5
                g = -1/cp * log( 1 + (   (exp(-v1*cp)-1) * (exp(-v2*cp)-1) * (exp(-v3*cp)-1) * (exp(-v4*cp)-1) * (exp(-v5*cp)-1)    /((exp(-cp)-1)^(DatasetSize-1))  )  );  
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
            case 6
                g = -1/cp * log( 1 + (   (exp(-v1*cp)-1) * (exp(-v2*cp)-1) * (exp(-v3*cp)-1) * (exp(-v4*cp)-1) * (exp(-v5*cp)-1) * (exp(-v6*cp)-1)    /((exp(-cp)-1)^(DatasetSize-1))  )  );  
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
            case 7
                g = -1/cp * log( 1 + (   (exp(-v1*cp)-1) * (exp(-v2*cp)-1) * (exp(-v3*cp)-1) * (exp(-v4*cp)-1) * (exp(-v5*cp)-1) * (exp(-v6*cp)-1) * (exp(-v7*cp)-1)    /((exp(-cp)-1)^(DatasetSize-1))  )  );  
                % Differenzieren der Copula Funktion um Dichte zu bestimmen
                gdiff = diff(g, v1); % 1 Variablen
                gdiff = diff(gdiff, v2); % 2 Variablen
                gdiff = diff(gdiff, v3); % 3 Variablen
                gdiff = diff(gdiff, v4); % 4 Variablen
                gdiff = diff(gdiff, v5); % 5 Variablen
                gdiff = diff(gdiff, v6); % 6 Variablen
                gdiff = diff(gdiff, v7); % 7 Variablen
            case 8
                g = -1/cp * log( 1 + (   (exp(-v1*cp)-1) * (exp(-v2*cp)-1) * (exp(-v3*cp)-1) * (exp(-v4*cp)-1) * (exp(-v5*cp)-1) * (exp(-v6*cp)-1) * (exp(-v7*cp)-1) * (exp(-v8*cp)-1)    /((exp(-cp)-1)^(DatasetSize-1))  )  );  
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
                g = -1/cp * log( 1 + (   (exp(-v1*cp)-1) * (exp(-v2*cp)-1) * (exp(-v3*cp)-1) * (exp(-v4*cp)-1) * (exp(-v5*cp)-1) * (exp(-v6*cp)-1) * (exp(-v7*cp)-1) * (exp(-v8*cp)-1) * (exp(-v9*cp)-1)    /((exp(-cp)-1)^(DatasetSize-1))  )  );  
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
                g = -1/cp * log( 1 + (   (exp(-v1*cp)-1) * (exp(-v2*cp)-1) * (exp(-v3*cp)-1) * (exp(-v4*cp)-1) * (exp(-v5*cp)-1) * (exp(-v6*cp)-1) * (exp(-v7*cp)-1) * (exp(-v8*cp)-1) * (exp(-v9*cp)-1) * (exp(-v10*cp)-1)    /((exp(-cp)-1)^(DatasetSize-1))  )  );  
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
            case 11
                g = -1/cp * log( 1 + (   (exp(-v1*cp)-1) * (exp(-v2*cp)-1) * (exp(-v3*cp)-1) * (exp(-v4*cp)-1) * (exp(-v5*cp)-1) * (exp(-v6*cp)-1) * (exp(-v7*cp)-1) * (exp(-v8*cp)-1) * (exp(-v9*cp)-1) * (exp(-v10*cp)-1)* (exp(-v11*cp)-1)     /((exp(-cp)-1)^(DatasetSize-1))  )  );
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
                gdiff = diff(gdiff, v11); % 11 Variablen
            case 12
                 g = -1/cp * log( 1 + (   (exp(-v1*cp)-1) * (exp(-v2*cp)-1) * (exp(-v3*cp)-1) * (exp(-v4*cp)-1) * (exp(-v5*cp)-1) * (exp(-v6*cp)-1) * (exp(-v7*cp)-1) * (exp(-v8*cp)-1) * (exp(-v9*cp)-1) * (exp(-v10*cp)-1)* (exp(-v11*cp)-1)* (exp(-v12*cp)-1)     /((exp(-cp)-1)^(DatasetSize-1))  )  );
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
                gdiff = diff(gdiff, v11); % 11 Variablen
                gdiff = diff(gdiff, v12); % 12 Variablen
            case 13 
                 g = -1/cp * log( 1 + (   (exp(-v1*cp)-1) * (exp(-v2*cp)-1) * (exp(-v3*cp)-1) * (exp(-v4*cp)-1) * (exp(-v5*cp)-1) * (exp(-v6*cp)-1) * (exp(-v7*cp)-1) * (exp(-v8*cp)-1) * (exp(-v9*cp)-1) * (exp(-v10*cp)-1)* (exp(-v11*cp)-1)* (exp(-v12*cp)-1)* (exp(-v13*cp)-1)     /((exp(-cp)-1)^(DatasetSize-1))  )  );
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
                gdiff = diff(gdiff, v11); % 11 Variablen
                gdiff = diff(gdiff, v12); % 12 Variablen
                gdiff = diff(gdiff, v13); % 13 Variablen
            case 14
                g = -1/cp * log( 1 + (   (exp(-v1*cp)-1) * (exp(-v2*cp)-1) * (exp(-v3*cp)-1) * (exp(-v4*cp)-1) * (exp(-v5*cp)-1) * (exp(-v6*cp)-1) * (exp(-v7*cp)-1) * (exp(-v8*cp)-1) * (exp(-v9*cp)-1) * (exp(-v10*cp)-1)* (exp(-v11*cp)-1)* (exp(-v12*cp)-1)* (exp(-v13*cp)-1)* (exp(-v14*cp)-1)     /((exp(-cp)-1)^(DatasetSize-1))  )  );
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
                gdiff = diff(gdiff, v11); % 11 Variablen
                gdiff = diff(gdiff, v12); % 12 Variablen
                gdiff = diff(gdiff, v13); % 13 Variablen
                gdiff = diff(gdiff, v14); % 14 Variablen
        end
end

% Einsetzen der Variablenvektoren in die bestimmten Dichtefunktionen (nur
% für multivariate Gumbel und Frank Copula)
switch lower(family)
    case {'gumbel', 'frank'}

        % gdiff in Vektorenform umwandeln um mit Daten rechnen zu können
        gdiff = vectorize(gdiff);
        % Löschen der numerischen Variablen
        clear v1 v2 v3 v4 v5 v6 v7 v8 v9 v10  cp g
        % Konvertieren des Formel Strings in nutzbare Formel
        gdiff = inline(gdiff);
        % Bestimmen der Variablenvektoren aus den empirischen
        % Marginalverteilungen entsprechend der Anzahl der Indices (max =
        % 10)
        v = [];
        for k = 1:DatasetSize
           i = genvarname('v', who);
           eval([i ' = z(:,k);'])
        end
        % Übergeben des Copula Parameters
        cp = varargin{1};
        % Einsetzen der variablenvektoren in neu definierte Funktionsgleichung gdiff
        switch DatasetSize
            case 2
                [CopulaDensity] = gdiff(cp,v1,v2);
            case 3
                [CopulaDensity] = gdiff(cp,v1,v2,v3);
            case 4
                [CopulaDensity] = gdiff(cp,v1,v2,v3,v4);
            case 5
                [CopulaDensity] = gdiff(cp,v1,v2,v3,v4,v5);
            case 6
                [CopulaDensity] = gdiff(cp,v1,v2,v3,v4,v5,v6);
            case 7
                [CopulaDensity] = gdiff(cp,v1,v2,v3,v4,v5,v6,v7);
            case 8
                [CopulaDensity] = gdiff(cp,v1,v2,v3,v4,v5,v6,v7,v8);
            case 9
                [CopulaDensity] = gdiff(cp,v1,v2,v3,v4,v5,v6,v7,v8,v9);
            case 10
                [CopulaDensity] = gdiff(cp,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10);
        end      
        varargout{1} = CopulaDensity;  

end
        
