% LL Function der dynamischen archimedischen multivariaten Copulas. 
% Zulässige Copula Familien: t, Gaussian, Clayton, Gumbel
%
% INPUT:
% - family: t, Gauss, Clayton, Gumbel
% - CopParam_1: bei archimedischen Copulas: erster CopulaParameter im Zeitablauf
%                            bei elliptischen Copulas bleibt dieser leer: []
% - tv_faktor: Parameter des ARMA Prozesses des Copulaparameters
% - data: [0 1] verteilte Zeitreihen
% - P: Lags der Residuen
% - Q: Lags der Covarianzmatrix
% P,Q werden nur für die elliptischen Copulas übergeben!
%
% OUTPUT:
% - LL: LogLikelihood Wert
% - Param: CopulaParameter
%
%   Author: Valentin Braun, Martin Grizska
%   Phd Student in finance Goethe Universität
function [LL, CopParam_tv, likelihood] = copulaLL_tv(tv_faktor, CopParam_1, family, data, P, Q);

family = lower(family);
tv_faktor = tv_faktor(:)';
[s1 s2] = size(data);

% Prüfen dass Daten als Zeit x Indices eingeben werden
if s2>s1
    data = data';
    [s1 s2] = size(data);
end

% Prüfen dass Daten [0 1] verteilt sind
if max(max(data))>1 || min(min(data))<0
    error('Daten müssen [0 1] verteilt sein');
end

% Prüfen dass max. 10 Indices verwendet werden
if s2<2 || s2>10
    error('Es können minimal 2 und maximal 10 Zeitreihen in den dynamischen Copulas verwendet werden');
end

% Prüfen dass nur erlaubte Copulafamilien verwendet werden
if sum(strcmp(family, {'clayton' 'gumbel' 't' 'gaussian'})) ~= 1
    error('angegebene Copulafunktion kann nicht benutzt werden');
end

% -----------------------------------------------------------------------
% Copula Log Likelihood
% -----------------------------------------------------------------------
switch family
    
% -----------------------------------------------------------------------
    case 'clayton'
        % Kendalls Tau über Zeithorizont anhand der ARMA Faktoren berechnen
        tau = copulaParam_tv_Patton(family,tv_faktor,CopParam_1,data); 
        tau = tau(:)'; % Sicherstellen dass Kendalls Tau als 1xn Vektor übergeben wird
        archCop_Param = 2*tau./(1-tau); % Kendalls Tau in Clayton Parameter umrechnen
        for i = 1:s1
            % Dichte der entsprechenden archimedischen Copula berechnen
            density(i,:) = copulapdfmultivariat(family, data(i,:), archCop_Param(i)); 
        end
        CopParam_tv = archCop_Param; % zeitvariablen Copulaparameter für den gesamten Zeithorizont übergeben
        likelihood = -log(density);
        LL = -sum(log(density)); % neg. LogLikelihood berechnen
        likelihoods = LL;
        
% -----------------------------------------------------------------------        
    case 'gumbel'
        % Kendalls Tau über Zeithorizont anhand der ARMA Faktoren berechnen
        tau = copulaParam_tv_Patton(family,tv_faktor,CopParam_1,data); 
        tau = tau(:)'; % Sicherstellen dass Kendalls Tau als 1xn Vektor übergeben wird
        archCop_Param = 1 ./(1-tau); % Kendalls Tau in Gumbel Parameter umrechnen
        Density_Func = copulapdffunc(family, s2); % Dichtefunktion herleiten und als Funktion abspeichern. Steigert Zeiteffizienz 
        % Datenmatrix in einzelne Datenvektoren zerlegen
        v = [];
        for i = 1:s2
           var = genvarname('v', who);
           eval([var ' = data(:, i);'])
        end
        for i = 1:s1
            % Dichte der entsprechenden archimedischen Copula berechnen
            switch s2
                case 2
                    density(i,:) = Density_Func(archCop_Param(i), v1(i),v2(i));
                case 3
                    density(i,:) = Density_Func(archCop_Param(i), v1(i),v2(i),v3(i));
                case 4
                    density(i,:) = Density_Func(archCop_Param(i), v1(i),v2(i),v3(i),v4(i));
                case 5
                    density(i,:) = Density_Func(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i));
                case 6
                    density(i,:) = Density_Func(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i));
                case 7
                    density(i,:) = Density_Func(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i));
                case 8
                    density(i,:) = Density_Func(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i));
                case 9
                    density(i,:) = Density_Func(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i));
                case 10
                    density(i,:) = Density_Func(archCop_Param(i), v1(i),v2(i),v3(i),v4(i),v5(i),v6(i),v7(i),v8(i),v9(i),v10(i));
            end
        end
        CopParam_tv = archCop_Param; % zeitvariablen Copulaparameter für den gesamten Zeithorizont übergeben
        likelihood = -log(density);
        LL = -sum(log(density)); % neg. LogLikelihood berechnen
        
% -----------------------------------------------------------------------        
    case {'gaussian' 't'}
        y = zeros(s1,s2);
        switch family
            case 't'
                DoF = tv_faktor(end); % DoF der t-Copula auslesen
                for i = 1:s2
                    y(:,i) = tinv(data(:,i),DoF);
                end
            case 'gaussian'
                for i = 1:s2
                    y(:,i) = norminv(data(:,i),0,1);
                end
        end

        a = tv_faktor(1:P);       % ARCH for all residuals
        a_negative = tv_faktor(P+1:2*P);   % ARCH for negative residuals
        b = tv_faktor(2*P+1:2*P+Q);        % GARCH
        sumA = eye(s2)*sum(a);
        sumA_negative = eye(s2)*sum(a_negative);
        sumB = eye(s2)*sum(b);

        % First compute Qbar, the correlation matrix of the standardized residuals,
        % and Nbar, the correlation matrix of the standardized residuals,
        % conditional on the standardized residuals being negative
        Qbar = cov(y);
        Nbar_negative = cov(y.*(y < 0));

        % Next compute Qt
        m = max(P,Q);
        Qt = zeros(s2,s2,s1+m);
        Corr = zeros(s2,s2,s1+m);
        Qt(:,:,1:m) = repmat(Qbar,[1 1 m]);
        Corr(:,:,1:m) = repmat(Qbar,[1 1 m]);
        violatePSD = 0; % Kontrollvariable für positive Semidefinitheit

        stdresid = [zeros(m,s2); y];
        negativey = stdresid .* (stdresid < 0); % Asymmetrie Faktor definieren
        Qinitial = Qbar * (eye(s2) - sumA - sumB) - Nbar_negative * sumA_negative;

        for j = (m+1):s1+m
            Qt(:,:,j) = Qinitial;
            for i=1:P
                Qt(:,:,j) = Qt(:,:,j) + a(i)*(stdresid(j-i,:)'*stdresid(j-i,:));
                Qt(:,:,j) = Qt(:,:,j) + a_negative(i)*(negativey(j-i,:)'*negativey(j-i,:));
            end
            for i = 1:Q
                Qt(:,:,j) = Qt(:,:,j) + b(i)*Qt(:,:,j-i);
            end
            Corremp = Qt(:,:,j)./(sqrt(diag(Qt(:,:,j)))*sqrt(diag(Qt(:,:,j)))');
            Corremp = Corremp - diag(diag(Corremp)) + eye(s2);
            Corr(:,:,j) = Corremp;
            maxmax = max(max(Corr(:,:,j)));
            minmin = min(min(Corr(:,:,j)));
            if maxmax > 1 || minmin < -1
                violatedPSD = 1;
            end
        end
        % Output vorbereiten
        Corr = Corr(:,:,(m+1:s1+m));

        % LL berechnen 
        switch family
            case 't'
                for i=1:s1
                    LL(i) = gammaln((DoF+s2)/2) + (s2-1)*gammaln(DoF/2) - s2*gammaln((DoF+1)/2) - 0.5*log(det(Corr(:,:,i)));
                    LL(i) = LL(i) - (DoF+s2)/2*log(1+y(i,:)*(Corr(:,:,i))^(-1)*y(i,:)'./(DoF-2));
                    LL(i) = LL(i) + (DoF+1)/2*sum(log(1+(y(i,:).^2/(DoF-2))));
                end
            case 'gaussian'
                LL=zeros(s1,1); 
                for i=1:s1
                    LL(i)=-.5*log(det(Corr(:,:,i)));
                    LL(i)=LL(i)-.5*y(i,:)*(inv(Corr(:,:,i))-eye(s2))*y(i,:)';    
                end
        end
        likelihood = -LL;
        LL = -sum(LL);
        CopParam_tv = Corr; % zeitvariablen Copulaparameter für den gesamten Zeithorizont übergeben
end
        
% Sicherstellen dass LL nicht NaN wird
if isnan(LL)
    LL = 1e6;
end






























