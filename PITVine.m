function [zt] = PITVine(X,CopulaSpec,VineOutput);
% Probability Integral transform for copula-vines after Aas et al (2006)
%
% USAGE:   [x] = PITVine(T,CopulaSpec,VineOutput)
%
% INPUT:
%          VineOutput: structure containing several elements ( CopParams,
%                      Correlation Params)
%                  X:  unif(0,1)-Daten
%                  T:  number of observations to be sampled
%          CopulaSpec: Structure of Copula specification, possible
%                      Copula-types: 'Clayton','t',
%          Rt    : special Matrix if correlation is time-varying; see
%                      VineVaR.m; if correlation='static': Rt=[]
%
% OUTPUT:  N x t matrix of simualted variables
%
% Author:  Martin Grziska, April/ 12/ 2010
% % ---------------------------------------------------------------------


N=size(VineOutput.VineParams,2)+1;
% matrix with dependence parameters is needed as cell, if not cell
% following code converts to one
phi = VineOutput.Rt;
% % Wenn die Abhängigkeitsstrukur kein Skalar sondern Matrix ist,
% % nehme(2,1,i) Element und schreibe als Skalar
T=size(X,1);
if size(VineOutput.Rt{1},2) > 1
    c=N-1;
    for j=1:N-1 %gehe in entsprechende Zeile
        for i=1:c %gehe in entsprehende Spalte; fürjede Zeile die weitergegangen wird nimmt die Spaltenzahl um eins ab
            for ii=1:T %gehe in der jeweiligen Zelle durch alle Zeitpunkte
                phi2{j,i}(ii,1) = phi{j,i}(2,1,ii);
            end
        end
        c=c-1;
    end
    clear phi
    phi = phi2;
    clear phi2;
end

% Wenn phi keine cell-Variable ist, wandle in cell-Variable um
if iscell(phi) == 0
    phi2 = cell(N-1,N-1);
    c=N-1;
    for j=1:N-1
        for i=1:c
            phi2{j,i} = phi(j,i);
        end
        c=c-1;
    end
    clear phi
    phi=phi2;
    clear phi2
end

% Problem: Clayton, rotated Clayton und Gumbel stellen nur positive
% Abhängigkeiten dar, negative Abhängigkeiten werden einfach = 0 gesetzt;
% damit funktioniert aber der PIT-Algorithmus nicht mehr; gehe also alle
% Abhängigkeitsparameter durch und überprüfe ob sie ungleich sind, wenn
% sich gleich null sind setzte sie 0 + epsilon
type = CopulaSpec.type;
if strcmp(type,'Clayton') || strcmp(type,'Rotated Clayton') || strcmp(type,'Gumbel')
    epsilon = 1e-10;
    c=N-1;
    for j=1:N-1
        for i=1:c
            for ii=1:T
                if phi{j,i}(ii,1)==0
                    phi{j,i}(ii,1) = epsilon;
                end
            end
        end
        c=c-1;
    end
end

% % sample unif(0,1) to start with
zt(:,1) = X(:,1);

% % ---------------------------------------------------------------------
if strcmp(CopulaSpec.decomposition,'DVine') == 1
    %     %     condtional sampling as in Aas et al (2006)
    %     Clayton-Copula
    if strcmp(type,'Clayton') == 1
        %         %         Algo from Aas et al (2006); formula for h and h^(-1)
        %         %         bivariate Clayton: Heinen, Valdesogo
        %         h-Funktion (conditional distribution)
        zt(:,2) = X(:,2).^(-1-phi{1,1}).*(X(:,1).^(-phi{1,1}) + X(:,2).^(-phi{1,1}) - 1).^(-1-1./phi{1,1});
        v{2,1}  = X(:,1);
        v{2,2}  = X(:,1).^(-1-phi{1,1}).*(X(:,2).^(-phi{1,1}) + X(:,1).^(-phi{1,1}) - 1).^(-1-1./phi{1,1});
        % h^(-1)-Funktion
        for i = 3:N
            zt(:,i)  = X(:,i-1).^(-1-phi{1,i-1}).*(X(:,i).^(-phi{1,i-1}) + X(:,i-1).^(-phi{1,i-1}) - 1).^(-1-1./phi{1,i-1});
            for j = 2:i-1
                zt(:,i)  = v{i-1,2*(j-1)}.^(-1-phi{j,i-j}).*(zt(:,i).^(-phi{j,i-j}) + v{i-1,2*(j-1)}.^(-phi{j,i-j}) - 1).^(-1-1./phi{j,i-j});
            end
            if i < N
                v{i,1} = X(:,i);
                v{i,2} = v{i,1}.^(-1-phi{1,i-1}).*(v{i-1,1}.^(-phi{1,i-1}) + v{i,1}.^(-phi{1,i-1}) - 1).^(-1-1./phi{1,i-1});
                v{i,3} = v{i-1,1}.^(-1-phi{1,i-1}).*(v{i,1}.^(-phi{1,i-1}) + v{i-1,1}.^(-phi{1,i-1}) - 1).^(-1-1./phi{1,i-1});
                for j = 1:i-3
                    %                         %  h-Funktionen (conditional distributions)
                    v{i,2*j+2} = v{i,2*j+1}.^(-1-phi{j+1,i-j-1}).*(v{i-1,2*j}.^(-phi{j+1,i-j-1}) + v{i,2*j+1}.^(-phi{j+1,i-j-1}) - 1).^(-1-1./phi{j+1,i-j-1});
                    v{i,2*j+3} = v{i-1,2*j}.^(-1-phi{j+1,i-j-1}).*(v{i,2*j+1}.^(-phi{j+1,i-j-1}) + v{i-1,2*j}.^(-phi{j+1,i-j-1}) - 1).^(-1-1./phi{j+1,i-j-1});
                end
                v{i,2*i-2} = v{i,2*i-3}.^(-1-phi{i-1,1}).*(v{i-1,2*i-4}.^(-phi{i-1,1}) + v{i,2*i-3}.^(-phi{i-1,1}) - 1).^(-1-1./phi{i-1,1});
            end
        end
        % ----------------------------------- Rotated Clayton
        % ------------------------------------------------------------------------
    elseif strcmp(type,'Rotated Clayton') == 1
        %         %         Algo from Aas et al (2006); formula for h and h^(-1)
        %         %         bivariate Clayton: Heinen, Valdesogo
        zt(:,2) = 1-((1-X(:,2)).^(-1-phi{1,1}).*((1-X(:,1)).^(-phi{1,1}) + (1-X(:,2)).^(-phi{1,1}) - 1).^(-1-1./phi{1,1}));
        v{2,1}  = 1-X(:,1);
        v{2,2}  = 1-((1-X(:,1)).^(-1-phi{1,1}).*((1-X(:,2)).^(-phi{1,1}) + (1-X(:,1)).^(-phi{1,1}) - 1).^(-1-1./phi{1,1}));
        % h^(-1)-Funktion
        for i = 3:N
            zt(:,i)  = 1-((1-X(:,i-1)).^(-1-phi{1,i-1}).*((1-X(:,i)).^(-phi{1,i-1}) + (1-X(:,i-1)).^(-phi{1,i-1}) - 1).^(-1-1./phi{1,i-1}));
            for j = 2:i-1
                zt(:,i)  = 1-((1-v{i-1,2*(j-1)}).^(-1-phi{j,i-j}).*((1-zt(:,i)).^(-phi{j,i-j}) + (1-v{i-1,2*(j-1)}).^(-phi{j,i-j}) - 1).^(-1-1./phi{j,i-j}));
            end
            if i < N
                v{i,1} = (1-X(:,i));
                v{i,1}(v{i,1}==1)
                v{i,2} = 1-((1-v{i,1}).^(-1-phi{1,i-1}).*((1-v{i-1,1}).^(-phi{1,i-1}) + (1-v{i,1}).^(-phi{1,i-1}) - 1).^(-1-1./phi{1,i-1}));
                v{i,3} = 1-((1-v{i-1,1}).^(-1-phi{1,i-1}).*((1-v{i,1}).^(-phi{1,i-1}) + (1-v{i-1,1}).^(-phi{1,i-1}) - 1).^(-1-1./phi{1,i-1}));
                for j = 1:i-3
                    %                         %  h-Funktionen (conditional distributions)
                    v{i,2*j+2} = 1-((1-v{i,2*j+1}).^(-1-phi{j+1,i-j-1}).*((1-v{i-1,2*j}).^(-phi{j+1,i-j-1}) + (1-v{i,2*j+1}).^(-phi{j+1,i-j-1}) - 1).^(-1-1./phi{j+1,i-j-1}));
                    v{i,2*j+3} = 1-((1-v{i-1,2*j}).^(-1-phi{j+1,i-j-1}).*((1-v{i,2*j+1}).^(-phi{j+1,i-j-1}) + (1-v{i-1,2*j}).^(-phi{j+1,i-j-1}) - 1).^(-1-1./phi{j+1,i-j-1}));
                end
                v{i,2*i-2} = 1-((1-v{i,2*i-3}).^(-1-phi{i-1,1}).*((1-v{i-1,2*i-4}).^(-phi{i-1,1}) + (1-v{i,2*i-3}).^(-phi{i-1,1}) - 1).^(-1-1./phi{i-1,1}));
            end
        end
        %     ---------------------------------------------------------------------
        %     t-Copula
    elseif strcmp(type,'t')
        %         if correlation is time varying read nu from vector[a,b,nu] and
        %         put it in cell
        if strcmp(CopulaSpec.corrspec,'static') == 1
            nu = VineOutput.VineParams;
        else
            nu=cell(N-1,N-1);
            c=N-1;
            for i=1:N-1
                for j=1:c
                    nu{i,j} = VineOutput.VineParams{i,j}{1}(end);
                end
                c=c-1;
            end
        end
        %         %         change double variables to cell, needed for
        %         %         following code
        if iscell(nu) == 0
            clear nu
            nu = cell(N-1,N-1);
            c=N-1;
            for i=1:N-1
                for j=1:c
                    nu{i,j} = VineOutput.VineParams{i,j}(end);
                end
                c=c-1;
            end
        end
        %         %         Algo from Aas et al (2006); formula for h and h^(-1) bivariate t: Heinen, Valdesogo
        v = cell(N,N+1);
        %         %         h-Funktion (conditional distribution)
        zt(:,2) = tcdf((tinv(X(:,2),nu{1,1})-phi{1,1}.*tinv(X(:,1),nu{1,1}))./sqrt(((nu{1,1}+tinv(X(:,1),nu{1,1}).^2).*(1-phi{1,1}.^2))./(nu{1,1}+1)),nu{1,1}+1);
        v{2,1} = X(:,2);
        v{2,2} = tcdf((tinv(X(:,1),nu{1,1})-phi{1,1}.*tinv(X(:,2),nu{1,1}))./sqrt(((nu{1,1}+tinv(X(:,2),nu{1,1}).^2).*(1-phi{1,1}.^2))./(nu{1,1}+1)),nu{1,1}+1);
        for i = 3:N
            zt(:,i) = tcdf((tinv(X(:,i),nu{1,i-1})-phi{1,i-1}.*tinv(X(:,i-1),nu{1,i-1}))./sqrt(((nu{1,i-1}+tinv(X(:,i-1),nu{1,i-1}).^2).*(1-phi{1,i-1}.^2))./(nu{1,i-1}+1)),nu{1,i-1}+1);
            for j=2:i-1
                zt(:,i) = tcdf((tinv(zt(:,i),nu{j,i-j})-phi{j,i-j}.*tinv(v{i-1,2*(j-1)},nu{j,i-j}))./sqrt(((nu{j,i-j}+tinv(v{i-1,2*(j-1)},nu{j,i-j}).^2)).*(1-phi{j,i-j}.^2)./(nu{j,i-j}+1)),nu{j,i-j}+1);
            end
            if i < N
                v{i,1} = X(:,i);
                v{i,2}= tcdf((tinv(v{i-1,1},nu{1,i-1})-phi{1,i-1}.*tinv(v{i,1},nu{1,i-1}))./sqrt((nu{1,i-1}+(tinv(v{i,1},nu{1,i-1}).^2).*(1-phi{1,i-1}.^2))./(nu{1,i-1}+1)),nu{1,i-1}+1);
                v{i,3}= tcdf((tinv(v{i,1},nu{1,i-1})-phi{1,i-1}.*tinv(v{i-1,1},nu{1,i-1}))./sqrt((nu{1,i-1}+(tinv(v{i,1},nu{1,i-1}).^2).*(1-phi{1,i-1}.^2))./(nu{1,i-1}+1)),nu{1,i-1}+1);
                for j = 1:i-3
                    %                         %  h-Funktionen (conditional distributions)
                    v{i,2*j+2} = tcdf((tinv(v{i-1,2*j},nu{j+1,i-j-1})-phi{j+1,i-j-1}.*tinv(v{i,2*j+1},nu{j+1,i-j-1}))./sqrt((nu{j+1,i-j-1}+(tinv(v{i,2*j+1},nu{j+1,i-j-1}).^2).*(1-phi{j+1,i-j-1}.^2))./(nu{j+1,i-j-1}+1)),nu{j+1,i-j-1}+1);
                    v{i,2*j+3} = tcdf((tinv(v{i,2*j+1},nu{j+1,i-j-1})-phi{j+1,i-j-1}.*tinv(v{i-1,2*j},nu{j+1,i-j-1}))./sqrt((nu{j+1,i-j-1}+(tinv(v{i-1,2*j},nu{j+1,i-j-1}).^2).*(1-phi{j+1,i-j-1}.^2))./(nu{j+1,i-j-1}+1)),nu{j+1,i-j-1}+1);
                end
                v{i,2*i-2} = tcdf((tinv(v{i-1,2*i-4},nu{i-1,1})-phi{i-1,1}.*tinv(v{i,2*i-3},nu{i-1,1}))./sqrt((nu{i-1,1}+(tinv(v{i,2*i-3},nu{i-1,1}).^2).*(1-phi{i-1,1}.^2))./(nu{i-1,1}+1)),nu{i-1,1}+1);
            end
        end
        %     --------------------------------------------------------------------
        %     Gaussian-Copula
    elseif strcmp(type,'Gaussian')
        v = cell(N,N+1);
        zt(:,2) = normcdf((norminv(X(:,2))-phi{1,1}.*norminv(X(:,1)))./(sqrt(1-phi{1,1}.^2)));
        v{2,1} = X(:,2);
        v{2,2} = normcdf((norminv(X(:,1))-phi{1,1}.*norminv(X(:,2)))./(sqrt(1-phi{1,1}.^2)));
        for i = 3:N
            zt(:,i) = normcdf((norminv(X(:,i))-phi{1,i-1}.*norminv(X(:,i-1)))./(sqrt(1-phi{1,i-1}.^2)));
            for j = 2:i-1
                zt(:,i) = normcdf((norminv(zt(:,i))-phi{j,i-j}.*norminv(v{i-1,2*(j-1)}))./(sqrt(1-phi{j,i-j}.^2)));
            end
            if i < N
                v{i,1} = X(:,i);
                v{i,2} = normcdf((norminv(v{i-1,1})-phi{1,i-1}.*norminv(v{i,1}))./(sqrt(1-phi{1,i-1}.^2)));
                v{i,3} = normcdf((norminv(v{i,1})-phi{1,i-1}.*norminv(v{i-1,1}))./(sqrt(1-phi{1,i-1}.^2)));
                for j = 1:i-3
                    v{i,2*j+2} = normcdf((norminv(v{i-1,2*j})-phi{j+1,i-j-1}.*norminv(v{i,2*j+1}))./(sqrt(1-phi{j+1,i-j-1}.^2)));
                    v{i,2*j+3} = normcdf((norminv(v{i,2*j+1})-phi{j+1,i-j-1}.*norminv(v{i-1,2*j}))./(sqrt(1-phi{j+1,i-j-1}.^2)));
                end
                v{i,2*i-2} = normcdf((norminv(v{i-1,2*i-4})-phi{i-1,1}.*norminv(v{i,2*i-3}))./(sqrt(1-phi{i-1,1}.^2)));
            end
        end
    end
end

%     ---------------------------------------------------------------------
%     Gumbel-Copula
%     elseif strcmp(type,'Gumbel')
%         %         Algo from Aas et al (2006); formula for h and h^(-1)
%         %         bivariate Gumbel: Heinen, Valdesogo
%         v = cell(N,N+1);
%         v{1,1} = w(:,1);
%         x(:,1) = v{1,1};
%         for i=1:T
%             uinitial = .5; %startingvalue für die fzero-Funktion
%             %             unterschiedliche Startwerte für die Funktion fzero sind
%             %             nötig, um die alle mögliche Werte abzudecken. Wenn
%             %             v{2,1}(i,:) z.B. sehr kelin ist, so ist auch ein sehr kleiner
%             %             Startwert nötig, um eine Lösung zu finden. Fange mit0.5 an,
%             %             wenn dass zu keiner Lösung führt,fange mit sehr kleinen Start
%             %            wertean an und werde dann größer
%             initial=0.5;
%             % die originalgleichung
%             % lautet: u2 = (h^(-1)(u1,w,theta) dies wird umgeformt zu:
%             % w-h(u1,u2,theta) = 0 : dies Gleichung löst  fzero(@u2)
%             % nach u2 auf
%             %             suche startwert der positiven Wert als Ergebnis hat, sonst
%             %             gibt es Probleme bei der fzero-Funktion
%             result = inv_hfunc_gumbel(v{1,1}(i),w(i,2),uinitial,phi{1,1});
%             if result <=0
%                 count = 1;
%                 while result <=0 && count<1000
%                     uinitial = random('unif',0,1);
%                     result = inv_hfunc_gumbel(v{1,1}(i),w(i,2),uinitial,phi{1,1});
%                     count = count + 1;
%                 end
%             end
%             v{2,1}(i,:) = fzero(@(uinitial) inv_hfunc_gumbel(v{1,1}(i),w(i,2),uinitial,phi{1,1}),initial);
%             if isnan(v{2,1}(i,:))
%                 initial = 1e-6;
%                 while isnan(v{2,1}(i,:)) && initial < (1-1e-4) %vergrößere die Startwerte in kleinen Schritten so lange, bis knapp 1 erreicht ist
%                     v{2,1}(i,:) = fzero(@(uinitial) inv_hfunc_gumbel(v{1,1}(i),w(i,2),uinitial,phi{1,1}),initial);
%                     initial = initial + 1e-4; %verändere Startwerte für fzero falls die optimierung keine Lösung findet
%                 end
%             end
%         end
%         x(:,2) = v{2,1};
%         %         %         h^(-1) - Funktion (zum samplen): Gumbel h^(-1)
%         %         %         muss numerisch berechnet werden (hier mit dem
%         %         Newton-Raphson-Verfahren)
%         %         Beispiel für die Logik des Newton-Raphsons-Verfahren zur
%         %         Berechnung einer Inversen: y=f(x); f^(-1)(y) = x; f(x)-y = 0;
%         %         Newton-Raphson sucht allgemein die Nullstellen für
%         %         Polynome und f(x)-y=0 entspricht dem Berechnnen von x bei
%         %         gegebenem y;
%         %         v{2,1} = newton(exp(-((-log(w(:,2))).^phi{1,1} + (-log(v{1,1})).^phi{1,1}).^(1/phi{1,1}))-fevalexp(-((-log(w(:,2))).^phi{1,1} + (-log(v{1,1})).^phi{1,1}).^(1/phi{1,1})),.1);
%
%
%         %         %         h-Funktion (conditional distribution)
%         v{2,2} = exp(-((-log(v{1,1})).^phi{1,1} + (-log(v{2,1})).^phi{1,1}).^(1/phi{1,1}));
%         v{2,2} = (v{2,2}./v{2,1}).*(-log(v{2,1})).^(phi{1,1}-1);
%         v{2,2} = v{2,2}.*((-log(v{1,1})).^phi{1,1}+(-log(v{2,1})).^phi{1,1}).^(1/phi{1,1}-1);
%         %         % h^(-1)-Funktion
%         %         g=1;
%         for i = 3:N
%             v{i,1} = w(:,i);
%             %             %             h^(-1)-Funktion (zum samplen)
%             index = 2:i;
%             index = sort(index,'descend');
%             for s = 1:(size(3:index(1),1))
%                 k = index(s)-1;
%                 v{i,1} = ((v{i,1}.*v{i-1,2*k-2}.^(phi{k,i-k}+1)).^(-phi{k,i-k}./(phi{k,i-k}+1)) + 1 - v{i-1,2*k-k}.^(-phi{k,i-k})).^(-1./phi{k,i-k});
%             end
%             v{i,1} = ((v{i,1}.*v{i-1,1}.^(phi{1,i-1}+1)).^(-phi{1,i-1}./(phi{1,i-1}+1)) + 1 - v{i-1,1}.^(-phi{1,i-1})).^(-1./phi{1,i-1});
%             x(:,i) = v{i,1};
%             if i < N
%                 v{i,2} = v{i,1}.^(-1-phi{1,i-1}).*(v{i-1,1}.^(-phi{1,i-1}) + v{i,1}.^(-phi{1,i-1}) - 1).^(-1-1./phi{1,i-1});
%                 v{i,3} =
%                 v{i-1,1}.^(-1-phi{1,i-1}).*(v{i,1}.^(-phi{1,i-1}) + v{i-1,1}.^(-phi{1,i-1}) - 1).^(-1-1./phi{1,i-1});
%                     for j = 2:i-2
%                         %  h-Funktionen (conditional distributions)
%                         v{i,2*j} = v{i,2*j-1}.^(-1-phi{j,i-j}).*(v{i-1,2*j-2}.^(-phi{j,i-j}) + v{i,2*j-1}.^(-phi{j,i-j}) - 1).^(-1-1./phi{j,i-j});
%                         v{i,2*j+1} = v{i-1,2*j-2}.^(-1-phi{j,i-j}).*(v{i,2*j-1}.^(-phi{j,i-j}) + v{i-1,2*j-2}.^(-phi{j,i-j}) - 1).^(-1-1./phi{j,i-j});
%                     end
%                 v{i,2*i-2} = v{i-1,2*i-4}.^(-1-phi{i-1,1}).*(v{i-1,2*i-4}.^(-phi{i-1,1}) + v{i,2*i-3}.^(-phi{i-1,1}) - 1).^(-1-1./phi{i-1,1});
%             end
%         end


% References:
% Aas, Czado, Frigessi, Bakken (2006): Pair-copula constructions for multiple
% dependence. Ssondeforschungsbereich 386, Paper 487.

% Heinen, Valdesogo: Asymmetric CAPM dependence for large dimensiona: the
% Canonical Vine Autoregressive Model.


