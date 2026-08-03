% Berechnen robuster Standardfehler (SE) nach White (1982)
%
% INPUTS:
%          Stat: Indikator ob SE berechnet werden sollen
%          MLE_Param: Vektor der optimalen Copulaparameter
%          data:       t x k matrix of unif(0,1) variables
%          P:          lag length of ARCH and Leverage-Term
%          Q:          lag length of GARCH term
%          CopParam_1: unconditional CopParam (needed only for estimation)
%          family:     string can be either t, Gauss, Clayton, Gumbel
%
% OUTPUTS:
%          V: robuste Covarianzmatrix der SE
%          StdError: White(1982) robust standard errors
%          p_Wert: robuster p-Wert der aus V bestimmt wird
%          t_Wert   : robust t-statistic for estimated parameters
%
%   Author: Valentin Braun, Martin Grizska
%   Phd Student in finance Goethe Universität, LMU
function [V StdError p_Wert t_Wert] = copulaStat_mix_tv_2(Stat, MLE_Param, CopParam_1, weights_1, family, U, P, Q, Dynamic);

[s1, s2] = size(U); % Beobachtungen pro Zeitreihe & # Zeitreihen
np = numel(MLE_Param);

% Prüfen ob StdError und p-Werte berechnet werden sollen
switch Stat
    case 'aus'
        V = [];
        StdError = NaN(1, np); % NaN Vektor der Größe des Param Vektors
        p_Wert = NaN(1, np); % NaN Vektor der Größe des Param Vektors
        t_Wert = NaN(1, np); % NaN Vektor der Größe des Param Vektors
        
    case 'an'
        % Sicherstellen dass der Paramtervektor als 1xn Vektor übergeben wird
        [p1 p2] = size(MLE_Param);
        if p1 ~= 1
            MLE_Param = MLE_Param';
            [p1 p2] = size(MLE_Param);
        end
        if p1 ~= 1
            error('Die MLE Parameter müssen für die Berechnung der Standard Fehler des RSC Modells als 1xn Vektor übergeben werden');
        end
        
        % Definieren einer kleinen Wertänderung pro MLE Parameter
        delta = 1e-5*abs(MLE_Param);
        delta(delta<1e-10) = 1e-10;
        
        % Prüfen dass die MLE Parameterwerte nicht die minimal zulässige Größe für
        % Matlab Variablen unterschreiten. Ansonsten ergibt sich für die Ableitung
        % der Loglikelihood Funtion NaN
        MLE_Param(abs(MLE_Param)<eps) = 0.0001;
        
        %--------------------------------------------------------------------------
        % Berechnen der äußeren Produktmatrix
        %--------------------------------------------------------------------------
        % Erste Ableitung der Informationsmatrix und berechnen
        n = zeros(s1, np);
        [H1, H2, H3, H4, H5, density] =  copulaLL_mix_tv_2(MLE_Param, CopParam_1, family, U, P, Q, [], Dynamic);
        logLikVec1 = log(density); % Loglikelihood Werte der dynamischen Mixture Copula
        for i = 1:np
            delta_Param = MLE_Param;
            delta_Param(i) = MLE_Param(i)+delta(i);
            [H1, H2, H3, H4, H5, density] =  copulaLL_mix_tv_2(delta_Param, CopParam_1,  family, U, P, Q, [], Dynamic);
            logLikVec2 = log(density); % Loglikelihood Werte der dynamischen Mixture Copula
            n(:,i)=(logLikVec2-logLikVec1)/(delta(i));
        end
        % Berechnen der äußeren Produktmatrix
        sum_n_Matrix=zeros(np);
        for i = 1:s1
            n_Matrix = n(i,:)'*n(i,:);
            sum_n_Matrix=sum_n_Matrix+n_Matrix;
        end
        OP_Matrix=sum_n_Matrix/s1;   % äußere Produktmatrix
        
        %--------------------------------------------------------------------------
        % Berechnen der Hesse Matrix (zweite Ableitung der Informationsmatrix)
        %--------------------------------------------------------------------------
        sde = zeros(np);
        for i = 1:np
            for j = 1:np
                delta_Param_1 = MLE_Param;
                delta_Param_2 = MLE_Param;
                % Numerische Berechnung der ersten Ableitung
                delta_Param_1(i)=MLE_Param(i)-delta(i);   %x-
                delta_Param_11=delta_Param_1;
                delta_Param_12=delta_Param_1;
                delta_Param_11(j)=delta_Param_1(j)-delta(j);     %x-,y-
                delta_Param_12(j)=delta_Param_1(j)+delta(j);     %x-,y+
                % Numerische Berechnung der zweiten Ableitung
                delta_Param_2(i)=MLE_Param(i)+delta(i);   %x+
                delta_Param_22=delta_Param_2;
                delta_Param_21=delta_Param_2;
                delta_Param_21(j)=delta_Param_2(j)-delta(j);     %x+,y-
                delta_Param_22(j)=delta_Param_2(j)+delta(j);     %x+,y+
                % Berechnen der LL
                [sumlik11] = copulaLL_mix_tv_2(delta_Param_11, CopParam_1,  family, U, P, Q, [], Dynamic);
                [sumlik12] = copulaLL_mix_tv_2(delta_Param_12, CopParam_1,  family, U, P, Q, [], Dynamic);
                [sumlik21] = copulaLL_mix_tv_2(delta_Param_21, CopParam_1,  family, U, P, Q, [], Dynamic);
                [sumlik22] = copulaLL_mix_tv_2(delta_Param_22, CopParam_1,  family, U, P, Q, [], Dynamic);
                % !!!ACHTUNG!!! RSC_Lik gibt die neg LL aus.
                sumlik11 = -sumlik11;
                sumlik12 = -sumlik12;
                sumlik21 = -sumlik21;
                sumlik22 = -sumlik22;
                % second derivative by one side finite difference
                sde(i,j) = (sumlik22-sumlik21-sumlik12+sumlik11)/(4*delta(i)*delta(j));
            end
        end
        H = -sde/s1; % Hesse Matrix nach Hamilton Formel 5.8.2, S. 143
        
        %--------------------------------------------------------------------------
        % Varianz-Kovarianz Matrix & StdError berechnen
        %--------------------------------------------------------------------------
        V = 1/s1*pinv((H*pinv(OP_Matrix)*H)); % Varianz-Kovarianz Matrix nach White (1982), Hamilton: Time Series Analysis, S. 145, Formel  5.8.7, S. 145
        StdError = sqrt(diag((V))); % der Std Vektor entpsircht der Wurzel der Diagonalen von V (Varianz-Kovarianz Matrix)
        % Kontrollieren der Kovarianzmatrix. Fehler in der Varianz werden auf 'inf'
        % gesetzt.
        StdError(isinf(StdError))=0;
        if ~isreal(StdError)
            for i=1:numel(MLE_Param)
                if ~isreal(StdError(i))
                    StdError(i)=Inf;
                end
            end
        end
        StdError = StdError(:)'; % Ausgeben als 1xn Vektor
        
        %--------------------------------------------------------------------------
        % Zweiseitigen p-Wert & t-Statistik berechnen
        %--------------------------------------------------------------------------
        p_Wert = 2*(1-tcdf(abs(MLE_Param)./StdError, s1-np)); % p-Wert
        t_Wert = MLE_Param(:)'./StdError; % t-Wert berechnen
        
        %--------------------------------------------------------------------------
        % Ausgabe als nx1 Vektoren vorbereiten
        %--------------------------------------------------------------------------
        StdError = StdError(:);
        p_Wert = p_Wert(:);
        t_Wert = t_Wert(:);
        
end
