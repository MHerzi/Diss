function [CopParam_pred, weights_pred] = CopulaVine_multistep_mix_LL_forecast(CopulaSpec,data,phi,weights)

% makes one-period forecast for D-Vine mixture
% INPUTS: 
%         CopulaSpec : Structure with Copula specific informations (which
%                      copula to estimat etc tec.
%         data       : TxN matrix if U(0,1) variabels  
%         phi        : cell variable with coefficients form Vine Estimation
%         weights    : Tx2 vector of weights
% 
% OUTPUTS:
%        CopParam_pred: 1-period forecast of copula parameters
%        weights_pred : 1-period forecast of weight parameters
% 
% Author: Martin Grziska
% Date of last modification: 10/07/2010

% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

%     Estimation of first D-Vine level
% put data into cell
[T,N]=size(data);
v=cell(N,N);
Dynamic=CopulaSpec.Dynamic;

for i=1:N
    v{1,i} = data(:,i);
end

% mache Vorhersage für Gewichte und Copula-Parameters für die erste Ebene
% der Vine-Copula; die Vorhersage steht in der letzten Zeile und ist die
% Vorhersage von t auf t+1
for i = 1:N-1
    [CopParam_pred{1,i}, weights_pred{1,i}] = copulamix_tv_paramforecast_2_grm(phi{1,i}, 2, CopulaSpec.family, [v{1,i} v{1,i+1}], 1, 1, 1, CopulaSpec.Dynamic);
end

g=1;
for i=1:2
    if  strcmp(CopulaSpec.family{i},'gaussian')
        if strcmp(Dynamic,'ADCC')
            v{2,1}(:,i) = hfunc_grm_mix(v{1,1},v{1,2},phi{1,1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+3;
        elseif strcmp(Dynamic,'DCC')
            v{2,1}(:,i) = hfunc_grm_mix(v{1,1},v{1,2},phi{1,1}(g:g+1),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+2;
        elseif strcmp(Dynamic,'GDCC')
            v{2,1}(:,i) = hfunc_grm_mix(v{1,1},v{1,2},phi{1,1}(g:g+3),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+4;
        elseif strcmp(Dynamic,'AGDCC')
            v{2,1}(:,i) = hfunc_grm_mix(v{1,1},v{1,2},phi{1,1}(g:g+5),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+6;
        end
    elseif strcmp(CopulaSpec.family{i},'t')
        if strcmp(Dynamic,'ADCC')
            v{2,1}(:,i) = hfunc_grm_mix(v{1,1},v{1,2},phi{1,1}(g:g+3),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+4;
        elseif strcmp(Dynamic,'DCC')
            v{2,1}(:,i) = hfunc_grm_mix(v{1,1},v{1,2},phi{1,1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+3;
        elseif strcmp(Dynamic,'GDCC')
            v{2,1}(:,i) = hfunc_grm_mix(v{1,1},v{1,2},phi{1,1}(g:g+4),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+5;
        elseif strcmp(Dynamic,'AGDCC')
            v{2,1}(:,i) = hfunc_grm_mix(v{1,1},v{1,2},phi{1,1}(g:g+6),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+7;
        end
    elseif strcmp(CopulaSpec.family{i},'clayton') || strcmp(CopulaSpec.family{i},'gumbel')
        v{2,1}(:,i) = hfunc_grm_mix(v{1,1},v{1,2},phi{1,1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
        g=g+3;
    end
end


% mixe die beiden h_funcs zu einer funktion - nehme nur den letzen
% (vorhergesagten) Wert von v
weight{1,1} = mnrnd(1,weights{1,1},T);
v{2,1} = weight{1,1}.*v{2,1};
v{2,1} = v{2,1}(:,1)+v{2,1}(:,2);

for k=1:N-3
    g=1;
    for i=1:2
        if  strcmp(CopulaSpec.family{i},'gaussian')
            if strcmp(Dynamic,'ADCC')
                v{2,2*k}(:,i)=hfunc_grm_mix(v{1,k+2},v{1,k+1},phi{1,k+1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
                v{2,2*k+1}(:,i)=hfunc_grm_mix(v{1,k+1},v{1,k+2},phi{1,k+1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
                g=g+3;
            elseif strcmp(Dynamic,'DCC')
                v{2,2*k}(:,i)=hfunc_grm_mix(v{1,k+2},v{1,k+1},phi{1,k+1}(g:g+1),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
                v{2,2*k+1}(:,i)=hfunc_grm_mix(v{1,k+1},v{1,k+2},phi{1,k+1}(g:g+1),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
                g=g+2;
            elseif strcmp(Dynamic,'GDCC')
                v{2,2*k}(:,i)=hfunc_grm_mix(v{1,k+2},v{1,k+1},phi{1,k+1}(g:g+3),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
                v{2,2*k+1}(:,i)=hfunc_grm_mix(v{1,k+1},v{1,k+2},phi{1,k+1}(g:g+3),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
                g=g+4;
            elseif strcmp(Dynamic,'AGDCC')
                v{2,2*k}(:,i)=hfunc_grm_mix(v{1,k+2},v{1,k+1},phi{1,k+1}(g:g+5),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
                v{2,2*k+1}(:,i)=hfunc_grm_mix(v{1,k+1},v{1,k+2},phi{1,k+1}(g:g+5),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
                g=g+6;
            end
        elseif strcmp(CopulaSpec.family{i},'t')
            if strcmp(Dynamic,'ADCC')
                v{2,2*k}(:,i)=hfunc_grm_mix(v{1,k+2},v{1,k+1},phi{1,k+1}(g:g+3),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
                v{2,2*k+1}(:,i)=hfunc_grm_mix(v{1,k+1},v{1,k+2},phi{1,k+1}(g:g+3),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
                g=g+4;
            elseif strcmp(Dynamic,'DCC')
                v{2,2*k}(:,i)=hfunc_grm_mix(v{1,k+2},v{1,k+1},phi{1,k+1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
                v{2,2*k+1}(:,i)=hfunc_grm_mix(v{1,k+1},v{1,k+2},phi{1,k+1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
                g=g+3;
            elseif strcmp(Dynamic,'GDCC')
                v{2,2*k}(:,i)=hfunc_grm_mix(v{1,k+2},v{1,k+1},phi{1,k+1}(g:g+4),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
                v{2,2*k+1}(:,i)=hfunc_grm_mix(v{1,k+1},v{1,k+2},phi{1,k+1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
                g=g+5;
            elseif strcmp(Dynamic,'AGDCC')
                v{2,2*k}(:,i)=hfunc_grm_mix(v{1,k+2},v{1,k+1},phi{1,k+1}(g:g+6),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
                v{2,2*k+1}(:,i)=hfunc_grm_mix(v{1,k+1},v{1,k+2},phi{1,k+1}(g:g+6),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
                g=g+7;
            end
        elseif strcmp(CopulaSpec.family{i},'clayton') || strcmp(CopulaSpec.family{i},'gumbel')
            v{2,2*k}(:,i)=hfunc_grm_mix(v{1,k+2},v{1,k+1},phi{1,k+1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            v{2,2*k+1}(:,i)=hfunc_grm_mix(v{1,k+1},v{1,k+2},phi{1,k+1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+3;
        end
    end
    weight{1,k+1} = mnrnd(1,weights{1,k+1},T); % die Gewichte orientieren sich an der Stelle des Abhängigkeitsparameters
    v{2,2*k} = weight{1,k+1}.*v{2,2*k};
    v{2,2*k} = v{2,2*k}(:,1)+v{2,2*k}(:,2);
    v{2,2*k+1} = weight{1,k+1}.*v{2,2*k+1};
    v{2,2*k+1} = v{2,2*k+1}(:,1)+v{2,2*k+1}(:,2);
end

g=1;
for i=1:2
    if  strcmp(CopulaSpec.family{i},'gaussian')
        if strcmp(Dynamic,'ADCC')
            v{2,2*N-4}(:,i)=hfunc_grm_mix(v{1,N},v{1,N-1},phi{1,N-1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+3;
        elseif strcmp(Dynamic,'DCC')
            v{2,2*N-4}(:,i)=hfunc_grm_mix(v{1,N},v{1,N-1},phi{1,N-1}(g:g+1),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+2;
        elseif strcmp(Dynamic,'GDCC')
            v{2,2*N-4}(:,i)=hfunc_grm_mix(v{1,N},v{1,N-1},phi{1,N-1}(g:g+3),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+4;
        elseif strcmp(Dynamic,'AGDCC')
            v{2,2*N-4}(:,i)=hfunc_grm_mix(v{1,N},v{1,N-1},phi{1,N-1}(g:g+5),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+6;
        end
    elseif strcmp(CopulaSpec.family{i},'t')
        if strcmp(Dynamic,'ADCC')
            v{2,2*N-4}(:,i)=hfunc_grm_mix(v{1,N},v{1,N-1},phi{1,N-1}(g:g+3),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+4;
        elseif strcmp(Dynamic,'DCC')
            v{2,2*N-4}(:,i)=hfunc_grm_mix(v{1,N},v{1,N-1},phi{1,N-1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+3;
        elseif strcmp(Dynamic,'GDCC')
            v{2,2*N-4}(:,i)=hfunc_grm_mix(v{1,N},v{1,N-1},phi{1,N-1}(g:g+4),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+5;
        elseif strcmp(Dynamic,'AGDCC')
            v{2,2*N-4}(:,i)=hfunc_grm_mix(v{1,N},v{1,N-1},phi{1,N-1}(g:g+6),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+7;
        end
    elseif strcmp(CopulaSpec.family{i},'clayton') || strcmp(CopulaSpec.family{i},'gumbel')
        v{2,2*N-4}(:,i)=hfunc_grm_mix(v{1,N},v{1,N-1},phi{1,N-1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
        g=g+2;
    end
end
% mixe die beiden h_funcs zu einer funktion
weight{1,N-1}=mnrnd(1,weights{1,N-1},T);
v{2,2*N-4}=weight{1,N-1}.*v{2,2*N-4};
v{2,2*N-4}=v{2,2*N-4}(:,1)+v{2,2*N-4}(:,2);

for j=2:N-1
    for i=1:N-j
        [CopParam_pred{j,i}, weights_pred{j,i}] = copulamix_tv_paramforecast_2_grm(phi{j,i}, 2, CopulaSpec.family, [v{j-1+1,2*i-1} v{j-1+1,2*i}], 1, 1, 1, CopulaSpec.Dynamic);
    end
    if j<N-1
        g=1;
        for k=1:2
            if  strcmp(CopulaSpec.family{k},'gaussian')
                if strcmp(Dynamic,'ADCC')
                    v{j+1,1}(:,k) = hfunc_grm_mix(v{j-1+1,1},v{j-1+1,2},phi{j,1}(g:g+2),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+3;
                elseif strcmp(Dynamic,'DCC')
                    v{j+1,1}(:,k) = hfunc_grm_mix(v{j-1+1,1},v{j-1+1,2},phi{j,1}(g:g+1),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+2;
                elseif strcmp(Dynamic,'GDCC')
                    v{j+1,1}(:,k) = hfunc_grm_mix(v{j-1+1,1},v{j-1+1,2},phi{j,1}(g:g+3),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+4;
                elseif strcmp(Dynamic,'AGDCC')
                    v{j+1,1}(:,k) = hfunc_grm_mix(v{j-1+1,1},v{j-1+1,2},phi{j,1}(g:g+5),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+6;
                end
            elseif strcmp(CopulaSpec.family{k},'t')
                if strcmp(Dynamic,'ADCC')
                    v{j+1,1}(:,k) = hfunc_grm_mix(v{j-1+1,1},v{j-1+1,2},phi{j,1}(g:g+3),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+4;
                elseif strcmp(Dynamic,'DCC')
                    v{j+1,1}(:,k) = hfunc_grm_mix(v{j-1+1,1},v{j-1+1,2},phi{j,1}(g:g+2),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+3;
                elseif strcmp(Dynamic,'GDCC')
                    v{j+1,1}(:,k) = hfunc_grm_mix(v{j-1+1,1},v{j-1+1,2},phi{j,1}(g:g+4),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+5;
                elseif strcmp(Dynamic,'AGDCC')
                    v{j+1,1}(:,k) = hfunc_grm_mix(v{j-1+1,1},v{j-1+1,2},phi{j,1}(g:g+6),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+7;
                end
            elseif strcmp(CopulaSpec.family{k},'clayton') || strcmp(CopulaSpec.family{k},'gumbel')
                v{j+1,1}(:,k) = hfunc_grm_mix(v{j-1+1,1},v{j-1+1,2},phi{j,1}(g:g+2),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                g=g+3;
            end
        end
        weight{j,1}=mnrnd(1,weights{j,1},T);
        v{j+1,1}=weight{j,1}.*v{j+1,1};
        v{j+1,1}=v{j+1,1}(:,1)+v{j+1,1}(:,2);
        if N>4
            for i=1:(N-2-j)
                g=1;
                for k=1:2
                    if  strcmp(CopulaSpec.family{k},'gaussian')
                        if strcmp(Dynamic,'ADCC')
                            v{j+1,2*i}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+2},v{j-1+1,2*i+1},phi{j,i+1}(g:g+2),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                            v{j+1,2*i+1}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+1},v{j-1+1,2*i+2},phi{j,i+1}(g:g+2),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                            g=g+3;
                        elseif strcmp(Dynamic,'DCC')
                            v{j+1,2*i}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+2},v{j-1+1,2*i+1},phi{j,i+1}(g:g+1),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                            v{j+1,2*i+1}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+1},v{j-1+1,2*i+2},phi{j,i+1}(g:g+1),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                            g=g+2;
                        elseif strcmp(Dynamic,'GDCC')
                            v{j+1,2*i}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+2},v{j-1+1,2*i+1},phi{j,i+1}(g:g+3),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                            v{j+1,2*i+1}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+1},v{j-1+1,2*i+2},phi{j,i+1}(g:g+3),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                            g=g+4;
                        elseif strcmp(Dynamic,'AGDCC')
                            v{j+1,2*i}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+2},v{j-1+1,2*i+1},phi{j,i+1}(g:g+5),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                            v{j+1,2*i+1}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+1},v{j-1+1,2*i+2},phi{j,i+1}(g:g+5),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                            g=g+6;
                        end
                    elseif strcmp(CopulaSpec.family{k},'t')
                        if strcmp(Dynamic,'ADCC')
                            v{j+1,2*i}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+2},v{j-1+1,2*i+1},phi{j,i+1}(g:g+3),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                            v{j+1,2*i+1}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+1},v{j-1+1,2*i+2},phi{j,i+1}(g:g+3),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                            g=g+4;
                        elseif strcmp(Dynamic,'DCC')
                            v{j+1,2*i}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+2},v{j-1+1,2*i+1},phi{j,i+1}(g:g+2),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                            v{j+1,2*i+1}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+1},v{j-1+1,2*i+2},phi{j,i+1}(g:g+2),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                            g=g+3;
                        elseif strcmp(Dynamic,'GDCC')
                            v{j+1,2*i}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+2},v{j-1+1,2*i+1},phi{j,i+1}(g:g+4),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                            v{j+1,2*i+1}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+1},v{j-1+1,2*i+2},phi{j,i+1}(g:g+4),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                            g=g+5;
                        elseif strcmp(Dynamic,'AGDCC')
                            v{j+1,2*i}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+2},v{j-1+1,2*i+1},phi{j,i+1}(g:g+6),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                            v{j+1,2*i+1}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+1},v{j-1+1,2*i+2},phi{j,i+1}(g:g+6),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                            g=g+7;
                        end
                    elseif strcmp(CopulaSpec.family{k},'clayton') || strcmp(CopulaSpec.family{k},'gumbel')
                        v{j+1,2*i}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+2},v{j-1+1,2*i+1},phi{j,i+1}(g:g+2),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                        v{j+1,2*i+1}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+1},v{j-1+1,2*i+2},phi{j,i+1}(g:g+2),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                        g=g+3;
                    end
                end
                % mixe die beiden h_funcs zu einer funktion
                weight{j,i+1}=mnrnd(1,weights{j,i+1},T);
                v{j+1,2*i}=weight{j,i+1}.*v{j+1,2*i};
                v{j+1,2*i}=v{j+1,2*i}(:,1)+v{j+1,2*i}(:,2);
                v{j+1,2*i+1}=weight{j,i+1}.*v{j+1,2*i+1};
                v{j+1,2*i+1}=v{j+1,2*i+1}(:,1)+v{j+1,2*i+1}(:,2);
            end
        end
    end
    if (2*N-2*j-2) > 0
        g=1;
        for k=1:2
            if  strcmp(CopulaSpec.family{k},'gaussian')
                if strcmp(Dynamic,'ADCC')
                    v{j+1,2*N-2*j-2}(:,k) = hfunc_grm_mix(v{j-1+1,2*N-2*j},v{j-1+1,2*N-2*j-1},phi{j,N-j}(g:g+2),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+3;
                elseif strcmp(Dynamic,'DCC')
                    v{j+1,2*N-2*j-2}(:,k) = hfunc_grm_mix(v{j-1+1,2*N-2*j},v{j-1+1,2*N-2*j-1},phi{j,N-j}(g:g+1),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+2;
                elseif strcmp(Dynamic,'GDCC')
                    v{j+1,2*N-2*j-2}(:,k) = hfunc_grm_mix(v{j-1+1,2*N-2*j},v{j-1+1,2*N-2*j-1},phi{j,N-j}(g:g+3),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+4;
                elseif strcmp(Dynamic,'AGDCC')
                    v{j+1,2*N-2*j-2}(:,k) = hfunc_grm_mix(v{j-1+1,2*N-2*j},v{j-1+1,2*N-2*j-1},phi{j,N-j}(g:g+5),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+6;
                end
            elseif  strcmp(CopulaSpec.family{k},'t')
                if strcmp(Dynamic,'ADCC')
                    v{j+1,2*N-2*j-2}(:,k) = hfunc_grm_mix(v{j-1+1,2*N-2*j},v{j-1+1,2*N-2*j-1},phi{j,N-j}(g:g+3),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+4;
                elseif strcmp(Dynamic,'DCC')
                    v{j+1,2*N-2*j-2}(:,k) = hfunc_grm_mix(v{j-1+1,2*N-2*j},v{j-1+1,2*N-2*j-1},phi{j,N-j}(g:g+2),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+3;
                elseif strcmp(Dynamic,'GDCC')
                    v{j+1,2*N-2*j-2}(:,k) = hfunc_grm_mix(v{j-1+1,2*N-2*j},v{j-1+1,2*N-2*j-1},phi{j,N-j}(g:g+4),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+5;
                elseif strcmp(Dynamic,'AGDCC')
                    v{j+1,2*N-2*j-2}(:,k) = hfunc_grm_mix(v{j-1+1,2*N-2*j},v{j-1+1,2*N-2*j-1},phi{j,N-j}(g:g+6),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+7;
                end
            elseif strcmp(CopulaSpec.family{k},'clayton') || strcmp(CopulaSpec.family{k},'gumbel')
                v{j+1,2*N-2*j-2}(:,k) = hfunc_grm_mix(v{j-1+1,2*N-2*j},v{j-1+1,2*N-2*j-1},phi{j,N-j}(g:g+2),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                g=g+3;
            end
        end
        % mixe die beiden h_funcs zu einer funktion
        weight{j,N-j}=mnrnd(1,weights{j,N-j},T);
        v{j+1,2*N-2*j-2}=weight{j,N-j}.*v{j+1,2*N-2*j-2};
        v{j+1,2*N-2*j-2}=v{j+1,2*N-2*j-2}(:,1)+v{j+1,2*N-2*j-2}(:,2);
    end
end

