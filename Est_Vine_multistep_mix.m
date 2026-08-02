function [StrOutput,phi] = Est_Vine_multistep_mix(CopulaSpec,N,data,T,Cop_Stat,PrtFig)

% Helper function for fitCopulaVine_multistep_mix_grm
% Estimates multistep D-Vine
%
% Author: Martin Grziska
% Date: 09/07/2010

% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

% Dynamik der elliptischen Copula
Dynamic=CopulaSpec.Dynamic;



%     Estimation of first D-Vine level
% put data into cell
v=cell(N,N);
phi = cell(N-1,N-1);
likhood = cell(N-1,N-1);

for i=1:N
    v{1,i} = data(:,i);
end

% schätze die erste Ebene der D-Vine
for i=1:N-1
    [CopParam_tv{1,i}, Weights_tv{1,i}, phi{1,i}, LL_Mix{1,i}, LL{1,i}, AIC{1,i}, BIC{1,i}, CopParam_1{1,i}] = copulafitmix_tv_grm2_DVine(CopulaSpec.family, [v{1,i} v{1,i+1}],Dynamic,Cop_Stat,PrtFig);
% setze Gewicht die < 0 sind (kann durch Rundungsfehler enstehen) = 0;
    for h=1:size(Weights_tv{1,i},1)
        for j=1:size(Weights_tv{1,i},2)
            if Weights_tv{1,i}(h,j)<0
                Weights_tv{1,i}(h,j)=0;
            end
        end
    end    
end

% clayton und gumbel folgen immer der Patton-Spezifikation; lese jeweils
% drei Elemente aus; gaussian hat ADCC-Spezifikation und ebenfalls drei Elemente; t folgt ADCC hat aber
% vier Elemente (d.o.f.); erzeuge für jede copula eine eigene Zeitreihe und
% "mixe" diese dann im Anschluss mit (multivariater) binomial-Verteilung
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

% mixe die beiden h_funcs zu einer funktion
weight{1,1}=mnrnd(1,Weights_tv{1,1},T); %die v's zu den weights sind immer um eine Reihe versetzt, da für die ersten v's keine Gewicht benötigt werden
v{2,1}=weight{1,1}.*v{2,1};
v{2,1}=v{2,1}(:,1)+v{2,1}(:,2);


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
    weight{1,k+1} = mnrnd(1,Weights_tv{1,k+1},T); % die Gewichte orientieren sich an der Stelle des Abhängigkeitsparameters
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
weight{1,N-1}=mnrnd(1,Weights_tv{1,N-1},T);
v{2,2*N-4}=weight{1,N-1}.*v{2,2*N-4};
v{2,2*N-4}=v{2,2*N-4}(:,1)+v{2,2*N-4}(:,2);

for j=2:N-1
    for i=1:N-j
        [CopParam_tv{j,i}, Weights_tv{j,i}, phi{j,i}, LL_Mix{j,i}, LL{j,i}, AIC{j,i}, BIC{j,i}, CopParam_1{j,i}] = copulafitmix_tv_grm2_DVine(CopulaSpec.family,[v{j-1+1,2*i-1} v{j-1+1,2*i}],Dynamic,Cop_Stat,PrtFig);
        % setze Gewicht die < 0 sind (kann durch Rundungsfehler enstehen) = 0;
        for h=1:size(Weights_tv{j,i},1)
            for l=1:size(Weights_tv{j,i},2)
                if Weights_tv{j,i}(h,l)<0
                    Weights_tv{j,i}(h,l)=0;
                end
            end
        end
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
        weight{j,1}=mnrnd(1,Weights_tv{j,1},T);
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
                weight{j,i+1}=mnrnd(1,Weights_tv{j,i+1},T);
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
        weight{j,N-j}=mnrnd(1,Weights_tv{j,N-j},T);
        v{j+1,2*N-2*j-2}=weight{j,N-j}.*v{j+1,2*N-2*j-2};
        v{j+1,2*N-2*j-2}=v{j+1,2*N-2*j-2}(:,1)+v{j+1,2*N-2*j-2}(:,2);
    end
end


%     calculate complete likelihood (see also Aas)
c=N-1;
likelihood=0;
for i=1:N-1
    for j=1:c
        likelihood = likelihood + likhood{i,j};
    end
    c=c-1;
end

% Generate Output
StrOutput.VineParams = phi;
StrOutput.VineDepParams = CopParam_tv;
StrOutput.Weights = Weights_tv;
% StrOutput.timeinseconds=toc;
StrOutput.CopParam_1 = CopParam_1;
StrOutput.phi = phi;
StrOutput.AIC = AIC;
StrOutput.BIC = BIC;
StrOutput.LogL = -likelihood;