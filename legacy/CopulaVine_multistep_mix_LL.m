function [StrOutput,phi] = CopulaVine_multistep_mix_LL(CopulaSpec,N,data,T,VineOutput)

% Helper function for fitCopulaVine_multistep_mix_grm
% Estimates multistep D-Vine
%
% Author: Martin Grziska
% Date: 07/27/2010

% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

%     Estimation of first D-Vine level
% put data into cell
v=cell(N,N);
phi = cell(N-1,N-1);
likhood = cell(N-1,N-1);

for i=1:N
    v{1,i} = data(:,i);
end

% genriere Weights, die auf dem gesamten Datensatz beruhen
for i = 1:N-1
    [LL_Mix{1,i}, LL{1,i} CopParam_tv{1,i}, weights_tv{1,i}, density{1,i}, likelihoods_mix{1,i}, likelihoods{1,i}] = copulaLL_mix_tv_grm(VineOutput.VineParams{1,i}, VineOutput.CopParam_1{1,i}, VineOutput.Weights{1,i}(1,:), CopulaSpec.family, [v{1,i} v{1,i+1}], 1, 1, CopulaSpec.Dynamic);
end

% clayton und gumbel folgen immer der Patton-Spezifikation; lese jeweils
% drei Elemente aus; gaussian hat ADCC-Spezifikation und ebenfalls drei Elemente; t folgt ADCC hat aber
% vier Elemente (d.o.f.); erzeuge für jede copula eine eigene Zeitreihe und
% "mixe" diese dann im Anschluss mit (multivariater) binomial-Verteilung
g=1;
for i=1:2
    if strcmp(CopulaSpec.family{i},'clayton') || strcmp(CopulaSpec.family{i},'gumbel') || strcmp(CopulaSpec.family{i},'gaussian')
        v{2,1}(:,i) = hfunc_grm_mix(v{1,1},v{1,2},VineOutput.VineParams{1,1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
        g=g+3;
    elseif strcmp(CopulaSpec.family{i},'t')
        v{2,1}(:,i) = hfunc_grm_mix(v{1,1},v{1,2},VineOutput.VineParams{1,1}(g:g+3),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
        g=g+4;
    end
end
% mixe die beiden h_funcs zu einer funktion
weight{1,1}=mnrnd(1,weights_tv{1,1},T);
v{2,1}=weight{1,1}.*v{2,1};
v{2,1}=v{2,1}(:,1)+v{2,1}(:,2);

g=1;
for k=1:N-3
    for i=1:2
        if strcmp(CopulaSpec.family{i},'clayton') || strcmp(CopulaSpec.family{i},'gumbel') || strcmp(CopulaSpec.family{i},'gaussian')
            v{2,2*k}(:,i)=hfunc_grm_mix(v{1,k+2},v{1,k+1},VineOutput.VineParams{1,k+1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            v{2,2*k+1}(:,i)=hfunc_grm_mix(v{1,k+1},v{1,k+2},VineOutput.VineParams{1,k+1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+3;
        elseif strcmp(CopulaSpec.family{i},'t')
            v{2,2*k}(:,i)=hfunc_grm_mix(v{1,k+2},v{1,k+1},VineOutput.VineParams{1,k+1}(g:g+3),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            v{2,2*k+1}(:,i)=hfunc_grm_mix(v{1,k+1},v{1,k+2},VineOutput.VineParams{1,k+1}(g:g+3),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
            g=g+4;
        end
    end
    weight{1,k+1}=mnrnd(1,weights_tv{1,k+1},T);
    v{2,2*k}=weight{1,k+1}.*v{2,2*k};
    v{2,2*k}=v{2,2*k}(:,1)+v{2,2*k}(:,2);
    v{2,2*k+1}=weight{1,k+1}.*v{2,2*k+1};
    v{2,2*k+1}=v{2,2*k+1}(:,1)+v{2,2*k+1}(:,2);
end

g=1;
for i=1:2
    if strcmp(CopulaSpec.family{i},'clayton') || strcmp(CopulaSpec.family{i},'gumbel') || strcmp(CopulaSpec.family{i},'gaussian')
        v{2,2*N-4}(:,i)=hfunc_grm_mix(v{1,N},v{1,N-1},VineOutput.VineParams{1,N-1}(g:g+2),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
        g=g+2;
    elseif strcmp(CopulaSpec.family{i},'t')
        v{2,2*N-4}(:,i)=hfunc_grm_mix(v{1,N},v{1,N-1},VineOutput.VineParams{1,N-1}(g:g+3),CopulaSpec.family{i},CopulaSpec.corrspec{i},CopulaSpec.optimizer);
        g=g+3;
    end
end
% mixe die beiden h_funcs zu einer funktion
weight{1,N-1}=mnrnd(1,weights_tv{1,N-1},T);
v{2,2*N-4}=weight{1,N-1}.*v{2,2*N-4};
v{2,2*N-4}=v{2,2*N-4}(:,1)+v{2,2*N-4}(:,2);

for j=2:N-1
    for i=1:N-j
        [LL_Mix{j,i}, LL{j,i} CopParam_tv{j,i}, weights_tv{j,i}, density{j,i}, likelihoods_mix{j,i}, likelihoods{j,i}] = copulaLL_mix_tv_grm(VineOutput.VineParams{j,i}, VineOutput.CopParam_1{j,i}, VineOutput.Weights{j,i}(1,:), CopulaSpec.family, [v{j-1+1,2*i-1} v{j-1+1,2*i}], 1, 1, CopulaSpec.Dynamic);
    end
    if j<N-1
        g=1;
        for k=1:2
            if strcmp(CopulaSpec.family{k},'clayton') || strcmp(CopulaSpec.family{k},'gumbel') || strcmp(CopulaSpec.family{k},'gaussian')
                v{j+1,1}(:,k) = hfunc_grm_mix(v{j-1+1,1},v{j-1+1,2},VineOutput.VineParams{j,1}(g:g+2),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                g=g+3;
            elseif strcmp(CopulaSpec.family{i},'t')
                v{j+1,1}(:,k) = hfunc_grm_mix(v{j-1+1,1},v{j-1+1,2},VineOutput.VineParams{j,1}(g:g+3),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                g=g+4;
            end
        end
        weight{j,i}=mnrnd(1,weights_tv{j,i},T);
        v{j+1,1}=weight{j,i}.*v{j+1,1};
        v{j+1,1}=v{j+1,1}(:,1)+v{j+1,1}(:,2);
    end
    if N>4
        g=1;
        for i=1:(N-2-j)
            for k=1:2
                if strcmp(CopulaSpec.family{i},'clayton') || strcmp(CopulaSpec.family{i},'gumbel') || strcmp(CopulaSpec.family{i},'gaussian')
                    v{j+1,2*i}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+2},v{j-1+1,2*i+1},VineOutput.VineParams{j,i+1}(g:g+2),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    v{j+1,2*i+1}(:,k) = hfunc_grm(v{j-1+1,2*i+1},v{j-1+1,2*i+2},VineOutput.VineParams{j,i+1}(g:g+2),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec);
                    g=g+3;
                elseif strcmp(CopulaSpec.family{i},'t')
                    v{j+1,2*i}(:,k) = hfunc_grm_mix(v{j-1+1,2*i+2},v{j-1+1,2*i+1},VineOutput.VineParams{j,i+1}(g:g+3),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    v{j+1,2*i+1}(:,k) = hfunc_grm(v{j-1+1,2*i+1},v{j-1+1,2*i+2},VineOutput.VineParams{j,i+1}(g:g+3),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                    g=g+4;
                end
            end
            % mixe die beiden h_funcs zu einer funktion
            weight{j,i+1}=mnrnd(1,weights_tv{j,i+1},T);
            v{j+1,2*i}=weight{j,i+1}.*v{j+1,2*i};
            v{j+1,2*i}=v{j+1,2*i}(:,1)+v{j+1,2*i}(:,2);
            v{j+1,2*i+1}=weight{j,i+1}.*v{j+1,2*i+1};
            v{j+1,2*i+1}=v{j+1,2*i+1}(:,1)+v{j+1,2*i+1}(:,2);
        end
    end
    if (2*N-2*j-2) > 0
        g=1;
        for k=1:2
            if strcmp(CopulaSpec.family{i},'clayton') || strcmp(CopulaSpec.family{i},'gumbel') || strcmp(CopulaSpec.family{i},'gaussian')
                v{j+1,2*N-2*j-2}(:,k) = hfunc_grm_mix(v{j-1+1,2*N-2*j},v{j-1+1,2*N-2*j-1},VineOutput.VineParams{j,N-j}(g:g+2),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                g=g+3;
            elseif  strcmp(CopulaSpec.family{i},'t')
                v{j+1,2*N-2*j-2}(:,k) = hfunc_grm_mix(v{j-1+1,2*N-2*j},v{j-1+1,2*N-2*j-1},VineOutput.VineParams{j,N-j}(g:g+3),CopulaSpec.family{k},CopulaSpec.corrspec{k},CopulaSpec.optimizer);
                g=g+4;
            end
        end
        % mixe die beiden h_funcs zu einer funktion
        weight{j,N-j}=mnrnd(1,weights_tv{j,N-j},T);
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
StrOutput.VineDepParams = CopParam_tv;
StrOutput.Weights = weights_tv;
StrOutput.timeinseconds=toc;
StrOutput.LogL = -likelihood;