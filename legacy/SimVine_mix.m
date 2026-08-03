function [x] = SimVine_mix(T,CopulaSpec,VineOutput,depparam,weight)
% Simulation for mixed copula-vines after Aas et al (2006)
%
% USAGE:   [x] = SimVine_mix(T,CopulaSpec,VineOutput,Rt,weight)
%
% INPUT:
%          VineOutput : structure containing several elements ( CopParams,
%                       Correlation Params)
%          T          : number of observations to be sampled
%          CopulaSpec : Structure of Copula specification, possible
%                       Copula-types: 'Clayton','t',
%          depparam   : special Matrix if correlation is time-varying; see
%                       VineVaR.m;
%          weight     : 1x2 vector containing weights for time t
%
% OUTPUT:  N x t matrix of simualted variables
%
% Author:  Martin Grziska, last modification: August/ 05/ 2010
% % ---------------------------------------------------------------------


N=size(VineOutput.VineParams,2)+1;

Dynamic = CopulaSpec.Dynamic;
w = cell(N-1,1);
for i = 1:N
    for j=1:2
        w{i}(:,j)= random('unif',0,1,T,1);
    end
end

for k=1:2
    if strcmp(CopulaSpec.family{k},'t')
        if k == 2
            nu=cell(N-1,N-1);
            c=N-1;
            for i=1:N-1
                for j=1:c
                    if strcmp(Dynamic,'DCC')
                        nu{i,j} = VineOutput.VineParams{i,j}(3);
                    elseif strcmp(Dynamic,'ADCC')
                        nu{i,j} = VineOutput.VineParams{i,j}(4);
                    elseif strcmp(Dynamic,'GDCC')
                        nu{i,j} = VineOutput.VineParams{i,j}(5);
                    elseif strcmp(Dynamic,'AGDCC')
                        nu{i,j} = VineOutput.VineParams{i,j}(7);
                    end
                end
                c=c-1;
            end
        elseif k == 1 %für statische Copula
            nu=cell(N-1,N-1);
            c=N-1;
            for i=1:N-1
                for j=1:c
                    if strcmp(Dynamic,'DCC')
                        nu{i,j} = VineOutput.VineParams{i,j}(3);
                    elseif strcmp(Dynamic,'ADCC')
                        nu{i,j} = VineOutput.VineParams{i,j}(4);
                    elseif strcmp(Dynamic,'GDCC')
                        nu{i,j} = VineOutput.VineParams{i,j}(5);
                    elseif strcmp(Dynamic,'AGDCC')
                        nu{i,j} = VineOutput.VineParams{i,j}(7);
                    end
                end
                c=c-1;
            end
        end
    end
end


% % ---------------------------------------------------------------------
if strcmp(CopulaSpec.decomposition,'DVine') || strcmp(CopulaSpec.decomposition,'DVine-Mix')  == 1
    v = cell(N,N+1);
    v{1,1} = w{1}(:,1);
    x(:,1) = v{1,1};
    for i=1:2
%         depparam needs to be a scalar, so change matrix to scalar if
%         necessary
        if size(depparam{1,1}{i},1) >1 % aus den bivariaten Korrelationsmatrizen von gauss und t wird der korrelationsparameter ausgelesen
            depparam{1,1}{i}=rho2theta(depparam{1,1}{i});
        end
        if strcmp(CopulaSpec.family{i},'clayton')
            v{2,1}(:,i) = ((w{2}(:,i).*v{1,1}.^(depparam{1,1}{i}+1)).^(-depparam{1,1}{i}./(depparam{1,1}{i}+1)) + 1 - v{1,1}.^(-depparam{1,1}{i})).^(-1./depparam{1,1}{i});
        elseif strcmp(CopulaSpec.family{i},'t')
            v{2,1}(:,i) = tcdf(tinv(w{2}(:,i),nu{1,1}+1) .*sqrt((nu{1,1}+((tinv(v{1,1},nu{1,1})).^2).*(1-depparam{1,1}{i}))./(nu{1,1}+1)) + depparam{1,1}{i}*tinv(v{1,1},nu{1,1}),nu{1,1});
        elseif strcmp(CopulaSpec.family{i},'gumbel')
            
        elseif strcmp(CopulaSpec.family{i},'gaussian')
            v{2,1}(:,i) = normcdf(norminv(w{2}(:,i))*sqrt(1-depparam{1,1}{i}^2)+depparam{1,1}{i}*norminv(v{1,1},0,1));
        end
    end
    nom = mnrnd(1,repmat(weight{1,1},T,1),T); %die vorhergesagten Gewichte müssen auf die datenlänge gebracht werden
    v{2,1} = nom(:,1).*v{2,1}(:,1) + nom(:,2).*v{2,1}(:,2);
    x(:,2) = v{2,1};
    for i=1:2
        if size(depparam{1,1}{i},1) >1
            depparam{1,1}{i}=rho2theta(depparam{1,1}{i});
        end
        if strcmp(CopulaSpec.family{i},'clayton')
            v{2,2}(:,i) = v{2,1}.^(-1-depparam{1,1}{i}).*(v{1,1}.^(-depparam{1,1}{i}) + v{2,1}.^(-depparam{1,1}{i}) - 1).^(-1-1./depparam{1,1}{i});
        elseif strcmp(CopulaSpec.family{i},'t')
            v{2,2}(:,i) = tcdf((tinv(v{1,1},nu{1,1})-depparam{1,1}{i}.*tinv(v{2,1},nu{1,1}))./sqrt(nu{1,1}+((tinv(v{2,1},nu{1,1})).^2).*(1-depparam{1,1}{i})./(nu{1,1}+1)),(nu{1,1}+1));
        elseif strcmp(CopulaSpec.family{i},'gumbel')
            
        elseif strcmp(CopulaSpec.family{i},'gaussian')
            v{2,2}(:,i) = normcdf((norminv(v{1,1})-depparam{1,1}{i}*norminv(v{2,1}))./(sqrt(1-depparam{1,1}{i}^2)));
        end
    end
    nom = mnrnd(1,repmat(weight{1,1},T,1),T);
    v{2,2} = nom(:,1).*v{2,2}(:,1) + nom(:,2).*v{2,2}(:,2);
    for i = 3:N
        index = 2:i;
        index = sort(index,'descend');
        for s = 1:(size(index,2)-1)
            k = index(s)-1;
            for j=1:2
                v{i,1}(:,j) = w{i}(:,j);
                if size(depparam{k,i-k}{j},1) >1
                    depparam{k,i-k}{j}=rho2theta(depparam{k,i-k}{j});
                end
                if strcmp(CopulaSpec.family{j},'clayton')
                    v{i,1}(:,j) = ((v{i,1}(:,j).*v{i-1,2*k-2}.^(depparam{k,i-k}{j}+1)).^(-depparam{k,i-k}{j}./(depparam{k,i-k}{j}+1)) + 1 - v{i-1,2*k-k}.^(-depparam{k,i-k}{j})).^(-1./depparam{k,i-k}{j});
                elseif strcmp(CopulaSpec.family{j},'t')
                    v{i,1}(:,j) =  tcdf(tinv(v{i,1}(:,j),nu{k,i-k}).*sqrt((nu{k,i-k}+(tinv(v{i-1,2*k-2},nu{k,i-k}).^2).*(1-depparam{k,i-k}))./(nu{k,i-k}+1)) + depparam{k,i-k}*tinv(v{i-1,2*k-2},nu{k,i-k}),nu{k,i-k}) ;
                elseif strcmp(CopulaSpec.family{j},'gumbel')
                    
                elseif strcmp(CopulaSpec.family{j},'gaussian')
                    v{i,1}(:,j) =  normcdf(norminv(v{i,1}(:,j))*sqrt(1-depparam{k,i-k}{j}^2)+depparam{k,i-k}{j}*norminv(v{i-1,2*k-2}));
                end
            end
            nom = mnrnd(1,repmat(weight{k,i-k},T,1),T);
            v{i,1} = nom(:,1).*v{i,1}(:,1)+nom(:,2).*v{i,1}(:,2);
        end
        for j=1:2
            if size(depparam{1,i-1}{j},1) >1
                depparam{1,i-1}{j}=rho2theta(depparam{1,i-1}{j});
            end
            if strcmp(CopulaSpec.family{j},'clayton')
                v{i,1}(:,j) = ((v{i,1}.*v{i-1,1}.^(depparam{1,i-1}{j}+1)).^(-depparam{1,i-1}{j}./(depparam{1,i-1}{j}+1)) + 1 - v{i-1,1}.^(-depparam{1,i-1}{j})).^(-1./depparam{1,i-1}{j});
            elseif strcmp(CopulaSpec.family{j},'t')
                v{i,1} =  tcdf(tinv(v{i,1},nu{1,i-1}).*sqrt((nu{1,i-1}+(tinv(v{i-1,1},nu{1,i-1}).^2).*(1-depparam{1,i-1}))./(nu{1,i-1}+1)) + depparam{1,i-1}*tinv(v{i-1,1},nu{1,i-1}),nu{1,i-1}) ;
            elseif strcmp(CopulaSpec.family{j},'gumbel')
                
            elseif strcmp(CopulaSpec.family{j},'gaussian')
                v{i,1}(:,j) =  normcdf(norminv(v{i,1})*sqrt(1-depparam{1,i-1}{j}^2)+depparam{1,i-1}{j}*norminv(v{i-1,1}));
            end
        end
        nom = mnrnd(1,repmat(weight{1,i-1},T,1),T);
        v{i,1} = nom(:,1).*v{i,1}(:,1)+nom(:,2).*v{i,1}(:,2);
        x(:,i) = v{i,1};
        if i < N
            for j=1:2
                if size(depparam{1,i-1}{j},1) >1
                    depparam{1,i-1}{j}=rho2theta(depparam{1,i-1}{j});
                end
                if strcmp(CopulaSpec.family{j},'clayton')
                    v{i,2}(:,j) = v{i,1}.^(-1-depparam{1,i-1}{j}).*(v{i-1,1}.^(-depparam{1,i-1}{j}) + v{i,1}.^(-depparam{1,i-1}{j}) - 1).^(-1-1./depparam{1,i-1}{j});
                    v{i,3}(:,j) = v{i-1,1}.^(-1-depparam{1,i-1}{j}).*(v{i,1}.^(-depparam{1,i-1}{j}) + v{i-1,1}.^(-depparam{1,i-1}{j}) - 1).^(-1-1./depparam{1,i-1}{j});
                elseif strcmp(CopulaSpec.family{j},'t')
                    v{i,2}(:,j) = tcdf((tinv(v{i-1,1},nu{1,i-1})-depparam{1,i-1}{j}.*tinv(v{i,1},nu{1,i-1}))./sqrt(nu{1,i-1}+(tinv(v{i,1},nu{1,i-1}).^2).*(1-depparam{1,i-1}{j}))./(nu{1,i-1}+1),nu{1,i-1}+1);
                    v{i,3}(:,j) = tcdf((tinv(v{i,1},nu{1,i-1})-depparam{1,i-1}(:,j).*tinv(v{i-1,1},nu{1,i-1}))./sqrt(nu{1,i-1}+(tinv(v{i-1,1},nu{1,i-1}).^2).*(1-depparam{1,i-1}))./(nu{1,i-1}+1),nu{1,i-1}+1);
                elseif strcmp(CopulaSpec.family{j},'gumbel')
                    
                elseif strcmp(CopulaSpec.family{j},'gaussian')
                    v{i,2}(:,j) = normcdf((norminv(v{i-1,1})-depparam{1,i-1}{j}*norminv(v{i,1}))./(sqrt(1-depparam{1,i-1}{j}^2)));
                    v{i,3}(:,j) = normcdf((norminv(v{i,1})-depparam{1,i-1}{j}*norminv(v{i-1,1}))./(sqrt(1-depparam{1,i-1}{j}^2)));
                end
            end
            nom = mnrnd(1,repmat(weight{1,i-1},T,1),T);
            v{i,2} = nom(:,1).*v{i,2}(:,1) + nom(:,2).*v{i,2}(:,2);
            v{i,3} = nom(:,1).*v{i,3}(:,1) + nom(:,2).*v{i,3}(:,2);
            if i>3
                for j = 2:i-2
                    for k = 1:2
                        if size(depparam{j,i-j}{k},1) >1
                            depparam{j,i-j}{k}=rho2theta(depparam{j,i-j}{k});
                        end
                        if strcmp(CopulaSpec.family{k},'clayton')
                            v{i,2*j}(:,k) = v{i,2*j-1}.^(-1-depparam{j,i-j}{k}).*(v{i-1,2*j-2}.^(-depparam{j,i-j}{k}) + v{i,2*j-1}.^(-depparam{j,i-j}{k}) - 1).^(-1-1./depparam{j,i-j}{k});
                            v{i,2*j+1}(:,k) = v{i-1,2*j-2}.^(-1-depparam{j,i-j}{k}).*(v{2*j-1,2*j-1}.^(-depparam{j,i-j}{k}) + v{i-1,2*j-2}.^(-depparam{j,i-j}{k}) - 1).^(-1-1./depparam{j,i-j}{k});
                        elseif strcmp(CopulaSpec.family{k},'t')
                            v{i,2*j}(:,k) = tcdf((tinv(v{i-1,2*j-2},nu{j,i-j})-depparam{j,i-j}{k}.*tinv(v{i,2*j-1},nu{j,i-j}))./(sqrt(nu{j,i-j}+(tinv(v{i,2*j-1},nu{j,i-j}).^2).*(1-depparam{j,i-j}{k}))./(nu{j,i-j}+1)),nu{j,i-j}+1);
                            v{i,2*j+1}(:,k) = tcdf((tinv(v{i,2*j-1},nu{j,i-j})-depparam{j,i-j}{k}.*tinv(v{i,2*j-2},nu{j,i-j}))./(sqrt(nu{j,i-j}+(tinv(v{i,2*j-2},nu{j,i-j}).^2).*(1-depparam{j,i-j}{k}))./(nu{j,i-j}+1)),nu{j,i-j}+1);
                        elseif strcmp(CopulaSpec.family{k},'gumbel')
                            
                        elseif strcmp(CopulaSpec.family{k},'gaussian')
                            v{i,2*j}(:,k) = normcdf((norminv(v{i-1,2*j-2})-depparam{j,i-j}{k}*norminv(v{i,2*j-1}))./(sqrt(1-depparam{j,i-j}{k}^2)));
                            v{i,2*j+1}(:,k) = normcdf((norminv(v{i,2*j-1})-depparam{j,i-j}{k}*norminv(v{i-1,2*j-2}))./(sqrt(1-depparam{j,i-j}{k}^2)));
                        end
                    end
                    nom = mnrnd(1,repmat(weight{j,i-j},T,1),T);
                    v{i,2*j} = nom(:,1).* v{i,2*j}(:,1) + nom(:,2).*v{i,2*j}(:,2);
                    v{i,2*j+1} = nom(:,1).*v{i,2*j+1}(:,2) + nom(:,2).*v{i,2*j+1}(:,2);
                end
            end
            for k=1:2
                if size(depparam{i-1,1}{k},1) >1
                    depparam{i-1,1}{k}=rho2theta(depparam{i-1,1}{k});
                end
                if strcmp(CopulaSpec.family{k},'clayton')
                    v{i,2*i-2}(:,k) = v{i-1,2*i-4}.^(-1-depparam{i-1,1}{k}).*(v{i-1,2*i-4}.^(-depparam{i-1,1}{k}) + v{i,2*i-3}.^(-depparam{i-1,1}{k}) - 1).^(-1-1./depparam{i-1,1}{k});
                elseif strcmp(CopulaSpec.family{k},'t')
                    v{i,2*i-2}(:,k) = tcdf((tinv(v{i-1,2*i-4},nu{i-1,1})-depparam{i-1,1}{k}.*tinv(v{i,2*i-3},nu{i-1,1}))./(sqrt(nu{i-1,1}+(tinv(v{i,2*i-3},nu{i-1,1}).^2).*(1-depparam{i-1,1}{k}))./(nu{i-1,1}+1)),nu{i-1,1}+1);
                elseif strcmp(CopulaSpec.family{k},'gumbel')
                    
                elseif strcmp(CopulaSpec.family{k},'gaussian')
                    v{i,2*i-2}(:,k) = normcdf((norminv(v{i-1,2*i-4})-depparam{i-1,1}{k}*norminv(v{i,2*i-3}))./(sqrt(1-depparam{i-1,1}{k}^2)));
                end
            end
            nom = mnrnd(1,repmat(weight{j,i-j},1,T),T);
            v{i,2*i-2} = nom(:,1).*v{i,2*i-2}(:,1) + nom(:,2).*v{i,2*i-2}(:,2);
        end
    end
end

% References:
% Aas, Czado, Frigessi, Bakken (2006): Pair-copula constructions for multiple
% dependence. Ssondeforschungsbereich 386, Paper 487.

% Heinen, Valdesogo: Asymmetric CAPM dependence for large dimensiona: the
% Canonical Vine Autoregressive Model.


