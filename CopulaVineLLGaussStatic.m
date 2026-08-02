function [LogL,Rt]=CopulaVineLLGaussStatic(data,CopulaSpec)
% ---- Log Likelihood functions of the supported copula Vines -----
% INPUTS:
% phi:          vector of parameters
% data:         matrix with U(0,1) margins
% CopulaSpec:   structured array that contains the various input arguments
%               that define the parameters. To obtain run the function
%               setCopulaLL_grm inputs.m
% OUTPUT:
% LogL:         The log - likelihood of the corresponding copula
% ------------------------------------------------------------------------
% author: Martin Grziska based on a code of Manthos Vogiatzoglou, February,
% 3nd, 2010
% contact at: vogia@yahoo.com
% ------------------------------------------------------------------------
dec=CopulaSpec.decomposition;
[T,N]=size(data);
if min(min(data))<0 || max(max(data))>1
    data=empiricalCDF(data);
end
lik=zeros(N-1,N-1);
Rt=cell(N-1,N-1);
if strcmp(dec,'CanonicalVine') == 1
    for j=1:N-1
        for i=1:N-j
            if j==1
                [lik(j,i), Rt{j,i}]=CopulaLL_grm(1,[data(:,1),data(:,i+1)],CopulaSpec);
            else
                [lik(j,i),Rt{j,i}]=CopulaLL_grm(1,[u{j-1,1},u{j-1,i+1}],CopulaSpec);
            end
        end
        if j<N-1
            for i=1:N-j
                if j==1
                    u{1,i}=hfunc_grm(data(:,i+1),data(:,1),1,CopulaSpec);
                else
                    u{j,i}=hfunc_grm(u{j-1,i+1},u{j-1,1},1,CopulaSpec);
                end
            end
        end
    end
elseif strcmp(dec,'DVine')==1
    %         Bilde die D-vine Struktur. wie in Aas et al., S.4
    for i=1:N-1
        %             Log-Likelihood-Funktion
            [lik(1,i),Rt{1,i}]=CopulaLL_grm(1,[data(:,i) data(:,i+1)],CopulaSpec);
    end
    %       h-Funktion wie in Aas et al., S.3
    %        Erstes Element der Dekomposition
        u{1,1}=hfunc_grm(data(:,1),data(:,2),1,CopulaSpec);
    %         Elemente zwischen dem Ersten und dem Letzten
    for k=1:N-3
            u{1,2*k}=hfunc_grm(data(:,k+2),data(:,k+1),1,CopulaSpec);
            u{1,2*k+1}=hfunc_grm(data(:,k+1),data(:,k+2),1,CopulaSpec);
    end
    %         erste Ebene der Dekomposition, letztes Element der ersten
    %         Zeile
        u{1,2*N-4}=hfunc_grm(data(:,N),data(:,N-1),1,CopulaSpec);
    for j=2:(N-1)
        for i=1:(N-j)
                [lik(j,i),Rt{j,i}]=CopulaLL_grm(1,[u{j-1,2*i-1} u{j-1,2*i}],CopulaSpec);
        end
        %             alle anderen Ebenen der Dekomposition
        %             (conditional Copula)
        if j<N-1
                u{j,1}=hfunc_grm(u{j-1,1},u{j-1,2},1,CopulaSpec);
            if N>4
                for i=1:(N-j-2)
                        u{j,2*i}=hfunc_grm(u{j-1,2*i+2},u{j-1,2*i+1},1,CopulaSpec);
                        u{j,2*i+1}=hfunc_grm(u{j-1,2*i+1},u{j-1,2*i+2},1,CopulaSpec);
                end
            end
                u{j,2*N-2*j-2}=hfunc_grm(u{j-1,2*N-2*j},u{j-1,2*N-2*j-1},1,CopulaSpec);
        end
    end
end
LogL=sum(sum(lik));

% Wandle unteres Element der Gauss- oder t-Korrelationsmatrix in Vektor um
if strcmp(CopulaSpec.type,'Gaussian')==1 || strcmp(CopulaSpec.type,'t')==1
    Rt2=zeros(size(Rt{1,1},3),1);
    c=size(Rt,1);
    for i=1:size(Rt,1)
        for j=1:c
            for h=1:size(Rt{j,i},3)
                Rt2(h,:) = Rt{j,i}(2,1,h);
            end
            clear Rt{j,i}
            Rt{j,i}=Rt2;
        end
        c=c-1;
    end
end



