function [LogL,Rt,likelihood]=CopulaVineLL_multistep_grm(phi,data,CopulaSpec)
% ---- Log Likelihood functions of the supported copula Vines -----
% USGAGE:
%         [LogL,Rt]=CopulaVineLL_grm(phi,data,CopulaSpec)
% INPUTS:
% phi:          vector of parameters
% data:         matrix with U(0,1) margins
% CopulaSpec:   structured array that contains the various input arguments
%               that define the parameters. To obtain run the function
%               setCopulaLL_grm inputs.m
% OUTPUT:
% LogL:         The log - likelihood of the corresponding copula
% ------------------------------------------------------------------------
% author: Martin Grziska
% last modification: 07/13/2010
% ------------------------------------------------------------------------
dec=CopulaSpec.decomposition;
type = CopulaSpec.type;
[T,N]=size(data);
lik=zeros(N-1,N-1);
u=cell(N,N);
if min(min(data))<0 || max(max(data))>1
    data=empiricalCDF(data);
end

% error-checking for static t-copula
if strcmp(CopulaSpec.type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static' == 1) && size(phi,2) ~= 2
    errors('t-copula needs starting values for d.o.f and correlation parameters')
elseif strcmp(CopulaSpec.type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static' == 1) && size(phi,1) ~= .5*N*(N-1)
    error('the parameters vector should be a vector with .5*N*(N-1) rows')
end

% error checking for other copulas beside SJC
if strcmp(CopulaSpec.type,'SJC')==0 && (strcmp(CopulaSpec.type,'t') == 0 && strcmp(CopulaSpec.corrspec,'static' == 0))
    if strcmp(CopulaSpec.corrspec,'static') && (size(phi,1)~=.5*N*(N-1) || size(phi,2)~=1)
        error('the parameters vector should be a vector with .5*N*(N-1) rows')
    elseif strcmp(CopulaSpec.corrspec,'Patton') == 1 && (size(phi,1)~= (.5*N*(N-1))*3 || size(phi,2)~=1)
        error('the parameters vector should be a vector with (.5*N*(N-1))*3 rows')
    elseif (strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1) && strcmp(CopulaSpec.type,'t') == 1 && size(phi,1)~= (.5*N*(N-1))*3 || size(phi,2)~=1
        error('the parameters vector should be a vector with ((.5*N*(N-1))*3 rows')
    elseif (strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1) && strcmp(CopulaSpec.type,'Gaussian') == 1 && size(phi,1)~= (.5*N*(N-1))*2 || size(phi,2)~=1
        error('the parameters vector should be a vector with ((.5*N*(N-1))*2 rows')
    elseif (strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1) && strcmp(CopulaSpec.type,'Clayton') == 1 && size(phi,1)~= (.5*N*(N-1))*2 || size(phi,2)~=1
        error('the parameters vector should be a vector with ((.5*N*(N-1))*2 rows')
    end
end

Rt=cell(N-1,N-1);
likelihood = zeros(T,N*(N-1)*.5);
% likelihoods need to be in a t*k array
% count for colums of likelihoods
countl = 1;

if strcmp(CopulaSpec.type,'SJC')==0
    %     create (Dynamic) Specification Structure, needed for CopulaLL_grm.
    %     Contians starting values for (Dynamic) specification
    if strcmp(CopulaSpec.corrspec,'Patton') == 1
        %     Die letzten drei Parameter sind nur für die
        %     Patton-Spezifikation; insgesamt 3 Parameter für die Dynamik
        %     des Copula-Parameters
        DynamicSpec = cell((.5*N*(N-1)),1);
        k=1;
        for i=1:3:size(phi,1)
            DynamicSpec{k,1} =phi(i:i+2);
            k=k+1;
        end
        DynamicSpec = crIPmat_grm(DynamicSpec);
    elseif strcmp(CopulaSpec.type,'t') == 1 && (strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1)
        %         Bei DCC oder TVC wird ein cell-array kreiert
        k=1;
        DynamicSpec=cell((.5*N*(N-1)),1);
        for i=1:3:size(phi,1)
            DynamicSpec{k,1} =phi(i:i+2);
            k=k+1;
        end
        DynamicSpec = crIPmat_grm(DynamicSpec);
    elseif (strcmp(CopulaSpec.corrspec,'DCC') == 1|| strcmp(CopulaSpec.corrspec,'TVC') == 1) && (strcmp(CopulaSpec.type,'Gaussian') == 1 ...
            ||  strcmp(CopulaSpec.type,'Clayton') || strcmp(CopulaSpec.type,'Gumbel'))
        k=1;
        DynamicSpec=cell((.5*N*(N-1)),1);
        for i=1:2:size(phi,1)
            DynamicSpec{k,1} =phi(i:i+1);
            k=k+1;
        end
        DynamicSpec = crIPmat_grm(DynamicSpec);
    elseif  strcmp(CopulaSpec.corrspec,'ADCC') == 1 && (strcmp(CopulaSpec.type,'Gaussian') == 1 || ...
            strcmp(CopulaSpec.type,'Clayton') == 1 || strcmp(CopulaSpec.type,'Gumbel') == 1)
        k=1;
        DynamicSpec=cell((.5*N*(N-1)),1);
        for i=1:3:size(phi,1)
            DynamicSpec{k,1} =phi(i:i+2);
            k=k+1;
        end
        DynamicSpec = crIPmat_grm(DynamicSpec);
    elseif  strcmp(CopulaSpec.corrspec,'ADCC') == 1 && strcmp(CopulaSpec.type,'t') == 1
        k=1;
        DynamicSpec=cell((.5*N*(N-1)),1);
        for i=1:4:size(phi,1)
            DynamicSpec{k,1} =phi(i:i+3);
            k=k+1;
        end
        DynamicSpec = crIPmat_grm(DynamicSpec);
        %         static startingvals if not t-copulas
    elseif strcmp(CopulaSpec.corrspec,'static') == 1 && strcmp(type,'t') == 0
        phi = crIPmat(phi);
        %         static startingvals if t-copula
    elseif strcmp(CopulaSpec.corrspec,'static') == 1 && strcmp(type,'t') == 1
        %             write double defvals into cell defvals
        for i=1:.5*(N*(N-1))
            for j=1:2
                phi1{i,:} = phi(i,:);
            end
        end
        clear phi
        phi=phi1;
        clear phi1;
        phi = crIPmat_grm(phi);
    end
    if strcmp(dec,'DVine')==1
        %         D-Vine structure as in Aas et al., p.4
        for i=1:N-1
            %             LogL and correlation
            if strcmp(CopulaSpec.corrspec,'Patton') == 1 || strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1
                [lik(1,i),Rt{1,i},likelihood(:,countl)]=CopulaLL_grm([DynamicSpec{1,i}],[data(:,i) data(:,i+1)],CopulaSpec);
                countl=countl+1;
            else
                if strcmp(type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
                    [lik(1,i),Rt{1,i},likelihood(:,countl)]=CopulaLL_grm(phi{1,i},[data(:,i) data(:,i+1)],CopulaSpec);
                    countl=countl+1;
                else
                    [lik(1,i),Rt{1,i},likelihood(:,countl)]=CopulaLL_grm(phi(1,i),[data(:,i) data(:,i+1)],CopulaSpec);
                    countl=countl+1;
                end
            end
        end
        %       h-function as in Aas et al., p.3
        %        first element of decomposition
        if strcmp(CopulaSpec.corrspec,'Patton') == 1 || strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1
            u{1,1}=hfunc_grm(data(:,1),data(:,2),[DynamicSpec{1,i}],CopulaSpec);
        else
            if strcmp(type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
                u{1,1}=hfunc_grm(data(:,1),data(:,2),phi{1,1},CopulaSpec);
            else
                u{1,1}=hfunc_grm(data(:,1),data(:,2),phi(1,1),CopulaSpec);
            end
        end
        %         elements between first and last
        for k=1:N-3
            if strcmp(CopulaSpec.corrspec,'Patton') == 1 || strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1
                u{1,2*k}=hfunc_grm(data(:,k+2),data(:,k+1),[DynamicSpec{1,k+1}],CopulaSpec);
                u{1,2*k+1}=hfunc_grm(data(:,k+1),data(:,k+2),[DynamicSpec{1,k+1}],CopulaSpec);
            else
                if strcmp(type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
                    u{1,2*k}=hfunc_grm(data(:,k+2),data(:,k+1),phi{1,k+1},CopulaSpec);
                    u{1,2*k+1}=hfunc_grm(data(:,k+1),data(:,k+2),phi{1,k+1},CopulaSpec);
                else
                    u{1,2*k}=hfunc_grm(data(:,k+2),data(:,k+1),phi(1,k+1),CopulaSpec);
                    u{1,2*k+1}=hfunc_grm(data(:,k+1),data(:,k+2),phi(1,k+1),CopulaSpec);
                end
            end
        end
        %         first elevel of decomposition, last element of first row
        if strcmp(CopulaSpec.corrspec,'Patton') == 1 || strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1
            u{1,2*N-4}=hfunc_grm(data(:,N),data(:,N-1),[DynamicSpec{1,N-1}],CopulaSpec);
        else
            if strcmp(type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
                u{1,2*N-4}=hfunc_grm(data(:,N),data(:,N-1),phi{1,N-1},CopulaSpec);
            else
                u{1,2*N-4}=hfunc_grm(data(:,N),data(:,N-1),phi(1,N-1),CopulaSpec);
            end
        end
        for j=2:(N-1)
            for i=1:(N-j)
                if strcmp(CopulaSpec.corrspec,'Patton') == 1 || strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1
                    [lik(j,i),Rt{j,i},likelihood(:,countl)]=CopulaLL_grm([DynamicSpec{j,i}],[u{j-1,2*i-1} u{j-1,2*i}],CopulaSpec);
                    countl=countl+1;
                else
                    if strcmp(type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
                        [lik(j,i),Rt{j,i},likelihood(:,countl)]=CopulaLL_grm(phi{j,i},[u{j-1,2*i-1} u{j-1,2*i}],CopulaSpec);
                        countl=countl+1;
                    else
                        [lik(j,i),Rt{j,i},likelihood(:,countl)]=CopulaLL_grm(phi(j,i),[u{j-1,2*i-1} u{j-1,2*i}],CopulaSpec);
                        countl=countl+1;
                    end
                end
            end
            %             all other levels of decomposition
            %             (conditional Copula)
            if j<N-1
                if strcmp(CopulaSpec.corrspec,'Patton') == 1 || strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1
                    u{j,1}=hfunc_grm(u{j-1,1},u{j-1,2},[DynamicSpec{j,1}],CopulaSpec);
                else
                    if strcmp(type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
                        u{j,1}=hfunc_grm(u{j-1,1},u{j-1,2},phi{j,1},CopulaSpec);
                    else
                        u{j,1}=hfunc_grm(u{j-1,1},u{j-1,2},phi(j,1),CopulaSpec);
                    end
                end
                if N>4
                    for i=1:(N-j-2)
                        if strcmp(CopulaSpec.corrspec,'Patton') == 1 || strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1
                            u{j,2*i}=hfunc_grm(u{j-1,2*i+2},u{j-1,2*i+1},[DynamicSpec{j,i+1}],CopulaSpec);
                            u{j,2*i+1}=hfunc_grm(u{j-1,2*i+1},u{j-1,2*i+2},[DynamicSpec{j,i+1}],CopulaSpec);
                        else
                            if strcmp(type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
                                u{j,2*i}=hfunc_grm(u{j-1,2*i+2},u{j-1,2*i+1},phi{j,i+1},CopulaSpec);
                                u{j,2*i+1}=hfunc_grm(u{j-1,2*i+1},u{j-1,2*i+2},phi{j,i+1},CopulaSpec);
                            else
                                u{j,2*i}=hfunc_grm(u{j-1,2*i+2},u{j-1,2*i+1},phi(j,i+1),CopulaSpec);
                                u{j,2*i+1}=hfunc_grm(u{j-1,2*i+1},u{j-1,2*i+2},phi(j,i+1),CopulaSpec);
                            end
                        end
                    end
                end
                if strcmp(CopulaSpec.corrspec,'Patton') == 1 || strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1
                    u{j,2*N-2*j-2}=hfunc_grm(u{j-1,2*N-2*j},u{j-1,2*N-2*j-1},[DynamicSpec{j,N-j}],CopulaSpec);
                else
                    if strcmp(type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
                        u{j,2*N-2*j-2}=hfunc_grm(u{j-1,2*N-2*j},u{j-1,2*N-2*j-1},phi{j,N-j},CopulaSpec);
                    else
                        u{j,2*N-2*j-2}=hfunc_grm(u{j-1,2*N-2*j},u{j-1,2*N-2*j-1},phi(j,N-j),CopulaSpec);
                    end
                end
            end
        end
    end
    LogL=sum(sum(lik));
end

% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
if strcmp(CopulaSpec.type,'SJC')
    % ----------------------  some comments ---------------------------------
    % In the SJC copula decomposition, each bivariate copula has two
    % parameters, unlike the t or Clayton copula which have only one. Therefore
    % the output and initial values are not in a matrix, but in a cell array, with
    % same dimension as the matrix. For example in the t - copula theta(1,1)
    % contains the dof parameter for the first copula in the cascade, whereas
    % in the SJC copula, theta{1,1} contains two values, the upper and lower
    % tail dependence of the first copula.
    % ------------------------------------------------------------------------
    phi=crIPmat_grm(phi);
    Rt = cell(size(phi,1));
    if strcmp(dec,'DVine')==1
        %         gilt das gleiche wie oben
        for i=1:N-1
            [lik(1,i),Rt{1,i}]=CopulaLL_grm(phi{1,i},[data(:,i) data(:,i+1)],CopulaSpec);
        end
        u{1,1}=hfunc_grm(data(:,1),data(:,2),phi{1,1},CopulaSpec);
        for k=1:N-3
            u{1,2*k}=hfunc_grm(data(:,k+2),data(:,k+1),phi{1,k+1},CopulaSpec);
            u{1,2*k+1}=hfunc_grm(data(:,k+1),data(:,k+2),phi{1,k+1},CopulaSpec);
        end
        u{1,2*N-4}=hfunc_grm(data(:,N),data(:,N-1),phi{1,N-1},CopulaSpec);
        for j=2:(N-1)
            for i=1:(N-j)
                [lik(j,i),Rt{j,i}]=CopulaLL_grm(phi{j,i},[u{j-1,2*i-1} u{j-1,2*i}],CopulaSpec);
            end
            if j<N-1
                u{j,1}=hfunc_grm(u{j-1,1},u{j-1,2},phi{j,1}, CopulaSpec);
                if N>4
                    for i=1:(N-j-2)
                        u{j,2*i}=hfunc_grm(u{j-1,2*i+2},u{j-1,2*i+1},phi{j,i+1},CopulaSpec);
                        u{j,2*i+1}=hfunc_grm(u{j-1,2*i+1},u{j-1,2*i+2},phi{j,i+1},CopulaSpec);
                    end
                end
                u{j,2*N-2*j-2}=hfunc_grm(u{j-1,2*N-2*j},u{j-1,2*N-2*j-1},phi{j,N-j},CopulaSpec);
            end
        end
    end
    LogL = sum(sum(lik));
end

% transform lower triangle matrix of Gauss or t-correlationmatrix to vector
if (strcmp(CopulaSpec.type,'Gaussian')==1 && strcmp(type,'t')) == 1 && strcmp(CopulaSpec.corrspec,'static') == 0
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
elseif strcmp(type,'t') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
    Rt2=zeros(size(Rt{1,1},3),1);
    c=size(Rt,1);
    for i=1:size(Rt,1)
        for j=1:c
            for h=1:size(Rt{j,i},3)
                Rt2(h,:) = Rt{j,i}(1,1,h);
            end
            clear Rt{j,i}
            Rt{j,i}=Rt2;
        end
        c=c-1;
    end
end

% make sure solution exists
if ~isreal(LogL)
    LogL=10E-6;
end

if isinf(LogL)
    LogL = 10E-6;
end

if isnan(LogL)
    LogL = 10E-6;
end

if ~isreal(phi)
    LogL = 10E-6;
end

