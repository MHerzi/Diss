function [Tstatistic_all, stderrors] = stderrors_ADCC(data,parameters,archP,garchQ,asymmG,dccP,dccQ,dccG,garchtype,errortype,univariate)
% Compute robust standard errors and associated t-stat
% Inputs :
%           data:        Txk matrix of stdresid
%           parameters:  vector of univariate parameters
%           archP:       # of ARCH lags
%           garchQ:      # of GARCH lags
%           asymmG:      1 if asymmetric term in univariate garch, 0 else
%           dccP:        # of ARCH lags in ADCC
%           dccQ:        # of GARCH lags in ADCC
%           dccG:        # of asymmetric lags in ADCC
%           garchtype:   type of univariate GARCH
%           errortype:   (assumed) Distribution of garchtype
%           univariate:  structure of univariate results
% 
% Outputs :
%          Tstat:     Tstat
%          stderrors: Bollerslev-Wooldridge stderrors
% Author: Martn Grziska, 02/25/2010


[t,k]= size(data);
A=zeros(length(parameters),length(parameters));
index=1;
for i=1:k
    workingsize=size(univariate{i}.stderrors);
    A(index:index+workingsize-1,index:index+workingsize-1)=univariate{i}.stderrors^(-1);
    index=index+workingsize;
end
otherA=dcc_hessian_grm('ADCC_full_likelihood', parameters, dccP+dccQ+dccG, data, archP, garchQ, asymmG, garchtype, errortype, dccP, dccQ, dccG);
A(length(parameters)-dccP-dccQ-dccG +1:length(parameters),:)=otherA;

jointscores=zeros(size(data,1),length(parameters));
index=1;

%  Bringe die univariaten scores auf die gleich Länge
for i=1:k
    lengthscore(i,:) = length(univariate{i}.scores);
end
minscore = min(lengthscore);
for i=1:k
    if length(univariate{i}.scores>minscore)
        univariate{i}.scores = univariate{i}.scores(length(univariate{i}.scores)-minscore+1:end,:);
    end
end

for i=1:k
    workingsize=size(univariate{i}.scores,2);
    jointscores(:,index:index+workingsize-1)=univariate{i}.scores;
    index=index+workingsize;
end

h=max(abs(parameters/2),1e-2)*eps^(1/3);
hplus=parameters+h;
hminus=parameters-h;
likelihoodsplus=zeros(t,length(parameters));
likelihoodsminus=zeros(t,length(parameters));
for i=length(parameters)-dccP-dccQ+1:length(parameters)
    hparameters=parameters;
    hparameters(i)=hplus(i);
    [HOLDER, HOLDER1, indivlike] = ADCC_full_likelihood(parameters, data, archP, garchQ, asymmG, garchtype, errortype, dccP, dccQ, dccG);
    likelihoodsplus(:,i)=indivlike;
end
for i=length(parameters)-dccP-dccQ+1:length(parameters)
    hparameters=parameters;
    hparameters(i)=hminus(i);
    [HOLDER, HOLDER1, indivlike] = ADCC_full_likelihood(parameters, data, archP, garchQ, asymmG, garchtype, errortype, dccP, dccQ, dccG);
    likelihoodsminus(:,i)=indivlike;
end
ADCCscores=(likelihoodsplus(:,length(parameters)-dccP-dccQ-dccG+1:length(parameters))-likelihoodsminus(:,length(parameters)-dccP-dccQ-dccG+1:length(parameters)))...
    ./(2*repmat(h(length(parameters)-dccP-dccQ-dccG+1:length(parameters))',t,1));
jointscores(:,length(parameters)-dccP-dccQ-dccG+1:length(parameters))=ADCCscores;
B=cov(jointscores);
A=A/t;
stderrors=A^(-1)*B*A'^(-1)*t^(-1);
%Done!
load tempHt
Tstatistic_all=parameters./diag(stderrors).^0.5;
