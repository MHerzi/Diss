function [Tstatistic_all, stderrors] = stderrors_AR_AGDCC(parameters, data,archP,garchQ,asymmG,dccP,dccQ,dccG,garchtype,errortype,GARCHOutput,const)
% Compute robust standard errors and associated t-stat
% Inputs :
%           data:        Txk matrix of stdresid
%           parameters:  vector of univariate parameters (WITH AR terms)
%           archP:       # of ARCH lags
%           garchQ:      # of GARCH lags
% %           asymmG:      1 if asymmetric term in univariate garch, 0 else
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
arlag=zeros(k,1);
for i=1:k
    arlag(i) = GARCHOutput{i}.arlag;
end
t=t-(max(arlag));
A=zeros(length(parameters),length(parameters));
index=1;
for i=1:k
    workingsize=size(GARCHOutput{i}.stderrors);
    A(index:index+workingsize-1,index:index+workingsize-1)=GARCHOutput{i}.stderrors^(-1);
    index=index+workingsize;
end
otherA = dcc_hessian_grm('AR_AGDCC_full_likelihood', parameters, dccP*k+dccQ*k+dccG*k, data, archP, garchQ, asymmG, garchtype, errortype, dccP, dccQ, dccG, arlag, const);
A(length(parameters)-dccP*k-dccQ*k-dccG*k +1:length(parameters),:)=otherA;

jointscores=zeros(t,length(parameters));
index=1;

for i=1:k
    workingsize=size(GARCHOutput{i}.Scores,2);
    jointscores(:,index:index+workingsize-1)=GARCHOutput{i}.Scores;
    index=index+workingsize;
end

h=max(abs(parameters/2),1e-2)*eps^(1/3);
hplus=parameters+h;
hminus=parameters-h;
likelihoodsplus=zeros(t,length(parameters));
likelihoodsminus=zeros(t,length(parameters));
for i=length(parameters)-2*dccP*k-k*dccQ+1:length(parameters)
    hparameters=parameters;
    hparameters(i)=hplus(i);
    [HOLDER, HOLDER1, indivlike] = AR_AGDCC_full_likelihood(hparameters, data, archP, garchQ, asymmG, garchtype, errortype, dccP, dccQ, dccG, arlag, const);
    likelihoodsplus(:,i)=indivlike;
end
for i=length(parameters)-2*dccP*k-k*dccQ+1:length(parameters)
    hparameters=parameters;
    hparameters(i)=hminus(i);
    [HOLDER, HOLDER1, indivlike] = AR_AGDCC_full_likelihood(hparameters, data, archP, garchQ, asymmG, garchtype, errortype, dccP, dccQ, dccG, arlag, const);
    likelihoodsminus(:,i)=indivlike;
end
AGDCCscores=(likelihoodsplus(:,length(parameters)-dccP*k-dccQ*k-dccG*k+1:length(parameters))-likelihoodsminus(:,length(parameters)-dccP*k-dccQ*k-dccG*k+1:length(parameters)))...
    ./(2*repmat(h(length(parameters)-dccP*k-dccQ*k-dccG*k+1:length(parameters))',t,1));
jointscores(:,length(parameters)-dccP*k-dccQ*k-dccG*k+1:length(parameters))=AGDCCscores;
B=cov(jointscores);
A=A/t;
stderrors=A^(-1)*B*A'^(-1)*t^(-1);
Tstatistic_all=parameters./(diag(stderrors).^0.5);

