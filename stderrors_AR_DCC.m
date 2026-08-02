function [Tstatistic_all, stderrors] = stderrors_AR_DCC(params,data,archP,garchQ,asymmG,dccP,dccQ,dccG,garchtype,errortype,GARCHOutput,const)
% Compute robust standard errors and associated t-stat
% 
% USAGE:
%       [Tstatistic_all, stderrors] =
%       stderrors_AR_DCC(params,data,archP,garchQ,asymmG,dccP,dccQ,dccG...
%       ,garchtype,errortype,scores,arlag,GARCHOutput)
% 
% Inputs :
%           data:        Txk matrix of stdresid
%           params:  vector of univariate params (WITH AR terms)
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
% GARCHOutputs :
%          Tstat:     Tstat
%          stderrors: Bollerslev-Wooldridge stderrors
% 
% Author: Martn Grziska, 02/25/2010

[t,k]= size(data);
arlag=zeros(k,1);
for i=1:k
    arlag(i) = GARCHOutput{i}.arlag;
end
t=t-(max(arlag));
% if const == 1
%     arlag=arlag+1;
% end
A=zeros(length(params),length(params));
index=1;
for i=1:k
    workingsize=size(GARCHOutput{i}.stderrors);
    A(index:index+workingsize-1,index:index+workingsize-1)=GARCHOutput{i}.stderrors^(-1);
    index=index+workingsize;
end
otherA = dcc_hessian_grm('AR_dcc_mvgarch_full_likelihood', params, dccP+dccQ, data, archP, garchQ, asymmG, garchtype, errortype, dccP, dccQ, arlag, const);
A(length(params)-dccP-dccQ-dccG+1:length(params),:)=otherA;

jointscores=zeros(t,length(params));
index=1;
for i=1:k
    workingsize=size(GARCHOutput{i}.Scores,2);
    jointscores(:,index:index+workingsize-1)=GARCHOutput{i}.Scores;
    index=index+workingsize;
end

h=max(abs(params/2),1e-2)*eps^(1/3);
hplus=params+h;
hminus=params-h;
likelihoodsplus=zeros(t,length(params));
likelihoodsminus=zeros(t,length(params));
for i=length(params)-dccP-dccQ+1:length(params)
    hparams=params;
    hparams(i)=hplus(i);
    [HOLDER, HOLDER1, indivlike] = AR_dcc_mvgarch_full_likelihood(hparams, data, archP, garchQ, asymmG, garchtype, errortype, dccP, dccQ, arlag, const);
    likelihoodsplus(:,i)=indivlike;
end
for i=length(params)-dccP-dccQ+1:length(params)
    hparams=params;
    hparams(i)=hminus(i);
    [HOLDER, HOLDER1, indivlike] = AR_dcc_mvgarch_full_likelihood(hparams, data, archP, garchQ, asymmG, garchtype, errortype, dccP, dccQ, arlag, const);
    likelihoodsminus(:,i)=indivlike;
end
DCCscores = (likelihoodsplus(:,length(params)-dccP-dccQ-dccG+1:length(params))-likelihoodsminus(:,length(params)-dccP-dccQ-dccG+1:length(params)))...
    ./(2*repmat(h(length(params)-dccP-dccQ-dccG+1:length(params))',t,1));
jointscores(:,length(params)-dccP-dccQ-dccG+1:length(params)) = DCCscores;
B=cov(jointscores);
A=A/t;
stderrors=A^(-1)*B*A'^(-1)*t^(-1);
Tstatistic_all=params./(diag(stderrors).^0.5);
