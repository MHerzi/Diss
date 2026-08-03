function [LogL,Rt]=CopulaLL_snoop_grm(theta,data,CopulaSpec)
% ---- Log Likelihood functions of the supported copulas -----
% INPUTS:
% theta:        vector of parameters
% data:         matrix with U(0,1) margins
% CopulaSpec:   structured array that contains the various input arguments
%               that define the parameters. To obtain run the function
%               setCopulaLLinputs.m
% OUTPUT:       The negative log - likelihood of the corresponding copula
% ------------------------------------------------------------------------
% author: Martin Grziska based on a code of Manthos Vogiatzoglou
% contact at: vogia@yahoo.com
% Last Modification: April, 09, 2010
% ------------------------------------------------------------------------
optimizer=CopulaSpec.optimizer;
% Wandle die "cell"-Variable in "double" um
if iscell(theta)
    theta=theta{1,1};
end
% -----------------------------------------------------------------------
% the Clayton Copula Log Likelihood
% -----------------------------------------------------------------------
if strcmp(CopulaSpec.type,'Clayton')==1
    T=size(data,1);
    u=data(:,1);
    v=data(:,2);
    if strcmp(CopulaSpec.corrspec,'static')==1 && strcmp(optimizer,'fmincon')==1
        tau=theta; tau=repmat(tau,[T,1]);
    elseif strcmp(CopulaSpec.corrspec,'DCC') == 1 && strcmp(optimizer,'fmincon')==1
        %         Annahme: DCC Gaussian Variablen
        trdata = norminv(data);
        %         Umwandlung der Startwerte als Clayton-Parameter in
        %         Gauss-Parameter
        %         theta = sin(pi*(theta./(theta+2))/2);
        [Holder, rho] = DCCeq_grm(theta,trdata,'fmincon');
        %         Clayton-Copula zeigt nur positive Abhängigkeiten: alle rho<0
        %         müssen 0 gesetzt werden
        rho = rho.*(rho>0);
        %         Setze Variablen die = 0 sind, größer Null, da Claytion-Copula
        %         ohne Nulle definiert ist
        rho1 = rho==0;
        rho1 = rho1*1e-4;
        rho = rho + rho1;
        %         Wandle den DCC-Korrelationsparameter in Kendalls Tau um,
        tau = 2*asin(rho)./pi;
        %         Wandle Kendalls Tau in Clayton-Copula Parameter um
        tau = 2*tau./(1-tau);
    elseif strcmp(CopulaSpec.corrspec,'static')==1 && strcmp(optimizer,'fminunc')==1
        tau=.0001+.85./(1+exp(-theta)); tau=repmat(tau,[T,1]);
    elseif strcmp(CopulaSpec.corrspec,'Patton')==1 && strcmp(optimizer,'fminunc')==1
        tau = Pattoneq_grm(theta,data,'fminunc');
    elseif strcmp(CopulaSpec.corrspec,'DCC') == 1 && strcmp(optimizer,'fminunc')==1
        trdata = norminv(data);
        %         theta = sin(pi*(theta./(theta+2))/2);
        [Holder, rho] = DCCeq_grm(theta,trdata,'fminunc');
        rho = rho.*(rho>0);
        rho1 = rho==0;
        rho1 = rho1*1e-4;
        rho = rho+ rho1;
        tau = 2*asin(rho)./pi;
        tau = 2*tau./(1-tau);
    end
    % log-likelihood der bivariaten Clayton-Copula
    LogL = log(1+tau) - (tau+1).*(log(u)+log(v));
    LogL = LogL - (2+1./tau).*log((u.^(-tau)) + (v.^(-tau)) -1);
    LogL = sum(LogL);
    LogL = -LogL;
    Rt=tau;
end
% -----------------------------------------------------------------------
% the SJC copula log likelihood
% -----------------------------------------------------------------------
if strcmp(CopulaSpec.type,'SJC')==1
    T=size(data,1);
    u=data(:,1); v=data(:,2);
    if strcmp(CopulaSpec.corrspec,'static')==1 && strcmp(optimizer,'fmincon')
        if size(theta,1)==2
            tauU=repmat(theta(1),[T,1]); tauL=repmat(theta(2),[T,1]);
        elseif size(theta,2)==2
            tauU=repmat(theta(1,1),[T,1]); tauL=repmat(theta(1,2),[T,1]);
        end
    elseif strcmp(CopulaSpec.corrspec,'static')==1 && strcmp(optimizer,'fminunc')
        theta=.0001+.85./(1+exp(-theta));
        if size(theta,1)==2
            tauU=repmat(theta(1),[T,1]); tauL=repmat(theta(2),[T,1]);
        elseif size(theta,2)==2
            tauU=repmat(theta(1,1),[T,1]); tauL=repmat(theta(1,2),[T,1]);
        end
    elseif strcmp(CopulaSpec.corrspec,'Patton')==1
        theta = repmat(theta,3,1);
        tau=Pattoneq(theta,data,'SJC');
        tauU=tau(:,1);
        tauL=tau(:,2);
    end
    k1 =  1./log2(2-tauU);
    k2 = -1./log2(tauL);
    JC1=(k1.*k2.*(1 - 1./(1./(1 - (1 - u).^k1).^k2 + 1./(1 - (1 - v).^k1).^k2 - 1).^(1./k2)).^(1./k1 - 1).*(1./k2 + 1).*(1 - u).^(k1 - 1).*(1 - v).^(k1 - 1))./((1 - (1 - u).^k1).^(k2 + 1).*(1 - (1 - v).^k1).^(k2 + 1).*(1./(1 - (1 - u).^k1).^k2 + 1./(1 - (1 - v).^k1).^k2 - 1).^(1./k2 + 2));
    JC2=(k1.*(1 - 1./(1./(1 - (1 - u).^k1).^k2 + 1./(1 - (1 - v).^k1).^k2 - 1).^(1./k2)).^(1./k1 - 2).*(1./k1 - 1).*(1 - u).^(k1 - 1).*(1 - v).^(k1 - 1))./((1 - (1 - u).^k1).^(k2 + 1).*(1 - (1 - v).^k1).^(k2 + 1).*(1./(1 - (1 - u).^k1).^k2 + 1./(1 - (1 - v).^k1).^k2 - 1).^(2./k2 + 2));
    out1=JC1-JC2;
    
    k1 =  1./log2(2-tauL);
    k2 = -1./log2(tauU);
    u  = 1-u;
    v  = 1-v;
    JC3=(k1.*k2.*(1 - 1./(1./(1 - (1 - u).^k1).^k2 + 1./(1 - (1 - v).^k1).^k2 - 1).^(1./k2)).^(1./k1 - 1).*(1./k2 + 1).*(1 - u).^(k1 - 1).*(1 - v).^(k1 - 1))./((1 - (1 - u).^k1).^(k2 + 1).*(1 - (1 - v).^k1).^(k2 + 1).*(1./(1 - (1 - u).^k1).^k2 + 1./(1 - (1 - v).^k1).^k2 - 1).^(1./k2 + 2));
    JC4=(k1.*(1 - 1./(1./(1 - (1 - u).^k1).^k2 + 1./(1 - (1 - v).^k1).^k2 - 1).^(1./k2)).^(1./k1 - 2).*(1./k1 - 1).*(1 - u).^(k1 - 1).*(1 - v).^(k1 - 1))./((1 - (1 - u).^k1).^(k2 + 1).*(1 - (1 - v).^k1).^(k2 + 1).*(1./(1 - (1 - u).^k1).^k2 + 1./(1 - (1 - v).^k1).^k2 - 1).^(2./k2 + 2));
    out2=JC3-JC4;
    LogL=-sum(log(.5*(out1+out2)));
    Rt=[tauU tauL];
end
[T,N]=size(data);
% -----------------------------------------------------------------------
% the t - Copula log - likelihood
% -----------------------------------------------------------------------
if strcmp(CopulaSpec.type,'t')==1
    if strcmp(CopulaSpec.corrspec,'static')==1
        nu=theta(1);
        rho=theta(2);
        trdata=tinv(data,nu);
        LL = zeros(T,1);
        for i=1:T
            LL(i) = gammaln((nu+2)/2) + gammaln(nu/2) - 2*gammaln((nu+1)/2) - 0.5*log((1-rho^2));
            LL(i) = LL(i) - (nu+2)/2*log(1+(trdata(i,1)^2 + trdata(i,2)^2 - 2*rho*trdata(i,1)*trdata(i,2))/(nu*(1-rho^2)));
            LL(i) = LL(i) + (nu+1)/2*log(1+trdata(i,1)^2/nu) + (nu+1)/2*log(1+trdata(i,2)^2/nu);
        end
        Rt=repmat(rho,[1 1 T]);
    else
        if strcmp(optimizer,'fminunc')==1
            nu=2.01+exp(theta(end));
        else
            nu=theta(end);
        end
        trdata=tinv(data,nu);
        if strcmp(optimizer,'fmincon')==1
            if strcmp(CopulaSpec.corrspec,'DCC')==1
                Rt=DCCeq_grm(theta(1:2),trdata,optimizer);
            elseif strcmp(CopulaSpec.corrspec,'TVC')==1
                Rt=TVCeq_grm(theta(1:2),trdata,optimizer);
            end
        elseif strcmp(optimizer,'fminunc')==1
            if strcmp(CopulaSpec.corrspec,'DCC')==1
                Rt=DCCeq_grm(theta(1:2),trdata,optimizer);
            elseif strcmp(CopulaSpec.corrspec,'TVC')==1
                Rt=TVCeq_grm(theta(1:2),trdata,optimizer);
            end
        end
        % The T Copula likelihood function
        LL=zeros(T,1);
        for i=1:T
            LL(i) = gammaln((nu+N)/2) + (N-1)*gammaln(nu/2) - N*gammaln((nu+1)/2) - 0.5*log(det(Rt(:,:,i)));
            LL(i) = LL(i) - (nu+N)/2*log(1+trdata(i,:)*(Rt(:,:,i))^(-1)*trdata(i,:)'./nu);
            LL(i) = LL(i) + (nu+1)/2*sum(log(1+(trdata(i,:).^2/nu)));
        end
    end
    likelihood=sum(LL);
    LogL=-likelihood;
end
% ------------------------------------------------------------------------
% the Gaussian Copula log likelihood
% ------------------------------------------------------------------------
if strcmp(CopulaSpec.type,'Gaussian')==1 && strcmp(optimizer,'SDPT3')==0
    trdata=norminv(data);
    if strcmp(optimizer,'fmincon')==1 && strcmp(CopulaSpec.corrspec,'static') == 0
        if strcmp(CopulaSpec.corrspec,'DCC')==1
            Rt=DCCeq_grm(theta(1:2),trdata,optimizer);
        elseif strcmp(CopulaSpec.corrspec,'TVC')==1
            Rt=TVCeq_grm(theta(1:2),trdata);
        end
    elseif strcmp(optimizer,'fminunc')==1 && strcmp(CopulaSpec.corrspec,'static') == 0
        if strcmp(CopulaSpec.corrspec,'DCC')==1
            Rt=DCCeq_grm(theta(1:2),trdata,optimizer);
        elseif strcmp(CopulaSpec.corrspec,'TVC')==1
            Rt=TVCeq_grm(theta(1:2),trdata,optimizer);
        end
    elseif strcmp(CopulaSpec.corrspec,'static') == 1
        rRt = corr(data,'type','kendall');
        Rt1 = sin(.5*pi*rRt);
        for i=1:T
            Rt(:,:,i) = Rt1;
        end
    end
    LL=zeros(T,1);
    for i=1:T
        LL(i)=-.5*log(det(Rt(:,:,i)));
        LL(i)=LL(i)-.5*trdata(i,:)*(inv(Rt(:,:,i))-eye(N))*trdata(i,:)';
    end
    likelihood=sum(LL);
    LogL=-likelihood;
end
% -----------------------------------------------------------------------
% the Gumbel Copula Log Likelihood
% ----------------------------------------------------------------------
if strcmp(CopulaSpec.type,'Gumbel') == 1
    trdata = norminv(data);
    u = data(:,1);
    v = data(:,2);
    if strcmp(optimizer,'fmincon')==1
        if strcmp(CopulaSpec.corrspec,'DCC')==1
            [Holder, rho]=DCCeq_grm(theta(1:2),trdata);
            tau = 2*asin(rho)./pi;
            tau = 1./(1-tau);
        elseif strcmp(CopulaSpec.corrspec,'TVC')==1
            [Holder, rho]=TVCeq_grm(theta(1:2),trdata);
            tau = 2*asin(rho)./pi;
            tau = 1./(1-tau);
        end
        if strcmp(CopulaSpec.corrspec,'static')==1
            tau = repmat(theta,[T,1]);
        end
    elseif strcmp(optimizer,'fminunc')==1
        if strcmp(CopulaSpec.corrspec,'DCC')==1
            [Holder, rho]=DCCeq_grm(theta(1:2),trdata,optimizer);
        elseif strcmp(CopulaSpec.corrspec,'TVC')==1
            [Holder, rho]=TVCeq_grm(theta(1:2),trdata,optimizer);
        else
            %             tau = tau=repmat(tau,[T,1]);theta;
        end
        tau = 2*asin(rho)./pi;
        tau = 1./(1-tau);
    end
    ut = -log(data(:,1));
    vt = -log(data(:,2));
    LL = log(Gumbel_cdf(u,v,tau)) - log(u) - log(v);
    LL = LL + (tau-1).*(log(ut)+log(vt)) - (2-1./tau.*(log(ut.^tau+vt.^tau)));
    LL = LL + log((ut.^tau + vt.^tau).^(1./tau) + tau - 1);
    likelihood=sum(LL);
    LogL=-likelihood;
    Rt=tau;
end


if ~isreal(LogL)
    LogL=10E-6;
end

if ~isreal(theta)
    LogL=10E-6;
end

if isinf(LogL)
    LogL = 10E-6;
end

if isnan(LogL)
    LogL = 10E-10;
end

