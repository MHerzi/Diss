function [Rt,likelihood]=CopulaLL_grm_1P_forecast(theta,data,CopulaSpec,epsilon)
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
% Last Modification: 09, 10, 2010
% ------------------------------------------------------------------------
optimizer=CopulaSpec.optimizer;
% Wandle die "cell"-Variable in "double" um
if iscell(theta)
    theta=theta{1,1};
end
% -----------------------------------------------------------------------
% the Clayton Copula Log Likelihood
% -----------------------------------------------------------------------
if strcmp(CopulaSpec.type,'Clayton')==1 || strcmp(CopulaSpec.type,'Rotated Clayton')==1
    T=size(data,1);
    u=data(:,1);
    v=data(:,2);
    if strcmp(CopulaSpec.corrspec,'static')==1 && strcmp(optimizer,'fmincon')==1
        tau=theta; tau=repmat(tau,[T,1]);
    elseif (strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1 ...
            || strcmp(CopulaSpec.corrspec,'GDCC') == 1 || strcmp(CopulaSpec.corrspec,'AGDCC') == 1) ...
            && strcmp(optimizer,'fmincon')==1
        %         Annahme: DCC Gaussian Variablen
        trdata = norminv([u,v]);
        if strcmp(CopulaSpec.corrspec,'DCC') == 1
            [Holder, rho] = DCCeq_grm_1P_forecast(theta,trdata,'fmincon');
        elseif strcmp(CopulaSpec.corrspec,'ADCC') == 1
            [Holder, rho] = ADCCeq_grm_1P_forecast(theta,trdata,'fmincon');
        elseif strcmp(CopulaSpec.corrspec,'GDCC') == 1
            [Holder, rho] = GDCCeq_grm_1P_forecast(theta,trdata,'fmincon');
        elseif strcmp(CopulaSpec.corrspec,'AGDCC') == 1
            [Holder, rho] = AGDCCeq_grm_1P_forecast(theta,trdata,'fmincon');
        end
        %         Setze Variablen die = 0 sind, größer Null, da Claytion-Copula
        %         ohne Nulle definiert ist
        rho = rho.*(rho>0);
        % %         setze Variablen die Null sind 0+epsilon, damit es keine
        % %         numerischen Probleme gibt
        rho1 = rho==0;
        rho1 = rho1*1e-5;
        rho = rho+ rho1;
        %         Wandle Gauß rho in Kendalls tau um
        ken = 2*asin(rho)./pi;
        tau = 2*ken./(1-ken);
    elseif strcmp(CopulaSpec.corrspec,'static')==1 && strcmp(optimizer,'fminunc')==1
        tau=.0001+.85./(1+exp(-theta)); tau=repmat(tau,[T,1]);
    elseif strcmp(CopulaSpec.corrspec,'Patton')==1 && strcmp(optimizer,'fminunc')==1
        tau = Pattoneq_grm(theta,data,CopulaSpec.type,'fminunc');
    elseif strcmp(CopulaSpec.corrspec,'Patton')==1 && strcmp(optimizer,'fmincon')==1
        tau = Pattoneq_grm(theta,data,CopulaSpec.type,'fmincon');
    elseif (strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1 ...
            || strcmp(CopulaSpec.corrspec,'GDCC') == 1 || strcmp(CopulaSpec.corrspec,'AGDCC') == 1) ...
            && strcmp(optimizer,'fminunc')==1
        %         Annahme: DCC Gaussian Variablen
        trdata = norminv([u,v]);
        if strcmp(CopulaSpec.corrspec,'DCC') == 1
            [Holder, rho] = DCCeq_grm_1P_forecast(theta,trdata,'fminunc');
        elseif strcmp(CopulaSpec.corrspec,'ADCC') == 1
            [Holder, rho] = ADCCeq_grm_1P_forecast(theta,trdata,'fminunc');
        elseif strcmp(CopulaSpec.corrspec,'GDCC') == 1
            [Holder, rho] = GDCCeq_grm_1P_forecast(theta,trdata,'fminunc');
        elseif strcmp(CopulaSpec.corrspec,'AGDCC') == 1
            [Holder, rho] = AGDCCeq_grm_1P_forecast(theta,trdata,'fminunc');
        end
        rho = rho*(rho>0);
        rho1 = rho==0;
        rho1 = rho1*1e-2;
        rho = rho+ rho1;
        ken = 2*asin(rho)./pi;
        tau = 2*ken./(1-ken);
    elseif strcmp(CopulaSpec.corrspec,'TVC') == 1
        trdata = norminv(data);
        [Holder ,rho] = TVCeq_grm_1P_forecast(theta,trdata,'fmincon');
        rho = rho.*(rho>0);
        %         setze Variablen die Null sind 0+epsilon, damit es keine
        %         numerischen Probleme gibt
        rho1 = rho==0;
        rho1 = rho1*1e-2;
        rho = rho+ rho1;
        %         Wandle Gauß rho in Kendalls tau um
        ken = 2*asin(rho)./pi;
        tau = 2*ken./(1-ken);
    end
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
    likelihood = log(.5*(out1+out2));
    LogL = -sum(likelihood);
    Rt=[tauU tauL];
end
[T,N]=size(data);
% -----------------------------------------------------------------------
% the t - Copula log - likelihood
% -----------------------------------------------------------------------
if strcmp(CopulaSpec.type,'t')==1
    if strcmp(CopulaSpec.corrspec,'static')==1
        nu=theta(1);
        trdata=tinv(data,nu);
        rho = corr(trdata);
        rho = rho(2,1);
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
                [Rt,rho] = DCCeq_grm_1P_forecast(theta(1:2),trdata,optimizer);
            elseif strcmp(CopulaSpec.corrspec,'TVC')==1
                [Rt,rho] = TVCeq_grm_1P_forecast(theta(1:2),trdata,optimizer);
            elseif  strcmp(CopulaSpec.corrspec,'ADCC')==1
                [Rt,rho] = ADCCeq_grm_1P_forecast(theta(1:3),trdata,optimizer);
            elseif strcmp(CopulaSpec.corrspec,'GDCC')
                [Rt,rho] = GDCCeq_grm_1P_forecast(theta(1:4),trdata,optimizer);
            elseif strcmp(CopulaSpec.corrspec,'AGDCC')
                [Rt,rho] = GDCCeq_grm_1P_forecast(theta(1:6),trdata,optimizer);
            end
        elseif strcmp(optimizer,'fminunc')==1
            if strcmp(CopulaSpec.corrspec,'DCC')==1
                [Rt,rho] = DCCeq_grm_1P_forecast(theta(1:2),trdata,optimizer);
            elseif strcmp(CopulaSpec.corrspec,'TVC')==1
                [Rt,rho] = TVCeq_grmv(theta(1:2),trdata,optimizer);
            elseif  strcmp(CopulaSpec.corrspec,'ADCC')==1
                [Rt,rho] = ADCCeq_grm_1P_forecast(theta(1:3),trdata,optimizer);
            elseif strcmp(CopulaSpec.corrspec,'GDCC')
                [Rt,rho] = GDCCeq_grm_1P_forecast(theta,trdata,optimizer);
            elseif strcmp(CopulaSpec.corrspec,'AGDCC')
                [Rt,rho] = GDCCeq_grm_1P_forecast(theta,trdata,optimizer);
            end
        end
        Rt=rho;
    end
end
% ------------------------------------------------------------------------
% the Gaussian Copula log likelihood
% ------------------------------------------------------------------------
if strcmp(CopulaSpec.type,'Gaussian')==1 && strcmp(optimizer,'SDPT3')==0
    trdata=norminv(data);
    if strcmp(optimizer,'fmincon')==1 && strcmp(CopulaSpec.corrspec,'static') == 0
        if strcmp(CopulaSpec.corrspec,'DCC')==1
            [Rt,rho] = DCCeq_grm_1P_forecast(theta(1:2),trdata,optimizer);
        elseif strcmp(CopulaSpec.corrspec,'TVC')==1
            [Rt,rho] = TVCeq_grm_1P_forecast(theta(1:2),trdata,optimizer);
        elseif  strcmp(CopulaSpec.corrspec,'ADCC')==1
            [Rt,rho] = ADCCeq_grm_1P_forecast(theta(1:3),trdata,optimizer);
        elseif strcmp(CopulaSpec.corrspec,'GDCC')==1
            [Rt,rho] = GDCCeq_grm_1P_forecast(theta(1:4),trdata,optimizer);
        elseif strcmp(CopulaSpec.corrspec,'AGDCC')
            [Rt,rho] = AGDCCeq_grm_1P_forecast(theta(1:6),trdata,optimizer);
        end
    elseif strcmp(optimizer,'fminunc')==1 && strcmp(CopulaSpec.corrspec,'static') == 0
        if strcmp(CopulaSpec.corrspec,'DCC')==1
            [Rt,rho] = DCCeq_grm_1P_forecast(theta(1:2),trdata,optimizer);
        elseif strcmp(CopulaSpec.corrspec,'TVC')==1
            [Rt,rho] = TVCeq_grm_1P_forecast(theta(1:2),trdata,optimizer);
        elseif  strcmp(CopulaSpec.corrspec,'ADCC')==1
            [Rt,rho] = ADCCeq_grm_1P_forecast(theta(1:3),trdata,optimizer);
        elseif strcmp(CopulaSpec.corrspec,'GDCC')==1
            [Rt,rho] = GDCCeq_grm_1P_forecast(theta(1:4),trdata,optimizer);
        elseif strcmp(CopulaSpec.corrspec,'AGDCC')
            [Rt,rho] = AGDCCeq_grm_1P_forecast(theta(1:6),trdata,optimizer);
        end
    elseif strcmp(CopulaSpec.corrspec,'static') == 1
        rho = corr(trdata);
        for i=1:T
            Rt(:,:,i)  = rho;
        end
    end
    Rt=rho;
end
% -----------------------------------------------------------------------
% the Gumbel Copula Log Likelihood
% ----------------------------------------------------------------------
if strcmp(CopulaSpec.type,'Gumbel') == 1
    trdata = norminv(data);
    u = data(:,1);
    v = data(:,2);
    if strcmp(optimizer,'fmincon')==1
        if (strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1 ...
                || strcmp(CopulaSpec.corrspec,'GDCC') == 1 || strcmp(CopulaSpec.corrspec,'AGDCC') == 1 ...
                || strcmp(CopulaSpec.corrspec,'TVC')) && strcmp(optimizer,'fmincon')==1
            %         Annahme: DCC Gaussian Variablen
            trdata = norminv([u,v]);
            if strcmp(CopulaSpec.corrspec,'DCC') == 1
                [Holder, rho] = DCCeq_grm_1P_forecast(theta,trdata,'fmincon');
            elseif strcmp(CopulaSpec.corrspec,'ADCC') == 1
                [Holder, rho] = ADCCeq_grm_1P_forecast(theta,trdata,'fmincon');
            elseif strcmp(CopulaSpec.corrspec,'GDCC') == 1
                [Holder, rho] = GDCCeq_grm_1P_forecast(theta,trdata,'fmincon');
            elseif strcmp(CopulaSpec.corrspec,'AGDCC') == 1
                [Holder, rho] = AGDCCeq_grm_1P_forecast(theta,trdata,'fmincon');
            elseif strcmp(CopulaSpec.corrspec,'TVC') == 1
                [Holder, rho] = TVCeq_grm_1P_forecast(theta,trdata,'fmincon');
            end
            %             restingiere Gumbel auf positive rho's
            rho = rho.*(rho>0);
            %         setze Variablen die Null sind 0+epsilon, damit es keine
            %         numerischen Probleme gibt
            rho1 = rho==0;
            rho1 = rho1*1e-5;
            rho = rho+ rho1;
            %         Wandle Gauß rho in Kendalls tau (Gumbel) um
            ken = 2*asin(rho)./pi;
            tau = 1./(1-ken);
        elseif strcmp(CopulaSpec.corrspec,'Patton')==1
            tau = Pattoneq_grm(theta,data,CopulaSpec.type,'fmincon');
        end
        if strcmp(CopulaSpec.corrspec,'static')==1
            tau = repmat(theta,[T,1]);
        end
    elseif strcmp(optimizer,'fminunc')==1
        if (strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1 ...
                || strcmp(CopulaSpec.corrspec,'GDCC') == 1 || strcmp(CopulaSpec.corrspec,'AGDCC') == 1 ...
                || strcmp(CopulaSpec.corrspec,'TVC'))
            if strcmp(CopulaSpec.corrspec,'DCC') == 1
                [Holder, rho] = DCCeq_grm_1P_forecast(theta,trdata,'fminunc');
            elseif strcmp(CopulaSpec.corrspec,'ADCC') == 1
                [Holder, rho] = ADCCeq_grm_1P_forecast(theta,trdata,'fminunc');
            elseif strcmp(CopulaSpec.corrspec,'GDCC') == 1
                [Holder, rho] = GDCCeq_grm_1P_forecast(theta,trdata,'fminunc');
            elseif strcmp(CopulaSpec.corrspec,'AGDCC') == 1
                [Holder, rho] = AGDCCeq_grm_1P_forecast(theta,trdata,'fminunc');
            elseif strcmp(CopulaSpec.corrspec,'TVC') == 1
                [Holder, rho] = TVCeq_grm(theta,trdata,'fminunc');
            end
            ken = 2*asin(rho)./pi;
            tau = 1./(1-ken);
        elseif strcmp(CopulaSpec.corrspec,'Patton')==1
            tau = Pattoneq_grm(theta,data,CopulaSpec.type,'fminunc');
        end
    end
    Rt=tau;
end

% setzet Dummy für likelihood
likelihood = 1;

