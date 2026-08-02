function out=hfunc_grm(u,v,theta,CopulaSpec)
% this function calculates the h function of some supported Copulas as
% introduced in Aas et al:"Pair - copula construction of multiple
% dependence"
% INPUTS:
% u,v:          uniform variables
% theta:        the copula parameters
% Author: Martin Grziska
% last modification: 09/01/2010


% Anmerkungen:
% generell: Die h-Funktion beschreibt die marginal conditional-Verteilungsfunktion:
% h(x,v,theta) = F(x|v) = (\partial C(x,v,theta))/(\partial v), wobei v die
% conditioning Variable beschreibt.
% Als Output erhält man ebenfalls eine U(0,1) gleichverteilte Variable
% Wenn man eine zeitvariierende Struktur der bzw. des
% Abhängigkeitsparameters unterstellt, so muss diese auch in der partiellen
% Ableitung auftauchen

[R,C]=size(theta);

if C>3
    error('theta is a scalar or a matrix with two columns at most')
end

type=CopulaSpec.type;

if iscell(theta) == 1
    theta = theta{1,1};
end

if strcmp(type, 't')==1
    if strcmp(CopulaSpec.corrspec,'static')
        nu=theta(1);
        rho=theta(2);
        if strcmp(CopulaSpec.optimizer,'fminunc') == 1
            nu=2.01+exp(nu);
        end
    elseif strcmp(CopulaSpec.corrspec,'DCC') == 1
        nu = theta(end);
        if strcmp(CopulaSpec.optimizer,'fminunc') == 1
            nu=2.01+exp(nu);
        end
        trdata = tinv([u,v],nu);
        [Rt, veclRt]=DCCeq_grm(theta(1:2),trdata,CopulaSpec.optimizer);
        rho=veclRt;
    elseif strcmp(CopulaSpec.corrspec,'TVC') == 1
        nu=theta(end);
        if strcmp(CopulaSpec.optimizer,'fminunc') == 1
            nu=2.01+exp(nu);
        end
        trdata = tinv([u,v],nu);
        [Rt, veclRt]=TVCeq_grm(theta(1:2),trdata,CopulaSpec.optimizer);
        rho=veclRt;
    elseif strcmp(CopulaSpec.corrspec,'ADCC') == 1
        nu=theta(end);
        if strcmp(CopulaSpec.optimizer,'fminunc') == 1
            nu=2.01+exp(nu);
        end
        trdata = tinv([u,v],nu);
        [Rt, veclRt]=ADCCeq_grm(theta(1:3),trdata,CopulaSpec.optimizer);
        rho=veclRt;
    elseif strcmp(CopulaSpec.corrspec,'GDCC')
        nu=theta(end);
        if strcmp(CopulaSpec.optimizer,'fminunc') == 1
            nu=2.01+exp(nu);
        end
        trdata = tinv([u,v],nu);
        [Rt, veclRt]=GDCCeq_grm(theta(1:4),trdata,CopulaSpec.optimizer);
        rho=veclRt;
    elseif strcmp(CopulaSpec.corrspec,'AGDCC')
        nu=theta(end);
        if strcmp(CopulaSpec.optimizer,'fminunc') == 1
            nu=2.01+exp(nu);
        end
        trdata = tinv([u,v],nu);
        [Rt, veclRt]=GDCCeq_grm(theta(1:6),trdata,CopulaSpec.optimizer);
        rho=veclRt;
    end
    %     siehe Aas et al (2006), S.35
    out1 = tinv(u,nu)-rho.*tinv(v,nu);
    out2 = sqrt(((nu+tinv(v,nu).^2).*(1-rho.^2))./(nu+1));
    
    out = tcdf(out1./out2,nu+1);
    
elseif strcmp(type, 'Clayton') == 1 || strcmp(type, 'Rotated Clayton') == 1
    if  strcmp(CopulaSpec.optimizer,'fminunc') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
        tau=.0001+.85./(1+exp(-theta));
    elseif strcmp(CopulaSpec.optimizer,'fminunc') == 1 && strcmp(CopulaSpec.corrspec,'Patton') == 1 %%Patton-Spezifikation: ARMA-type-3Parameter(omega,alpha,beta)
        tau = Pattoneq_grm(theta,[u,v],type,'fminunc');
        tau = .0001+.85./(1+exp(-tau));
    elseif strcmp(CopulaSpec.optimizer,'fminunc') == 1 && strcmp(CopulaSpec.corrspec,'DCC') == 1 %%Patton-Spezifikation: ARMA-type-3Parameter(omega,alpha,beta)
        theta = sin(pi*(theta./(theta+2))/2);
        [Holder, rho] = DCCeq_grm(theta,[u,v],'fminunc');
        rho = rho.*(rho>0);
        rho1 = rho==0;
        rho1 = rho1*1e-4;
        rho = rho + rho1;
        ken = 2*asin(rho)./pi;
        tau = 2*ken./(1-ken);
    elseif strcmp(CopulaSpec.optimizer,'fmincon') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
        tau=theta;
    elseif strcmp(CopulaSpec.optimizer,'fmincon') == 1 && strcmp(CopulaSpec.corrspec,'Patton') == 1 %%Patton-Spezifikation: ARMA-type-3Parameter(omega,alpha,beta)
        tau = Pattoneq_grm(theta,[u,v],CopulaSpec.type,'fmincon');
    elseif strcmp(CopulaSpec.optimizer,'fmincon') == 1 && (strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') || strcmp(CopulaSpec.corrspec,'TVC') == 1) ...
            || strcmp(CopulaSpec.corrspec,'GDCC') || strcmp(CopulaSpec.corrspec,'AGDCC')
        trdata = norminv([u,v]);
        if strcmp(CopulaSpec.corrspec,'DCC') == 1
            [Holder, rho] = DCCeq_grm(theta,trdata,'fmincon');
        elseif strcmp(CopulaSpec.corrspec,'ADCC') == 1
            [Holder, rho] = ADCCeq_grm(theta,trdata,'fmincon');
        elseif strcmp(CopulaSpec.corrspec,'TVC') == 1
            [Holder, rho] = TVCeq_grm(theta,trdata,'fmincon');
        elseif strcmp(CopulaSpec.corrspec,'GDCC') == 1
            [Holder, rho] = GDCCeq_grm(theta,trdata,'fmincon');
        elseif strcmp(CopulaSpec.corrspec,'AGDCC') == 1
            [Holder, rho] = AGDCCeq_grm(theta,trdata,'fmincon');
        end
        rho = rho.*(rho>0);
        rho1 = rho==0;
        rho1 = rho1*1e-4;
        rho = rho+ rho1;
        ken = 2*asin(rho)./pi;
        tau = 2*ken./(1-ken);
    end
    % %     Partielle Ableitung der Clayton-Copula nach v Gleichung(7) bei Aas et
    %     al, S.3 (mit Matlab symbolic)
    if strcmp(type,'Clayton')
        out = 1./(v.^(tau +1).*(1./u.^tau + 1./v.^tau - 1).^(1./tau + 1));
    elseif strcmp(type,'Rotated Clayton')
        out = 1 - 1./((1-v).^(tau +1).*(1./(1-u).^tau + 1./(1-v).^tau - 1).^(1./tau + 1));
    end
    
elseif strcmp(type,'Gumbel') == 1
    if strcmp(CopulaSpec.optimizer,'fmincon') == 1 && strcmp(CopulaSpec.corrspec,'static') == 1
        tau = theta;
    elseif strcmp(CopulaSpec.optimizer,'fmincon') == 1 && strcmp(CopulaSpec.corrspec,'Patton') == 1 %%Patton-Spezifikation: ARMA-type-3Parameter(omega,alpha,beta)
        tau = Pattoneq_grm(theta,[u,v],type,'fmincon');
    elseif strcmp(CopulaSpec.optimizer,'fmincon') == 1 && (strcmp(CopulaSpec.corrspec,'DCC') == 1 || strcmp(CopulaSpec.corrspec,'ADCC') == 1 || strcmp(CopulaSpec.corrspec,'TVC') == 1 ...
            || strcmp(CopulaSpec.corrspec,'GDCC') || strcmp(CopulaSpec.corrspec,'AGDCC'))
        trdata = norminv([u,v]);
        if strcmp(CopulaSpec.corrspec,'DCC') == 1
            [Holder, rho] = DCCeq_grm(theta,trdata,'fmincon');
        elseif strcmp(CopulaSpec.corrspec,'ADCC') == 1
            [Holder, rho] = DCCeq_grm(theta,trdata,'fmincon');
        elseif strcmp(CopulaSpec.corrspec,'TVC') == 1
            [Holder, rho] = TVCeq_grm(theta,trdata,'fmincon');
        elseif strcmp(CopulaSpec.corrspec,'GDCC')
            [Holder, rho] = GDCCeq_grm(theta,trdata,'fmincon');
        elseif strcmp(CopulaSpec.corrspec,'AGDCC')
            [Holder, rho] = AGDCCeq_grm(theta,trdata,'fmincon');
        end
        rho = rho.*(rho>0);
        rho1 = rho==0;
        rho1 = rho1*1e-5;
        rho = rho+ rho1;
        ken= 2*asin(rho)./pi;
        tau = 1./(1-ken);
    elseif strcmp(CopulaSpec.optimizer,'fminunc') == 1  && strcmp(CopulaSpec.corrspec,'Patton') == 1
        tau = Pattoneq_grm(theta,[u,v],type,'fminunc');
    end
    out1 = 1./exp(((-log(u)).^tau + (-log(v)).^tau).^(1./tau));
    out2 = ((-log(v)).^(tau - 1).*((-log(u)).^tau + (-log(v)).^tau).^(1./tau - 1))./v;
    out = out1.*out2;
    
elseif strcmp(type, 'SJC') == 1
    if strcmp(CopulaSpec.optimizer,'fminunc')
        theta=.0001+.85./(1+exp(-theta));
    end
    out1=hfuncJC_grm(u,v,theta,CopulaSpec.corrspec);
    out2=hfuncJC_grm(1-u,1-v,[theta(2);theta(1)]);
    out=.5*(out1 - out2 + 1);
    
elseif strcmp(type, 'Gaussian') == 1
    trdata=norminv([u,v]);
    if strcmp(CopulaSpec.corrspec,'DCC') == 1
        [Rt, veclRt]=DCCeq_grm(theta(1:2),trdata,CopulaSpec.optimizer);
        rho=veclRt;
    elseif strcmp(CopulaSpec.corrspec,'ADCC') == 1
        [Rt, veclRt]=ADCCeq_grm(theta(1:3),trdata,CopulaSpec.optimizer);
        rho=veclRt;
    elseif strcmp(CopulaSpec.corrspec,'TVC') == 1
        [Rt, veclRt]=TVCeq_grm(theta(1:2),trdata,CopulaSpec.optimizer);
        rho=veclRt;
    elseif strcmp(CopulaSpec.corrspec,'GDCC') == 1
        [Rt, veclRt]=GDCCeq_grm(theta(1:4),trdata,CopulaSpec.optimizer);
        rho=veclRt;
    elseif strcmp(CopulaSpec.corrspec,'AGDCC') == 1
        [Rt, veclRt]=AGDCCeq_grm(theta(1:6),trdata,CopulaSpec.optimizer);
        rho=veclRt;
    elseif strcmp(CopulaSpec.corrspec,'static') == 1
        ken=corr(u,v,'type','kendall');
        rho = sin(ken*pi/2);
    end
    out1 = norminv(u,0,1) - rho.*norminv(v,0,1);
    out2 = sqrt(1-rho.^2);
    out = normcdf(out1./out2);
end

%clear rounding erros
% problem: manchmal treten Rundungsfehler auf, so dass u<0 oder u>1
T=size(out,1);
for i=1:T
    if out(i)>0.9999
        out(i)=0.9999;
    elseif out(i)<0.0001
        out(i)=0.0001;
    end
end
