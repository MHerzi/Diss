function out=hfunc_grm_mix(uu,vv,theta,type,depspec,optimizer)
% this function calculates the h function of some supported Copulas as
% introduced in Aas et al:"Pair - copula construction of multiple
% dependence"
% INPUTS:
% u,v:          uniform variables
% theta:        the copula parameters
% Author: Martin Grziska
% last modification: 07/20/2010


% Anmerkungen:
% generell: Die h-Funktion beschreibt die marginal conditional-Verteilungsfunktion:
% h(x,v,theta) = F(x|v) = (\partial C(x,v,theta))/(\partial v), wobei v die
% conditioning Variable beschreibt.
% Als Output erhält man ebenfalls eine U(0,1) gleichverteilte Variable
% Wenn man eine zeitvariierende Struktur der bzw. des
% Abhängigkeitsparameters unterstellt, so muss diese auch in der partiellen
% Ableitung auftauchen

[R,C]=size(theta);

if C>2
    error('theta is a scalar or a matrix with two columns at most')
end


if iscell(theta) == 1
    theta = theta{1,1};
end

if strcmp(type, 't')==1
    if strcmp(depspec,'static') == 1
        nu=theta(1);
        rho=theta(2);        
    elseif strcmp(depspec,'DCC') == 1
        nu=theta(end);
        trdata = tinv([uu,vv],nu);
        [Rt, veclRt]=DCCeq_grm(theta(1:2),trdata,optimizer);
        rho=veclRt;
    elseif strcmp(depspec,'TVC') == 1
        nu=theta(end);
        trdata = tinv([uu,vv],nu);
        [Rt, veclRt]=TVCeq_grm(theta(1:2),trdata,optimizer);
        rho=veclRt;
    elseif strcmp(depspec,'ADCC') == 1
        nu=theta(end);
        trdata = tinv([uu,vv],nu);
        [Rt, veclRt]=ADCCeq_grm(theta(1:3),trdata,optimizer);
        rho=veclRt;
    elseif strcmp(depspec,'AGDCC') == 1
        nu=theta(end);
        trdata = tinv([uu,vv],nu);
        [Rt, veclRt]=AGDCCeq_grm(theta(1:3),trdata,optimizer);
        rho=veclRt;
    end
    %     siehe Aas et al (2006), S.35
    out1 = tinv(uu,nu)-rho.*tinv(vv,nu);
    out2 = sqrt(((nu+tinv(vv,nu).^2).*(1-rho.^2))./(nu+1));
    if out1 < -1
        out1=-.99999;
    elseif out2 < -1
        out2 = -.99999;
    elseif out1 > 1
        out1 = .99999;
    elseif out2 > 1
        out2 = .99999;
    end
    
    out = tcdf(out1./out2,nu+1);
    
elseif strcmp(type, 'clayton') == 1
    taucl = Pattoneq_mix_grm(theta,[uu,vv],type);
    out = 1./(vv.^(taucl +1).*(1./uu.^taucl + 1./vv.^taucl - 1).^(1./taucl + 1));    
    
elseif strcmp(type,'gumbel') == 1
    taugu = Pattoneq_mix_grm(theta,[uu,vv],type);
    out1 = 1./exp(((-log(uu)).^taugu + (-log(vv)).^taugu).^(1./taugu));
    out2 = ((-log(vv)).^(taugu - 1).*((-log(uu)).^taugu + (-log(vv)).^taugu).^(1./taugu - 1))./vv;
    out = out1.*out2;
    
elseif strcmp(type, 'gaussian') == 1
    trdata=norminv([uu,vv]);
    if strcmp(depspec,'DCC') == 1
        [Rt, veclRt]=DCCeq_grm(theta(1:2),trdata,optimizer);
        rho=veclRt;
    elseif strcmp(depspec,'ADCC') == 1
        [Rt, veclRt]=ADCCeq_grm(theta(1:3),trdata,optimizer);
        rho=veclRt;
    elseif strcmp(depspec,'AGDCC') == 1
        [Rt, veclRt]=AGDCCeq_grm(theta(1:6),trdata,optimizer);
        rho=veclRt;
    elseif strcmp(depspec,'TVC') == 1
        [Rt, veclRt]=TVCeq_grm(theta(1:2),trdata,optimizer);
        rho=veclRt;
    elseif strcmp(depspec,'static') == 1
        ken=corr(uu,vv,'type','kendall');
        rho = sin(ken*pi/2);
    end
    out1 = norminv(uu,0,1) - rho.*norminv(vv,0,1);
    out2 = sqrt(1-rho.^2);
    out = normcdf(out1./out2);
end

%clear rounding erros
T=size(out,1);
for i=1:T
    if out(i)>.99999
        out(i)=.99999;
    elseif out(i)==0
        out(i) = min(out)-1e-3;
    end
end
