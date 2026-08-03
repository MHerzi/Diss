function h=egarchcore_grm(data, parameters, stdEstimate,p, o, q ,m , T, errortype)
% PURPOSE:
%     Core routing for egarch(use MEX file)
%
% USAGE:
%     h = egarchcore_grm(data, parameters, stdEstimate,p, q ,m , T, errotype);
%
% INPUTS:
%     See egarch
%
% OUTPUTS:
%     See egarch
%
% COMMENTS:
%     Helper function part of UCSD_GARCH toolbox. Used if you do not use the MEX file.
%     You should use the MEX file.
% ACHTUNG: Modifikation im gegensatz zum Original werden hier die
% EGARCH-Parameter in folgender Reihenfolge verlangt: Omega, ARCH, GAMMA, GARCH
%
%
%
% Author: Kevin Sheppard, Martin Grziska
% kevin.sheppard@economics.ox.ac.uk
% Revision: 2    Date: 12/31/2001
% Modification (Martin Grziska): 04/30/2010

T=T+500;
parameters(find(parameters(1:1+p+q+o) <= 0)) = realmin;
garchparameters=parameters(1:p+q+o+1);
remainingparams=parameters(p+q+2:length(parameters));

constp = garchparameters(1);
archp = garchparameters(2);
garchp = garchparameters(4);

if errortype ~=1 && errortype ~= 4
   nu2 = remainingparams(length(remainingparams));
end

if errortype == 4
    nu3 = remainingparams(length(remainingparams)-1:end);
end

if errortype == 1
    randnum=randn(T,1);
elseif errortype == 2
    randnum=stdtdis_rnd(T,nu2);
elseif errortype == 3
    randnum=ged_rnd(T,nu2);
elseif errortype == 4
    randnum = skewtdis_rnd(nu3(1),nu3(2),T);
else
    error('Do not know how to simulate the type of errors')
end

UncondStd   = sqrt(constp/(1-sum(archp)-sum(garchp)));
h           = UncondStd*ones(T,1);
m=max([p o q]);

dataneg=zeros(size(h));
dataneg(1:m)=UncondStd/2;
datamb=zeros(size(h));
datamb(1:m)=UncondStd;
data=zeros(size(h));
% for t = m+1:T
%     h(t) = (constp   +  archp'*datamb(t-(1:p)).^nu  + garchp'*h(t-(1:q)).^lambda)^(1/lambda);
%     data(t)=h(t)*randnum(t);
%     datamb(t)=abs(data(t)-b);
%     dataneg(t)=abs((data(t)<0)*data(t));
% end

for t = (m + 1):T
    h(t) = exp(garchparameters' * [1 ; abs(datamb(t-(1:p)))./sqrt(h(t-(1:p))); datamb(t-(1:o))./sqrt(h(t-(1:o))); log(h(t-(1:q)))]);
    data(t)=h(t)*randnum(t);
%     datamb(t)=abs(data(t)-b);
%     dataneg(t)=abs((data(t)<0)*data(t));
end

data=randnum.*h;
data=data(501:T);
h=h(501:T).^2;

% 
% h=zeros(T,1);
% h(1:m)=stdEstimate^2;
% 
% for t = (m + 1):T
%     h(t) = exp(parameters' * [1 ; abs(data(t-(1:p)))./sqrt(h(t-(1:p))); data(t-(1:o))./sqrt(h(t-(1:o))); log(h(t-(1:q)))]);
%     if h(t)==0
%         h(t)=stdEstimate*.00001;
%     end
% end