function out=hfuncJC_grm(u,v,theta,corrspec)
T=size(u,1);
if iscell(theta)==0
k1=1./log2(2-theta(1)); k2=-1./log2(theta(2));
else
k1=1./log2(2-theta(1,1)); k2=-1./log2(theta(1,2));
end
sig1=1./(1 - (1 - u).^k1).^k2 + 1./(1 - (1 - v).^k1).^k2 - 1;
out=((1 - 1./sig1.^(1/k2)).^(1/k1 - 1).*(1 - v).^(k1 - 1))./(sig1.^(1/k2 + 1).*(1 - (1 - v).^k1).^(k2 + 1));
for i=1:T
    if out(i)>.9999
        out(i)=.9999;
    end
end