function out=hfuncSJC(u,v,theta)
if max(u)>1 || min(u)<0
    error('u is uniform U(0,1)')
end
if max(v)>1 || min(v)<0
    error('v is uniform U(0,1)')
end
out1=hfuncJC(u,v,theta);
out2=hfuncJC(1-u,1-v,[theta(2);theta(1)]);
out=.5*(out1 - out2 + 1);
