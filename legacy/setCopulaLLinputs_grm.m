function CopulaSpec=setCopulaLLinputs_grm(dimension)
% this function creates the CopulaSpec structure array that contains the
% various specifications for the copula log likelihood.
% INPUT:
% dinension:        positive integer, num of columns of data
% ----------------------------------------------------------------------
% author: Manthos Vogiatzoglou
% contact at: vogia@yahoo.com
% -----------------------------------------------------------------------
% ************************************************************************
% bivariate copulas
% -------------------------------------------------------------------------
CopulaSpec.ll='copula';
if dimension==2
    j=menu('choose the copula family','Gaussian','t','Clayton','SJC');
if j==1
    CopulaSpec.type='Gaussian';
    k=menu('choose the specification for the dependency parameter','TVC','DCC');
    if k==1
       CopulaSpec.depspec='TVC';
    elseif k==2
       CopulaSpec.depspec='DCC';
    end
elseif j==2
    CopulaSpec.type='t';
    l=menu('choose the specification for the dependency parameter','static','TVC','DCC');
    if l==1
       CopulaSpec.depspec='static';
    elseif l==2
       CopulaSpec.depspec='TVC';
    elseif l==3
       CopulaSpec.depspec='DCC';
    end
elseif j==3
    CopulaSpec.type='Clayton';
    l=menu('choose the specification for the dependency parameter','static','Patton');
    if l==1
       CopulaSpec.depspec='static';
    elseif l==2
       CopulaSpec.depspec='Patton';
    end
elseif j==4
    CopulaSpec.type='SJC';
    l=menu('choose the specification for the dependency parameter','static','Patton');
    if l==1
       CopulaSpec.depspec='static';
    elseif l==2
       CopulaSpec.depspec='Patton';
    end
end
end
% -------------------------------------------------------------------------
% multivariate copulas
% -------------------------------------------------------------------------
if dimension>2
    m=menu('choose the copula family','Gaussian','t');
if m==1
    CopulaSpec.type='Gaussian';
    n=menu('choose the specification for the dependency parameter','static','TVC','DCC','penalized');
    if n==1
       CopulaSpec.depspec='static';
    elseif n==2
       CopulaSpec.depspec='TVC';
    elseif n==3
       CopulaSpec.depspec='DCC';
    elseif n==4
       CopulaSpec.depspec='Penalized'; 
    end
elseif m==2
    CopulaSpec.type='t';
    o=menu('choose the specification for the dependency parameter','static','TVC','DCC');
    if o==1
       CopulaSpec.depspec='static';
    elseif o==2
       CopulaSpec.depspec='TVC';
    elseif o==3
       CopulaSpec.depspec='DCC';
    end
end
end 
if strcmp(CopulaSpec.type,'Gaussian') && (strcmp(CopulaSpec.depspec,'Penalized') || strcmp(CopulaSpec.depspec,'static'))
display(' for this specification, the SDPT3 solver is used')
CopulaSpec.optimizer='SDPT3';
else
p=menu('choose optimizer','fmincon','fminunc');
if p==1
    CopulaSpec.optimizer='fmincon';
else
    CopulaSpec.optimizer='fminunc';
end
end
q=menu('save Gradient and Hessian?','yes','no');
if q==1
    CopulaSpec.derivatives='on';
elseif q==2
    CopulaSpec.derivatives='off';
end