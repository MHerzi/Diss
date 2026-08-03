function CopulaSpec = setCopulaVineLLinputs_grm(dimension)
% Define copula vine and estimation procedure parameters
% This functions creates a structure array that contains all the
% specification of the model. It is in a menu form therefore it is very
% easy to use.
%
% USAGE:     CopulaSpec = setCopulaVineLLinputs_grm(dimension)
%
% INPUTS:
%            dimension:        A positive integer, number of columns in the data
%                              matrix.
%
% OUTPUTS:
%            CopulaSpec:       structure containing all the information
%            n                 ecessary to use fitCopulaVine_grm.m
% ----------------------------------------------------------------------
% author: Martin Grziska based on a code from Manthos Vogiatzoglou
% -----------------------------------------------------------------------
CopulaSpec.ll='copulavine';

if isscalar(dimension)==0;
    error('dimension is a positive integer, the num of data columns')
end

if dimension<=2
    error('Copula Vines have meaning for sizes greater than two')
end

CopulaSpec.use='fit copula vine';

m=menu('choose the copula Vine decomposition','DVine','DVine-Mixture');
% if m==1
%     CopulaSpec.decomposition='CanonicalVine';
if m==1
    CopulaSpec.decomposition='DVine';
elseif m==2
    CopulaSpec.decomposition='DVine-Mix';
    CopulaSpec.optimizer='fmincon';
    n=menu('choose the first family of the mixture - choose elliptical first if you wish to estimate one ','Clayton','Gumbel','t','Gaussian');
    if n==1
        CopulaSpec.family{1} = 'clayton';
        CopulaSpec.corrspec{1}='Patton';
        o=menu('chosse second family of mixture','Gumbel','t','Gaussian');
        if o==1
            CopulaSpec.family{2} = 'gumbel';
            CopulaSpec.corrspec{2}='Patton';
        elseif o==2
            CopulaSpec.family{2} = 't';
            h=menu('choose dynamic structure for t','ADCC','DCC','GDCC','AGDCC');
            if h==1
                CopulaSpec.corrspec{2}='ADCC';
                CopulaSpec.Dynamic='ADCC';
            elseif h==2
                CopulaSpec.corrspec{2}='DCC';
                CopulaSpec.Dynamic='DCC';
            elseif h==3
                CopulaSpec.Corrspec{2}='GDCC';
                CopulaSpec.Dynamic='GDCC';
            end
        elseif o==3
            CopulaSpec.family{2} = 'gaussian';
            h=menu('choose dynamic structure for gaussian','ADCC','DCC','GDCC','AGDCC');
            if h==1
                CopulaSpec.corrsepc{2}='ADCC';
                CopulaSpec.Dynamic='ADCC';
            elseif h==2
                CopulaSpec.corrspec{2}='DCC';
                CopulaSpec.Dynamic='DCC';
            elseif h==3
                CopulaSpec.corrspec{2}='GDCC';
                CopulaSpec.Dynamic='GDCC';
            elseif h==4
                CopulaSpec.corrspec{2}='AGDCC';
                CopulaSpec.Dynamic='AGDCC';
            end
        end
    elseif n==2
        CopulaSpec.familiy{1}='gumbel';
        CopulaSpec.corrspec{1}='Patton';
        o=menu('chosse second family of mixture','Clayton','t','Gaussian');
        if o==1
            CopulaSpec.family{2} = 'clayton';
            CopulaSpec.corrspec{2}='Patton';
        elseif o==2
            CopulaSpec.family{2} = 't';
            h=menu('choose dynamic structure for t','ADCC','DCC','GDCC','AGDCC');
            if h==1
                CopulaSpec.corrsepc{2}='ADCC';
                CopulaSpec.Dynamic='ADCC';
            elseif h==2
                CopulaSpec.corrspec{2}='DCC';
                CopulaSpec.Dynamic='DCC';
            elseif h==3
                CopulaSpec.corrspec{2}='GDCC';
                CopulaSpec.Dynamic='GDCC';
            elseif h==4
                CopulaSpec.corrspec{2}='AGDCC';
                CopulaSpec.Dynamic='AGDCC';
            end
        elseif o==3
            CopulaSpec.family{2} = 'gaussian';
            h=menu('choose dynamic structure for gaussian','ADCC','DCC','GDCC','AGDCC');
            if h==1
                CopulaSpec.corrsepc{2}='ADCC';
                CopulaSpec.Dynamic='ADCC';
            elseif h==2
                CopulaSpec.corrspec{2}='DCC';
                CopulaSpec.Dynamic='DCC';
            elseif h==3
                CopulaSpec.corrspec{2}='GDCC';
                CopulaSpec.Dynamic='GDCC';
            elseif h==4
                CopulaSpec.corrspec{2}='AGDCC';
                CopulaSpec.Dynamic='AGDCC';
            end
        end
    elseif n==3
        CopulaSpec.family{1}='t';
        h=menu('choose dynamic structure for t','ADCC','DCC','GDCC','AGDCC');
        if h==1
            CopulaSpec.corrspec{1}='ADCC';
            CopulaSpec.Dynamic='ADCC';
        elseif h==2
            CopulaSpec.corrspec{1}='DCC';
            CopulaSpec.Dynamic='DCC';
        elseif h==3
            CopulaSpec.corrspec{1}='GDCC';
            CopulaSpec.Dynamic='GDCC';
        elseif h==4 
            CopulaSpec.corrspec{1}='AGDCC';
            CopulaSpec.Dynamic='AGDCC';
        end
        o=menu('chosse second family of mixture','Gumbel','Clayton');
        CopulaSpec.corrspec{2}='Patton';
        if o==1
            CopulaSpec.family{2} = 'gumbel';
            CopulaSpec.corrspec{2}='Patton';
        elseif o==2
            CopulaSpec.family{2} = 'clayton';
            CopulaSpec.corrspec{2}='Patton';
        end
    elseif n==4
        CopulaSpec.family{1}='gaussian';
        h=menu('choose dynamic structure for gaussian','ADCC','DCC','GDCC','AGDCC');
        if h==1
            CopulaSpec.corrspec{1}='ADCC';
            CopulaSpec.Dynamic='ADCC';
        elseif h==2
            CopulaSpec.corrspec{1}='DCC';
            CopulaSpec.Dynamic='DCC';
        elseif h==3
            CopulaSpec.corrspec{1}='GDCC';
            CopulaSpec.Dynamic='GDCC';
        elseif h==4
            CopulaSpec.corrspec{1}='AGDCC';
            CopulaSpec.Dynamic='AGDCC';
        end
        o=menu('chosse second family of mixture','Gumbel','clayton');
        CopulaSpec.corrspec{2}='Patton';
        if o==1
            CopulaSpec.family{2} = 'gumbel';
            CopulaSpec.corrspec{2}='Patton';
        elseif o==2
            CopulaSpec.family{2} = 'clayton';
            CopulaSpec.corrspec{2}='Patton';
        end
    end
    return
end
n=menu('choose the copula family','Clayton','SJC','t','Gaussian','Rotated Clayton','Gumbel' );
if n==1
    CopulaSpec.type='Clayton';
    o=menu('choose dynamic structure','static','Patton','DCC','ADCC','GDCC','AGDCC','TVC');
    if o==1
        CopulaSpec.corrspec='static';
    elseif o==2
        CopulaSpec.corrspec='Patton';
        CopulaSpec.optimizer='fminunc';
    elseif o==3
        CopulaSpec.corrspec='DCC';
        CopulaSpec.optimizer='fmincon';
    elseif o==4
        CopulaSpec.corrspec='ADCC';
        CopulaSpec.optimizer='fmincon';
    elseif o==5
        CopulaSpec.corrspec='GDCC';
        CopulaSpec.optimizer='fmincon';
    elseif o==6
        CopulaSpec.corrspec='AGDCC';
        CopulaSpec.optimizer='fmincon';
    elseif o==7
        CopulaSpec.corrspec='TVC';
        CopulaSpec.optimizer='fmincon';
    end
elseif n==2
    CopulaSpec.type='SJC';
    o=menu('choose dynamic structure','static','Patton');
    if o==1
        CopulaSpec.corrspec='static';
    else
        CopulaSpec.corrspec='Patton';
    end
elseif n==3
    CopulaSpec.type='t';
    o=menu('choose dynamic structure','static','DCC','TVC','ADCC','GDCC','AGDCC');
    if o==1
        CopulaSpec.corrspec='static';
    elseif o==2
        CopulaSpec.corrspec='DCC';
        CopulaSpec.optimizer='fmincon';
    elseif o==3
        CopulaSpec.corrspec='TVC';
        CopulaSpec.optimizer='fmincon';
    elseif o==4
        CopulaSpec.corrspec='ADCC';
        CopulaSpec.optimizer='fmincon';
    elseif o==5
        CopulaSpec.corrspec='GDCC';
        CopulaSpec.optimizer='fmincon';
    elseif o==6
        CopulaSpec.corrspec='AGDCC';
        CopulaSpec.optimizer='fmincon';
    end
elseif n==4
    CopulaSpec.type = 'Gaussian';
    o=menu('choose dynamic structure','static','DCC','TVC','ADCC','GDCC','AGDCC');
    if o==1
        CopulaSpec.corrspec='static';
    elseif o==2
        CopulaSpec.corrspec='DCC';
        CopulaSpec.optimizer='fmincon';
    elseif o==3
        CopulaSpec.corrspec='TVC';
        CopulaSpec.optimizer='fmincon';
    elseif o==4
        CopulaSpec.corrspec='ADCC';
        CopulaSpec.optimizer='fmincon';
    elseif o==5
        CopulaSpec.corrspec='GDCC';
        CopulaSpec.optimizer='fmincon';
    elseif o==6
        CopulaSpec.corrspec='AGDCC';
        CopulaSpec.optimizer='fmincon';
    end
elseif n==5
    CopulaSpec.type = 'Rotated Clayton';
    o=menu('choose dynamic structure','static','Patton','DCC','ADCC','TVC');
    if o==1
        CopulaSpec.corrspec='static';
    elseif o==2
        CopulaSpec.corrspec='Patton';
        CopulaSpec.optimizer='fminunc';
    elseif o==3
        CopulaSpec.corrspec='DCC';
        CopulaSpec.optimizer='fmincon';
    elseif o==4
        CopulaSpec.corrspec='ADCC';
        CopulaSpec.optimizer='fmincon';
        
    elseif o==5
        CopulaSpec.corrspec='TVC';
        CopulaSpec.optimizer='fmincon';
    end
elseif n==6
    CopulaSpec.type = 'Gumbel';
    o=menu('choose dynamic structure','static','Patton','DCC','ADCC','TVC','GDCC','AGDCC');
    if o==1
        CopulaSpec.corrspec='static';
    elseif o==2
        CopulaSpec.corrspec='Patton';
        CopulaSpec.optimizer='fminunc';
    elseif o==3
        CopulaSpec.corrspec='DCC';
        CopulaSpec.optimizer='fmincon';
    elseif o==4
        CopulaSpec.corrspec='ADCC';
        CopulaSpec.optimizer='fmincon';
    elseif o==5
        CopulaSpec.corrspec='TVC';
        CopulaSpec.optimizer='fmincon';
    elseif o==6
        CopulaSpec.corrspec='GDCC';
        CopulaSpec.optimizer='fmincon';
    elseif o==7
        CopulaSpec.corrspec='AGDCC';
        CopulaSpec.optimizer='fmincon';
    end
end

if strcmp(CopulaSpec.corrspec,'static') || strcmp(CopulaSpec.corrspec,'Patton')
    k=menu('choose the optimization function','fminunc','fmincon');
    if k==1
        CopulaSpec.optimizer='fminunc';
    else
        CopulaSpec.optimizer='fmincon';
    end
end

q=menu('save Gradient and Hessian? (if "yes" stderrors will be calculated automatically with fminon hessian)','yes','no');
if q==1
    CopulaSpec.derivatives='on';
elseif q==2
    CopulaSpec.derivatives='off';
end

if strcmp(CopulaSpec.derivatives,'off')
    r=menu('calculate stderrors (with own hessian)?','yes','no');
    if r==1
        CopulaSpec.stderrors='on';
    elseif r==2
        CopulaSpec.stderrors = 'off';
    end
end
