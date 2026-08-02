function [TstatWhite,TstatGodambe,robustWhiteSE,GodambeSE,WhiteSE] = TstatVine(parameters,data,CopulaSpec,hess)

% Computes (robust) stderrors and Tstat for Vine-Copulas
%
% USAGE:
%       [TstatWhite,TstatGodambe,robustWhiteSE,GodambeSE] =...
%                    TstatVine(parameters,data,CopulaSpec,hess)
%
% INPUTS:
%        parameters:    columnvector of parameters
%        data:          t x k array of unif(0,1)-data
%        CopulaSpec:    structure used to estimate Vine-Copula
%        hess:          user supplied hessian
%        error:         string('white','godambe')
%
% OUTPUTS:
%         TstatWhite:    robust White Tstat
%       TstatGodambe:    Godambe Tstat
%      robustWhiteSE:    robust White stderrors
%          GodambeSE:    Godambe stderrors
%            WhiteSE:    White stderrors (not robust)
%
% Author: Martin Grziska, 07/20/2010

[t,k]=size(data);

if isempty(hess)
    hess = hessian_2sided('CopulaVineLL_grm',parameters,data,CopulaSpec);
end

eps=1e-5;
WhiteSE = hess^(-1);
h = min(abs(parameters/2) + 1e-4,max(parameters,1e-2))*eps^(1/3);
hplus = parameters+h;
hminus = parameters-h;
likelihoodsplus1 = cell(k-1,k-1);
likelihoodsminus1 = cell(k-1,k-1);
likelihoodsplus = zeros(t,(length(parameters)));
likelihoodsminus = zeros(t,(length(parameters)));

% create special matrix to select correct likelihood-columns; e.g. for DCC:
% first parameter is in first array, but second parameter is also in first
% arras; third parameter is in second array and fourth parameter is in
% scond array ...
if strcmp(CopulaSpec.corrspec,'DCC') == 1
    if strcmp(CopulaSpec.type,'Gaussian') || strcmp(CopulaSpec.type,'Gumbel') || strcmp(CopulaSpec.type,'Clayton')
        special = repmat(1,2,1);
        for i=2:(length(parameters)/2)
            special = [special; repmat(i,2,1)];
        end
    elseif strcmp(CopulaSpec.type,'t')
        special = repmat(1,3,1);
        for i=2:(length(parameters)/3)
            special = [special; repmat(i,3,1)];
        end
    end
    %     for ADCC: first parameter is in first array, second parameter is in
    %     first array, third parameter is in third array...
elseif strcmp(CopulaSpec.corrspec,'ADCC') == 1
    if strcmp(CopulaSpec.type,'Gaussian') || strcmp(CopulaSpec.type,'Gumbel') || strcmp(CopulaSpec.type,'Clayton')
        special = repmat(1,3,1);
        for i=2:(length(parameters)/3)
            special = [special; repmat(i,3,1)];
        end
    elseif strcmp(CopulaSpec.type,'t')
        special = repmat(1,4,1);
        for i=2:(length(parameters)/4)
            special = [special; repmat(i,4,1)];
        end
    end
end

for i=1:length(parameters)
    hparameters = parameters;
    hparameters(i) = hplus(i);
    [Holder,Holder,indivlike] = CopulaVineLL_grm(hparameters,data,CopulaSpec);
    likelihoodsplus1{i} = indivlike;
    likelihoodsplus(:,i) = likelihoodsplus1{i}(:,special(i));
end
for i=1:length(parameters)
    hparameters = parameters;
    hparameters(i) = hminus(i);
    [HOLDER, HOLDER1, indivlike] = CopulaVineLL_grm(hparameters,data,CopulaSpec);
    likelihoodsminus1{i} = indivlike;
    likelihoodsminus(:,i) = likelihoodsminus1{i}(:,special(i));
end

scores = (likelihoodsplus-likelihoodsminus)./(2*repmat(h',t,1));
B=scores'*scores;

robustWhiteSE = diag(sqrt(WhiteSE*B*WhiteSE));
TstatWhite = parameters./robustWhiteSE;
GodambeSE = diag(sqrt((-hess)^(-1)*B*(-hess)^(-1)));
TstatGodambe = parameters./GodambeSE;


