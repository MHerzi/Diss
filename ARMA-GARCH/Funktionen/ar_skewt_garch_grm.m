function [parameters, LLF, stderrors, robustSE, ht, scores, resid, likelihood, EXITFLAG] = ar_skewt_garch_grm(data , p , q , arlag, const, startingvals, options)
% PURPOSE:
%     SKEWT_GARCH(P,Q) parameter estimation with different error distributions, the NOrmal, The T,
%     and the Generalized Error Distribution
%
% USAGE:
%     [parameters, likelihood, stderrors, robustSE, ht, scores] = skewt_garch(data , p , q , startingvals, options, type)
%
%
% INPUTS:
%     data: A single column of zero mean random data, normal or not for quasi likelihood
%
%     P: Non-negative, scalar integer representing a model order of the ARCH
%       process
%
%     Q: Positive, scalar integer representing a model order of the GARCH
%       process: Q is the number of lags of the lagged conditional variances included
%       Can be empty([]) for ARCH process
%
%     startingvals: A (1+p+q) (plus 1 if STUDENTT OR GED is selected for the nu parameter) vector of starting vals.
%       If you do not provide, a naieve guess of 1/(2*max(p,q)+1) is used for the arch and garch parameters,
%       and omega is set to make the real unconditional variance equal
%       to the garch expectation of the expectation.
%
%     options: default options are below.  You can provide an options vector.  See HELP OPTIMSET
%
%     arlag: laglength of AR(k)-model
%     const : 1 - estimate AR(k) with const, else 0
%
% OUTPUTS:
%     parameters : a [1+p+q X 1] column of parameters with omega, alpha1, alpha2, ..., alpha(p)
%                 beta1, beta2, ... beta(q), nu, lambda
%
%     likelihood = the loglikelihood evaluated at he parameters
%
%     robustSE = QuasiLikelihood std errors which are robust to some forms of misspecification(see White 94)
%
%     stderrors = the inverse analytical hessian, not for quasi maximum liklihood
%
%     ht = the estimated time varying VARIANCES
%
%     scores = The numberical scores(# of params by t) for M testing
%
% COMMENTS:
%   SKEWT_GARCH(P,Q) the following(wrong) constratins are used(they are right for the (1,1) case or any Arch case
%     (1) Omega > 0
%     (2) Alpha(i) >= 0 for i = 1,2,...P
%     (3) Beta(i)  >= 0 for i = 1,2,...Q
%     (4) sum(Alpha(i) + Beta(j)) < 1 for i = 1,2,...P and j = 1,2,...Q
%     (5) nu>2 of Students T and nu>1 for GED
%
%   The time-conditional variance, H(t), of a GARCH(P,Q) process is modeled
%   as follows:
%
%     H(t) = Omega + Alpha(1)*r_{t-1}^2 + Alpha(2)*r_{t-2}^2 +...+ Alpha(P)*r_{t-p}^2+...
%                    Beta(1)*H(t-1)+ Beta(2)*H(t-2)+...+ Beta(Q)*H(t-q)
%
%   Default Options
%
%   options  =  optimset('fmincon');
%   options  =  optimset(options , 'TolFun'      , 1e-003);
%   options  =  optimset(options , 'Display'     , 'iter');
%   options  =  optimset(options , 'Diagnostics' , 'on');
%   options  =  optimset(options , 'LargeScale'  , 'off');
%   options  =  optimset(options , 'MaxFunEvals' , '400*numberOfVariables');
%
%
%  uses SKEWT_GARCHLIKELIHOOD and GARCHCORE.  You should MEX, mex 'path\garchcore.c', the MEX source
%  The included MEX is for R12 Windows and was compiled with VC++6. It gives a 10-15 times speed improvement
%
% Author: Andrew Patton
% Author: Kevin Sheppard
% Author: Martin Grziska
% kevin.sheppard@economics.ox.ac.uk
% Revision: 2    Date: 12/31/2001
% Modifications by Martin Grziska, 05/02/2010

t=size(data,1);

if nargin<7
    options=[];
end

if size(data,2) > 1
    error('Data series must be a column vector.')
elseif isempty(data)
    error('Data Series is Empty.')
end

if (length(q) > 1) || any(q < 0)
    error('Q must ba a single positive scalar or 0 for ARCH.')
end

if (length(p) > 1) || any(p <  0)
    error('P must be a single positive number.')
elseif isempty(p)
    error('P is empty.')
end

if isempty(q) || q==0;
    q=0;
    m=p;
else
    m  =  max(p,q);
end

if const < 0 || const > 1
    error('const has to be either 0 or 1')
end

if const==1
    arlag=arlag+1;
end
stdEstimate =  std(data,1);

if nargin<=5 || isempty(startingvals)
    %     [parameters] = skewt_garch(data , p , q);
    %     omega = parameters(1);
    %     alpha = parameters(2);
    %     beta = parameters(3);
    %     nu = parameters(4);
    %     lambda = parameters(5);
    [omega, beta, alpha]  =  garch0(p, q, stdEstimate);
    nu=5;
    lambda=0;
    
else
    omega=startingvals(1);
    alpha=startingvals(2:p+1);
    beta=startingvals(p+2:p+q+1);
    nu  = startingvals(p+q+2);
    lambda = startingvals(p+q+3);
end


sumA =[[-eye(1+p+q) zeros(1+p+q,arlag+1);...
    0  ones(1,p)  ones(1,q) zeros(1,p+q+arlag-1);...
    zeros(1,1+p+q+arlag) -1]...
    zeros(1+p+q+2,1)];

sumB =  [zeros(1+p+q,1); 1 ; -4.1];

if (nargin <= 5) ||isempty(options)
    options  =  optimset('fmincon');
    options  =  optimset(options , 'TolFun'      , 1e-006);
    options  =  optimset(options , 'Display'     , 'iter');
    options  =  optimset(options , 'Diagnostics' , 'on');
    options  =  optimset(options , 'LargeScale'  , 'off');
    options  =  optimset(options , 'MaxFunEvals' , 400*(2+p+q));
    options  =  optimset(options , 'Algorithm'   , 'interior-point');
end

% if constraints with sum(abs(AR(k))) work set 0 to 1
sumB = sumB - [zeros(1+p+q,1); 1; 1]*2*optimget(options, 'TolCon', 1e-6);

% startingvals for AR
optionsAR=optimset('fmincon');
optionsAR=optimset(optionsAR,'Display','off','Diagnostics','off','LargeScale','off','MaxFunEvals',5000*(0+arlag-1),'MaxIter',1000*(0+arlag-1),'Algorithm','interior-point');
[parameters_AR] = armaxfilter(data,const,arlag-1,0,[],optionsAR);
AR = parameters_AR';
startingvals = [omega ; alpha ; beta;  AR; nu; lambda];
m=max(p,q);

warning off;

[parameters, LLF, EXITFLAG, OUTPUT, LAMBDA, GRAD] =  fmincon('ar_skewt_garchlikelihood_grm', startingvals , sumA, sumB,[],[],[],[],'ar_garch_skewt_nonlincon',options, data, p, q, m, arlag, const);
[LLF, Holder, likelihood, resid] = ar_skewt_garchlikelihood_grm(parameters , data , p , q, m, arlag, const);

if EXITFLAG<=0
    EXITFLAG
    fprintf(1,'Not Sucessful! \n')
end

if nargout>1
    parameters(find(parameters(1:1+p+q) <  0)) = 0;
    parameters(find(parameters(1) <= 0)) = realmin;
    hess = hessian_2sided('ar_skewt_garchlikelihood_grm',parameters,data,p , q, m, arlag, const);
    [Holder, ht, likelihood]=ar_skewt_garchlikelihood_grm(parameters,data,p , q, m, arlag, const);
    likelihood=-likelihood;
    stderrors=hess^(-1);
end

if const==1
    t=t-(arlag-1);
else
    t=t-arlag;
end

if nargout > 4
    h=max(abs(parameters/2),1e-2)*eps^(1/3);
    hplus=parameters+h;
    hminus=parameters-h;
    likelihoodsplus=zeros(t,length(parameters));
    likelihoodsminus=zeros(t,length(parameters));
    for i=1:length(parameters)
        hparameters=parameters;
        hparameters(i)=hplus(i);
        [HOLDER, HOLDER1, indivlike] = ar_skewt_garchlikelihood_grm(hparameters, data, p , q, m, arlag, const);
        likelihoodsplus(:,i)=indivlike;
    end
    for i=1:length(parameters)
        hparameters=parameters;
        hparameters(i)=hminus(i);
        [HOLDER, HOLDER1, indivlike] = ar_skewt_garchlikelihood_grm(hparameters, data, p , q, m, arlag, const);
        likelihoodsminus(:,i)=indivlike;
    end
    scores=(likelihoodsplus-likelihoodsminus)./(2*repmat(h',t,1));
    scores=scores-repmat(mean(scores),t,1);
    B=scores'*scores;
    robustSE=stderrors*B*stderrors;
end

function [K, GARCH, ARCH] = garch0(p, q,stdEstimate)

GARCH =  0.85;
GARCH =  GARCH(ones(p,1)) / max(p,1);
GARCH =  GARCH(:);

ARCH  =  0.90 - sum(GARCH);
ARCH  =  ARCH(ones(q,1)) / max(q,1);
ARCH  =  ARCH(:);


if isempty(stdEstimate) ||  (stdEstimate <= 0)
   K  =  1e-3;   % A decent assumption for daily returns.
else
   K  =  stdEstimate * (1 - sum([GARCH ; ARCH]));
end
