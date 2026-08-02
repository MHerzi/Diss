function [parameters, LLF, stderrors, robustSE, ht, scores, resid, likelihood, EXITFLAG]=ar_egarch_grm(data, p, o, q,errors, arlag, const, options, startingvals)
% PURPOSE:
%     E_GARCH(P,Q) parameter estimation with different error distributions, the Normal, The T,
%     and the Generalized Error Distribution
%
% USAGE:
%     [parameters, likelihood, stderrors, robustSE, ht, scores]=egarch(data,p,o,q,errors, options, startingvals);
%
% INPUTS:
%     data: A single column of zero mean random data, normal or not for quasi likelihood
%
%     P: Non-negative, scalar integer representing a model order of the ARCH
%        process
%
%     O: Number of assymetric terms to include
%
%     Q: Positive, scalar integer representing a model order of the GARCH
%        process: Q is the number of lags of the lagged conditional variances included
%        Can be empty([]) for ARCH process
%
%     error:  The type of error being assumed, valid types are:
%             'NORMAL' - Gaussian Innovations
%             'STUDENTST' - T-distributed errors
%             'GED' - General Error Distribution
%
%     arlag: laglength of AR(k)-model
%     const : 1 - estimate AR(k) with const, else 0
%
%
%     startingvals: A (1+p+q) (plus 1 if STUDENTT OR GED is selected for the nu parameter) vector of starting vals.
%        If you do not provide, a naieve guess of 1/(2*max(p,q)+1) is used for the arch and garch parameters,
%        and omega is set to make the real unconditional variance equal
%        to the garch expectation of the expectation.
%        %%%%%%%%% Modification: k -additional parameters of the AR(k)
%        model %%%%%%%%%%%%%%%%%
%
%     options: default options are below.  You can provide an options vector.  See HELP OPTIMSET
%
%
% OUTPUTS:
%     parameters : a [1+2*p+q X 1] column of parameters with omega, alpha1, alpha2, ..., alpha(p), absolute alpha(1),
%              absolute alpha(2), ... , absolute alpha(p), beta1, beta2, ... beta(q)
%
%     likelihood = the loglikelihood evaluated at he parameters
%
%     robustSE = QuasiLikelihood std errors which are robust to some forms of misspecification(see White 94)
%
%     stderrors = the inverse analytical hessian, not for quasi maximum liklihood
%
%     ht = the estimated time varying VARIANCES
%
%     scores = The numberical scores(# fo params by t) for M testing
%
%     resid = residuals from AR(k) estimation
%
%
% COMMENTS:
%   EGARCH(P,Q) the following(wrong) constratins are used(they are right for the (1,1) case or any Arch case
%     (1) nu>2 of Students T and nu>1 for GED
%
%   The time-conditional variance, H(t), of a EGARCH(P,Q) process is modeled
%   as follows:
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% !!!! Absolute alpha is the SECOND parameter and NOT the third as often
%      modeled !!!!!!!!!!!!!!!!!
%     log h(t) = Omega + Absolute Alpha(1)* abs(r_{t-1}/(sqrt(h(t-1)))) + ...
%                    + Absolute Alpha(P)* abs(r_{t-p}/(sqrt(h(t-p))))+ Alpha(1)*r_{t-1}/(sqrt(h(t-1))) + Alpha(2)*r_{t-2}/(sqrt(h(t-2))) +...
%                    + Alpha(P)*r_{t-p}/(sqrt(h(t-p)))+ +  Beta(1)* log(H(t-1))
%                    + Beta(2)*log(H(t-2))+...+ Beta(Q)*log(H(t-q))
%  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
% Author: Martin Grziska based on a code of Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 2    Date: 12/31/2001
% Modification: Martin Grziska, 02/21/2010

t=size(data,1);

if strcmp(errors,'NORMAL') || strcmp(errors,'STUDENTST') || strcmp(errors,'GED') ||  strcmp(errors,'SKEWT')
    if strcmp(errors,'NORMAL')
        errortype = 1;
    elseif strcmp(errors,'STUDENTST')
        errortype = 2;
    elseif strcmp(errors,'GED')
        errortype = 3;
    elseif strcmp(errors,'SKEWT')
        errortype =4;
    end
else
    error('error must be one of the four strings NORMAL, STUDENTST, GED, or SKEWT');
end


if size(data,2) > 1
    error('Data series must be a column vector.')
elseif isempty(data)
    error('Data Series is Empty.')
end


if (length(q) > 1) || any(q < 0)
    error('Q must ba a single positive scalar or an empty vector for ARCH.')
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


if nargin<8 || isempty(options)
    options  =  optimset('fmincon');
    options  =  optimset(options , 'TolFun'      , 1e-6);
    options  =  optimset(options , 'Display'     , 'iter');
    options  =  optimset(options , 'Diagnostics' , 'on');
    options  =  optimset(options , 'LargeScale'  , 'off');
    options  =  optimset(options , 'MaxFunEvals' , 400*(2+p+q)) ;
    options  =  optimset(options , 'Algorithm'   ,'interior-point');   
%     options  =  optimset(options , 'Algorithm'   ,'active-set');   
end


% startingvals for AR
optionsAR=optimset('fmincon');
optionsAR=optimset(optionsAR,'Display','off','Diagnostics','off','LargeScale','off','MaxFunEvals',5000*(0+arlag-1),'MaxIter',1000*(0+arlag-1));

%UCSD-Toolbox
[parameters_AR] = armaxfilter(data,const,arlag,0,[],optionsAR);
AR = parameters_AR';

%%%MFE-Toolbox
%[parameters_AR] = armaxfilter(data,const,arlag,0,[],[],optionsAR);
%AR = parameters_AR;

stdEstimate =  std(data,1);

if nargin<9 || isempty(startingvals)
    
    [omega, beta,alpha, talpha]  =  egarch0(p, q, stdEstimate);
    
    if strcmp(errors,'STUDENTST')
        EX = exist('garchset_grm');
        if EX == 5 % Wenn Funktion garchset vorhanden ist benutze garchset  
        Spec = garchset_grm('VarianceModel','EGARCH','Distribution','T','P',p,'Q',q,'Leverage',p);
        [Coeff] = garchfit_grm(Spec,data);
        nu = Coeff.DoF;
        else 
            [omega, beta,alpha, talpha]  =  egarch0(p, q, stdEstimate);
            nu = 6;
        end
    elseif strcmp(errors,'GED')
        nu=1.5;
    elseif strcmp(errors,'SKEWT')
        EX = exist('garchset_grm');
        if EX == 5 % Wenn Funktion garchset vorhanden ist benutze garchset  
        Spec = garchset_grm('VarianceModel','EGARCH','Distribution','T','P',p,'Q',q,'Leverage',p);
        [Coeff] = garchfit_grm(Spec,data);
        nu = Coeff.DoF;
        else 
            [omega, beta,alpha, talpha]  =  egarch0(p, q, stdEstimate);
            nu = 6;
        end
        lambda=0;       
    else
        nu=[];
    end
else
    omega=startingvals(1);
    alpha=startingvals(2:p+1);
    talpha=startingvals(p+2:p+1+o);
    beta=startingvals(p+o+2:p+o+q+1);
end

% -------------------------------------------------------------------------
% ------------------------------------------------------------------------

% % Constraints
%     Konstante mean  GARCH-process
lower(1) = -5;
%  Leverage und ARCH-term
lower(2:1+p+o) =  -2*ones(p+o,1);
% GARCH-term
lower(2+p+o:1+p+o+q) = -1*ones(q,1);
%  Konstante AR
if const == 1
    lower(1+p+o+q+1) = -10;
    lower(1+p+o+q+2:1+p+o+q+1+arlag) = -1*(ones(arlag,1));
else
    lower(1+p+o+q+1:1+p+o+q+arlag) = -1*(ones(arlag,1));
end
upper = -lower;
%  AR-terme
if errortype == 1
    %     keine constraints bei Gauss-Verteilung
    startingvals = [omega ; alpha ; talpha; beta; AR];
elseif errortype == 2
    %     Student-t Verteilung
    startingvals = [omega ; alpha ; talpha; beta; AR; nu;];
    %     constraint: nu>2.1
    lower(end+1) = 2.1;
    upper(end+1) = 200;
elseif errortype == 3
    %     GED-Verteilung
    startingvals = [omega ; alpha ; talpha; beta; AR; nu;];
    lower(end+1) = 1.05;
    upper(end+1) = 50;
elseif errortype == 4
    %     skew-t Verteilung
    startingvals = [omega ; alpha ; talpha; beta; AR; nu; lambda];
    % DoF-Parameters
    lower(end+1) = 4.01;
    % skew-Parameter
    lower(end+1) = -.99;
    upper(end+1) = 200;
    upper(end+1) = .99;
end
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
if const==1
    arlag=arlag+1;
end
% Estimate the parameters
[parameters,LLF,EXITFLAG,OUTPUT,lambda,GRAD,hessian] = fmincon('ar_egarchEstLikelihood_grm', startingvals,[],[],[],[],lower,upper,'ar_egarch_nonlincon', options, data, p ,o, q, errortype, arlag, const);

[LLF, ht, likelihood, resid]= ar_egarchlikelihood_grm(parameters,data,p,o,q,errortype,arlag,const);

if EXITFLAG<=0
    EXITFLAG
    fprintf(1,'Not Sucessful! \n')
end


hess = hessian_2sided('ar_egarchlikelihood_grm',parameters,data,p,o,q,errortype,arlag,const);

if errortype ~=4
    likelihood=-likelihood;
end

if const==1
    t = t-(arlag-1);
else
    t = t-arlag;
end

stderrors=hess^(-1);
if nargout > 4
    h=min(abs(parameters/2),max(parameters,1e-2))*eps^(1/3);
    hplus=parameters+h;
    hminus=parameters-h;
    likelihoodsplus=zeros(t,length(parameters));
    likelihoodsminus=zeros(t,length(parameters));
    for i=1:length(parameters)
        hparameters=parameters;
        hparameters(i)=hplus(i);
        [HOLDER, HOLDER1, indivlike] = ar_egarchEstLikelihood_grm(hparameters,data,p,o,q,errortype, arlag, const);
        likelihoodsplus(:,i)=indivlike;
    end
    for i=1:length(parameters)
        hparameters=parameters;
        hparameters(i)=hminus(i);
        [HOLDER, HOLDER1, indivlike] = ar_egarchEstLikelihood_grm(hparameters,data,p,o,q,errortype, arlag, const);
        likelihoodsminus(:,i)=indivlike;
    end
    scores=(likelihoodsplus-likelihoodsminus)./(2*repmat(h',t,1));
    scores=scores-repmat(mean(scores),t,1);
    B=scores'*scores;
    robustSE=stderrors*B*stderrors;
end
warning on

function [K, GARCH, ARCH, Leverage] = egarch0(p, q, stdEstimate)
GARCH =  0.9;
GARCH =  GARCH(ones(p,1)) / max(p,1);
GARCH =  GARCH(:);

ARCH  =  0.2;
ARCH  =  ARCH(ones(q,1)) / max(q,1);
ARCH  =  ARCH(:);

if isempty(stdEstimate) ||  (stdEstimate <= 0)
    K  =  -0.01;   % A decent assumption for daily returns.
else
    K  =  (1 - sum(GARCH)) * log(stdEstimate);
end

Leverage  =  zeros(q,1);
