function [parameters, LLF, stderrors, robustSE, ht, scores, resid, likelihood, EXITFLAG]=ar_multigarch_grm(data,p,o,q,type,errors,arlag,const,options,startingvals)
% PURPOSE:
%     This is a multi use univariate GARCH function which can estimate
%     GARCH(you should use garchpq though), EGARCH(Nelson), Threshold GARCH(Zakoian),
%     Absolute Value GARCH(Taylor/Schwert), Non-Linear Asymetric GARCH(Engle Ng),
%     GJR-GARCH(G,J &R), Nonlinear GARCH(Higgins Bera),
%     and asymetric power GARCH(Ding Engle and Granger),
%     and a flexible garch whiel allows for non-linearities, threshold effects,
%     news impact rotation and recentering of the news impact curve
%
% USAGE:
%      [parameters, LLF, stderrors, robustSE, ht, scores, resid, likelihood, EXITFLAG]=ar_multigarch_grm(data,p,o,q,type,errors,arlag,const,options,startingvals)
%
% INPUTS:
%     data: Zero Mean series of regression residuals or other zero mean series
%     p :  The order of the ARCH(innovations) process
%     o :  The order of the TARCH process(only for thoses models with threshold effects)
%     q :  The order of the GARCH process
%     const : 1 - estimate AR(k) with const, else 0
%     type :  A string telling the proc which type of model is to be estimated
%             Can be one of the following(note: ALL CAPS)
%                   Without assymetric terms
%             'GARCH'    -  Normal GARCH Model(see garchpq or fattailed_garch instead)
%             'AVGARCH'  -  Absolute Value GARCH
%             'NGARCH'   -  Non-linear GARCH
%             'NAGARCH'  -  Non-Linear Asymetric GARCH
%                   With assymetric terms
%             'EGARCH'   -  Exponential GARCH
%             'TGARCH'   -  Threshold GARCH
%             'GJRCARCH' -  GJR Representation of TARCH
%             'APGARCH'  -  Asymetric Power GARCH
%             'ALLGARCH' -  Asymetric Power GARCH with news impact centering parameter
%
% OUTPUTS:
%    parameters: a 1+p+q+special column vector of estimated model parameters, the size
%                of special depends on the model being estimated
%    LLF: The (negative) log-likelihood of the likelihood function.
%    parameters : a [1+p+o+q X 1] column of parameters with omega, alpha1, alpha2, ..., alpha(p)
%                 tarch(1), tarch(2), ... tarch(o) beta1, beta2, ... beta(q)
%    likelihood = the loglikelihood evaluated at he parameters
%    ht = the estimated time varying VARIANCES
%    stderrors = the inverse analytical hessian, not for quasi maximum liklihood
%    robustSE = robust standard errors of form A^-1*B*A^-1*T^-1
%              where A is the analytic hessian
%              and B is the covariance of the scores
%    scores = the list of T scores for use in M testing
%
%    the special parameter will contain estimates of the following, in this order
%    (non-estimated parameters are not reported), range is in paranthesis
%
%    lambda(0, infty] -  this is the parameter which determines the power of sigma being estimaded(i.e. 2 for GARCH)
%    b(-infty, infty) -  The centering parameter for the effect of the new impact on volatility
%
%
% COMMENTS:
%
%     This proceedure estimates conditional volatility of the following form:
%
%     h(t)^(lambda)-1                                                 h(t-1)^(lambda)-1
%     ------------  = omega + a*h(t-1)^(lambda)+f(data(t-1))^(nu)+ b*-------------------
%          lambda                                                           lambda
%
%     f(data(t))=abs(data(t)-b) - c*(data(t)-b)
%
%     This program is in no small part influenced by the work of
%     L. Hentschel J. of Empirical Finance 95
%
%
%     Author: Kevin Sheppard
%     kevin.sheppard@economics.ox.ac.uk
%     Revision: 2    Date: 12/31/2001
% Modifications by: Martin Grziska, 03/01/2010

[t,k]=size(data);

if o~=0
    if strcmp(type,'GARCH') || strcmp(type,'AVGARCH') || strcmp(type,'NGARCH') || strcmp(type,'NAGARCH')
        error('Selected model does not allow for asymetric terms')
    end
end

if strcmp(errors,'NORMAL') || strcmp(errors,'STUDENTST') || strcmp(errors,'GED') || strcmp(errors,'SKEWT')
    if strcmp(errors,'NORMAL')
        errortype = 1;
    elseif strcmp(errors,'STUDENTST')
        errortype = 2;
    elseif strcmp(errors,'GED')
        errortype = 3;
    elseif strcmp(errors,'SKEWT')
        errortype = 4;
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

if (length(p) > 1) || any(p <=  0)
    error('P must be a single positive number.')
elseif isempty(p)
    error('P is empty.')
end

if (length(o) > 1) || any(o <  0)
    error('O must be a single positive number.')
end

if const < 0 || const > 1
    error('const has to be either 0 or 1')
end


if (nargin <= 9) || isempty(options)
    options  =  optimset('fmincon');
    options  =  optimset(options , 'TolFun'      , 1e-006);
    options  =  optimset(options , 'Display'     , 'iter');
    options  =  optimset(options , 'Diagnostics' , 'on');
    options  =  optimset(options , 'LargeScale'  , 'off');
    options  =  optimset(options , 'MaxFunEvals' , 400*(2+p+q));
    options  =  optimset(options , 'Algorithm'   , 'interior-point');
    %         options  =  optimset(options , 'Algorithm'   , 'active-set');
end

if strcmp(type,'EGARCH');
    [parameters, LLF, stderrors, robustSE, ht, scores, resid, likelihood, EXITFLAG]=ar_egarch_grm(data,p,o,q,errors,arlag,const);
    return
elseif strcmp(type,'GARCH')
    if strcmp(errors,'SKEWT')
        [parameters, LLF, stderrors, robustSE, ht, scores, resid, likelihood, EXITFLAG] = ar_skewt_garch_grm(data,p,q,arlag,const);
        return
    else
        [parameters, LLF, stderrors, robustSE, ht, scores, resid, likelihood, EXITFLAG] = ar_fattailed_garch_grm(data,p,q,errors,arlag,const);
        return
    end
elseif strcmp(type,'AVGARCH')
    if strcmp(errors,'SKEWT')
        [parameters, LLF, stderrors, robustSE, ht, scores, resid, likelihood, EXITFLAG] = ar_skewt_avgarch_grm(data,p,q,arlag,const);
    else
        [parameters, LLF, stderrors, robustSE, ht, scores, resid, likelihood, EXITFLAG] = ar_avgarch_garch_grm(data,p,q,errors,arlag,const);
    end
    return
end

if isempty(q)
    q=0;
    m=p;
else
    m  =  max(p,q);
end

% Estimate starting values for AR
optionsAR=optimset('fmincon');
optionsAR=optimset(optionsAR,'Algorithm','interior-point','Display','off','Diagnostics','off','LargeScale','off','MaxFunEvals',5000*(0+arlag),'MaxIter',1000*(0+arlag));

%MFE-Toolbox
%[parameters_AR] = armaxfilter(data,const,arlag,0,[],[],optionsAR);
%AR = parameters_AR; % Modifikation am 04.04.2013, da paramters_AR als Zeilenvektor ausgegeben wird

%%%UCSD-Toolbox
[parameters_AR] = armaxfilter(data,const,arlag,0,[],optionsAR);
AR = parameters_AR';

stdEstimate = std(data);

% calc startinvals
if nargin<=10 || isempty(startingvals)
    [omega, beta,alpha]  =  garch0(p, q, stdEstimate);
    tarchp=0;
    tarchp  =  tarchp*ones(o,1)/o;
    %     nu startinvals for student-t and skew-t
    if (errortype == 4 || errortype == 2)
        EX = exist('garchset');
        if EX == 5 % Wenn Funktion garchset vorhanden ist benutze garchset
            Spec = garchset('VarianceModel','EGARCH','Distribution','T','P',p,'Q',q,'Leverage',p);
            [Coeff] = garchfit(Spec,data);
            nu2 = Coeff.DoF;
        else
            [omega, beta,alpha, tarchp]  =  egarch0(p, q, stdEstimate);
            nu2 = 6;
        end
        lambda2 = 0;
    end
else
    omega=startingvals(1);
    alpha=startingvals(2:p+1);
    tarchp=startingvals(p+2:p+o+1);
    beta=startingvals(p+o+2:p+q+o+1);
end

startingvalues=[omega ; alpha; tarchp ; beta];

[lambda, nu, b, garchtype, indicator]=multi_garch_paramsetup(type);
[sumA, sumB, startingvalues, LB, UB, garchtype]=multi_garch_constraints( startingvalues, p,o, q, data, type);
sumB = sumB-2*options.TolCon;
startingvalues=[startingvalues; AR];

% % % constraints for other models than GARCH/EGARCH (constraints are written
% % with upper and lower bounds)
% % % parameters are in order: omega, alpha, talpha, garch, (nonlinear), AR,
% % % nu, (skew-t)
% % [sumA_shep, sumB_shep, startingvalues, LB, UB, garchtype]=multi_garch_constraints( startingvalues, p,o, q, data, type);
% % sumB_shep = sumB_shep-2*options.TolCon;
% % if strcmp(type,'TGARCH') || strcmp(type,'GJRGARCH')
% %     LB = [0 zeros(1,p) zeros(1,o) zeros(1,q)]+1e-5;
% %     UB = LB+1-2*1e-5;
% %     sumA = [0 ones(p,1) ones(p,1)*0.5 ones(q,1)];
% %     sumB = 1-1e-6;
% % elseif strcmp(type,'AVGARCH')
% %     LB = [0 zeros(1,p) zeros(1,q) -sumB_shep(end)]+1e-6;
% %     UB = [1 ones(1,p) ones(1,q) sumB_shep(end-1)]-1e-6;
% %     sumA = [0 ones(p,1) ones(q,1) 0];
% %     sumB = 1-1e-6;
% % elseif strcmp(type,'NGARCH')
% %     LB = [0 zeros(1,p) zeros(1,q) -sumB_shep(end)]+1e-6;
% %     UB = [1 ones(1,p) ones(1,q) 10]-1e-6;
% %     sumA = [0 ones(1,p) ones(1,q) 0];
% %     sumB = 1-1e6;
% % elseif strcmp(type,'NAGARCH')
% %     LB = [0 zeros(1,p) zeros(1,q) -sumB_shep(end)]+1e-6;
% %     UB = [1 ones(1,p) ones(1,q) sumB_shep(end-1)]-1e-6;
% %     sumA = [0 ones(p,1) ones(q,1) 0];
% %     sumB = 1-1e-6;
% % elseif strcmp(type,'APGARCH')
% %     LB = [0 zeros(1,p) zeros(1,o) zeros(1,q) -sumB_shep(end)]+1e-5;
% %     UB = [0 zeros(1,p) zeros(1,o) zeros(1,q) 5]+1e-5;
% %     sumA = [0 ones(p,1) ones(p,1)*0.5 ones(q,1) 0];
% %     sumB = 1-1e-6;
% % end
% % % Add AR-constraint
% % if const==1
% %     sumA = [sumA zeros(size(sumA,1),arlag+1)];
% % else
% %     sumA = [sumA zeros(size(sumA,1),arlag)];
% % end
% %
% % if strcmp(errors,'STUDENTST')
% %     sumA = [sumA 0];
% %     startingvalues=[startingvalues;nu2];
% % elseif strcmp(errors,'GED')
% %     sumA = [sumA 0];
% %     startingvalues=[startingvalues;2];
% % elseif strcmp(errors,'SKEWT')
% %     startingvalues=[startingvalues;nu2;lambda2];
% %     sumA = [sumA zeros(1,2)];
% %     LB = [LB 4.1 -.999];
% %     UB = [UB 200 .999];
% % end


% Add AR-constraint
sumA = [sumA zeros(size(sumA,1),arlag+const)];

if strcmp(errors,'STUDENTST')
    sumA=[sumA';zeros(1,size(sumA,1))]';
    %     constraint nu>2.1
    nuconst=zeros(1,size(sumA,2));
    nuconst(size(sumA,2))=-1;
    %     constraint: nu <200
    nuconst=[nuconst;zeros(1,size(nuconst,2)-1) 1];
    sumA=[sumA;nuconst];
    sumB=[sumB;-2.1;200];
    startingvalues = [startingvalues; nu2];
elseif strcmp(errors,'GED')
    startingvalues=[startingvalues;2];
    sumA=[sumA';zeros(1,size(sumA,1))]';
    %     constraint: nu>1.1
    nuconst=zeros(1,size(sumA,2));
    nuconst(size(sumA,2))=-1;
    %     constraint nu<5
    nuconst=[nuconst;zeros(1,size(nuconst,2)-1) 1];
    sumA=[sumA;nuconst];
    sumB=[sumB;-1.01;5];
elseif strcmp(errors,'SKEWT')
    startingvalues=[startingvalues;nu2;lambda2];
    sumA = [sumA zeros(size(sumA,1),2)];
    %     constraint: nu>4.1
    nuconst=zeros(1,size(sumA,2));
    nuconst(end-1)=-1;
    %     constraint: nu< 200
    nuconst=[nuconst;zeros(1,size(nuconst,2)-2) 1 0];
    sumA=[sumA;nuconst];
    sumB=[sumB;-4.1;200];
end
% additional constraint can be found in "ar_multigarch_nonlincon"

[parameters, LLF, EXITFLAG, OUTPUT, LAMBDA, GRAD] =  fmincon('ar_multigarch_likelihood_grm', startingvalues ,sumA  , sumB ,[] , [] , LB , UB,'ar_multigarch_nonlincon',options,data, p , o, q, garchtype, errortype, arlag, const);

if EXITFLAG<=0
    EXITFLAG
    fprintf(1,'Not Sucessful! \n')
end

parameters(find(parameters(1:p+o+q+1)<  0)) = 0;
parameters(find(parameters(1) <= 0)) = realmin;
[LLF, ht,likelihood, resid] = ar_multigarch_likelihood_grm(parameters,data,p,o,q,garchtype, errortype, arlag, const);

t=t-arlag;

likelihood=-likelihood;
if nargout >= 3
    %Calculate std errors if needed
    hess = hessian_2sided('ar_multigarch_likelihood_grm',parameters,data,p,o,q,garchtype,errortype, arlag, const);
    stderrors=hess^(-1);
    h=min(abs(parameters/2)+1e-4,max(parameters,1e-2))*eps^(1/3);
    hplus=parameters+h;
    hminus=parameters-h;
    likelihoodsplus=zeros(t,length(parameters));
    likelihoodsminus=zeros(t,length(parameters));
    for i=1:length(parameters)
        hparameters=parameters;
        hparameters(i)=hplus(i);
        [HOLDER, HOLDER1, indivlike] = ar_multigarch_likelihood_grm(hparameters,data,p,o,q,garchtype,errortype, arlag, const);
        likelihoodsplus(:,i)=indivlike;
    end
    for i=1:length(parameters)
        hparameters=parameters;
        hparameters(i)=hminus(i);
        [HOLDER, HOLDER1, indivlike] = ar_multigarch_likelihood_grm(hparameters,data,p,o,q,garchtype,errortype, arlag, const);
        likelihoodsminus(:,i)=indivlike;
    end
    %     (f(x+h) - f(x-h))/2h
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
