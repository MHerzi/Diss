function [residuals, UnResiduals,GARCHspec,likelihoods]=filtReturnsGARCH(returns,distr,model,method)
%Estimates standardize residuals and transforms them to uniform. 
%For a returns matrix with size TxN each series (column) is filtered by a 
%volatility model.Supported models are c/ar(1)-GARCH(1,1), c/ar(1)-GJR(1,1) with 
%Gaussian or T residuals. The choice between c and AR(1) for the mean is
%been done by the Ljung - Box test. If no autocorrelation is present c is
%chosen, else AR(1) is chosen.Another model that is supported is GARCH(1,1) with
%skewT(Hansen's Skew-T distribution) residuals. Since this distribution is
%not supported by GARCH toolbox only the simplest specification for this
%model is used. If you want to use it on real (not simulated) data you have
%to filter the (mean eq of) returns to deal with possible autocorrelation 
%or at least demean your returns (function 'demeanRets'). 

%INPUTS:            
%returns:               Matrix with TxN size
%distr:                 String with values 'Gaussian','T', or 'SkewT'
%model:                 String with values 'GJR' or 'GARCH'
%method:                String with values 'CML' or 'IFM'
%OUTPUTS:
%residuals:             Matrix TxN with the standardized residuals
%UnResiduals:           Matrix TxN with uniform marginals (input for copula)
%spec:                  The parameters of the univariate GARCH. If distr is
%                       SkewT spec is a 5xN matrix, else is a cell containing structure 
%likelihoods:           Vector Nx1, with log likelihood values
% ------------------------------------------------------------------------
% Author: Manthos Vogiatzoglou, UoM, 2008 - 2009
% contact at: vogia@yahoo.com

[T N]=size(returns);
UnResiduals=zeros(T,N);
%---------------Error checking--------------
if isempty(returns)
    error('---------- IMPUT RETURNS MATRIX ----------------')
end
if isempty(model);
    model='GARCH';
end
if isempty(distr);
    distr='Gaussian';
end
if strcmp(distr,'T')==0 && strcmp(distr,'Gaussian')==0 && strcmp(distr,'SkewT')==0
    error('distr can be either Gaussian, T, or SkewT')
end
if strcmp(model,'GARCH')==0 && strcmp(model,'GJR')==0
    error('--model should be either GARCH or GJR ----')
end
if strcmp(method, 'CML')==0 && strcmp(distr,'Gaussian')==1
    display('Warning! you have chosen the IFM method assuming normal residuals')
    display('this probably lead to infinite values on the next step')
    display('that means you will not get any estimates!!')
    display('Use the IFM method for a heavy - tailed distribution, only!')
end
    
% The following code segment extracts the filtered residuals and 
% volatilities from the returns of each stock.

if strcmp(distr,'T')==1 || strcmp(distr,'Gaussian')==1  
residuals = NaN(T, N);  % preallocate storage
sigmas    = NaN(T, N);
likelihoods=zeros(N,1);
H=-99*ones(1,N);
GARCHspec=cell(1,N);
if strcmp(model,'GJR')==1
for i=1:N
    test=lbqtest(returns(:,i),15,.05);
    H(i)=test;
    spec = garchset('Distribution' , distr  , 'Display', 'off', ...
                            'VarianceModel', 'GJR', 'P', 1, 'Q', 1, 'R', H(1,i));
    [spec , errors, likelihoods(i,1), ...
     residuals(:,i), sigmas(:,i)] = garchfit(spec, returns(:,i));
 garchdisp(spec,errors)
 GARCHspec{i}=spec;
 clear spec
end
else
for i=1:N
    test=lbqtest(returns(:,i),15,.05);
    H(i)=test;
    spec = garchset('Distribution' , distr  , 'Display', 'off', ...
                            'VarianceModel', 'GARCH', 'P', 1, 'Q', 1, 'R', H(1,i));
    [spec , errors, likelihoods(i,1), ...
     residuals(:,i), sigmas(:,i)] = garchfit(spec, returns(:,i));
 garchdisp(spec,errors)
 GARCHspec{i}=spec;
 clear spec
end
end
residuals = residuals ./ sigmas;
else
    fprintf('-----------GARCH(1,1)-SkewT SPECIFICATION-------------')
    % MATLAB's GARCH toolbox only supports Normal and Student T residulas
    % therefore we use skewt_garch function from UCSD GARCH toolbox of
    % Kevin Sheppard. You can download this amazing toolbox from:
    %http://www.kevinsheppard.com/wiki/UCSD_GARCH
    uu=mean(returns);
    parameters=zeros(5,N); robustStErrors=zeros(5,N); likelihoods=zeros(N,1);
    ht=zeros(T,N);
    GARCHspec=cell(1,2);
    H=zeros(1,N);
    for i=1:N
    test=lbqtest(returns(:,i),15,.05);
    H(i)=test;
    if H(i)==0
        returns(:,i)=returns(:,i)-mean(returns(:,i));
    else
         returns(:,i)=returns(:,i)-mean(returns(:,i));
        display('your returns are autocorrelated')
    end
    end
    for i=1:N
    [parameters(:,i), likelihoods(i,1), stderrors, robustSE, ht(:,i)] = skewt_garch(returns(:,i) , 1,1);
    robustStErrors(:,i)=diag(robustSE);
    if parameters(4,i)>200;
        parameters(4,i)=200;
    end
    end
    residuals=returns./sqrt(ht); %the standardized residuals of the GARCH
    clear robustSE scores
    GARCHspec{1}=[uu;parameters];
    GARCHspec{2}=robustStErrors;
end
% the transformation of iid residuals to U(0,1)
if strcmp(method,'CML')==1
   for i = 1:N
    UnResiduals(:,i) =empiricalCDF(residuals(:,i)); % transform margin to uniform
   end
elseif strcmp(method,'IFM')==1
if strcmp(distr,'Gaussian')==1
for i = 1:N
    UnResiduals(:,i) =normcdf(residuals(:,i)); % transform margin to uniform
end
elseif strcmp(distr,'T')==1
    DoF=zeros(1,N);
 for i = 1:N
    DoF(i)=GARCHspec{1,i}.DoF;
    UnResiduals(:,i) =tcdf(residuals(:,i),DoF(i)); % transform margin to uniform
end   
elseif strcmp(distr,'SkewT')==1
    display('GARCHspec{1} is the estimates and GARCHspec{2} are the st errors')
for i = 1:N
    UnResiduals(:,i) =skewtdis_cdf(residuals(:,i),parameters(4,i),parameters(5,i)); % transform margin to uniform
end 
else
    error('distr can be either Gaussian, T, or SkewT')
end
end