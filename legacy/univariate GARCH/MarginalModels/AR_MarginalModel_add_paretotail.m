
function [Output] = AR_MarginalModel_add_paretotail(data,archP,garchQ,GARCHOutput)
% % % % %
% Estimate marginal Distribution for univariate time series
%
% USAGE:
%        [Output] = AR_MarginalModel_add(data,archP,garchQ,GARCHOutput)
%
% INPUTS:
%         data:   t x k matirx of return series
%         const:  1 if univariate AR-GARH is to be estimated with const,
%                 else 0
%         arlag:  # of lags in AR(k)
%         archP:  # of ARCH lags
%         garchQ: # of GARCH lags
% OUTPUT:
%        Structure
%         Output.Params:       params of estimated AR-GARCH;
%         Output.ParamsGARCH:  parameters_GARCH_Gauss(1:end-GARCHOutput{j}.arlag_Gauss);
%         Output.LLF:          neg log-likelihoodLLF_GARCH_Gauss;
%         Output.stderrors:    stderrors of GARCH;
%         Output.robustSE:     robustSE (White);
%         Output.ht:           estimated Variance;
%         Output.Scores:       scores
%         Output.Innovations:  residuals of AR(k)-GARCH;
%         Output.GARCH:        string shows estimated garchtype (GARCH; EGARCH;
%                              TGARCH; GJRGARCH)
%         Output.U:            unif(0,1)
%         Output.Tstat:        robust Tstat (White);
%         Output.garchtype:    # of garchtype
%         Output.leverage:     1 if GARCH is estimated with leverage term;
%                              else 0
%         Output.errortype:    errortype for GARCH estimations (1:NORMAL, 2:
%                              STUDENTST, 3: GED, 4:SKEWT
%         Output.likelihood:   negative log-likelihoods
%         Output. EXITFLAG
%
% Author: Martin Grziska, 03/19/2010

[t,k]=size(data);
Output = cell(k,1);
% Fitte die verschiedenen GARCH-Modelle
% 1.GARCH-Gauss
% checke ob AR-GARCH mit Konstante geschätzt wurde; wenn ein GARCH-Modell
% mit const geschätzt wurde, wurden auch alle anderen mit const geschätzt
if GARCHOutput{1}.const == 1
    const=1;
else const=0;
end

for j=1:k
    if strcmp(GARCHOutput{j}.GARCH,'GARCH')==1 && strcmp(GARCHOutput{j}.dist,'GAUSS') == 1
        [parameters_GARCH_Gauss, LLF_GARCH_Gauss, stderrors_GARCH_Gauss, robustSE_GARCH_Gauss, ht_GARCH_Gauss, scores_GARCH_Gauss, resid_GARCH_Gauss, likelihood_GARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),archP,0,garchQ,'GARCH','NORMAL',GARCHOutput{j}.arlag,const);
        BIC_GARCH_Gauss = 2*LLF_GARCH_Gauss+ log(t)*size(parameters_GARCH_Gauss,1);
        Tstatistic_GARCH_Gauss=parameters_GARCH_Gauss./diag(robustSE_GARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_GARCH_Gauss./sqrt(ht_GARCH_Gauss);
Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_GARCH_Gauss = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_GARCH_Gauss;
        Output{j}.ParamsGARCH = parameters_GARCH_Gauss(1:end-GARCHOutput{j}.arlag-const);
        Output{j}.ParamsAR = parameters_GARCH_Gauss(size(Output{j}.ParamsGARCH,1)+1:size(parameters_GARCH_Gauss,1));
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_GARCH_Gauss;
        Output{j}.stderrors=stderrors_GARCH_Gauss;
        Output{j}.robustSE=robustSE_GARCH_Gauss;
        Output{j}.ht=ht_GARCH_Gauss;
        Output{j}.Scores=scores_GARCH_Gauss;
        Output{j}.Innovations=resid_GARCH_Gauss;
        Output{j}.GARCH = 'GARCH';
        Output{j}.U = U_GARCH_Gauss;
        Output{j}.Tstat = Tstatistic_GARCH_Gauss;
        Output{j}.garchtype = 1;
        Output{j}.leverage = 0;
        Output{j}.errortype = 1;
        Output{j}.likelihoods = likelihood_GARCH_Gauss;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GAUSS';
        Output{j}.Innovations = resid_GARCH_Gauss;
        Output{j}.BIC = BIC_GARCH_Gauss;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'EGARCH')==1 && strcmp(GARCHOutput{j}.dist,'GAUSS') == 1
        [parameters_EGARCH_Gauss, LLF_EGARCH_Gauss, stderrors_EGARCH_Gauss, robustSE_EGARCH_Gauss, ht_EGARCH_Gauss, scores_EGARCH_Gauss, resid_EGARCH_Gauss, likelihood_EGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),archP,1,garchQ,'EGARCH','NORMAL',GARCHOutput{j}.arlag,const);
        BIC_EGARCH_Gauss = 2*LLF_EGARCH_Gauss+log(t)*size(parameters_EGARCH_Gauss,1);
        Tstatistic_EGARCH_Gauss=parameters_EGARCH_Gauss./diag(robustSE_EGARCH_Gauss).^0.5;
        stdresid = resid_EGARCH_Gauss./sqrt(ht_EGARCH_Gauss);
Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_EGARCH_Gauss = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_EGARCH_Gauss;
        Output{j}.ParamsGARCH = parameters_EGARCH_Gauss(1:end-GARCHOutput{j}.arlag-const);
        Output{j}.ParamsAR = parameters_EGARCH_Gauss(size(Output{j}.ParamsGARCH,1)+1:size(parameters_EGARCH_Gauss,1));
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_EGARCH_Gauss;
        Output{j}.stderrors=stderrors_EGARCH_Gauss;
        Output{j}.robustSE=robustSE_EGARCH_Gauss;
        Output{j}.ht=ht_EGARCH_Gauss;
        Output{j}.Scores=scores_EGARCH_Gauss;
        Output{j}.Innovations=resid_EGARCH_Gauss;
        Output{j}.GARCH = 'EGARCH';
        Output{j}.U = U_EGARCH_Gauss;
        Output{j}.Tstat = Tstatistic_EGARCH_Gauss;
        Output{j}.garchtype = 0;
        Output{j}.leverage = 1;
        Output{j}.errortype = 1;
        Output{j}.likelihoods = likelihood_EGARCH_Gauss;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GAUSS';
        Output{j}.Innovations = resid_EGARCH_Gauss;
        Output{j}.BIC = BIC_EGARCH_Gauss;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'TGARCH')==1 && strcmp(GARCHOutput{j}.dist,'GAUSS') == 1
        [parameters_TGARCH_Gauss, LLF_TGARCH_Gauss, stderrors_TGARCH_Gauss, robustSE_TGARCH_Gauss, ht_TGARCH_Gauss, scores_TGARCH_Gauss, resid_TGARCH_Gauss, likelihood_TGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),archP,1,garchQ,'TGARCH','NORMAL',GARCHOutput{j}.arlag,const);
        BIC_TGARCH_Gauss = 2*LLF_TGARCH_Gauss+log(t)*size(parameters_TGARCH_Gauss,1);
        Tstatistic_TGARCH_Gauss=parameters_TGARCH_Gauss./diag(robustSE_TGARCH_Gauss).^0.5;
        stdresid = resid_TGARCH_Gauss./sqrt(ht_TGARCH_Gauss);
Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_GARCH_TGauss = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_TGARCH_Gauss;
        Output{j}.ParamsGARCH = parameters_TGARCH_Gauss(1:end-GARCHOutput{j}.arlag-const);
        Output{j}.ParamsAR = parameters_TGARCH_Gauss(size(Output{j}.ParamsGARCH,1)+1:size(parameters_TGARCH_Gauss,1));
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_TGARCH_Gauss;
        Output{j}.stderrors=stderrors_TGARCH_Gauss;
        Output{j}.robustSE=robustSE_TGARCH_Gauss;
        Output{j}.ht=ht_TGARCH_Gauss;
        Output{j}.Scores=scores_TGARCH_Gauss;
        Output{j}.Innovations=resid_TGARCH_Gauss;
        Output{j}.GARCH = 'TGARCH';
        Output{j}.U = U_TGARCH_Gauss;
        Output{j}.Tstat = Tstatistic_TGARCH_Gauss;
        Output{j}.garchtype = 2;
        Output{j}.leverage = 1;
        Output{j}.errortype = 1;
        Output{j}.likelihoods = likelihood_TGARCH_Gauss;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GAUSS';
        Output{j}.Innnovations = resid_TGARCH_Gauss;
        Output{j}.BIC = BIC_TGARCH_Gauss;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'GJRGARCH')==1 && strcmp(GARCHOutput{j}.dist,'GAUSS') == 1
        [parameters_GJRGARCH_Gauss, LLF_GJRGARCH_Gauss, stderrors_GJRGARCH_Gauss, robustSE_GJRGARCH_Gauss, ht_GJRGARCH_Gauss, scores_GJRGARCH_Gauss, resid_GJRGARCH_Gauss, likelihood_GJRGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),archP,archP,garchQ,'GJRGARCH','NORMAL',GARCHOutput{j}.arlag,const);
        BIC_GJRGARCH_Gauss = 2*LLF_GJRGARCH_Gauss+log(t)*size(parameters_GJRGARCH_Gauss,1);
        Tstatistic_GJRGARCH_Gauss=parameters_GJRGARCH_Gauss./diag(robustSE_GJRGARCH_Gauss).^0.5;
        stdresid = resid_GJRGARCH_Gauss./sqrt(ht_GJRGARCH_Gauss);
Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_GJRGARCH_Gauss = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_GJRGARCH_Gauss;
        Output{j}.ParamsGARCH = parameters_GJRGARCH_Gauss(1:end-GARCHOutput{j}.arlag-const);
        Output{j}.ParamsAR = parameters_GJRGARCH_Gauss(size(Output{j}.ParamsGARCH,1)+1:size(parameters_GJRGARCH_Gauss,1));
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_GJRGARCH_Gauss;
        Output{j}.stderrors=stderrors_GJRGARCH_Gauss;
        Output{j}.robustSE=robustSE_GJRGARCH_Gauss;
        Output{j}.ht=ht_GJRGARCH_Gauss;
        Output{j}.Scores=scores_GJRGARCH_Gauss;
        Output{j}.Innovations=resid_GJRGARCH_Gauss;
        Output{j}.GARCH = 'GJRGARCH';
        Output{j}.U = U_GJRGARCH_Gauss;
        Output{j}.Tstat = Tstatistic_GJRGARCH_Gauss;
        Output{j}.garchtype = 8;
        Output{j}.leverage = 1;
        Output{j}.errortype = 1;
        Output{j}.likelihoods = likelihood_GJRGARCH_Gauss;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GAUSS';
        Output{j}.Innovations = resid_GJRGARCH_Gauss;
        Output{j}.BIC = BIC_GJRGARCH_Gauss;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'AVGARCH')==1 && strcmp(GARCHOutput{j}.dist,'GAUSS') == 1
        [parameters_AVGARCH_Gauss, LLF_AVGARCH_Gauss, stderrors_AVGARCH_Gauss, robustSE_AVGARCH_Gauss, ht_AVGARCH_Gauss, scores_AVGARCH_Gauss, resid_AVGARCH_Gauss, likelihood_AVGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'AVGARCH','NORMAL',GARCHOutput{j}.arlag,const);
        BIC_AVGARCH_Gauss = 2*LLF_AVGARCH_Gauss+log(t)*size(parameters_AVGARCH_Gauss,1);
        Tstatistic_AVGARCH_Gauss=parameters_AVGARCH_Gauss./diag(robustSE_AVGARCH_Gauss).^0.5;
        stdresid = resid_AVGARCH_Gauss./sqrt(ht_AVGARCH_Gauss);
Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_AVGARCH_Gauss = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_AVGARCH_Gauss;
        Output{j}.ParamsGARCH = parameters_AVGARCH_Gauss(1:end-GARCHOutput{j}.arlag-const);
        Output{j}.ParamsAR = parameters_AVGARCH_Gauss(size(Output{j}.ParamsGARCH,1)+1:size(parameters_AVGARCH_Gauss,1));
        Output{j}.LLF=LLF_AVGARCH_Gauss;
        Output{j}.stderrors=stderrors_AVGARCH_Gauss;
        Output{j}.robustSE=robustSE_AVGARCH_Gauss;
        Output{j}.ht=ht_AVGARCH_Gauss;
        Output{j}.Scores=scores_AVGARCH_Gauss;
        Output{j}.Innovations=resid_AVGARCH_Gauss;
        Output{j}.GARCH = 'AVGARCH';
        Output{j}.U = U_AVGARCH_Gauss;
        Output{j}.Tstat = Tstatistic_AVGARCH_Gauss;
        Output{j}.garchtype = 3;
        Output{j}.leverage = 0;
        Output{j}.errortype = 1;
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.likelihoods = likelihood_AVGARCH_Gauss;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GAUSS';
        Output{j}.Innovations = resid_AVGARCH_Gauss;
        Output{j}.BIC = BIC_AVGARCH_Gauss;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'NGARCH')==1 && strcmp(GARCHOutput{j}.dist,'GAUSS') == 1
        [parameters_NGARCH_Gauss, LLF_NGARCH_Gauss, stderrors_NGARCH_Gauss, robustSE_NGARCH_Gauss, ht_NGARCH_Gauss, scores_NGARCH_Gauss, resid_NGARCH_Gauss, likelihood_NGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NGARCH','NORMAL',GARCHOutput{j}.arlag,const);
        BIC_NGARCH_Gauss = 2*LLF_NGARCH_Gauss+log(t)*size(parameters_NGARCH_Gauss,1);
        Tstatistic_NGARCH_Gauss=parameters_NGARCH_Gauss./diag(robustSE_NGARCH_Gauss).^0.5;
        U_NGARCH_Gauss=normcdf(stdresid);
        Output{j}.Params=parameters_NGARCH_Gauss;
        Output{j}.ParamsGARCH = parameters_NGARCH_Gauss(1:end-GARCHOutput{j}.arlag-const);
        Output{j}.ParamsAR = parameters_NGARCH_Gauss(size(Output{j}.ParamsGARCH,1)+1:size(parameters_NGARCH_Gauss,1));
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_NGARCH_Gauss;
        Output{j}.stderrors=stderrors_NGARCH_Gauss;
        Output{j}.robustSE=robustSE_NGARCH_Gauss;
        Output{j}.ht=ht_NGARCH_Gauss;
        Output{j}.Scores=scores_NGARCH_Gauss;
        Output{j}.Innovations=resid_NGARCH_Gauss;
        Output{j}.GARCH = 'NGARCH';
        Output{j}.U = U_NGARCH_Gauss;
        Output{j}.Tstat = Tstatistic_NGARCH_Gauss;
        Output{j}.garchtype = 4;
        Output{j}.leverage = 1;
        Output{j}.errortype = 1;
        GARCHOutput{j}.arlag
        Output{j}.likelihoods = likelihood_NGARCH_Gauss;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GAUSS';
        Output{j}.Innovations = resid_NGARCH_Gauss;
        Output{j}.BIC = BIC_NGARCH_Gauss;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'NAGARCH')==1 && strcmp(GARCHOutput{j}.dist,'GAUSS') == 1
        [parameters_NAGARCH_Gauss, LLF_NAGARCH_Gauss, stderrors_NAGARCH_Gauss, robustSE_NAGARCH_Gauss, ht_NAGARCH_Gauss, scores_NAGARCH_Gauss, resid_NAGARCH_Gauss, likelihood_NAGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NAGARCH','NORMAL',GARCHOutput{j}.arlag,const);
        BIC_NAGARCH_Gauss = 2*LLF_NAGARCH_Gauss+log(t)*size(parameters_NAGARCH_Gauss,1);
        Tstatistic_NAGARCH_Gauss=parameters_NAGARCH_Gauss./diag(robustSE_NAGARCH_Gauss).^0.5;
        stdresid = resid_NAGARCH_Gauss./sqrt(ht_NAGARCH_Gauss);
        U_NAGARCH_Gauss=normcdf(stdresid);
        Output{j}.Params=parameters_NAGARCH_Gauss;
        Output{j}.ParamsGARCH = parameters_NAGARCH_Gauss(1:end-GARCHOutput{j}.arlag-const);
        Output{j}.ParamsAR = parameters_NAGARCH_Gauss(size(Output{j}.ParamsGARCH,1)+1:size(parameters_NAGARCH_Gauss,1));
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_NAGARCH_Gauss;
        Output{j}.stderrors=stderrors_NAGARCH_Gauss;
        Output{j}.robustSE=robustSE_NAGARCH_Gauss;
        Output{j}.ht=ht_NAGARCH_Gauss;
        Output{j}.Scores=scores_NAGARCH_Gauss;
        Output{j}.Innovations=resid_NAGARCH_Gauss;
        Output{j}.GARCH = 'NAGARCH';
        Output{j}.U = U_NAGARCH_Gauss;
        Output{j}.Tstat = Tstatistic_NAGARCH_Gauss;
        Output{j}.garchtype = 5;
        Output{j}.leverage = 1;
        Output{j}.errortype = 1;
        GARCHOutput{j}.arlag
        Output{j}.likelihoods = likelihood_NAGARCH_Gauss;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GAUSS';
        Output{j}.Innovations = resid_NAGARCH_Gauss;
        Output{j}.BIC = BIC_NAGARCH_Gauss;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'APGARCH')==1 && strcmp(GARCHOutput{j}.dist,'GAUSS') == 1
        [parameters_APGARCH_Gauss, LLF_APGARCH_Gauss, stderrors_APGARCH_Gauss, robustSE_APGARCH_Gauss, ht_APGARCH_Gauss, scores_APGARCH_Gauss, resid_APGARCH_Gauss, likelihood_APGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'APGARCH','NORMAL',GARCHOutput{j}.arlag,const);
        BIC_APGARCH_Gauss = 2*LLF_APGARCH_Gauss+log(t)*size(parameters_APGARCH_Gauss,1);
        Tstatistic_APGARCH_Gauss=parameters_APGARCH_Gauss./diag(robustSE_APGARCH_Gauss).^0.5;
        stdresid = resid_APGARCH_Gauss./sqrt(ht_APGARCH_Gauss);
        U_APGARCH_Gauss=normcdf(stdresid);
        Output{j}.Params=parameters_APGARCH_Gauss;
        Output{j}.ParamsGARCH = parameters_APGARCH_Gauss(1:end-GARCHOutput{j}.arlag-const);
        Output{j}.ParamsAR = parameters_APGARCH_Gauss(size(Output{j}.ParamsGARCH,1)+1:size(parameters_APGARCH_Gauss,1));
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_APGARCH_Gauss;
        Output{j}.stderrors=stderrors_APGARCH_Gauss;
        Output{j}.robustSE=robustSE_APGARCH_Gauss;
        Output{j}.ht=ht_APGARCH_Gauss;
        Output{j}.Scores=scores_APGARCH_Gauss;
        Output{j}.Innovations=resid_APGARCH_Gauss;
        Output{j}.GARCH = 'APGARCH';
        Output{j}.U = U_APGARCH_Gauss;
        Output{j}.Tstat = Tstatistic_APGARCH_Gauss;
        Output{j}.garchtype = 6;
        Output{j}.leverage = 2;
        Output{j}.errortype = 1;
        Output{j}.likelihoods = likelihood_APGARCH_Gauss;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GAUSS';
        Output{j}.Innovations = resid_APGARCH_Gauss;
        Output{j}.BIC = BIC_APGARCH_Gauss;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'GARCH')==1 && strcmp(GARCHOutput{j}.dist,'STUDENTST') == 1
        [parameters_GARCH_STUDENTST, LLF_GARCH_STUDENTST, stderrors_GARCH_STUDENTST, robustSE_GARCH_STUDENTST, ht_GARCH_STUDENTST, scores_GARCH_STUDENTST, resid_GARCH_STUDENTST likelihood_GARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'GARCH','STUDENTST',GARCHOutput{j}.arlag,const);
        BIC_GARCH_STUDENTST = 2*LLF_GARCH_STUDENTST+log(t)*size(parameters_GARCH_STUDENTST,1);
        Tstatistic_GARCH_STUDENTST=parameters_GARCH_STUDENTST./diag(robustSE_GARCH_STUDENTST).^0.5;
        stdresid = resid_GARCH_STUDENTST./sqrt(ht_GARCH_STUDENTST);
        Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_GARCH_STUDENTST = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_GARCH_STUDENTST;
        Output{j}.ParamsGARCH = [parameters_GARCH_STUDENTST(1:end-GARCHOutput{j}.arlag-const-1); parameters_GARCH_STUDENTST(end)];
        Output{j}.ParamsAR = parameters_GARCH_STUDENTST(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_GARCH_STUDENTST,1)-1);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_GARCH_STUDENTST;
        Output{j}.stderrors=stderrors_GARCH_STUDENTST;
        Output{j}.robustSE=robustSE_GARCH_STUDENTST;
        Output{j}.ht=ht_GARCH_STUDENTST;
        Output{j}.Scores=scores_GARCH_STUDENTST;
        Output{j}.Innovations=resid_GARCH_STUDENTST;
        Output{j}.GARCH = 'GARCH';
        Output{j}.U = U_GARCH_STUDENTST;
        Output{j}.Tstat = Tstatistic_GARCH_STUDENTST;
        Output{j}.garchtype = 1;
        Output{j}.leverage = 0;
        Output{j}.errortype = 2;
        Output{j}.likelihoods = likelihood_GARCH_STUDENTST;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'STUDENTST';
        Output{j}.Innovations = resid_GARCH_STUDENTST;
        Output{j}.BIC = BIC_GARCH_STUDENTST;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'EGARCH')==1 && strcmp(GARCHOutput{j}.dist,'STUDENTST') == 1
        [parameters_EGARCH_STUDENTST, LLF_EGARCH_STUDENTST, stderrors_EGARCH_STUDENTST, robustSE_EGARCH_STUDENTST, ht_EGARCH_STUDENTST, scores_EGARCH_STUDENTST, resid_EGARCH_STUDENTST, likelihood_EGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'EGARCH','STUDENTST',GARCHOutput{j}.arlag,const);
        BIC_EGARCH_STUDENTST = 2*LLF_EGARCH_STUDENTST+log(t)*size(parameters_EGARCH_STUDENTST,1);
        Tstatistic_EGARCH_STUDENTST=parameters_EGARCH_STUDENTST./diag(robustSE_EGARCH_STUDENTST).^0.5;
        stdresid = resid_EGARCH_STUDENTST./sqrt(ht_EGARCH_STUDENTST);
  Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_EGARCH_STUDENTST = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_EGARCH_STUDENTST;
        Output{j}.ParamsGARCH = [parameters_EGARCH_STUDENTST(1:end-GARCHOutput{j}.arlag-const-1); parameters_EGARCH_STUDENTST(end)];
        Output{j}.ParamsAR = parameters_EGARCH_STUDENTST(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_EGARCH_STUDENTST,1)-1);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_EGARCH_STUDENTST;
        Output{j}.stderrors=stderrors_EGARCH_STUDENTST;
        Output{j}.robustSE=robustSE_EGARCH_STUDENTST;
        Output{j}.ht=ht_EGARCH_STUDENTST;
        Output{j}.Scores=scores_EGARCH_STUDENTST;
        Output{j}.Innovations=resid_EGARCH_STUDENTST;
        Output{j}.GARCH = 'EGARCH';
        Output{j}.U = U_EGARCH_STUDENTST;
        Output{j}.Tstat = Tstatistic_EGARCH_STUDENTST;
        Output{j}.garchtype = 0;
        Output{j}.leverage = 1;
        Output{j}.errortype = 2;
        Output{j}.likelihoods = likelihood_EGARCH_STUDENTST;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'STUDENTST';
        Output{j}.Innovations = resid_EGARCH_STUDENTST;
        Output{j}.BIC = BIC_EGARCH_STUDENTST;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'TGARCH')==1 && strcmp(GARCHOutput{j}.dist,'STUDENTST') == 1
        [parameters_TGARCH_STUDENTST, LLF_TGARCH_STUDENTST, stderrors_TGARCH_STUDENTST, robustSE_TGARCH_STUDENTST, ht_TGARCH_STUDENTST, scores_TGARCH_STUDENTST, resid_TGARCH_STUDENTST, likelihood_TGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'TGARCH','STUDENTST',GARCHOutput{j}.arlag,const);
        BIC_TGARCH_STUDENTST = 2*LLF_TGARCH_STUDENTST+log(t)*size(parameters_TGARCH_STUDENTST,1);
        Tstatistic_TGARCH_STUDENTST=parameters_TGARCH_STUDENTST./diag(robustSE_TGARCH_STUDENTST).^0.5;
        stdresid = resid_TGARCH_STUDENTST./sqrt(ht_TGARCH_STUDENTST);
  Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_TGARCH_STUDENTST = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_TGARCH_STUDENTST;
        Output{j}.ParamsGARCH = [parameters_TGARCH_STUDENTST(1:end-GARCHOutput{j}.arlag-const-1); parameters_TGARCH_STUDENTST(end)];
        Output{j}.ParamsAR = parameters_TGARCH_STUDENTST(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_TGARCH_STUDENTST,1)-1);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_TGARCH_STUDENTST;
        Output{j}.stderrors=stderrors_TGARCH_STUDENTST;
        Output{j}.robustSE=robustSE_TGARCH_STUDENTST;
        Output{j}.ht=ht_TGARCH_STUDENTST;
        Output{j}.Scores=scores_TGARCH_STUDENTST;
        Output{j}.Innovations=resid_TGARCH_STUDENTST;
        Output{j}.GARCH = 'TGARCH';
        Output{j}.U = U_TGARCH_STUDENTST;
        Output{j}.Tstat = Tstatistic_TGARCH_STUDENTST;
        Output{j}.garchtype = 2;
        Output{j}.leverage = 1;
        Output{j}.errortype = 2;
        Output{j}.likelihoods = likelihood_TGARCH_STUDENTST;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'STUDENTST';
        Output{j}.Innovations = resid_TGARCH_STUDENTST;
        Output{j}.BIC = BIC_TGARCH_STUDENTST;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'GJRGARCH')==1 && strcmp(GARCHOutput{j}.dist,'STUDENTST') == 1
        [parameters_GJRGARCH_STUDENTST, LLF_GJRGARCH_STUDENTST, stderrors_GJRGARCH_STUDENTST, robustSE_GJRGARCH_STUDENTST, ht_GJRGARCH_STUDENTST, scores_GJRGARCH_STUDENTST, resid_GJRGARCH_STUDENTST,likelihood_GJRGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'GJRGARCH','STUDENTST',GARCHOutput{j}.arlag,const);
        BIC_GJRGARCH_STUDENTST = 2*LLF_GJRGARCH_STUDENTST+log(t)*size(parameters_GJRGARCH_STUDENTST,1);
        Tstatistic_GJRGARCH_STUDENTST=parameters_GJRGARCH_STUDENTST./diag(robustSE_GJRGARCH_STUDENTST).^0.5;
        stdresid = resid_GJRGARCH_STUDENTST./sqrt(ht_GJRGARCH_STUDENTST);
  Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_GJRGARCH_STUDENTST = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_GJRGARCH_STUDENTST;
        Output{j}.ParamsGARCH = [parameters_GJRGARCH_STUDENTST(1:end-GARCHOutput{j}.arlag-const-1); parameters_GJRGARCH_STUDENTST(end)];
        Output{j}.ParamsAR = parameters_GJRGARCH_STUDENTST(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_GJRGARCH_STUDENTST,1)-1);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_GJRGARCH_STUDENTST;
        Output{j}.stderrors=stderrors_GJRGARCH_STUDENTST;
        Output{j}.robustSE=robustSE_GJRGARCH_STUDENTST;
        Output{j}.ht=ht_GJRGARCH_STUDENTST;
        Output{j}.Scores=scores_GJRGARCH_STUDENTST;
        Output{j}.Innovations=resid_GJRGARCH_STUDENTST;
        Output{j}.GARCH = 'GJRGARCH';
        Output{j}.U = U_GJRGARCH_STUDENTST;
        Output{j}.Tstat = Tstatistic_GJRGARCH_STUDENTST;
        Output{j}.garchtype = 8;
        Output{j}.leverage = 1;
        Output{j}.errortype = 2;
        Output{j}.likelihoods = likelihood_GJRGARCH_STUDENTST;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.datat = data(:,j);
        Output{j}.BIC =  2*LLF_GJRGARCH_STUDENTST+ log(t)*size(parameters_GJRGARCH_STUDENTST,1);
        Output{j}.Innnovations = resid_GJRGARCH_STUDENTST;
        Output{j}.BIC = BIC_GJRGARCH_STUDENTST;
        Output{j}.dist = 'STUDENTST';
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'AVGARCH')==1 && strcmp(GARCHOutput{j}.dist,'STUDENTST') == 1
        [parameters_AVGARCH_STUDENTST, LLF_AVGARCH_STUDENTST, stderrors_AVGARCH_STUDENTST, robustSE_AVGARCH_STUDENTST, ht_AVGARCH_STUDENTST, scores_AVGARCH_STUDENTST, resid_AVGARCH_STUDENTST, likelihood_AVGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'AVGARCH','STUDENTST',GARCHOutput{j}.arlag,const);
        BIC_AVGARCH_STUDENTST = 2*LLF_AVGARCH_STUDENTST+log(t)*size(parameters_AVGARCH_STUDENTST,1);
        Tstatistic_AVGARCH_STUDENTST=parameters_AVGARCH_STUDENTST./diag(robustSE_AVGARCH_STUDENTST).^0.5;
        stdresid = resid_AVGARCH_STUDENTST./sqrt(ht_AVGARCH_STUDENTST);
  Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_AVGARCH_STUDENTST = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_AVGARCH_STUDENTST;
        Output{j}.ParamsGARCH = [parameters_AVGARCH_STUDENTST(1:end-GARCHOutput{j}.arlag-const-1); parameters_AVGARCH_STUDENTST(end)];
        Output{j}.ParamsAR = parameters_AVGARCH_STUDENTST(size(Output{j}.ParamsGARCH,1):size(parameters_AVGARCH_STUDENTST,1)-1);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_AVGARCH_STUDENTST;
        Output{j}.stderrors=stderrors_AVGARCH_STUDENTST;
        Output{j}.robustSE=robustSE_AVGARCH_STUDENTST;
        Output{j}.ht=ht_AVGARCH_STUDENTST;
        Output{j}.Scores=scores_AVGARCH_STUDENTST;
        Output{j}.Innovations=resid_AVGARCH_STUDENTST;
        Output{j}.GARCH = 'AVGARCH';
        Output{j}.U = U_AVGARCH_STUDENTST;
        Output{j}.Tstat = Tstatistic_AVGARCH_STUDENTST;
        Output{j}.garchtype = 3;
        Output{j}.leverage = 0;
        Output{j}.errortype = 2;
        Output{j}.likelihoods = likelihood_AVGARCH_STUDENTST;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'STUDENTST';
        Output{j}.Innovations = resid_AVGARCH_STUDENTST;
        Output{j}.BIC = BIC_AVGARCH_STUDENTST;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'NGARCH')==1 && strcmp(GARCHOutput{j}.dist,'STUDENTST') == 1
        [parameters_NGARCH_STUDENTST, LLF_NGARCH_STUDENTST, stderrors_NGARCH_STUDENTST, robustSE_NGARCH_STUDENTST, ht_NGARCH_STUDENTST, scores_NGARCH_STUDENTST, resid_NGARCH_STUDENTST, likelihood_NGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NGARCH','STUDENST',GARCHOutput{j}.arlag,const);
        BIC_NGARCH_STUDENTST = 2*LLF_NGARCH_STUDENTST+log(t)*size(parameters_NGARCH_STUDENTST,1);
        Tstatistic_NGARCH_STUDENTST=parameters_NGARCH_STUDENTST./diag(robustSE_NGARCH_STUDENTST).^0.5;
        stdresid = resid_NGARCH_STUDENTST./sqrt(ht_NGARCH_STUDENTST);
        U_NGARCH_STUDENTST=tcdf(stdresid,parameters_NGARCH_STUDENTST(end));
        Output{j}.Params=parameters_NGARCH_STUDENTST;
        Output{j}.ParamsGARCH = [parameters_NGARCH_STUDENTST(1:end-GARCHOutput{j}.arlag-const-1); parameters_NGARCH_STUDENTST(end)];
        Output{j}.ParamsAR = parameters_NGARCH_STUDENTST(size(Output{j}.ParamsGARCH,1):size(parameters_NGARCH_STUDENTST,1)-1);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_NGARCH_STUDENTST;
        Output{j}.stderrors=stderrors_NGARCH_STUDENTST;
        Output{j}.robustSE=robustSE_NGARCH_STUDENTST;
        Output{j}.ht=ht_NGARCH_STUDENTST;
        Output{j}.Scores=scores_NGARCH_STUDENTST;
        Output{j}.Innovations=resid_NGARCH_STUDENTST;
        Output{j}.GARCH = 'NGARCH';
        Output{j}.U = U_NGARCH_STUDENTST;
        Output{j}.Tstat = Tstatistic_NGARCH_STUDENTST;
        Output{j}.garchtype = 4;
        Output{j}.leverage = 1;
        Output{j}.errortype = 2;
        Output{j}.likelihoods = likelihood_NGARCH_STUDENTST;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'STUDENTST';
        Output{j}.Innovations = resid_NGARCH_STUDENTST;
        Output{j}.BIC = BIC_NGARCH_STUDENTST;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'NAGARCH')==1 && strcmp(GARCHOutput{j}.dist,'STUDENTST') == 1
        [parameters_NAGARCH_STUDENTST, LLF_NAGARCH_STUDENTST, stderrors_NAGARCH_STUDENTST, robustSE_NAGARCH_STUDENTST, ht_NAGARCH_STUDENTST, scores_NAGARCH_STUDENTST, resid_NAGARCH_STUDENTST, likelihood_NAGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NAGARCH','STUDENTST',GARCHOutput{j}.arlag,const);
        BIC_NAGARCH_STUDENTST = 2*LLF_NAGARCH_STUDENTST+log(t)*size(parameters_NAGARCH_STUDENTST,1);
        Tstatistic_NAGARCH_STUDENTST=parameters_NAGARCH_STUDENTST./diag(robustSE_NAGARCH_STUDENTST).^0.5;
        stdresid = resid_NAGARCH_STUDENTST./sqrt(ht_NAGARCH_STUDENTST);
        U_NAGARCH_STUDENTST=tcdf(stdresid,parameters_NAGARCH_STUDENTST(end));
        Output{j}.Params=parameters_NAGARCH_STUDENTST;
        Output{j}.ParamsGARCH = [parameters_NAGARCH_STUDENTST(1:end-GARCHOutput{j}.arlag-const-1); parameters_NAGARCH_STUDENTST(end)];
        Output{j}.ParamsAR = parameters_NAGARCH_STUDENTST(size(Output{j}.ParamsGARCH,1):size(parameters_NAGARCH_STUDENTST,1)-1);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_NAGARCH_STUDENTST;
        Output{j}.stderrors=stderrors_NAGARCH_STUDENTST;
        Output{j}.robustSE=robustSE_NAGARCH_STUDENTST;
        Output{j}.ht=ht_NAGARCH_STUDENTST;
        Output{j}.Scores=scores_NAGARCH_STUDENTST;
        Output{j}.Innovations=resid_NAGARCH_STUDENTST;
        Output{j}.GARCH = 'NAGARCH';
        Output{j}.U = U_NAGARCH_STUDENTST;
        Output{j}.Tstat = Tstatistic_NAGARCH_STUDENTST;
        Output{j}.garchtype = 5;
        Output{j}.leverage = 1;
        Output{j}.errortype = 2;
        Output{j}.likelihoods = likelihood_NAGARCH_STUDENTST;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'STUDENTST';
        Output{j}.Innovations = resid_NAGARCH_STUDENTST;
        Output{j}.BIC = BIC_NAGARCH_STUDENTST;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'APGARCH')==1 && strcmp(GARCHOutput{j}.dist,'STUDENTST') == 1
        [parameters_APGARCH_STUDENTST, LLF_APGARCH_STUDENTST, stderrors_APGARCH_STUDENTST, robustSE_APGARCH_STUDENTST, ht_APGARCH_STUDENTST, scores_APGARCH_STUDENTST, resid_APGARCH_STUDENTST, likelihood_APGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'APGARCH','STUDENTST',GARCHOutput{j}.arlag,const);
        BIC_APGARCH_STUDENTST = 2*LLF_APGARCH_STUDENTST+log(t)*size(parameters_APGARCH_STUDENTST,1);
        Tstatistic_APGARCH_STUDENTST=parameters_APGARCH_STUDENTST./diag(robustSE_APGARCH_STUDENTST).^0.5;
        stdresid = resid_APGARCH_STUDENTST./sqrt(ht_APGARCH_STUDENTST);
        U_APGARCH_STUDENTST=tcdf(stdresid,parameters_APGARCH_STUDENTST(end));
        Output{j}.Params=parameters_APGARCH_STUDENTST;
        Output{j}.ParamsGARCH = [parameters_APGARCH_STUDENTST(1:end-GARCHOutput{j}.arlag-const-1); parameters_APGARCH_STUDENTST(end)];
        Output{j}.ParamsAR = parameters_APGARCH_STUDENTST(size(Output{j}.ParamsGARCH,1):size(parameters_APGARCH_STUDENTST,1)-1);        Output{j}.LLF=LLF_APGARCH_STUDENTST;
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.stderrors=stderrors_APGARCH_STUDENTST;
        Output{j}.robustSE=robustSE_APGARCH_STUDENTST;
        Output{j}.ht=ht_APGARCH_STUDENTST;
        Output{j}.Scores=scores_APGARCH_STUDENTST;
        Output{j}.Innovations=resid_APGARCH_STUDENTST;
        Output{j}.GARCH = 'APGARCH';
        Output{j}.U = U_APGARCH_STUDENTST;
        Output{j}.Tstat = Tstatistic_APGARCH_STUDENTST;
        Output{j}.garchtype = 6;
        Output{j}.leverage = 2;
        Output{j}.errortype = 2;
        Output{j}.likelihoods = likelihood_APGARCH_STUDENTST;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'STUDENTST';
        Output{j}.Innovations = resid_APGARCH_STUDENTST;
        Output{j}.BIC = BIC_APGARCH_STUDENTST;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'GARCH')==1 && strcmp(GARCHOutput{j}.dist,'GED') == 1
        % 1.GARCH-GED
        [parameters_GARCH_GED, LLF_GARCH_GED, stderrors_GARCH_GED, robustSE_GARCH_GED, ht_GARCH_GED, scores_GARCH_GED, resid_GARCH_GED, likelihood_GARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'GARCH','GED',GARCHOutput{j}.arlag,const);
        BIC_GARCH_GED = 2*LLF_GARCH_GED+log(t)*size(parameters_GARCH_GED,1);
        Tstatistic_GARCH_GED=parameters_GARCH_GED./diag(robustSE_GARCH_GED).^0.5;
        stdresid = resid_GARCH_GED./sqrt(ht_GARCH_GED);
  Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_GARCH_GED = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_GARCH_GED;
        Output{j}.ParamsGARCH = [parameters_GARCH_GED(1:end-GARCHOutput{j}.arlag-const-1); parameters_GARCH_GED(end)];
        Output{j}.ParamsAR = parameters_GARCH_GED(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_GARCH_GED,1)-1);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_GARCH_GED;
        Output{j}.stderrors=stderrors_GARCH_GED;
        Output{j}.robustSE=robustSE_GARCH_GED;
        Output{j}.ht=ht_GARCH_GED;
        Output{j}.Scores=scores_GARCH_GED;
        Output{j}.Innovations=resid_GARCH_GED;
        Output{j}.GARCH = 'GARCH';
        Output{j}.U = U_GARCH_GED;
        Output{j}.Tstat = Tstatistic_GARCH_GED;
        Output{j}.garchtype = 1;
        Output{j}.leverage = 0;
        Output{j}.errortype = 3;
        Output{j}.likelihoods = likelihood_GARCH_GED;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GED';
        Output{j}.Innovations = resid_GARCH_GED;
        Output{j}.BIC = BIC_GARCH_GED;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'EGARCH')==1 && strcmp(GARCHOutput{j}.dist,'GED') == 1
        [parameters_EGARCH_GED, LLF_EGARCH_GED, stderrors_EGARCH_GED, robustSE_EGARCH_GED, ht_EGARCH_GED, scores_EGARCH_GED, resid_EGARCH_GED, likelihood_EGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'EGARCH','GED',GARCHOutput{j}.arlag,const);
        BIC_EGARCH_GED = 2*LLF_EGARCH_GED+log(t)*size(parameters_EGARCH_GED,1);
        Tstatistic_EGARCH_GED=parameters_EGARCH_GED./diag(robustSE_EGARCH_GED).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_EGARCH_GED./sqrt(ht_EGARCH_GED);
  Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_EGARCH_GED = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_EGARCH_GED;
        Output{j}.ParamsGARCH = [parameters_EGARCH_GED(1:end-GARCHOutput{j}.arlag-const-1); parameters_EGARCH_GED(end)];
        Output{j}.ParamsAR = parameters_EGARCH_GED(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_EGARCH_GED,1)-1);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_EGARCH_GED;
        Output{j}.stderrors=stderrors_EGARCH_GED;
        Output{j}.robustSE=robustSE_EGARCH_GED;
        Output{j}.ht=ht_EGARCH_GED;
        Output{j}.Scores=scores_EGARCH_GED;
        Output{j}.Innovations=resid_EGARCH_GED;
        Output{j}.GARCH = 'EGARCH';
        Output{j}.U = U_EGARCH_GED;
        Output{j}.Tstat = Tstatistic_EGARCH_GED;
        Output{j}.garchtype = 0;
        Output{j}.leverage = 1;
        Output{j}.errortype = 3;
        Output{j}.likelihoods = likelihood_EGARCH_GED;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GED';
        Output{j}.Innovations = resid_EGARCH_GED;
        Output{j}.BIC = BIC_EGARCH_GED;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'TGARCH')==1 && strcmp(GARCHOutput{j}.dist,'GED') == 1
        [parameters_TGARCH_GED, LLF_TGARCH_GED, stderrors_TGARCH_GED, robustSE_TGARCH_GED, ht_TGARCH_GED, scores_TGARCH_GED, resid_TGARCH_GED, likelihood_TGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'TGARCH','GED',GARCHOutput{j}.arlag,const);
        BIC_TGARCH_GED = 2*LLF_TGARCH_GED+log(t)*size(parameters_TGARCH_GED,1);
        Tstatistic_TGARCH_GED=parameters_TGARCH_GED./diag(robustSE_TGARCH_GED).^0.5;
        stdresid = resid_TGARCH_GED./sqrt(ht_TGARCH_GED);
  Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_TGARCH_GED = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_TGARCH_GED;
        Output{j}.ParamsGARCH = [parameters_TGARCH_GED(1:end-GARCHOutput{j}.arlag-const-1); parameters_TGARCH_GED(end)];
        Output{j}.ParamsAR = parameters_TGARCH_GED(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_TGARCH_GED,1)-1);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_TGARCH_GED;
        Output{j}.stderrors=stderrors_TGARCH_GED;
        Output{j}.robustSE=robustSE_TGARCH_GED;
        Output{j}.ht=ht_TGARCH_GED;
        Output{j}.Scores=scores_TGARCH_GED;
        Output{j}.Innovations=resid_TGARCH_GED;
        Output{j}.GARCH = 'TGARCH';
        Output{j}.U = U_TGARCH_GED;
        Output{j}.Tstat = Tstatistic_TGARCH_GED;
        Output{j}.garchtype = 2;
        Output{j}.leverage = 1;
        Output{j}.errortype = 3;
        Output{j}.likelihoods = likelihood_TGARCH_GED;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GED';
        Output{j}.Innovations = resid_TGARCH_GED;
        Output{j}.BIC = BIC_TGARCH_GED;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'GJRGARCH')==1 && strcmp(GARCHOutput{j}.dist,'GED') == 1
        [parameters_GJRGARCH_GED, LLF_GJRGARCH_GED, stderrors_GJRGARCH_GED, robustSE_GJRGARCH_GED, ht_GJRGARCH_GED, scores_GJRGARCH_GED, resid_GJRGARCH_GED, likelihood_GJRGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'GJRGARCH','GED',GARCHOutput{j}.arlag,const);
        BIC_GJRGARCH_GED = 2*LLF_GJRGARCH_GED+log(t)*size(parameters_GJRGARCH_GED,1);
        Tstatistic_GJRGARCH_GED=parameters_GJRGARCH_GED./diag(robustSE_GJRGARCH_GED).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_GJRGARCH_GED./sqrt(ht_GJRGARCH_GED);
  Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_GJRGARCH_GED = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_GJRGARCH_GED;
        Output{j}.ParamsGARCH = [parameters_GJRGARCH_GED(1:end-GARCHOutput{j}.arlag-const-1); parameters_GJRGARCH_GED(end)];
        Output{j}.ParamsAR = parameters_GJRGARCH_GED(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_GJRGARCH_GED,1)-1);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_GJRGARCH_GED;
        Output{j}.stderrors=stderrors_GJRGARCH_GED;
        Output{j}.robustSE=robustSE_GJRGARCH_GED;
        Output{j}.ht=ht_GJRGARCH_GED;
        Output{j}.Scores=scores_GJRGARCH_GED;
        Output{j}.Innovations=resid_GJRGARCH_GED;
        Output{j}.GARCH = 'GJRGARCH';
        Output{j}.U = U_GJRGARCH_GED;
        Output{j}.Tstat = Tstatistic_GJRGARCH_GED;
        Output{j}.garchtype = 8;
        Output{j}.leverage = 1;
        Output{j}.errortype = 3;
        Output{j}.likelihoods = likelihood_GJRGARCH_GED;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GED';
        Output{j}.Innovations = resid_GJRGARCH_GED;
        Output{j}.BIC = BIC_GJRGARCH_GED;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'AVGARCH')==1 && strcmp(GARCHOutput{j}.dist,'GED') == 1
        [parameters_AVGARCH_GED, LLF_AVGARCH_GED, stderrors_AVGARCH_GED, robustSE_AVGARCH_GED, ht_AVGARCH_GED, scores_AVGARCH_GED, resid_AVGARCH_GED, likelihood_AVGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'AVGARCH','GED',GARCHOutput{j}.arlag,const);
        BIC_AVGARCH_GED = 2*LLF_AVGARCH_GED+log(t)*size(parameters_AVGARCH_GED,1);
        Tstatistic_AVGARCH_GED=parameters_AVGARCH_GED./diag(robustSE_AVGARCH_GED).^0.5;
        stdresid = resid_AVGARCH_GED./sqrt(ht_AVGARCH_GED);
  Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_AVGARCH_GED = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_AVGARCH_GED;
        Output{j}.ParamsGARCH = [parameters_AVGARCH_GED(1:end-GARCHOutput{j}.arlag-const-1); parameters_AVGARCH_GED(end)];
        Output{j}.ParamsAR = parameters_AVGARCH_GED(size(Output{j}.ParamsGARCH,1):size(parameters_AVGARCH_GED,1)-1);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_AVGARCH_GED;
        Output{j}.stderrors=stderrors_AVGARCH_GED;
        Output{j}.robustSE=robustSE_AVGARCH_GED;
        Output{j}.ht=ht_AVGARCH_GED;
        Output{j}.Scores=scores_AVGARCH_GED;
        Output{j}.Innovations=resid_AVGARCH_GED;
        Output{j}.GARCH = 'AVGARCH';
        Output{j}.U = U_AVGARCH_GED;
        Output{j}.Tstat = Tstatistic_AVGARCH_GED;
        Output{j}.garchtype = 3;
        Output{j}.leverage = 0;
        Output{j}.errortype = 3;
        Output{j}.likelihoods = likelihood_AVGARCH_GED;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GED';
        Output{j}.Innovations = resid_AVGARCH_GED;
        Output{j}.BIC = BIC_AVGARCH_GED;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'NGARCH')==1 && strcmp(GARCHOutput{j}.dist,'GED') == 1
        [parameters_NGARCH_GED, LLF_NGARCH_GED, stderrors_NGARCH_GED, robustSE_NGARCH_GED, ht_NGARCH_GED, scores_NGARCH_GED, resid_NGARCH_GED, likelihood_NGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NGARCH','GED',GARCHOutput{j}.arlag,const);
        BIC_NGARCH_GED = 2*LLF_NGARCH_GED+log(t)*size(parameters_NGARCH_GED,1);
        Tstatistic_NGARCH_GED=parameters_NGARCH_GED./diag(robustSE_NGARCH_GED).^0.5;
        stdresid = resid_NGARCH_GED./sqrt(ht_NGARCH_GED);
        U_NGARCH_GED=gedcdf(stdresid,parameters_NGARCH_GED(end));
        Output{j}.Params=parameters_NGARCH_GED;
        Output{j}.ParamsGARCH = [parameters_NGARCH_GED(1:end-GARCHOutput{j}.arlag-const-1); parameters_NGARCH_GED(end)];
        Output{j}.ParamsAR = parameters_NGARCH_GED(size(Output{j}.ParamsGARCH,1):size(parameters_NGARCH_GED,1)-1);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_NGARCH_GED;
        Output{j}.stderrors=stderrors_NGARCH_GED;
        Output{j}.robustSE=robustSE_NGARCH_GED;
        Output{j}.ht=ht_NGARCH_GED;
        Output{j}.Scores=scores_NGARCH_GED;
        Output{j}.Innovations=resid_NGARCH_GED;
        Output{j}.GARCH = 'NGARCH';
        Output{j}.U = U_NGARCH_GED;
        Output{j}.Tstat = Tstatistic_NGARCH_GED;
        Output{j}.garchtype = 4;
        Output{j}.leverage = 1;
        Output{j}.errortype = 3;
        Output{j}.likelihoods = likelihood_NGARCH_GED;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GED';
        Output{j}.Innovations = resid_NGARCH_GED;
        Output{j}.BIC = BIC_NGARCH_GED;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'NAGARCH')==1 && strcmp(GARCHOutput{j}.dist,'GED') == 1
        [parameters_NAGARCH_GED, LLF_NAGARCH_GED, stderrors_NAGARCH_GED, robustSE_NAGARCH_GED, ht_NAGARCH_GED, scores_NAGARCH_GED, resid_NAGARCH_GED, likelihood_NAGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NAGARCH','GED',GARCHOutput{j}.arlag,const);
        BIC_NAGARCH_GED = 2*LLF_NAGARCH_GED+log(t)*size(parameters_NAGARCH_GED,1);
        Tstatistic_NAGARCH_GED=parameters_NAGARCH_GED./diag(robustSE_NAGARCH_GED).^0.5;
        stdresid = resid_NAGARCH_GED./sqrt(ht_NAGARCH_GED);
        U_NAGARCH_GED=gedcdf(stdresid,parameters_NAGARCH_GED(end));
        Output{j}.Params=parameters_NAGARCH_GED;
        Output{j}.ParamsGARCH = [parameters_NAGARCH_GED(1:end-GARCHOutput{j}.arlag-const-1); parameters_NAGARCH_GED(end)];
        Output{j}.ParamsAR = parameters_NAGARCH_GED(size(Output{j}.ParamsGARCH,1):size(parameters_NAGARCH_GED,1)-1);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_NAGARCH_GED;
        Output{j}.stderrors=stderrors_NAGARCH_GED;
        Output{j}.robustSE=robustSE_NAGARCH_GED;
        Output{j}.ht=ht_NAGARCH_GED;
        Output{j}.Scores=scores_NAGARCH_GED;
        Output{j}.Innovations=resid_NAGARCH_GED;
        Output{j}.GARCH = 'NAGARCH';
        Output{j}.U = U_NAGARCH_GED;
        Output{j}.Tstat = Tstatistic_NAGARCH_GED;
        Output{j}.garchtype = 5;
        Output{j}.leverage = 1;
        Output{j}.errortype = 3;
        Output{j}.likelihoods = likelihood_NAGARCH_GED;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GED';
        Output{j}.Innovations = resid_NAGARCH_GED;
        Output{j}.BIC = BIC_NAGARCH_GED;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'APGARCH')==1 && strcmp(GARCHOutput{j}.dist,'GED') == 1
        [parameters_APGARCH_GED, LLF_APGARCH_GED, stderrors_APGARCH_GED, robustSE_APGARCH_GED, ht_APGARCH_GED, scores_APGARCH_GED, resid_APGARCH_GED, likelihood_APGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'APGARCH','GED',GARCHOutput{j}.arlag,const);
        BIC_APGARCH_GED = 2*LLF_APGARCH_GED+log(t)*size(parameters_APGARCH_GED,1);
        Tstatistic_APGARCH_GED=parameters_APGARCH_GED./diag(robustSE_APGARCH_GED).^0.5;
        stdresid = resid_APGARCH_GED./sqrt(ht_APGARCH_GED);
        U_APGARCH_GED=gedcdf(stdresid,parameters_APGARCH_GED(end));
        Output{j}.Params=parameters_APGARCH_GED;
        Output{j}.ParamsGARCH = [parameters_APGARCH_GED(1:end-GARCHOutput{j}.arlag-const-1); parameters_APGARCH_GED(end)];
        Output{j}.ParamsAR = parameters_APGARCH_GED(size(Output{j}.ParamsGARCH,1):size(parameters_APGARCH_GED,1)-1);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_APGARCH_GED;
        Output{j}.stderrors=stderrors_APGARCH_GED;
        Output{j}.robustSE=robustSE_APGARCH_GED;
        Output{j}.ht=ht_APGARCH_GED;
        Output{j}.Scores=scores_APGARCH_GED;
        Output{j}.Innovations=resid_APGARCH_GED;
        Output{j}.GARCH = 'APGARCH';
        Output{j}.U = U_APGARCH_GED;
        Output{j}.Tstat = Tstatistic_APGARCH_GED;
        Output{j}.garchtype = 6;
        Output{j}.leverage = 2;
        Output{j}.errortype = 3;
        Output{j}.likelihoods = likelihood_APGARCH_GED;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GED';
        Output{j}.Innovations = resid_APGARCH_GED;
        Output{j}.BIC = BIC_APGARCH_GED;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'GARCH')==1 && strcmp(GARCHOutput{j}.dist,'SKEWT') == 1
        [parameters_GARCH_SKEWT, LLF_GARCH_SKEWT, stderrors_GARCH_SKEWT, robustSE_GARCH_SKEWT, ht_GARCH_SKEWT, scores_GARCH_SKEWT, resid_GARCH_SKEWT, likelihood_GARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'GARCH','SKEWT',GARCHOutput{j}.arlag,const);
        BIC_GARCH_SKEWT = 2*LLF_GARCH_SKEWT+log(t)*size(parameters_GARCH_SKEWT,1);
        Tstatistic_GARCH_SKEWT=parameters_GARCH_SKEWT./diag(robustSE_GARCH_SKEWT).^0.5;
        stdresid = resid_GARCH_SKEWT./sqrt(ht_GARCH_SKEWT);
  Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_GARCH_SKEWT = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_GARCH_SKEWT;
        Output{j}.ParamsGARCH = [parameters_GARCH_SKEWT(1:end-GARCHOutput{j}.arlag-const-2); parameters_GARCH_SKEWT(end-1:end)];
        Output{j}.ParamsAR = parameters_GARCH_SKEWT(size(Output{j}.ParamsGARCH,1)-2+1:size(parameters_GARCH_SKEWT,1)-2);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_GARCH_SKEWT;
        Output{j}.stderrors=stderrors_GARCH_SKEWT;
        Output{j}.robustSE=robustSE_GARCH_SKEWT;
        Output{j}.ht=ht_GARCH_SKEWT;
        Output{j}.Scores=scores_GARCH_SKEWT;
        Output{j}.Innovations=resid_GARCH_SKEWT;
        Output{j}.GARCH = 'GARCH';
        Output{j}.U = U_GARCH_SKEWT;
        Output{j}.Tstat = Tstatistic_GARCH_SKEWT;
        Output{j}.garchtype = 1;
        Output{j}.leverage = 0;
        Output{j}.errortype = 4;
        Output{j}.likelihoods = likelihood_GARCH_SKEWT;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'SKEWT';
        Output{j}.Innovations = resid_GARCH_SKEWT;
        Output{j}.BIC =  BIC_GARCH_SKEWT;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'EGARCH')==1 && strcmp(GARCHOutput{j}.dist,'SKEWT') == 1
        [parameters_EGARCH_SKEWT, LLF_EGARCH_SKEWT, stderrors_EGARCH_SKEWT, robustSE_EGARCH_SKEWT, ht_EGARCH_SKEWT, scores_EGARCH_SKEWT, resid_EGARCH_SKEWT, likelihood_EGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'EGARCH','SKEWT',GARCHOutput{j}.arlag,const);
        BIC_EGARCH_SKEWT = 2*LLF_EGARCH_SKEWT+log(t)*size(parameters_EGARCH_SKEWT,1);
        Tstatistic_EGARCH_SKEWT=parameters_EGARCH_SKEWT./diag(robustSE_EGARCH_SKEWT).^0.5;
        stdresid = resid_EGARCH_SKEWT./sqrt(ht_EGARCH_SKEWT);
 Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_EGARCH_SKEWT = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_EGARCH_SKEWT;
        Output{j}.ParamsGARCH = [parameters_EGARCH_SKEWT(1:end-GARCHOutput{j}.arlag-const-2); parameters_EGARCH_SKEWT(end-1:end)];
        Output{j}.ParamsAR = parameters_EGARCH_SKEWT(size(Output{j}.ParamsGARCH,1)-2+1:size(parameters_EGARCH_SKEWT,1)-2);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_EGARCH_SKEWT;
        Output{j}.stderrors=stderrors_EGARCH_SKEWT;
        Output{j}.robustSE=robustSE_EGARCH_SKEWT;
        Output{j}.ht=ht_EGARCH_SKEWT;
        Output{j}.Scores=scores_EGARCH_SKEWT;
        Output{j}.Innovations=resid_EGARCH_SKEWT;
        Output{j}.GARCH = 'EGARCH';
        Output{j}.U = U_EGARCH_SKEWT;
        Output{j}.Tstat = Tstatistic_EGARCH_SKEWT;
        Output{j}.garchtype = 0;
        Output{j}.leverage = 1;
        Output{j}.errortype = 4;
        Output{j}.likelihoods = likelihood_EGARCH_SKEWT;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'SKEWT';
        Output{j}.Innovations = resid_EGARCH_SKEWT;
        Output{j}.BIC = BIC_EGARCH_SKEWT;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'TGARCH')==1 && strcmp(GARCHOutput{j}.dist,'SKEWT') == 1        
        [parameters_TGARCH_SKEWT, LLF_TGARCH_SKEWT, stderrors_TGARCH_SKEWT, robustSE_TGARCH_SKEWT, ht_TGARCH_SKEWT, scores_TGARCH_SKEWT, resid_TGARCH_SKEWT, likelihood_TGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'TGARCH','SKEWT',GARCHOutput{j}.arlag,const);
        BIC_TGARCH_SKEWT = 2*LLF_TGARCH_SKEWT+log(t)*size(parameters_TGARCH_SKEWT,1);
        Tstatistic_TGARCH_SKEWT=parameters_TGARCH_SKEWT./diag(robustSE_TGARCH_SKEWT).^0.5;
        stdresid = resid_TGARCH_SKEWT./sqrt(ht_TGARCH_SKEWT);
 Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_TGARCH_SKEWT = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_TGARCH_SKEWT;
        Output{j}.ParamsGARCH = [parameters_TGARCH_SKEWT(1:end-GARCHOutput{j}.arlag-const-2); parameters_TGARCH_SKEWT(end-1:end)];
        Output{j}.ParamsAR = parameters_TGARCH_SKEWT(size(Output{j}.ParamsGARCH,1)-2+1:size(parameters_TGARCH_SKEWT,1)-2);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_TGARCH_SKEWT;
        Output{j}.stderrors=stderrors_TGARCH_SKEWT;
        Output{j}.robustSE=robustSE_TGARCH_SKEWT;
        Output{j}.ht=ht_TGARCH_SKEWT;
        Output{j}.Scores=scores_TGARCH_SKEWT;
        Output{j}.Innovations=resid_TGARCH_SKEWT;
        Output{j}.GARCH = 'TGARCH';
        Output{j}.U = U_TGARCH_SKEWT;
        Output{j}.Tstat = Tstatistic_TGARCH_SKEWT;
        Output{j}.garchtype = 2;
        Output{j}.leverage = 1;
        Output{j}.errortype = 4;
        Output{j}.likelihoods = likelihood_TGARCH_SKEWT;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'SKEWT';
        Output{j}.Innovations = resid_TGARCH_SKEWT;
        Output{j}.BIC = BIC_TGARCH_SKEWT;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'GJRGARCH')==1 && strcmp(GARCHOutput{j}.dist,'SKEWT') == 1
        [parameters_GJRGARCH_SKEWT, LLF_GJRGARCH_SKEWT, stderrors_GJRGARCH_SKEWT, robustSE_GJRGARCH_SKEWT, ht_GJRGARCH_SKEWT, scores_GJRGARCH_SKEWT, resid_GJRGARCH_SKEWT,likelihood_GJRGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'GJRGARCH','SKEWT',GARCHOutput{j}.arlag,const);
        BIC_GJRGARCH_SKEWT = 2*LLF_GJRGARCH_SKEWT+log(t)*size(parameters_GJRGARCH_SKEWT,1);
        Tstatistic_GJRGARCH_SKEWT=parameters_GJRGARCH_SKEWT./diag(robustSE_GJRGARCH_SKEWT).^0.5;
        stdresid = resid_GJRGARCH_SKEWT./sqrt(ht_GJRGARCH_SKEWT);
 Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_GJRGARCH_SKEWT = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_GJRGARCH_SKEWT;
        Output{j}.ParamsGARCH = [parameters_GJRGARCH_SKEWT(1:end-GARCHOutput{j}.arlag-const-2); parameters_GJRGARCH_SKEWT(end-1:end)];
        Output{j}.ParamsAR = parameters_GJRGARCH_SKEWT(size(Output{j}.ParamsGARCH,1)-2+1:size(parameters_GJRGARCH_SKEWT,1)-2);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_GJRGARCH_SKEWT;
        Output{j}.stderrors=stderrors_GJRGARCH_SKEWT;
        Output{j}.robustSE=robustSE_GJRGARCH_SKEWT;
        Output{j}.ht=ht_GJRGARCH_SKEWT;
        Output{j}.Scores=scores_GJRGARCH_SKEWT;
        Output{j}.Innovations=resid_GJRGARCH_SKEWT;
        Output{j}.GARCH = 'GJRGARCH';
        Output{j}.U = U_GJRGARCH_SKEWT;
        Output{j}.Tstat = Tstatistic_GJRGARCH_SKEWT;
        Output{j}.garchtype = 8;
        Output{j}.errortype = 4;
        Output{j}.arlag = Output{j}.arlag;
        Output{j}.likelihoods = likelihood_GJRGARCH_SKEWT;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'SKEWT';
        Output{j}.Innovations = resid_GJRGARCH_SKEWT;
        Output{j}.BIC = BIC_GJRGARCH_SKEWT;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'AVGARCH')==1 && strcmp(GARCHOutput{j}.dist,'SKEWT') == 1
        [parameters_AVGARCH_SKEWT, LLF_AVGARCH_SKEWT, stderrors_AVGARCH_SKEWT, robustSE_AVGARCH_SKEWT, ht_AVGARCH_SKEWT, scores_AVGARCH_SKEWT, resid_AVGARCH_SKEWT, likelihood_AVGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'AVGARCH','SKEWT',GARCHOutput{j}.arlag,const);
        BIC_AVGARCH_SKEWT = 2*LLF_AVGARCH_SKEWT+log(t)*size(parameters_AVGARCH_SKEWT,1);
        Tstatistic_AVGARCH_SKEWT=parameters_AVGARCH_SKEWT./diag(robustSE_AVGARCH_SKEWT).^0.5;
        stdresid = resid_AVGARCH_SKEWT./sqrt(ht_AVGARCH_SKEWT);
 Output{j}.u_par = paretotails(stdresid(:,1),.1,.9,'kernel');
                u_par = Output{j}.u_par;
        U_AVGARCH_SKEWT = u_par.cdf(stdresid(:,1));
        Output{j}.Params=parameters_AVGARCH_SKEWT;
        Output{j}.ParamsGARCH = [parameters_AVGARCH_SKEWT(1:end-GARCHOutput{j}.arlag-const-2);parameters_AVGARCH_SKEWT(end-1:end)] ;
        Output{j}.ParamsAR = parameters_AVGARCH_SKEWT(size(Output{j}.ParamsGARCH,1)-2+1:size(parameters_AVGARCH_SKEWT,1)-2);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_AVGARCH_SKEWT;
        Output{j}.stderrors=stderrors_AVGARCH_SKEWT;
        Output{j}.robustSE=robustSE_AVGARCH_SKEWT;
        Output{j}.ht=ht_AVGARCH_SKEWT;
        Output{j}.Scores=scores_AVGARCH_SKEWT;
        Output{j}.Innovations=resid_AVGARCH_SKEWT;
        Output{j}.GARCH = 'AVGARCH';
        Output{j}.U = U_AVGARCH_SKEWT;
        Output{j}.Tstat = Tstatistic_AVGARCH_SKEWT;
        Output{j}.garchtype = 3;
        Output{j}.leverage = 0;
        Output{j}.errortype = 4;
        Output{j}.likelihoods = likelihood_AVGARCH_SKEWT;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'SKEWT';
        Output{j}.Innovations = resid_AVGARCH_SKEWT;
        Output{j}.BIC = BIC_AVGARCH_SKEWT;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'NGARCH')==1 && strcmp(GARCHOutput{j}.dist,'SKEWT') == 1
        [parameters_NGARCH_SKEWT, LLF_NGARCH_SKEWT, stderrors_NGARCH_SKEWT, robustSE_NGARCH_SKEWT, ht_NGARCH_SKEWT, scores_NGARCH_SKEWT, resid_NGARCH_SKEWT, likelihood_NGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NGARCH','SKEWT',GARCHOutput{j}.arlag,const);
        BIC_NGARCH_SKEWT = 2*LLF_NGARCH_SKEWT+log(t)*size(parameters_NGARCH_SKEWT,1);
        Tstatistic_NGARCH_SKEWT=parameters_NGARCH_SKEWT./diag(robustSE_NGARCH_SKEWT).^0.5;
        stdresid = resid_NGARCH_SKEWT./sqrt(ht_NGARCH_SKEWT);
        U_NGARCH_SKEWT = skewtdis_cdf(stdresid,parameters_NGARCH_SKEWT(end-1),parameters_NGARCH_SKEWT(end));
        Output{j}.Params=parameters_NGARCH_SKEWT;
        Output{j}.ParamsGARCH = [parameters_NGARCH_SKEWT(1:end-GARCHOutput{j}.arlag-const-2);parameters_NGARCH_SKEWT(end-1:end)];
        Output{j}.ParamsAR = parameters_NGARCH_SKEWT(size(Output{j}.ParamsGARCH,1)-2+1:size(parameters_NGARCH_SKEWT,1)-2);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_NGARCH_SKEWT;
        Output{j}.stderrors=stderrors_NGARCH_SKEWT;
        Output{j}.robustSE=robustSE_NGARCH_SKEWT;
        Output{j}.ht=ht_NGARCH_SKEWT;
        Output{j}.Scores=scores_NGARCH_SKEWT;
        Output{j}.Innovations=resid_NGARCH_SKEWT;
        Output{j}.GARCH = 'NGARCH';
        Output{j}.U = U_NGARCH_SKEWT;
        Output{j}.Tstat = Tstatistic_NGARCH_SKEWT;
        Output{j}.garchtype = 4;
        Output{j}.leverage = 1;
        Output{j}.errortype = 1;
        Output{j}.likelihoods = likelihood_NGARCH_SKEWT;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'SKEWT';
        Output{j}.Innovations = resid_NGARCH_SKEWT;
        Output{j}.BIC = BIC_NGARCH_SKEWT;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'NAGARCH')==1 && strcmp(GARCHOutput{j}.dist,'SKEWT') == 1
        [parameters_NAGARCH_SKEWT, LLF_NAGARCH_SKEWT, stderrors_NAGARCH_SKEWT, robustSE_NAGARCH_SKEWT, ht_NAGARCH_SKEWT, scores_NAGARCH_SKEWT, resid_NAGARCH_SKEWT, likelihood_NAGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NAGARCH','SKEWT',GARCHOutput{j}.arlag,const);
        BIC_NAGARCH_SKEWT = 2*LLF_NAGARCH_SKEWT+log(t)*size(parameters_NAGARCH_SKEWT,1);
        Tstatistic_NAGARCH_SKEWT=parameters_NAGARCH_SKEWT./diag(robustSE_NAGARCH_SKEWT).^0.5;
        stdresid = resid_NAGARCH_SKEWT./sqrt(ht_NAGARCH_SKEWT);
        U_NAGARCH_SKEWT = skewtdis_cdf(stdresid,parameters_NAGARCH_SKEWT(end-1),parameters_NAGARCH_SKEWT(end));
        Output{j}.Params=parameters_NAGARCH_SKEWT;
        Output{j}.ParamsGARCH = [parameters_NAGARCH_SKEWT(1:end-GARCHOutput{j}.arlag-const-2);parameters_NAGARCH_SKEWT(end-1:end)];
        Output{j}.ParamsAR = parameters_NAGARCH_SKEWT(size(Output{j}.ParamsGARCH,1)-2+1:size(parameters_NAGARCH_SKEWT,1)-2);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_NAGARCH_SKEWT;
        Output{j}.stderrors=stderrors_NAGARCH_SKEWT;
        Output{j}.robustSE=robustSE_NAGARCH_SKEWT;
        Output{j}.ht=ht_NAGARCH_SKEWT;
        Output{j}.Scores=scores_NAGARCH_SKEWT;
        Output{j}.Innovations=resid_NAGARCH_SKEWT;
        Output{j}.GARCH = 'NAGARCH';
        Output{j}.U = U_NAGARCH_SKEWT;
        Output{j}.Tstat = Tstatistic_NAGARCH_SKEWT;
        Output{j}.garchtype = 5;
        Output{j}.leverage = 1;
        Output{j}.errortype = 1;
        Output{j}.likelihoods = likelihood_NAGARCH_SKEWT;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'SKEWT';
        Output{j}.Innovations = resid_NAGARCH_SKEWT;
        Output{j}.BIC = BIC_NAGARCH_SKEWT;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    elseif strcmp(GARCHOutput{j}.GARCH,'APGARCH')==1 && strcmp(GARCHOutput{j}.dist,'SKEWT') == 1
        [parameters_APGARCH_SKEWT, LLF_APGARCH_SKEWT, stderrors_APGARCH_SKEWT, robustSE_APGARCH_SKEWT, ht_APGARCH_SKEWT, scores_APGARCH_SKEWT, resid_APGARCH_SKEWT, likelihood_APGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'APGARCH','SKEWT',GARCHOutput{j}.arlag,const);
        BIC_APGARCH_SKEWT = 2*LLF_APGARCH_SKEWT+log(t)*size(parameters_APGARCH_SKEWT,1);
        Tstatistic_APGARCH_SKEWT=parameters_APGARCH_SKEWT./diag(robustSE_APGARCH_SKEWT).^0.5;n
        stdresid = resid_APGARCH_SKEWT./sqrt(ht_APGARCH_SKEWT);
        U_APGARCH_SKEWT = skewtdis_cdf(stdresid,parameters_APGARCH_SKEWT(end-1),parameters_APGARCH_SKEWT(end));
        Output{j}.Params=parameters_APGARCH_SKEWT;
        Output{j}.ParamsGARCH = [parameters_APGARCH_SKEWT(1:end-GARCHOutput{j}.arlag-const-2);parameters_APGARCH_SKEWT(end-1:end)];
        Output{j}.ParamsAR = parameters_APGARCH_SKEWT(size(Output{j}.ParamsGARCH,1)-2+1:size(parameters_APGARCH_SKEWT,1)-2);
        Output{j}.arlag = GARCHOutput{j}.arlag;
        Output{j}.LLF=LLF_APGARCH_SKEWT;
        Output{j}.stderrors=stderrors_APGARCH_SKEWT;
        Output{j}.robustSE=robustSE_APGARCH_SKEWT;
        Output{j}.ht=ht_APGARCH_SKEWT;
        Output{j}.Scores=scores_APGARCH_SKEWT;
        Output{j}.Innovations=resid_APGARCH_SKEWT;
        Output{j}.GARCH = 'APGARCH';
        Output{j}.U = U_APGARCH_SKEWT;
        Output{j}.Tstat = Tstatistic_APGARCH_SKEWT;
        Output{j}.garchtype = 8;
        Output{j}.leverage = 2;
        Output{j}.errortype = 1;
        Output{j}.likelihoods = likelihood_APGARCH_SKEWT;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'SKEWT';
        Output{j}.Innovations = resid_APGARCH_SKEWT;
        Output{j}.BIC = BIC_APGARCH_SKEWT;
        Output{j}.stdresid = stdresid;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    end
end
