function [Output,Output_Gauss] = AR_MarginalModel(data,const,arlag)

% Estimate marginal distribution for return series with different
% AR-GARCH(1,1)-models
%
% GARCH models: GARCH, EGARCH, TGARCH, GJRGARCH, AVGARCH, NGARCH, NAGARCH,
%               APGARCH (activated are only GARCH, EGARCH, TGARCH, AVGARCH)
%
% Distributions: Gauss, Student T, GED, SKEWT
%
% USAGE: [Output,Output_Gauss] = AR_MarginalModel(data,const,arlag)
%
% INPUTS:
%          data:   t x k array of returns
%          const:  1 if AR is to be estimated with cont, 0 else
%          arlag:  # of lags in AR model
%
% OUTPUTS:
%          Output: Structure all infos about estimated return series
%
% Author: Martin Grziska, based on a code of Kevin Sheppard
%
% Last Modification: 06/09/2010

[t,k]=size(data);
BIC_ALL_opt = cell(k,1);
BIC_opt = cell(k,1);
BIC_gesamt_Gauss = cell(k,1);
BIC_gesamt_STUDENTST = cell(k,1);
BIC_gesamt_GED = cell(k,1);
BIC_gesamt_SKEWT = cell(k,1);
% Fitte die verschiedenen GARCH-Modelle
% 1.GARCH-Gauss
for j=1:k;
    BIC_GARCH_Gauss = zeros(arlag,1);
    LLF_GARCH_Gauss = zeros(arlag,1);
    %     teste Modell mit Gauss-Verteilung und bis zu arlag, arlags
    for i=1:arlag
        [parameters_GARCH_Gauss, LLF_GARCH_Gauss(i), stderrors_GARCH_Gauss, robustSE_GARCH_Gauss, ht_GARCH_Gauss, scores_GARCH_Gauss, resid_GARCH_Gauss, likelihood_GARCH_Gauss, EXITFLAG] = ar_multigarch_grm(data(:,j),1,0,1,'GARCH','NORMAL',i,const,[],[],false);
        %         wenn exitflag <0 war die Optimierung noicht erfolgreich; sete
        %         BIC-Wert auf 1e100. BIC wird nach dem min-Wert ausgesucht und mit
        %         BIC = 1e100 wird das jeweilige Modell auf keinen Fall ausgesucht
        if EXITFLAG <= 0
            BIC_GARCH_Gauss(i) = 1e100;
        else
%             es wird die negative log-likelihood ausgegeben, deshalb darf
%             beim BIC nicjt *(-2) genommen werden, sondern nur *2
            BIC_GARCH_Gauss(i) = 2*LLF_GARCH_Gauss(i) + log(t)*size(parameters_GARCH_Gauss,1);
        end
        %         wenn ein Parameter = 0 ist wird der BIC-Wert ebenfalls 1e100
        %         gesetzt
        if sum(parameters_GARCH_Gauss == 0)>= 1
            BIC_GARCH_Gauss(i) = 1e100;
        end
    end
    %     wähle das Minimum über die verschiedenen parameter (die sich durch
    %     die verschiedenen arlags ergeben) aus; in nGARCH_gauss steht die
    %     optimale AR
    [m,nGARCH_Gauss(j)] = min(BIC_GARCH_Gauss);

    % % 2.EGARCH-Gauss
    BIC_EGARCH_Gauss = zeros(arlag,1);
    LLF_EGARCH_Gauss = zeros(arlag,1);
    for i=1:arlag
        [parameters_EGARCH_Gauss, LLF_EGARCH_Gauss(i), stderrors_EGARCH_Gauss, robustSE_EGARCH_Gauss, ht_EGARCH_Gauss, scores_EGARCH_Gauss, resid_EGARCH_Gauss, likelihood_EGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'EGARCH','NORMAL',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_EGARCH_Gauss(i) = 1e100;
        else
            BIC_EGARCH_Gauss(i) = 2*LLF_EGARCH_Gauss(i)+log(t)*size(parameters_EGARCH_Gauss,1);
        end
    end
    [m,nEGARCH_Gauss(j)] = min(BIC_EGARCH_Gauss);

    % %     3.TGARCH-Gauss
    BIC_TGARCH_Gauss = zeros(arlag,1);
    LLF_TGARCH_Gauss = zeros(arlag,1);
    for i=1:arlag
        [parameters_TGARCH_Gauss, LLF_TGARCH_Gauss(i), stderrors_TGARCH_Gauss, robustSE_TGARCH_Gauss, ht_TGARCH_Gauss, scores_TGARCH_Gauss, resid_TGARCH_Gauss, likelihood_TGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'TGARCH','NORMAL',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_TGARCH_Gauss(i) = 1e100;
        else
            BIC_TGARCH_Gauss(i) = 2*LLF_TGARCH_Gauss(i)+log(t)*size(parameters_TGARCH_Gauss,1);
        end
        if sum(parameters_TGARCH_Gauss == 0)>= 1
            BIC_TGARCH_Gauss(i) = 1e100;
        end
    end
    [m,nTGARCH_Gauss(j)] = min(BIC_TGARCH_Gauss);


    %     % 4.GJRGARCH-Gauss
    BIC_GJRGARCH_Gauss = zeros(arlag,1);
    LLF_GJRGARCH_Gauss = zeros(arlag,1);
    for i=1:arlag
        [parameters_GJRGARCH_Gauss, LLF_GJRGARCH_Gauss(i), stderrors_GJRGARCH_Gauss, robustSE_GJRGARCH_Gauss, ht_GJRGARCH_Gauss, scores_GJRGARCH_Gauss, resid_GJRGARCH_Gauss, likelihood_GJRGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'GJRGARCH','NORMAL',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_GJRGARCH_Gauss(i) = 1e100;
        else
            BIC_GJRGARCH_Gauss(i) = 2*LLF_GJRGARCH_Gauss(i)+log(t)*size(parameters_GJRGARCH_Gauss,1);
        end
    end
    [m,nGJRGARCH_Gauss(j)] = min(BIC_GJRGARCH_Gauss);

    %   5. AVGARCH-Gauss
    BIC_AVGARCH_Gauss = zeros(arlag,1);
    LLF_AVGARCH_Gauss = zeros(arlag,1);
    for i=1:arlag
        [parameters_AVGARCH_Gauss, LLF_AVGARCH_Gauss(i), stderrors_AVGARCH_Gauss, robustSE_AVGARCH_Gauss, ht_AVGARCH_Gauss, scores_AVGARCH_Gauss, resid_AVGARCH_Gauss, likelihood_AVGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'AVGARCH','NORMAL',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_AVGARCH_Gauss(i) = 1e100;
        else
            BIC_AVGARCH_Gauss(i) = 2*LLF_AVGARCH_Gauss(i)+log(t)*size(parameters_AVGARCH_Gauss,1);
        end
    end
    [m,nAVGARCH_Gauss(j)] = min(BIC_AVGARCH_Gauss);

    %   6. NGARCH-Gauss
%     BIC_NGARCH_Gauss = zeros(arlag,1);
%     LLF_NGARCH_Gauss = zeros(arlag,1);
%     for i=1:arlag
%         [parameters_NGARCH_Gauss, LLF_NGARCH_Gauss(i), stderrors_NGARCH_Gauss, robustSE_NGARCH_Gauss, ht_NGARCH_Gauss, scores_NGARCH_Gauss, resid_NGARCH_Gauss, likelihood_NGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NGARCH','NORMAL',i,const,[],[],false);
%         if EXITFLAG <= 0
%             BIC_NGARCH_Gauss(i) = 1e100;
%         else
%             BIC_NGARCH_Gauss(i) = 2*LLF_NGARCH_Gauss(i)+log(t)*size(parameters_NGARCH_Gauss,1);
%         end
%     end
%     [m,nNGARCH_Gauss(j)] = min(BIC_NGARCH_Gauss);

    %   7. NAGARCH-Gauss
%     BIC_NAGARCH_Gauss = zeros(arlag,1);
%     LLF_NAGARCH_Gauss = zeros(arlag,1);
%     for i=1:arlag
%         [parameters_NAGARCH_Gauss, LLF_NAGARCH_Gauss(i), stderrors_NAGARCH_Gauss, robustSE_NAGARCH_Gauss, ht_NAGARCH_Gauss, scores_NAGARCH_Gauss, resid_NAGARCH_Gauss, likelihood_NAGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NAGARCH','NORMAL',i,const,[],[],false);
%         if EXITFLAG <= 0
%             BIC_NAGARCH_Gauss(i) = 1e100;
%         else
%             BIC_NAGARCH_Gauss(i) = 2*LLF_NAGARCH_Gauss(i)+log(t)*size(parameters_NAGARCH_Gauss,1);
%         end
%     end
%     [m,nNAGARCH_Gauss(j)] = min(BIC_NAGARCH_Gauss);


    %   8. APGARCH-Gauss
%     BIC_APGARCH_Gauss = zeros(arlag,1);
%     LLF_APGARCH_Gauss = zeros(arlag,1);
%     for i=1:arlag
%         [parameters_APGARCH_Gauss, LLF_APGARCH_Gauss(i), stderrors_APGARCH_Gauss, robustSE_APGARCH_Gauss, ht_APGARCH_Gauss, scores_APGARCH_Gauss, resid_APGARCH_Gauss, likelihood_APGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'APGARCH','NORMAL',i,const,[],[],false);
%         if EXITFLAG <= 0
%             BIC_APGARCH_Gauss(i) = 1e100;
%         else
%             BIC_APGARCH_Gauss(i) = 2*LLF_APGARCH_Gauss(i)+log(t)*size(parameters_APGARCH_Gauss,1);
%         end
%     end
%     [m,nAPGARCH_Gauss(j)] = min(BIC_APGARCH_Gauss);

    %     wähle das optimale Modell über die verschiedenen Modelle mit
    %     Gauß-Verteilung aus - die optimale arlag-Länge wurde vorher bestimmt
    %     und steht in, z.B. nGARCH_Gauß
%     BIC_gesamt_Gauss{j} = [BIC_GARCH_Gauss(nGARCH_Gauss(j));BIC_EGARCH_Gauss(nEGARCH_Gauss(j));BIC_TGARCH_Gauss(nTGARCH_Gauss(j));BIC_GJRGARCH_Gauss(nGJRGARCH_Gauss(j));BIC_AVGARCH_Gauss(nAVGARCH_Gauss(j));BIC_NGARCH_Gauss(nNGARCH_Gauss(j));BIC_NAGARCH_Gauss(nNAGARCH_Gauss(j));BIC_APGARCH_Gauss(nAPGARCH_Gauss(j))];
        BIC_gesamt_Gauss{j} = [BIC_GARCH_Gauss(nGARCH_Gauss(j));BIC_EGARCH_Gauss(nEGARCH_Gauss(j));BIC_TGARCH_Gauss(nTGARCH_Gauss(j));BIC_GJRGARCH_Gauss(nGJRGARCH_Gauss(j));BIC_AVGARCH_Gauss(nAVGARCH_Gauss(j));1e100;1e100;1e100];
    [Holder, BIC_opt_Gauss{j}] = min(BIC_gesamt_Gauss{j});

    % -------------------------------------------------------------------------
    % -------------------------------------------------------------------------

    % Verteilung STUDENTST
    % 9.GARCH-STUDENTST
    BIC_GARCH_STUDENTST = zeros(arlag,1);
    LLF_GARCH_STUDENTST = zeros(arlag,1);

    for i=1:arlag
        [parameters_GARCH_STUDENTST, LLF_GARCH_STUDENTST(i), stderrors_GARCH_STUDENTST, robustSE_GARCH_STUDENTST, ht_GARCH_STUDENTST, scores_GARCH_STUDENTST, resid_GARCH_STUDENTST, liklihood_GARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'GARCH','STUDENTST',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_GARCH_STUDENTST(i) = 1e100;
        else
            BIC_GARCH_STUDENTST(i) = 2*LLF_GARCH_STUDENTST(i)+log(t)*size(parameters_GARCH_STUDENTST,1);
        end
    end
    [m,nGARCH_STUDENTST(j)] = min(BIC_GARCH_STUDENTST);


    % % % 10.EGARCH-STUDENTST
    BIC_EGARCH_STUDENTST = zeros(arlag,1);
    LLF_EGARCH_STUDENTST = zeros(arlag,1);
    for i=1:arlag
        [parameters_EGARCH_STUDENTST, LLF_EGARCH_STUDENTST(i), stderrors_EGARCH_STUDENTST, robustSE_EGARCH_STUDENTST, ht_EGARCH_STUDENTST, scores_EGARCH_STUDENTST, resid_EGARCH_STUDENTST, likelihood_EGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'EGARCH','STUDENTST',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_EGARCH_STUDENTST(i) = 1e100;
        else
            BIC_EGARCH_STUDENTST(i) = 2*LLF_EGARCH_STUDENTST(i)+log(t)*size(parameters_EGARCH_STUDENTST,1);
        end
    end
    [m,nEGARCH_STUDENTST(j)] = min(BIC_EGARCH_STUDENTST);


    %     11.TGARCH-STUDENTST
    BIC_TGARCH_STUDENTST = zeros(arlag,1);
    LLF_TGARCH_STUDENTST = zeros(arlag,1);
    for i=1:arlag
        [parameters_TGARCH_STUDENTST, LLF_TGARCH_STUDENTST(i), stderrors_TGARCH_STUDENTST, robustSE_TGARCH_STUDENTST, ht_TGARCH_STUDENTST, scores_TGARCH_STUDENTST, resid_TGARCH_STUDENTST, likelihood_TGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'TGARCH','STUDENTST',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_TGARCH_STUDENTST(i) = 1e100;
        else
            BIC_TGARCH_STUDENTST(i) = 2*LLF_TGARCH_STUDENTST(i)+log(t)*size(parameters_TGARCH_STUDENTST,1);
        end
    end
    [m,nTGARCH_STUDENTST(j)] = min(BIC_TGARCH_STUDENTST);

    % 12. GJRGARCH_STUDENTST
    BIC_GJRGARCH_STUDENTST = zeros(arlag,1);
    LLF_GJRGARCH_STUDENTST = zeros(arlag,1);
    for i=1:arlag
        [parameters_GJRGARCH_STUDENTST, LLF_GJRGARCH_STUDENTST(i), stderrors_GJRGARCH_STUDENTST, robustSE_GJRGARCH_STUDENTST, ht_GJRGARCH_STUDENTST, scores_GJRGARCH_STUDENTST, resid_GJRGARCH_STUDENTST, likelihood_GJRGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'GJRGARCH','STUDENTST',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_GJRGARCH_STUDENTST(i) = 1e100;
        else
            BIC_GJRGARCH_STUDENTST(i) = 2*LLF_GJRGARCH_STUDENTST(i)+log(t)*size(parameters_GJRGARCH_STUDENTST,1);
        end
    end
    [m,nGJRGARCH_STUDENTST(j)] = min(BIC_GJRGARCH_STUDENTST);

%     %   13. AVGARCH-STUDENST
    BIC_AVGARCH_STUDENTST = zeros(arlag,1);
    LLF_AVGARCH_STUDENTST = zeros(arlag,1);
    for i=1:arlag
        [parameters_AVGARCH_STUDENTST, LLF_AVGARCH_STUDENTST(i), stderrors_AVGARCH_STUDENTST, robustSE_AVGARCH_STUDENTST, ht_AVGARCH_STUDENTST, scores_AVGARCH_STUDENTST, resid_AVGARCH_STUDENTST, likelihood_AVGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'AVGARCH','STUDENTST',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_AVGARCH_STUDENTST(i) = 1e100;
        else
            BIC_AVGARCH_STUDENTST(i) = 2*LLF_AVGARCH_STUDENTST(i)+log(t)*size(parameters_AVGARCH_STUDENTST,1);
        end
    end
    [m,nAVGARCH_STUDENTST(j)] = min(BIC_AVGARCH_STUDENTST);

    %   14. NGARCH-STUDENTST
%     BIC_NGARCH_STUDENTST = zeros(arlag,1);
%     LLF_NGARCH_STUDENTST = zeros(arlag,1);
%     for i=1:arlag
%         [parameters_NGARCH_STUDENTST, LLF_NGARCH_STUDENTST(i), stderrors_NGARCH_STUDENTST, robustSE_NGARCH_STUDENTST, ht_NGARCH_STUDENTST, scores_NGARCH_STUDENTST, resid_NGARCH_STUDENTST, likelihood_NGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NGARCH','STUDENTST',i,const,[],[],false);
%         if EXITFLAG <= 0
%             BIC_NGARCH_STUDENTST(i) = 1e100;
%         else
%             BIC_NGARCH_STUDENTST(i) = 2*LLF_NGARCH_STUDENTST(i)+log(t)*size(parameters_NGARCH_STUDENTST,1);
%         end
%     end
%     [m,nNGARCH_STUDENTST(j)] = min(BIC_NGARCH_STUDENTST);

    %   15. NAGARCH-STUDENTST
%     BIC_NAGARCH_STUDENTST = zeros(arlag,1);
%     LLF_NAGARCH_STUDENTST = zeros(arlag,1);
%     for i=1:arlag
%         [parameters_NAGARCH_STUDENTST, LLF_NAGARCH_STUDENTST(i), stderrors_NAGARCH_STUDENTST, robustSE_NAGARCH_STUDENTST, ht_NAGARCH_STUDENTST, scores_NAGARCH_STUDENTST, resid_NAGARCH_STUDENTST, likelihood_NAGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NAGARCH','STUDENTST',i,const,[],[],false);
%         if EXITFLAG <= 0
%             BIC_NAGARCH_STUDENTST(i) = 1e100;
%         else
%             BIC_NAGARCH_STUDENTST(i) = 2*LLF_NAGARCH_STUDENTST(i)+log(t)*size(parameters_NAGARCH_STUDENTST,1);
%         end
%     end
%     [m,nNAGARCH_STUDENTST(j)] = min(BIC_NAGARCH_STUDENTST);

    %   16. APGARCH-STUDENTST
%     BIC_APGARCH_STUDENTST = zeros(arlag,1);
%     LLF_APGARCH_STUDENTST = zeros(arlag,1);
%     for i=1:arlag
%         [parameters_APGARCH_STUDENTST, LLF_APGARCH_STUDENTST(i), stderrors_APGARCH_STUDENTST, robustSE_APGARCH_STUDENTST, ht_APGARCH_STUDENTST, scores_APGARCH_STUDENTST, resid_APGARCH_STUDENTST, likelihood_APGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'APGARCH','STUDENTST',i,const,[],[],false);
%         if EXITFLAG <= 0
%             BIC_APGARCH_STUDENTST(i) = 1e100;
%         else
%             BIC_APGARCH_STUDENTST(i) = 2*LLF_APGARCH_STUDENTST(i)+log(t)*size(parameters_APGARCH_STUDENTST,1);
%         end
%     end
%     [m,nAPGARCH_STUDENTST(j)] = min(BIC_APGARCH_STUDENTST);

%     BIC_gesamt_STUDENTST{j} = [BIC_GARCH_STUDENTST(nGARCH_STUDENTST(j));BIC_EGARCH_STUDENTST(nEGARCH_STUDENTST(j));BIC_TGARCH_STUDENTST(nTGARCH_STUDENTST(j));BIC_GJRGARCH_STUDENTST(nGJRGARCH_STUDENTST(j));BIC_AVGARCH_STUDENTST(nAVGARCH_STUDENTST(j));BIC_NGARCH_STUDENTST(nNGARCH_STUDENTST(j));BIC_NAGARCH_STUDENTST(nNAGARCH_STUDENTST(j));BIC_APGARCH_STUDENTST(nAPGARCH_STUDENTST(j))];
        BIC_gesamt_STUDENTST{j} = [BIC_GARCH_STUDENTST(nGARCH_STUDENTST(j));BIC_EGARCH_STUDENTST(nEGARCH_STUDENTST(j));BIC_TGARCH_STUDENTST(nTGARCH_STUDENTST(j));BIC_GJRGARCH_STUDENTST(nGJRGARCH_STUDENTST(j));BIC_AVGARCH_STUDENTST(nAVGARCH_STUDENTST(j));1e100;1e100;1e100];

    % ------------------------------------------------------------------------
    % ------------------------------------------------------------------------
    % Verteilung GED
    % 17.GARCH-GED
    BIC_GARCH_GED = zeros(arlag,1);
    LLF_GARCH_GED = zeros(arlag,1);
    for i=1:arlag
        [parameters_GARCH_GED, LLF_GARCH_GED(i), stderrors_GARCH_GED, robustSE_GARCH_GED, ht_GARCH_GED, scores_GARCH_GED, resid_GARCH_GED, likelihood_GARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'GARCH','GED',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_GARCH_GED(i) = 1e100;
        else
            BIC_GARCH_GED(i) = 2*LLF_GARCH_GED(i)+log(t)*size(parameters_GARCH_GED,1);
        end
    end
    [m,nGARCH_GED(j)] = min(BIC_GARCH_GED);


    % % % 18.EGARCH-GED
    BIC_EGARCH_GED = zeros(arlag,1);
    LLF_EGARCH_GED = zeros(arlag,1);
    for i=1:arlag
        [parameters_EGARCH_GED, LLF_EGARCH_GED(i), stderrors_EGARCH_GED, robustSE_EGARCH_GED, ht_EGARCH_GED, scores_EGARCH_GED, resid_EGARCH_GED, likelihood_EGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'EGARCH','GED',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_EGARCH_GED(i) = 1e100;
        else
            BIC_EGARCH_GED(i) = 2*LLF_EGARCH_GED(i)+log(t)*size(parameters_EGARCH_GED,1);
        end
    end
    [m,nEGARCH_GED(j)] = min(BIC_EGARCH_GED);

    % 19.TGARCH-GED
    BIC_TGARCH_GED = zeros(arlag,1);
    LLF_TGARCH_GED = zeros(arlag,1);
    for i=1:arlag
        [parameters_TGARCH_GED, LLF_TGARCH_GED(i), stderrors_TGARCH_GED, robustSE_TGARCH_GED, ht_TGARCH_GED, scores_TGARCH_GED, resid_TGARCH_GED, likelihood_TGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'TGARCH','GED',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_TGARCH_GED(i) = 1e100;
        else
            BIC_TGARCH_GED(i) = 2*LLF_TGARCH_GED(i)+log(t)*size(parameters_TGARCH_GED,1);
        end
    end
    [m,nTGARCH_GED(j)] = min(BIC_TGARCH_GED);

    % 20.GJRGARCH-GED
    BIC_GJRGARCH_GED = zeros(arlag,1);
    LLF_GJRGARCH_GED = zeros(arlag,1);
    for i=1:arlag
        [parameters_GJRGARCH_GED, LLF_GJRGARCH_GED(i), stderrors_GJRGARCH_GED, robustSE_GJRGARCH_GED, ht_GJRGARCH_GED, scores_GJRGARCH_GED, resid_GJRGARCH_GED, likelihood_GJRGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'GJRGARCH','GED',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_GJRGARCH_GED(i) = 1e100;
        else
            BIC_GJRGARCH_GED(i) = 2*LLF_GJRGARCH_GED(i)+log(t)*size(parameters_GJRGARCH_GED,1);
        end
    end
    [m,nGJRGARCH_GED(j)] = min(BIC_GJRGARCH_GED);


%     %   21. AVGARCH-GED
    BIC_AVGARCH_GED = zeros(arlag,1);
    LLF_AVGARCH_GED = zeros(arlag,1);
    for i=1:arlag
        [parameters_AVGARCH_GED, LLF_AVGARCH_GED(i), stderrors_AVGARCH_GED, robustSE_AVGARCH_GED, ht_AVGARCH_GED, scores_AVGARCH_GED, resid_AVGARCH_GED, likelihood_AVGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'AVGARCH','GED',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_AVGARCH_GED(i) = 1e100;
        else
            BIC_AVGARCH_GED(i) = 2*LLF_AVGARCH_GED(i)+log(t)*size(parameters_AVGARCH_GED,1);
        end
    end
    [m,nAVGARCH_GED(j)] = min(BIC_AVGARCH_GED);

    %   22. NGARCH-GED
%     BIC_NGARCH_GED = zeros(arlag,1);
%     LLF_NGARCH_GED = zeros(arlag,1);
%     for i=1:arlag
%         [parameters_NGARCH_GED, LLF_NGARCH_GED(i), stderrors_NGARCH_GED, robustSE_NGARCH_GED, ht_NGARCH_GED, scores_NGARCH_GED, resid_NGARCH_GED, likelihood_NGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NGARCH','GED',i,const,[],[],false);
%         if EXITFLAG <= 0
%             BIC_NGARCH_GED(i) = 1e100;
%         else
%             BIC_NGARCH_GED(i) = 2*LLF_NGARCH_GED(i)+log(t)*size(parameters_NGARCH_GED,1);
%         end
%     end
%     [m,nNGARCH_GED(j)] = min(BIC_NGARCH_GED);

    %   23. NAGARCH-GED
%     BIC_NAGARCH_GED = zeros(arlag,1);
%     LLF_NAGARCH_GED = zeros(arlag,1);
%     for i=1:arlag
%         [parameters_NAGARCH_GED, LLF_NAGARCH_GED(i), stderrors_NAGARCH_GED, robustSE_NAGARCH_GED, ht_NAGARCH_GED, scores_NAGARCH_GED, resid_NAGARCH_GED, likelihood_NAGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NAGARCH','GED',i,const,[],[],false);
%         if EXITFLAG <= 0
%             BIC_NAGARCH_GED(i) = 1e100;
%         else
%             BIC_NAGARCH_GED(i) = 2*LLF_NAGARCH_GED(i)+log(t)*size(parameters_NAGARCH_GED,1);
%         end
%     end
%     [m,nNAGARCH_GED(j)] = min(BIC_NAGARCH_GED);

    %   24. APGARCH-GED
%     BIC_APGARCH_GED = zeros(arlag,1);
%     LLF_APGARCH_GED = zeros(arlag,1);
%     for i=1:arlag
%         [parameters_APGARCH_GED, LLF_APGARCH_GED(i), stderrors_APGARCH_GED, robustSE_APGARCH_GED, ht_APGARCH_GED, scores_APGARCH_GED, resid_APGARCH_GED, likelihood_APGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'APGARCH','GED',i,const,[],[],false);
%         if EXITFLAG <= 0
%             BIC_APGARCH_GED(i) = 1e100;
%         else
%             BIC_APGARCH_GED(i) = 2*LLF_APGARCH_GED(i)+log(t)*size(parameters_APGARCH_GED,1);
%         end
%     end
%     [m,nAPGARCH_GED(j)] = min(BIC_APGARCH_GED);

%     BIC_gesamt_GED{j} = [BIC_GARCH_GED(nGARCH_GED(j));BIC_EGARCH_GED(nEGARCH_GED(j));BIC_TGARCH_GED(nTGARCH_GED(j));BIC_GJRGARCH_GED(nGJRGARCH_GED(j));BIC_AVGARCH_GED(nAVGARCH_GED(j));BIC_NGARCH_GED(nNGARCH_GED(j));BIC_NAGARCH_GED(nNAGARCH_GED(j));BIC_APGARCH_GED(nAPGARCH_GED(j))];
    BIC_gesamt_GED{j} = [BIC_GARCH_GED(nGARCH_GED(j));BIC_EGARCH_GED(nEGARCH_GED(j));BIC_TGARCH_GED(nTGARCH_GED(j));BIC_GJRGARCH_GED(nGJRGARCH_GED(j));BIC_AVGARCH_GED(nAVGARCH_GED(j));1e100;1e100;1e100];

    %     optGARCH_GED = find(BIC_gesamt_GED == min(BIC_gesamt_GED));
    %     BIC_GED = BIC_gesamt_GED(optGARCH_GED);

    %     ---------------------------------------------------------------------
    %     --------------------------------------------------------------------
    % Verteilung SKEWT
    % 25.GARCH-SKEWT
    BIC_GARCH_SKEWT = zeros(arlag,1);
    LLF_GARCH_SKEWT = zeros(arlag,1);
    for i=1:arlag
        [parameters_GARCH_SKEWT, LLF_GARCH_SKEWT(i), stderrors_GARCH_SKEWT, robustSE_GARCH_SKEWT, ht_GARCH_SKEWT, scores_GARCH_SKEWT, resid_GARCH_SKEWT, likelihood_GARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'GARCH','SKEWT',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_GARCH_SKEWT(i) = 1e100;
        else
            BIC_GARCH_SKEWT(i) = 2*LLF_GARCH_SKEWT(i)+log(t)*size(parameters_GARCH_SKEWT,1);
        end
    end
    [m,nGARCH_SKEWT(j)] = min(BIC_GARCH_SKEWT);

    % % % 26.EGARCH-SKEWT
    BIC_EGARCH_SKEWT = zeros(arlag,1);
    LLF_EGARCH_SKEWT = zeros(arlag,1);
    for i=1:arlag
        [parameters_EGARCH_SKEWT, LLF_EGARCH_SKEWT(i), stderrors_EGARCH_SKEWT, robustSE_EGARCH_SKEWT, ht_EGARCH_SKEWT, scores_EGARCH_SKEWT, resid_EGARCH_SKEWT, likelihood_EGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'EGARCH','SKEWT',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_EGARCH_SKEWT(i) = 1e100;
        else
            BIC_EGARCH_SKEWT(i) = 2*LLF_EGARCH_SKEWT(i)+log(t)*size(parameters_EGARCH_SKEWT,1);
        end
    end
    [m,nEGARCH_SKEWT(j)] = min(BIC_EGARCH_SKEWT);

    % 27.TGARCH-SKEWT
    BIC_TGARCH_SKEWT = zeros(arlag,1);
    LLF_TGARCH_SKEWT = zeros(arlag,1);
    for i=1:arlag
        [parameters_TGARCH_SKEWT, LLF_TGARCH_SKEWT(i), stderrors_TGARCH_SKEWT, robustSE_TGARCH_SKEWT, ht_TGARCH_SKEWT, scores_TGARCH_SKEWT, resid_TGARCH_SKEWT, likelihood_TGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'TGARCH','SKEWT',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_TGARCH_SKEWT(i) = 1e100;
        else
            BIC_TGARCH_SKEWT(i) = 2*LLF_TGARCH_SKEWT(i)+log(t)*size(parameters_TGARCH_SKEWT,1);
        end
    end
    [m,nTGARCH_SKEWT(j)] = min(BIC_TGARCH_SKEWT);

    % 28.GJRGARCH-SKEWT
    BIC_GJRGARCH_SKEWT = zeros(arlag,1);
    LLF_GJRGARCH_SKEWT = zeros(arlag,1);
    for i=1:arlag
        [parameters_GJRGARCH_SKEWT, LLF_GJRGARCH_SKEWT(i), stderrors_GJRGARCH_SKEWT, robustSE_GJRGARCH_SKEWT, ht_GJRGARCH_SKEWT, scores_GJRGARCH_SKEWT, resid_GJRGARCH_SKEWT, likelihood_GJRGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'GJRGARCH','SKEWT',i,const,[],[],false);
        if EXITFLAG <= 0
            BIC_GJRGARCH_SKEWT(i) = 1e100;
        else
            BIC_GJRGARCH_SKEWT(i) = 2*LLF_GJRGARCH_SKEWT(i)+log(t)*size(parameters_GJRGARCH_SKEWT,1);
        end
    end
    [m,nGJRGARCH_SKEWT(j)] = min(BIC_GJRGARCH_SKEWT);


%     %   29. AVGARCH-SKEWT
    BIC_AVGARCH_SKEWT = zeros(arlag,1);
    LLF_AVGARCH_SKEWT = zeros(arlag,1);
    for i=1:arlag
        [parameters_AVGARCH_SKEWT, LLF_AVGARCH_SKEWT(i), stderrors_AVGARCH_SKEWT, robustSE_AVGARCH_SKEWT, ht_AVGARCH_SKEWT, scores_AVGARCH_SKEWT, resid_AVGARCH_SKEWT, likelihood_AVGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'AVGARCH','SKEWT',i,const,[],[],false);
        if EXITFLAG <= 0 || any(isnan(parameters_AVGARCH_SKEWT))
            BIC_AVGARCH_SKEWT(i) = 1e100;
        else
            BIC_AVGARCH_SKEWT(i) = 2*LLF_AVGARCH_SKEWT(i)+log(t)*size(parameters_AVGARCH_SKEWT,1);
        end
    end
    [m,nAVGARCH_SKEWT(j)] = min(BIC_AVGARCH_SKEWT);
% %
%     %   30. NGARCH-SKEWT
%     BIC_NGARCH_SKEWT = zeros(arlag,1);
%     LLF_NGARCH_SKEWT = zeros(arlag,1);
%     for i=1:arlag
%         [parameters_NGARCH_SKEWT, LLF_NGARCH_SKEWT(i), stderrors_NGARCH_SKEWT, robustSE_NGARCH_SKEWT, ht_NGARCH_SKEWT, scores_NGARCH_SKEWT, resid_NGARCH_SKEWT, likelihood_NGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NGARCH','SKEWT',i,const,[],[],false);
%         if EXITFLAG <= 0
%             BIC_NGARCH_SKEWT(i) = 1e100;
%         else
%             BIC_NGARCH_SKEWT(i) = 2*LLF_NGARCH_SKEWT(i)+log(t)*size(parameters_NGARCH_SKEWT,1);
%         end
%     end
%     [m,nNGARCH_SKEWT(j)] = min(BIC_NGARCH_SKEWT);

    %   31. NAGARCH-SKEWT
%     BIC_NAGARCH_SKEWT = zeros(arlag,1);
%     LLF_NAGARCH_SKEWT = zeros(arlag,1);
%     for i=1:arlag
%         [parameters_NAGARCH_SKEWT, LLF_NAGARCH_SKEWT(i), stderrors_NAGARCH_SKEWT, robustSE_NAGARCH_SKEWT, ht_NAGARCH_SKEWT, scores_NAGARCH_SKEWT, resid_NAGARCH_SKEWT, likelihood_NAGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NAGARCH','SKEWT',i,const,[],[],false);
%         if EXITFLAG <= 0
%             BIC_NAGARCH_SKEWT(i) = 1e100;
%         else
%             BIC_NAGARCH_SKEWT(i) = 2*LLF_NAGARCH_SKEWT(i)+log(t)*size(parameters_NAGARCH_SKEWT,1);
%         end
%     end
%     [m,nNAGARCH_SKEWT(j)] = min(BIC_NAGARCH_SKEWT);

    %   32. APGARCH-SKEWT
%     BIC_APGARCH_SKEWT = zeros(arlag,1);
%     LLF_APGARCH_SKEWT = zeros(arlag,1);
%     for i=1:arlag
%         [parameters_APGARCH_SKEWT, LLF_APGARCH_SKEWT(i), stderrors_APGARCH_SKEWT, robustSE_APGARCH_SKEWT, ht_APGARCH_SKEWT, scores_APGARCH_SKEWT, resid_APGARCH_SKEWT, likelihood_APGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'APGARCH','SKEWT',i,const,[],[],false);
%         if EXITFLAG <= 0
%             BIC_APGARCH_SKEWT(i) = 1e100;
%         else
%             BIC_APGARCH_SKEWT(i) = 2*LLF_APGARCH_SKEWT(i)+log(t)*size(parameters_APGARCH_SKEWT,1);
%         end
%     end
%     [m,nAPGARCH_SKEWT(j)] = min(BIC_APGARCH_SKEWT);

%     BIC_gesamt_SKEWT{j} = [BIC_GARCH_SKEWT(nGARCH_SKEWT(j));BIC_EGARCH_SKEWT(nEGARCH_SKEWT(j));BIC_TGARCH_SKEWT(nTGARCH_SKEWT(j));BIC_GJRGARCH_SKEWT(nGJRGARCH_SKEWT(j));BIC_AVGARCH_SKEWT(nAVGARCH_SKEWT(j));BIC_NGARCH_SKEWT(nNGARCH_SKEWT(j));BIC_NAGARCH_SKEWT(nNAGARCH_SKEWT(j));BIC_APGARCH_SKEWT(nAPGARCH_SKEWT(j))];
    BIC_gesamt_SKEWT{j} = [BIC_GARCH_SKEWT(nGARCH_SKEWT(j));BIC_EGARCH_SKEWT(nEGARCH_SKEWT(j));BIC_TGARCH_SKEWT(nTGARCH_SKEWT(j));BIC_GJRGARCH_SKEWT(nGJRGARCH_SKEWT(j));BIC_AVGARCH_SKEWT(nAVGARCH_SKEWT(j));1e100;1e100;1e100];

    % -------------------------------------------------------------------------
    % -------------------------------------------------------------------------
    % % % Bestimmung des optimalen Modells über alle Modelle mit allen
    % Verteilungen
    BIC_ALL_opt{j} = [BIC_gesamt_Gauss{j}; BIC_gesamt_STUDENTST{j}; BIC_gesamt_GED{j}; BIC_gesamt_SKEWT{j}];
    [m,BIC_opt{j}] = min(BIC_ALL_opt{j});
end

% % für multivariate GARCH wähle optimales Modell mit Normalverteilung aus
Output_Gauss = cell(k,1);
for j=1:k
    if BIC_opt_Gauss{j}==1
        [parameters_GARCH_Gauss, LLF_GARCH_Gauss, stderrors_GARCH_Gauss, robustSE_GARCH_Gauss, ht_GARCH_Gauss, scores_GARCH_Gauss, resid_GARCH_Gauss, likelihood_GARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'GARCH','NORMAL',nGARCH_Gauss(j),const);
        BIC_GARCH_Gauss = 2*LLF_GARCH_Gauss+ log(t)*size(parameters_GARCH_Gauss,1);
        Tstatistic_GARCH_Gauss=parameters_GARCH_Gauss./diag(robustSE_GARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_GARCH_Gauss./sqrt(ht_GARCH_Gauss);
        result_stderrors_GARCH_Gauss=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'norm_cdf');
        U_GARCH_Gauss=normcdf(stdresid,0,1);
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_GARCH_Gauss,.05,'unifcdf',0,1);
        U_GARCH_Gauss_mean = U_GARCH_Gauss-mean(U_GARCH_Gauss);
        U_GARCH_Gauss_mean2 = (U_GARCH_Gauss-mean(U_GARCH_Gauss)).^2;
        U_GARCH_Gauss_mean3 = (U_GARCH_Gauss-mean(U_GARCH_Gauss)).^3;
        U_GARCH_Gauss_mean4 = (U_GARCH_Gauss-mean(U_GARCH_Gauss)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_GARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output_Gauss{j}.Params=parameters_GARCH_Gauss;
        Output_Gauss{j}.ParamsGARCH = parameters_GARCH_Gauss(1:end-nGARCH_Gauss(j)-const);
        Output_Gauss{j}.ParamsAR = parameters_GARCH_Gauss(size(Output_Gauss{j}.ParamsGARCH,1)+1:size(parameters_GARCH_Gauss,1));
        Output_Gauss{j}.LLF=LLF_GARCH_Gauss;
        Output_Gauss{j}.stderrors=stderrors_GARCH_Gauss;
        Output_Gauss{j}.robustSE=robustSE_GARCH_Gauss;
        Output_Gauss{j}.ht=ht_GARCH_Gauss;
        Output_Gauss{j}.Scores=scores_GARCH_Gauss;
        Output_Gauss{j}.Innovations=resid_GARCH_Gauss;
        Output_Gauss{j}.GARCH = 'GARCH';
        Output_Gauss{j}.U = U_GARCH_Gauss;
        Output_Gauss{j}.Tstat = Tstatistic_GARCH_Gauss;
        Output_Gauss{j}.garchtype = 1;
        Output_Gauss{j}.leverage = 0;
        Output_Gauss{j}.errortype = 1;
        Output_Gauss{j}.arlag = nGARCH_Gauss(j);
        Output_Gauss{j}.likelihoods = likelihood_GARCH_Gauss;
        Output_Gauss{j}.EXITFLAG = EXITFLAG;
        Output_Gauss{j}.data = data(:,j);
        Output_Gauss{j}.dist = 'GAUSS';
        Output_Gauss{j}.Innovations = resid_GARCH_Gauss;
        Output_Gauss{j}.BIC = BIC_GARCH_Gauss;
        Output_Gauss{j}.stdresid = stdresid;
        Output_Gauss{j}.statKolmResid = stat_Kolmogorov_resid;
        Output_Gauss{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output_Gauss{j}.HKolmResid = H_Kolmogorov_resid;
        Output_Gauss{j}.statKolmU= stat_Kolmogorov_U;
        Output_Gauss{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output_Gauss{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output_Gauss{j}.const = 1;
        else
            Output_Gauss{j}.const = 0;
        end

        % % % 5.EGARCH-Gauss
    elseif BIC_opt_Gauss{j}==2
        [parameters_EGARCH_Gauss, LLF_EGARCH_Gauss, stderrors_EGARCH_Gauss, robustSE_EGARCH_Gauss, ht_EGARCH_Gauss, scores_EGARCH_Gauss, resid_EGARCH_Gauss, likelihood_EGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'EGARCH','NORMAL',nEGARCH_Gauss(j),const);
        BIC_EGARCH_Gauss = 2*LLF_EGARCH_Gauss+log(t)*size(parameters_EGARCH_Gauss,1);
        Tstatistic_EGARCH_Gauss=parameters_EGARCH_Gauss./diag(robustSE_EGARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_EGARCH_Gauss./sqrt(ht_EGARCH_Gauss);
        result_stderrors_EGARCH_Gauss=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'norm_cdf');
        U_EGARCH_Gauss=normcdf(stdresid,0,1);
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_EGARCH_Gauss,.05,'unifcdf',0,1);
        U_EGARCH_Gauss_mean = U_EGARCH_Gauss-mean(U_EGARCH_Gauss);
        U_EGARCH_Gauss_mean2 = (U_EGARCH_Gauss-mean(U_EGARCH_Gauss)).^2;
        U_EGARCH_Gauss_mean3 = (U_EGARCH_Gauss-mean(U_EGARCH_Gauss)).^3;
        U_EGARCH_Gauss_mean4 = (U_EGARCH_Gauss-mean(U_EGARCH_Gauss)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_EGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output_Gauss{j}.Params=parameters_EGARCH_Gauss;
        Output_Gauss{j}.ParamsGARCH = parameters_EGARCH_Gauss(1:end-nEGARCH_Gauss(j)-const);
        Output_Gauss{j}.ParamsAR = parameters_EGARCH_Gauss(size(Output_Gauss{j}.ParamsGARCH,1)+1:size(parameters_EGARCH_Gauss,1));
        Output_Gauss{j}.LLF=LLF_EGARCH_Gauss;
        Output_Gauss{j}.stderrors=stderrors_EGARCH_Gauss;
        Output_Gauss{j}.robustSE=robustSE_EGARCH_Gauss;
        Output_Gauss{j}.ht=ht_EGARCH_Gauss;
        Output_Gauss{j}.Scores=scores_EGARCH_Gauss;
        Output_Gauss{j}.Innovations=resid_EGARCH_Gauss;
        Output_Gauss{j}.GARCH = 'EGARCH';
        Output_Gauss{j}.U = U_EGARCH_Gauss;
        Output_Gauss{j}.Tstat = Tstatistic_EGARCH_Gauss;
        Output_Gauss{j}.garchtype = 0;
        Output_Gauss{j}.leverage = 1;
        Output_Gauss{j}.errortype = 1;
        Output_Gauss{j}.arlag = nEGARCH_Gauss(j);
        Output_Gauss{j}.likelihoods = likelihood_EGARCH_Gauss;
        Output_Gauss{j}.EXITFLAG = EXITFLAG;
        Output_Gauss{j}.data = data(:,j);
        Output_Gauss{j}.dist = 'GAUSS';
        Output_Gauss{j}.Innovations = resid_EGARCH_Gauss;
        Output_Gauss{j}.BIC = BIC_EGARCH_Gauss;
        Output_Gauss{j}.stdresid = stdresid;
        Output_Gauss{j}.statKolmResid = stat_Kolmogorov_resid;
        Output_Gauss{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output_Gauss{j}.HKolmResid = H_Kolmogorov_resid;
        Output_Gauss{j}.statKolmU= stat_Kolmogorov_U;
        Output_Gauss{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output_Gauss{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output_Gauss{j}.const = 1;
        else
            Output_Gauss{j}.const = 0;
        end

        % 6.TGARCH-Gauss
    elseif BIC_opt_Gauss{j}==3
        [parameters_TGARCH_Gauss, LLF_TGARCH_Gauss, stderrors_TGARCH_Gauss, robustSE_TGARCH_Gauss, ht_TGARCH_Gauss, scores_TGARCH_Gauss, resid_TGARCH_Gauss, likelihood_TGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'TGARCH','NORMAL',nTGARCH_Gauss(j),const);
        BIC_TGARCH_Gauss = 2*LLF_TGARCH_Gauss+log(t)*size(parameters_TGARCH_Gauss,1);
        Tstatistic_TGARCH_Gauss=parameters_TGARCH_Gauss./diag(robustSE_TGARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_TGARCH_Gauss./sqrt(ht_TGARCH_Gauss);
        result_stderrors_TGARCH_Gauss=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'norm_cdf');
        U_TGARCH_Gauss=normcdf(stdresid,0,1);
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_TGARCH_Gauss,.05,'unifcdf',0,1);
        U_TGARCH_Gauss_mean = U_TGARCH_Gauss-mean(U_TGARCH_Gauss);
        U_TGARCH_Gauss_mean2 = (U_TGARCH_Gauss-mean(U_TGARCH_Gauss)).^2;
        U_TGARCH_Gauss_mean3 = (U_TGARCH_Gauss-mean(U_TGARCH_Gauss)).^3;
        U_TGARCH_Gauss_mean4 = (U_TGARCH_Gauss-mean(U_TGARCH_Gauss)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_TGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output_Gauss{j}.Params=parameters_TGARCH_Gauss;
        Output_Gauss{j}.ParamsGARCH = parameters_TGARCH_Gauss(1:end-nTGARCH_Gauss(j)-const);
        Output_Gauss{j}.ParamsAR = parameters_TGARCH_Gauss(size(Output_Gauss{j}.ParamsGARCH,1)+1:size(parameters_TGARCH_Gauss,1));
        Output_Gauss{j}.LLF=LLF_TGARCH_Gauss;
        Output_Gauss{j}.stderrors=stderrors_TGARCH_Gauss;
        Output_Gauss{j}.robustSE=robustSE_TGARCH_Gauss;
        Output_Gauss{j}.ht=ht_TGARCH_Gauss;
        Output_Gauss{j}.Scores=scores_TGARCH_Gauss;
        Output_Gauss{j}.Innovations=resid_TGARCH_Gauss;
        Output_Gauss{j}.GARCH = 'TGARCH';
        Output_Gauss{j}.U = U_TGARCH_Gauss;
        Output_Gauss{j}.Tstat = Tstatistic_TGARCH_Gauss;
        Output_Gauss{j}.garchtype = 2;
        Output_Gauss{j}.leverage = 1;
        Output_Gauss{j}.errortype = 1;
        Output_Gauss{j}.arlag = nTGARCH_Gauss(j);
        Output_Gauss{j}.likelihoods = likelihood_TGARCH_Gauss;
        Output_Gauss{j}.EXITFLAG = EXITFLAG;
        Output_Gauss{j}.data = data(:,j);
        Output_Gauss{j}.dist = 'GAUSS';
        Output_Gauss{j}.Innnovations = resid_TGARCH_Gauss;
        Output_Gauss{j}.BIC = BIC_TGARCH_Gauss;
        Output_Gauss{j}.stdresid = stdresid;
        Output_Gauss{j}.statKolmResid = stat_Kolmogorov_resid;
        Output_Gauss{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output_Gauss{j}.HKolmResid = H_Kolmogorov_resid;
        Output_Gauss{j}.statKolmU= stat_Kolmogorov_U;
        Output_Gauss{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output_Gauss{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output_Gauss{j}.const = 1;
        else
            Output_Gauss{j}.const = 0;
        end

        % 7.GJRGARCH-Gauss
    elseif BIC_opt_Gauss{j}==4
        [parameters_GJRGARCH_Gauss, LLF_GJRGARCH_Gauss, stderrors_GJRGARCH_Gauss, robustSE_GJRGARCH_Gauss, ht_GJRGARCH_Gauss, scores_GJRGARCH_Gauss, resid_GJRGARCH_Gauss, likelihood_GJRGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'GJRGARCH','NORMAL',nGJRGARCH_Gauss(j),const);
        BIC_GJRGARCH_Gauss = 2*LLF_GJRGARCH_Gauss+log(t)*size(parameters_GJRGARCH_Gauss,1);
        Tstatistic_GJRGARCH_Gauss=parameters_GJRGARCH_Gauss./diag(robustSE_GJRGARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_GJRGARCH_Gauss./sqrt(ht_GJRGARCH_Gauss);
        result_stderrors_GJRGARCH_Gauss=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'norm_cdf');
        U_GJRGARCH_Gauss=normcdf(stdresid,0,1);
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_GJRGARCH_Gauss,.05,'unifcdf',0,1);
        U_GJRGARCH_Gauss_mean = U_GJRGARCH_Gauss-mean(U_GJRGARCH_Gauss);
        U_GJRGARCH_Gauss_mean2 = (U_GJRGARCH_Gauss-mean(U_GJRGARCH_Gauss)).^2;
        U_GJRGARCH_Gauss_mean3 = (U_GJRGARCH_Gauss-mean(U_GJRGARCH_Gauss)).^3;
        U_GJRGARCH_Gauss_mean4 = (U_GJRGARCH_Gauss-mean(U_GJRGARCH_Gauss)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_GJRGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output_Gauss{j}.Params=parameters_GJRGARCH_Gauss;
        Output_Gauss{j}.ParamsGARCH = parameters_GJRGARCH_Gauss(1:end-nGJRGARCH_Gauss(j)-const);
        Output_Gauss{j}.ParamsAR = parameters_GJRGARCH_Gauss(size(Output_Gauss{j}.ParamsGARCH,1)+1:size(parameters_GJRGARCH_Gauss,1));
        Output_Gauss{j}.LLF=LLF_GJRGARCH_Gauss;
        Output_Gauss{j}.stderrors=stderrors_GJRGARCH_Gauss;
        Output_Gauss{j}.robustSE=robustSE_GJRGARCH_Gauss;
        Output_Gauss{j}.ht=ht_GJRGARCH_Gauss;
        Output_Gauss{j}.Scores=scores_GJRGARCH_Gauss;
        Output_Gauss{j}.Innovations=resid_GJRGARCH_Gauss;
        Output_Gauss{j}.GARCH = 'GJRGARCH';
        Output_Gauss{j}.U = U_GJRGARCH_Gauss;
        Output_Gauss{j}.Tstat = Tstatistic_GJRGARCH_Gauss;
        Output_Gauss{j}.garchtype = 8;
        Output_Gauss{j}.leverage = 1;
        Output_Gauss{j}.errortype = 1;
        Output_Gauss{j}.arlag = nGJRGARCH_Gauss(j);
        Output_Gauss{j}.likelihoods = likelihood_GJRGARCH_Gauss;
        Output_Gauss{j}.EXITFLAG = EXITFLAG;
        Output_Gauss{j}.data = data(:,j);
        Output_Gauss{j}.dist = 'GAUSS';
        Output_Gauss{j}.Innovations = resid_GJRGARCH_Gauss;
        Output_Gauss{j}.BIC = BIC_GJRGARCH_Gauss;
        Output_Gauss{j}.stdresid = stdresid;
        Output_Gauss{j}.statKolmResid = stat_Kolmogorov_resid;
        Output_Gauss{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output_Gauss{j}.HKolmResid = H_Kolmogorov_resid;
        Output_Gauss{j}.statKolmU= stat_Kolmogorov_U;
        Output_Gauss{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output_Gauss{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output_Gauss{j}.const = 1;
        else
            Output_Gauss{j}.const = 0;
        end

    elseif BIC_opt_Gauss{j}==5
        [parameters_AVGARCH_Gauss, LLF_AVGARCH_Gauss, stderrors_AVGARCH_Gauss, robustSE_AVGARCH_Gauss, ht_AVGARCH_Gauss, scores_AVGARCH_Gauss, resid_AVGARCH_Gauss, likelihood_AVGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'AVGARCH','NORMAL',nAVGARCH_Gauss(j),const);
        BIC_AVGARCH_Gauss = 2*LLF_AVGARCH_Gauss+log(t)*size(parameters_AVGARCH_Gauss,1);
        Tstatistic_AVGARCH_Gauss=parameters_AVGARCH_Gauss./diag(robustSE_AVGARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_AVGARCH_Gauss./sqrt(ht_AVGARCH_Gauss);
        result_stderrors_AVGARCH_Gauss=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'norm_cdf');
        U_AVGARCH_Gauss=normcdf(stdresid,0,1);
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_AVGARCH_Gauss,.05,'unifcdf',0,1);
        U_AVGARCH_Gauss_mean = U_AVGARCH_Gauss-mean(U_AVGARCH_Gauss);
        U_AVGARCH_Gauss_mean2 = (U_AVGARCH_Gauss-mean(U_AVGARCH_Gauss)).^2;
        U_AVGARCH_Gauss_mean3 = (U_AVGARCH_Gauss-mean(U_AVGARCH_Gauss)).^3;
        U_AVGARCH_Gauss_mean4 = (U_AVGARCH_Gauss-mean(U_AVGARCH_Gauss)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_AVGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output_Gauss{j}.Params=parameters_AVGARCH_Gauss;
        Output_Gauss{j}.ParamsGARCH = parameters_AVGARCH_Gauss(1:end-nAVGARCH_Gauss(j)-const);
        Output_Gauss{j}.ParamsAR = parameters_AVGARCH_Gauss(size(Output_Gauss{j}.ParamsGARCH,1)+1:size(parameters_AVGARCH_Gauss,1));
        Output_Gauss{j}.LLF=LLF_AVGARCH_Gauss;
        Output_Gauss{j}.stderrors=stderrors_AVGARCH_Gauss;
        Output_Gauss{j}.robustSE=robustSE_AVGARCH_Gauss;
        Output_Gauss{j}.ht=ht_AVGARCH_Gauss;
        Output_Gauss{j}.Scores=scores_AVGARCH_Gauss;
        Output_Gauss{j}.Innovations=resid_AVGARCH_Gauss;
        Output_Gauss{j}.GARCH = 'AVGARCH';
        Output_Gauss{j}.U = U_AVGARCH_Gauss;
        Output_Gauss{j}.Tstat = Tstatistic_AVGARCH_Gauss;
        Output_Gauss{j}.garchtype = 3;
        Output_Gauss{j}.leverage = 1;
        Output_Gauss{j}.errortype = 1;
        Output_Gauss{j}.arlag = nAVGARCH_Gauss(j);
        Output_Gauss{j}.likelihoods = likelihood_AVGARCH_Gauss;
        Output_Gauss{j}.EXITFLAG = EXITFLAG;
        Output_Gauss{j}.data = data(:,j);
        Output_Gauss{j}.dist = 'GAUSS';
        Output_Gauss{j}.Innovations = resid_AVGARCH_Gauss;
        Output_Gauss{j}.BIC = BIC_AVGARCH_Gauss;
        Output_Gauss{j}.stdresid = stdresid;
        Output_Gauss{j}.statKolmResid = stat_Kolmogorov_resid;
        Output_Gauss{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output_Gauss{j}.HKolmResid = H_Kolmogorov_resid;
        Output_Gauss{j}.statKolmU= stat_Kolmogorov_U;
        Output_Gauss{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output_Gauss{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output_Gauss{j}.const = 1;
        else
            Output_Gauss{j}.const = 0;
        end


    elseif BIC_opt_Gauss{j}==6
        [parameters_NGARCH_Gauss, LLF_NGARCH_Gauss, stderrors_NGARCH_Gauss, robustSE_NGARCH_Gauss, ht_NGARCH_Gauss, scores_NGARCH_Gauss, resid_NGARCH_Gauss, likelihood_NGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NGARCH','NORMAL',nNGARCH_Gauss(j),const);
        BIC_NGARCH_Gauss = 2*LLF_NGARCH_Gauss+log(t)*size(parameters_NGARCH_Gauss,1);
        Tstatistic_NGARCH_Gauss=parameters_NGARCH_Gauss./diag(robustSE_NGARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_NGARCH_Gauss./sqrt(ht_NGARCH_Gauss);
        result_stderrors_NGARCH_Gauss=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'norm_cdf');
        U_NGARCH_Gauss=normcdf(stdresid,0,1);
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_NGARCH_Gauss,.05,'unifcdf',0,1);
        U_NGARCH_Gauss_mean = U_NGARCH_Gauss-mean(U_NGARCH_Gauss);
        U_NGARCH_Gauss_mean2 = (U_NGARCH_Gauss-mean(U_NGARCH_Gauss)).^2;
        U_NGARCH_Gauss_mean3 = (U_NGARCH_Gauss-mean(U_NGARCH_Gauss)).^3;
        U_NGARCH_Gauss_mean4 = (U_NGARCH_Gauss-mean(U_NGARCH_Gauss)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_NGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output_Gauss{j}.Params=parameters_NGARCH_Gauss;
        Output_Gauss{j}.ParamsGARCH = parameters_NGARCH_Gauss(1:end-nNGARCH_Gauss(j)-const);
        Output_Gauss{j}.ParamsAR = parameters_NGARCH_Gauss(size(Output_Gauss{j}.ParamsGARCH,1)+1:size(parameters_NGARCH_Gauss,1));
        Output_Gauss{j}.LLF=LLF_NGARCH_Gauss;
        Output_Gauss{j}.stderrors=stderrors_NGARCH_Gauss;
        Output_Gauss{j}.robustSE=robustSE_NGARCH_Gauss;
        Output_Gauss{j}.ht=ht_NGARCH_Gauss;
        Output_Gauss{j}.Scores=scores_NGARCH_Gauss;
        Output_Gauss{j}.Innovations=resid_NGARCH_Gauss;
        Output_Gauss{j}.GARCH = 'NGARCH';
        Output_Gauss{j}.U = U_NGARCH_Gauss;
        Output_Gauss{j}.Tstat = Tstatistic_NGARCH_Gauss;
        Output_Gauss{j}.garchtype = 4;
        Output_Gauss{j}.leverage = 1;
        Output_Gauss{j}.errortype = 1;
        Output_Gauss{j}.arlag = nNGARCH_Gauss(j);
        Output_Gauss{j}.likelihoods = likelihood_NGARCH_Gauss;
        Output_Gauss{j}.EXITFLAG = EXITFLAG;
        Output_Gauss{j}.data = data(:,j);
        Output_Gauss{j}.dist = 'GAUSS';
        Output_Gauss{j}.Innovations = resid_NGARCH_Gauss;
        Output_Gauss{j}.BIC = BIC_NGARCH_Gauss;
        Output_Gauss{j}.stdresid = stdresid;
        Output_Gauss{j}.statKolmResid = stat_Kolmogorov_resid;
        Output_Gauss{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output_Gauss{j}.HKolmResid = H_Kolmogorov_resid;
        Output_Gauss{j}.statKolmU= stat_Kolmogorov_U;
        Output_Gauss{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output_Gauss{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output_Gauss{j}.const = 1;
        else
            Output_Gauss{j}.const = 0;
        end

    elseif BIC_opt_Gauss{j}==7
        [parameters_NAGARCH_Gauss, LLF_NAGARCH_Gauss, stderrors_NAGARCH_Gauss, robustSE_NAGARCH_Gauss, ht_NAGARCH_Gauss, scores_NAGARCH_Gauss, resid_NAGARCH_Gauss, likelihood_NAGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NAGARCH','NORMAL',nNAGARCH_Gauss(j),const);
        BIC_NAGARCH_Gauss = 2*LLF_NAGARCH_Gauss+log(t)*size(parameters_NAGARCH_Gauss,1);
        Tstatistic_NAGARCH_Gauss=parameters_NAGARCH_Gauss./diag(robustSE_NAGARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_NAGARCH_Gauss./sqrt(ht_NAGARCH_Gauss);
        result_stderrors_NAGARCH_Gauss=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'norm_cdf');
        U_NAGARCH_Gauss=normcdf(stdresid,0,1);
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_NAGARCH_Gauss,.05,'unifcdf',0,1);
        U_NAGARCH_Gauss_mean = U_NAGARCH_Gauss-mean(U_NAGARCH_Gauss);
        U_NAGARCH_Gauss_mean2 = (U_NAGARCH_Gauss-mean(U_NAGARCH_Gauss)).^2;
        U_NAGARCH_Gauss_mean3 = (U_NAGARCH_Gauss-mean(U_NAGARCH_Gauss)).^3;
        U_NAGARCH_Gauss_mean4 = (U_NAGARCH_Gauss-mean(U_NAGARCH_Gauss)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 NAullhypothese wird akzeptiert
        [H_NAGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output_Gauss{j}.Params=parameters_NAGARCH_Gauss;
        Output_Gauss{j}.ParamsGARCH = parameters_NAGARCH_Gauss(1:end-nNAGARCH_Gauss(j)-const);
        Output_Gauss{j}.ParamsAR = parameters_NAGARCH_Gauss(size(Output_Gauss{j}.ParamsGARCH,1)+1:size(parameters_NAGARCH_Gauss,1));
        Output_Gauss{j}.LLF=LLF_NAGARCH_Gauss;
        Output_Gauss{j}.stderrors=stderrors_NAGARCH_Gauss;
        Output_Gauss{j}.robustSE=robustSE_NAGARCH_Gauss;
        Output_Gauss{j}.ht=ht_NAGARCH_Gauss;
        Output_Gauss{j}.Scores=scores_NAGARCH_Gauss;
        Output_Gauss{j}.Innovations=resid_NAGARCH_Gauss;
        Output_Gauss{j}.GARCH = 'NAGARCH';
        Output_Gauss{j}.U = U_NAGARCH_Gauss;
        Output_Gauss{j}.Tstat = Tstatistic_NAGARCH_Gauss;
        Output_Gauss{j}.garchtype = 5;
        Output_Gauss{j}.leverage = 1;
        Output_Gauss{j}.errortype = 1;
        Output_Gauss{j}.arlag = nNAGARCH_Gauss(j);
        Output_Gauss{j}.likelihoods = likelihood_NAGARCH_Gauss;
        Output_Gauss{j}.EXITFLAG = EXITFLAG;
        Output_Gauss{j}.data = data(:,j);
        Output_Gauss{j}.dist = 'GAUSS';
        Output_Gauss{j}.Innovations = resid_NAGARCH_Gauss;
        Output_Gauss{j}.BIC = BIC_NAGARCH_Gauss;
        Output_Gauss{j}.stdresid = stdresid;
        Output_Gauss{j}.statKolmResid = stat_Kolmogorov_resid;
        Output_Gauss{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output_Gauss{j}.HKolmResid = H_Kolmogorov_resid;
        Output_Gauss{j}.statKolmU= stat_Kolmogorov_U;
        Output_Gauss{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output_Gauss{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output_Gauss{j}.const = 1;
        else
            Output_Gauss{j}.const = 0;
        end

    elseif BIC_opt_Gauss{j}==8
        [parameters_APGARCH_Gauss, LLF_APGARCH_Gauss, stderrors_APGARCH_Gauss, robustSE_APGARCH_Gauss, ht_APGARCH_Gauss, scores_APGARCH_Gauss, resid_APGARCH_Gauss, likelihood_APGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'APGARCH','NORMAL',nAPGARCH_Gauss(j),const);
        BIC_APGARCH_Gauss = 2*LLF_APGARCH_Gauss+log(t)*size(parameters_APGARCH_Gauss,1);
        Tstatistic_APGARCH_Gauss=parameters_APGARCH_Gauss./diag(robustSE_APGARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_APGARCH_Gauss./sqrt(ht_APGARCH_Gauss);
        result_stderrors_APGARCH_Gauss=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'norm_cdf');
        U_APGARCH_Gauss=normcdf(stdresid,0,1);
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_APGARCH_Gauss,.05,'unifcdf',0,1);
        U_APGARCH_Gauss_mean = U_APGARCH_Gauss-mean(U_APGARCH_Gauss);
        U_APGARCH_Gauss_mean2 = (U_APGARCH_Gauss-mean(U_APGARCH_Gauss)).^2;
        U_APGARCH_Gauss_mean3 = (U_APGARCH_Gauss-mean(U_APGARCH_Gauss)).^3;
        U_APGARCH_Gauss_mean4 = (U_APGARCH_Gauss-mean(U_APGARCH_Gauss)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 NAullhypothese wird akzeptiert
        [H_APGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output_Gauss{j}.Params=parameters_APGARCH_Gauss;
        Output_Gauss{j}.ParamsGARCH = parameters_APGARCH_Gauss(1:end-nAPGARCH_Gauss(j)-const);
        Output_Gauss{j}.ParamsAR = parameters_APGARCH_Gauss(size(Output_Gauss{j}.ParamsGARCH,1)+1:size(parameters_APGARCH_Gauss,1));
        Output_Gauss{j}.LLF=LLF_APGARCH_Gauss;
        Output_Gauss{j}.stderrors=stderrors_APGARCH_Gauss;
        Output_Gauss{j}.robustSE=robustSE_APGARCH_Gauss;
        Output_Gauss{j}.ht=ht_APGARCH_Gauss;
        Output_Gauss{j}.Scores=scores_APGARCH_Gauss;
        Output_Gauss{j}.Innovations=resid_APGARCH_Gauss;
        Output_Gauss{j}.GARCH = 'APGARCH';
        Output_Gauss{j}.U = U_APGARCH_Gauss;
        Output_Gauss{j}.Tstat = Tstatistic_APGARCH_Gauss;
        Output_Gauss{j}.garchtype = 6;
        Output_Gauss{j}.leverage = 2;
        Output_Gauss{j}.errortype = 1;
        Output_Gauss{j}.arlag = nAPGARCH_Gauss(j);
        Output_Gauss{j}.likelihoods = likelihood_APGARCH_Gauss;
        Output_Gauss{j}.EXITFLAG = EXITFLAG;
        Output_Gauss{j}.data = data(:,j);
        Output_Gauss{j}.dist = 'GAUSS';
        Output_Gauss{j}.Innovations = resid_APGARCH_Gauss;
        Output_Gauss{j}.BIC = BIC_APGARCH_Gauss;
        Output_Gauss{j}.stdresid = stdresid;
        Output_Gauss{j}.statKolmResid = stat_Kolmogorov_resid;
        Output_Gauss{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output_Gauss{j}.HKolmResid = H_Kolmogorov_resid;
        Output_Gauss{j}.statKolmU= stat_Kolmogorov_U;
        Output_Gauss{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output_Gauss{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output_Gauss{j}.const = 1;
        else
            Output_Gauss{j}.const = 0;
        end
    end
end

% für multivariate GARCH wähle optimales Modell aus allen Verteilungen aus
Output=cell(k,1);
for j=1:k

    if BIC_opt{j}==1
        [parameters_GARCH_Gauss, LLF_GARCH_Gauss, stderrors_GARCH_Gauss, robustSE_GARCH_Gauss, ht_GARCH_Gauss, scores_GARCH_Gauss, resid_GARCH_Gauss, likelihood_GARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'GARCH','NORMAL',nGARCH_Gauss(j),const);
        BIC_GARCH_Gauss = 2*LLF_GARCH_Gauss+ log(t)*size(parameters_GARCH_Gauss,1);
        Tstatistic_GARCH_Gauss=parameters_GARCH_Gauss./diag(robustSE_GARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_GARCH_Gauss./sqrt(ht_GARCH_Gauss);
        result_stderrors_GARCH_Gauss=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'norm_cdf');
        U_GARCH_Gauss=normcdf(stdresid,0,1);
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_GARCH_Gauss,.05,'unifcdf',0,1);
        U_GARCH_Gauss_mean = U_GARCH_Gauss-mean(U_GARCH_Gauss);
        U_GARCH_Gauss_mean2 = (U_GARCH_Gauss-mean(U_GARCH_Gauss)).^2;
        U_GARCH_Gauss_mean3 = (U_GARCH_Gauss-mean(U_GARCH_Gauss)).^3;
        U_GARCH_Gauss_mean4 = (U_GARCH_Gauss-mean(U_GARCH_Gauss)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_GARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_GARCH_Gauss;
        Output{j}.ParamsGARCH = parameters_GARCH_Gauss(1:end-nGARCH_Gauss(j)-const);
        Output{j}.ParamsAR = parameters_GARCH_Gauss(size(Output{j}.ParamsGARCH,1)+1:size(parameters_GARCH_Gauss,1));
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
        Output{j}.arlag = nGARCH_Gauss(j);
        Output{j}.likelihoods = likelihood_GARCH_Gauss;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GAUSS';
        Output{j}.Innovations = resid_GARCH_Gauss;
        Output{j}.BIC = BIC_GARCH_Gauss;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

        % % % 5.EGARCH-Gauss
    elseif BIC_opt{j}==2
        [parameters_EGARCH_Gauss, LLF_EGARCH_Gauss, stderrors_EGARCH_Gauss, robustSE_EGARCH_Gauss, ht_EGARCH_Gauss, scores_EGARCH_Gauss, resid_EGARCH_Gauss, likelihood_EGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'EGARCH','NORMAL',nEGARCH_Gauss(j),const);
        BIC_EGARCH_Gauss = 2*LLF_EGARCH_Gauss+log(t)*size(parameters_EGARCH_Gauss,1);
        Tstatistic_EGARCH_Gauss=parameters_EGARCH_Gauss./diag(robustSE_EGARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_EGARCH_Gauss./sqrt(ht_EGARCH_Gauss);
        result_stderrors_EGARCH_Gauss=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'norm_cdf');
        U_EGARCH_Gauss=normcdf(stdresid);
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_EGARCH_Gauss,.05,'unifcdf',0,1);
        U_EGARCH_Gauss_mean = U_EGARCH_Gauss-mean(U_EGARCH_Gauss);
        U_EGARCH_Gauss_mean2 = (U_EGARCH_Gauss-mean(U_EGARCH_Gauss)).^2;
        U_EGARCH_Gauss_mean3 = (U_EGARCH_Gauss-mean(U_EGARCH_Gauss)).^3;
        U_EGARCH_Gauss_mean4 = (U_EGARCH_Gauss-mean(U_EGARCH_Gauss)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_EGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_EGARCH_Gauss;
        Output{j}.ParamsGARCH = parameters_EGARCH_Gauss(1:end-nEGARCH_Gauss(j)-const);
        Output{j}.ParamsAR = parameters_EGARCH_Gauss(size(Output{j}.ParamsGARCH,1)+1:size(parameters_EGARCH_Gauss,1));
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
        Output{j}.arlag = nEGARCH_Gauss(j);
        Output{j}.likelihoods = likelihood_EGARCH_Gauss;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GAUSS';
        Output{j}.Innovations = resid_EGARCH_Gauss;
        Output{j}.BIC = BIC_EGARCH_Gauss;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end


        % 6.TGARCH-Gauss
    elseif BIC_opt{j}==3
        [parameters_TGARCH_Gauss, LLF_TGARCH_Gauss, stderrors_TGARCH_Gauss, robustSE_TGARCH_Gauss, ht_TGARCH_Gauss, scores_TGARCH_Gauss, resid_TGARCH_Gauss, likelihood_TGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'TGARCH','NORMAL',nTGARCH_Gauss(j),const);
        BIC_TGARCH_Gauss = 2*LLF_TGARCH_Gauss+log(t)*size(parameters_TGARCH_Gauss,1);
        Tstatistic_TGARCH_Gauss=parameters_TGARCH_Gauss./diag(robustSE_TGARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_TGARCH_Gauss./sqrt(ht_TGARCH_Gauss);
        result_stderrors_TGARCH_Gauss=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'norm_cdf');
        U_TGARCH_Gauss = normcdf(stdresid);
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_TGARCH_Gauss,.05,'unifcdf',0,1);
        U_TGARCH_Gauss_mean = U_TGARCH_Gauss-mean(U_TGARCH_Gauss);
        U_TGARCH_Gauss_mean2 = (U_TGARCH_Gauss-mean(U_TGARCH_Gauss)).^2;
        U_TGARCH_Gauss_mean3 = (U_TGARCH_Gauss-mean(U_TGARCH_Gauss)).^3;
        U_TGARCH_Gauss_mean4 = (U_TGARCH_Gauss-mean(U_TGARCH_Gauss)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_TGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_TGARCH_Gauss;
        Output{j}.ParamsGARCH = parameters_TGARCH_Gauss(1:end-nTGARCH_Gauss(j)-const);
        Output{j}.ParamsAR = parameters_TGARCH_Gauss(size(Output{j}.ParamsGARCH,1)+1:size(parameters_TGARCH_Gauss,1));
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
        Output{j}.arlag = nTGARCH_Gauss(j);
        Output{j}.likelihoods = likelihood_TGARCH_Gauss;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GAUSS';
        Output{j}.Innnovations = resid_TGARCH_Gauss;
        Output{j}.BIC = BIC_TGARCH_Gauss;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

        % 7.GJRGARCH-Gauss
    elseif BIC_opt{j}==4
        [parameters_GJRGARCH_Gauss, LLF_GJRGARCH_Gauss, stderrors_GJRGARCH_Gauss, robustSE_GJRGARCH_Gauss, ht_GJRGARCH_Gauss, scores_GJRGARCH_Gauss, resid_GJRGARCH_Gauss, likelihood_GJRGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'GJRGARCH','NORMAL',nGJRGARCH_Gauss(j),const);
        BIC_GJRGARCH_Gauss = 2*LLF_GJRGARCH_Gauss+log(t)*size(parameters_GJRGARCH_Gauss,1);
        Tstatistic_GJRGARCH_Gauss=parameters_GJRGARCH_Gauss./diag(robustSE_GJRGARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_GJRGARCH_Gauss./sqrt(ht_GJRGARCH_Gauss);
        result_stderrors_GJRGARCH_Gauss=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'norm_cdf');
        U_GJRGARCH_Gauss=normcdf(stdresid);
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_GJRGARCH_Gauss,.05,'unifcdf',0,1);
        U_GJRGARCH_Gauss_mean = U_GJRGARCH_Gauss-mean(U_GJRGARCH_Gauss);
        U_GJRGARCH_Gauss_mean2 = (U_GJRGARCH_Gauss-mean(U_GJRGARCH_Gauss)).^2;
        U_GJRGARCH_Gauss_mean3 = (U_GJRGARCH_Gauss-mean(U_GJRGARCH_Gauss)).^3;
        U_GJRGARCH_Gauss_mean4 = (U_GJRGARCH_Gauss-mean(U_GJRGARCH_Gauss)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_GJRGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_GJRGARCH_Gauss;
        Output{j}.ParamsGARCH = parameters_GJRGARCH_Gauss(1:end-nGJRGARCH_Gauss(j)-const);
        Output{j}.ParamsAR = parameters_GJRGARCH_Gauss(size(Output{j}.ParamsGARCH,1)+1:size(parameters_GJRGARCH_Gauss,1));
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
        Output{j}.arlag = nGJRGARCH_Gauss(j);
        Output{j}.likelihoods = likelihood_GJRGARCH_Gauss;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GAUSS';
        Output{j}.Innovations = resid_GJRGARCH_Gauss;
        Output{j}.BIC = BIC_GJRGARCH_Gauss;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

    elseif BIC_opt{j}==5
        [parameters_AVGARCH_Gauss, LLF_AVGARCH_Gauss, stderrors_AVGARCH_Gauss, robustSE_AVGARCH_Gauss, ht_AVGARCH_Gauss, scores_AVGARCH_Gauss, resid_AVGARCH_Gauss, likelihood_AVGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'AVGARCH','NORMAL',nAVGARCH_Gauss(j),const);
        BIC_AVGARCH_Gauss = 2*LLF_AVGARCH_Gauss+log(t)*size(parameters_AVGARCH_Gauss,1);
        Tstatistic_AVGARCH_Gauss=parameters_AVGARCH_Gauss./diag(robustSE_AVGARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_AVGARCH_Gauss./sqrt(ht_AVGARCH_Gauss);
        result_stderrors_AVGARCH_Gauss=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'norm_cdf');
        U_AVGARCH_Gauss=normcdf(stdresid);
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_AVGARCH_Gauss,.05,'unifcdf',0,1);
        U_AVGARCH_Gauss_mean = U_AVGARCH_Gauss-mean(U_AVGARCH_Gauss);
        U_AVGARCH_Gauss_mean2 = (U_AVGARCH_Gauss-mean(U_AVGARCH_Gauss)).^2;
        U_AVGARCH_Gauss_mean3 = (U_AVGARCH_Gauss-mean(U_AVGARCH_Gauss)).^3;
        U_AVGARCH_Gauss_mean4 = (U_AVGARCH_Gauss-mean(U_AVGARCH_Gauss)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_AVGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_AVGARCH_Gauss;
        Output{j}.ParamsGARCH = parameters_AVGARCH_Gauss(1:end-nAVGARCH_Gauss(j)-const);
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
        Output{j}.leverage = 1;
        Output{j}.errortype = 1;
        Output{j}.arlag = nAVGARCH_Gauss(j);
        Output{j}.likelihoods = likelihood_AVGARCH_Gauss;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GAUSS';
        Output{j}.Innovations = resid_AVGARCH_Gauss;
        Output{j}.BIC = BIC_AVGARCH_Gauss;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end


    elseif BIC_opt{j}==6
        [parameters_NGARCH_Gauss, LLF_NGARCH_Gauss, stderrors_NGARCH_Gauss, robustSE_NGARCH_Gauss, ht_NGARCH_Gauss, scores_NGARCH_Gauss, resid_NGARCH_Gauss, likelihood_NGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NGARCH','NORMAL',nNGARCH_Gauss(j),const);
        BIC_NGARCH_Gauss = 2*LLF_NGARCH_Gauss+log(t)*size(parameters_NGARCH_Gauss,1);
        Tstatistic_NGARCH_Gauss=parameters_NGARCH_Gauss./diag(robustSE_NGARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_NGARCH_Gauss./sqrt(ht_NGARCH_Gauss);
        result_stderrors_NGARCH_Gauss=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'norm_cdf');
        U_NGARCH_Gauss=normcdf(stdresid);
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_NGARCH_Gauss,.05,'unifcdf',0,1);
        U_NGARCH_Gauss_mean = U_NGARCH_Gauss-mean(U_NGARCH_Gauss);
        U_NGARCH_Gauss_mean2 = (U_NGARCH_Gauss-mean(U_NGARCH_Gauss)).^2;
        U_NGARCH_Gauss_mean3 = (U_NGARCH_Gauss-mean(U_NGARCH_Gauss)).^3;
        U_NGARCH_Gauss_mean4 = (U_NGARCH_Gauss-mean(U_NGARCH_Gauss)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_NGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_NGARCH_Gauss;
        Output{j}.ParamsGARCH = parameters_NGARCH_Gauss(1:end-nNGARCH_Gauss(j)-const);
        Output{j}.ParamsAR = parameters_NGARCH_Gauss(size(Output{j}.ParamsGARCH,1)+1:size(parameters_NGARCH_Gauss,1));
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
        Output{j}.arlag = nNGARCH_Gauss(j);
        Output{j}.likelihoods = likelihood_NGARCH_Gauss;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GAUSS';
        Output{j}.Innovations = resid_NGARCH_Gauss;
        Output{j}.BIC = BIC_NGARCH_Gauss;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

    elseif BIC_opt{j}==7
        [parameters_NAGARCH_Gauss, LLF_NAGARCH_Gauss, stderrors_NAGARCH_Gauss, robustSE_NAGARCH_Gauss, ht_NAGARCH_Gauss, scores_NAGARCH_Gauss, resid_NAGARCH_Gauss, likelihood_NAGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NAGARCH','NORMAL',nNAGARCH_Gauss(j),const);
        BIC_NAGARCH_Gauss = 2*LLF_NAGARCH_Gauss+log(t)*size(parameters_NAGARCH_Gauss,1);
        Tstatistic_NAGARCH_Gauss=parameters_NAGARCH_Gauss./diag(robustSE_NAGARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_NAGARCH_Gauss./sqrt(ht_NAGARCH_Gauss);
        result_stderrors_NAGARCH_Gauss=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'norm_cdf');
        U_NAGARCH_Gauss=normcdf(stdresid);
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_NAGARCH_Gauss,.05,'unifcdf',0,1);
        U_NAGARCH_Gauss_mean = U_NAGARCH_Gauss-mean(U_NAGARCH_Gauss);
        U_NAGARCH_Gauss_mean2 = (U_NAGARCH_Gauss-mean(U_NAGARCH_Gauss)).^2;
        U_NAGARCH_Gauss_mean3 = (U_NAGARCH_Gauss-mean(U_NAGARCH_Gauss)).^3;
        U_NAGARCH_Gauss_mean4 = (U_NAGARCH_Gauss-mean(U_NAGARCH_Gauss)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 NAullhypothese wird akzeptiert
        [H_NAGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_NAGARCH_Gauss;
        Output{j}.ParamsGARCH = parameters_NAGARCH_Gauss(1:end-nNAGARCH_Gauss(j)-const);
        Output{j}.ParamsAR = parameters_NAGARCH_Gauss(size(Output{j}.ParamsGARCH,1)+1:size(parameters_NAGARCH_Gauss,1));
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
        Output{j}.arlag = nNAGARCH_Gauss(j);
        Output{j}.likelihoods = likelihood_NAGARCH_Gauss;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GAUSS';
        Output{j}.Innovations = resid_NAGARCH_Gauss;
        Output{j}.BIC = BIC_NAGARCH_Gauss;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

    elseif BIC_opt{j}==8
        [parameters_APGARCH_Gauss, LLF_APGARCH_Gauss, stderrors_APGARCH_Gauss, robustSE_APGARCH_Gauss, ht_APGARCH_Gauss, scores_APGARCH_Gauss, resid_APGARCH_Gauss, likelihood_APGARCH_Gauss, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'APGARCH','NORMAL',nAPGARCH_Gauss(j),const);
        BIC_APGARCH_Gauss = 2*LLF_APGARCH_Gauss+log(t)*size(parameters_APGARCH_Gauss,1);
        Tstatistic_APGARCH_Gauss=parameters_APGARCH_Gauss./diag(robustSE_APGARCH_Gauss).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_APGARCH_Gauss./sqrt(ht_APGARCH_Gauss);
        result_stderrors_APGARCH_Gauss=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der Gauss; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'norm_cdf');
        U_APGARCH_Gauss=normcdf(stdresid);
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_APGARCH_Gauss,.05,'unifcdf',0,1);
        U_APGARCH_Gauss_mean = U_APGARCH_Gauss-mean(U_APGARCH_Gauss);
        U_APGARCH_Gauss_mean2 = (U_APGARCH_Gauss-mean(U_APGARCH_Gauss)).^2;
        U_APGARCH_Gauss_mean3 = (U_APGARCH_Gauss-mean(U_APGARCH_Gauss)).^3;
        U_APGARCH_Gauss_mean4 = (U_APGARCH_Gauss-mean(U_APGARCH_Gauss)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 NAullhypothese wird akzeptiert
        [H_APGARCH_Gauss_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_APGARCH_Gauss;
        Output{j}.ParamsGARCH = parameters_APGARCH_Gauss(1:end-nAPGARCH_Gauss(j)-const);
        Output{j}.ParamsAR = parameters_APGARCH_Gauss(size(Output{j}.ParamsGARCH,1)+1:size(parameters_APGARCH_Gauss,1));
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
        Output{j}.arlag = nAPGARCH_Gauss(j);
        Output{j}.likelihoods = likelihood_APGARCH_Gauss;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GAUSS';
        Output{j}.Innovations = resid_APGARCH_Gauss;
        Output{j}.BIC = BIC_APGARCH_Gauss;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

        % Verteilung STUDENTST
        % 1.GARCH-STUDENTST
    elseif BIC_opt{j}==9
        [parameters_GARCH_STUDENTST, LLF_GARCH_STUDENTST, stderrors_GARCH_STUDENTST, robustSE_GARCH_STUDENTST, ht_GARCH_STUDENTST, scores_GARCH_STUDENTST, resid_GARCH_STUDENTST, likelihood_GARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'GARCH','STUDENTST',nGARCH_STUDENTST(j),const);
        BIC_GARCH_STUDENTST = 2*LLF_GARCH_STUDENTST+log(t)*size(parameters_GARCH_STUDENTST,1);
        Tstatistic_GARCH_STUDENTST=parameters_GARCH_STUDENTST./diag(robustSE_GARCH_STUDENTST).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_GARCH_STUDENTST./sqrt(ht_GARCH_STUDENTST);
        result_stderrors_GARCH_STUDENTST=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der STUDENTST; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'tdis_cdf',parameters_GARCH_STUDENTST(end));
        U_GARCH_STUDENTST = tcdf(stdresid,parameters_GARCH_STUDENTST(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_GARCH_STUDENTST,.05,'unifcdf',0,1);
        U_GARCH_STUDENTST_mean = U_GARCH_STUDENTST-mean(U_GARCH_STUDENTST);
        U_GARCH_STUDENTST_mean2 = (U_GARCH_STUDENTST-mean(U_GARCH_STUDENTST)).^2;
        U_GARCH_STUDENTST_mean3 = (U_GARCH_STUDENTST-mean(U_GARCH_STUDENTST)).^3;
        U_GARCH_STUDENTST_mean4 = (U_GARCH_STUDENTST-mean(U_GARCH_STUDENTST)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_GARCH_STUDENTST_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_GARCH_STUDENTST;
        Output{j}.ParamsGARCH = [parameters_GARCH_STUDENTST(1:end-nGARCH_STUDENTST(j)-const-1); parameters_GARCH_STUDENTST(end)];
        Output{j}.ParamsAR = parameters_GARCH_STUDENTST(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_GARCH_STUDENTST,1)-1);
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
        Output{j}.arlag = nGARCH_STUDENTST(j);
        Output{j}.likelihoods = likelihood_GARCH_STUDENTST;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'STUDENTST';
        Output{j}.Innovations = resid_GARCH_STUDENTST;
        Output{j}.BIC = BIC_GARCH_STUDENTST;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

        %

        % % % 5.EGARCH-STUDENTST
    elseif BIC_opt{j}==10
        [parameters_EGARCH_STUDENTST, LLF_EGARCH_STUDENTST, stderrors_EGARCH_STUDENTST, robustSE_EGARCH_STUDENTST, ht_EGARCH_STUDENTST, scores_EGARCH_STUDENTST, resid_EGARCH_STUDENTST, likelihood_EGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'EGARCH','STUDENTST',nEGARCH_STUDENTST(j),const);
        BIC_EGARCH_STUDENTST = 2*LLF_EGARCH_STUDENTST+log(t)*size(parameters_EGARCH_STUDENTST,1);
        Tstatistic_EGARCH_STUDENTST=parameters_EGARCH_STUDENTST./diag(robustSE_EGARCH_STUDENTST).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_EGARCH_STUDENTST./sqrt(ht_EGARCH_STUDENTST);
        result_stderrors_EGARCH_STUDENTST=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der STUDENTST; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'tdis_cdf',parameters_EGARCH_STUDENTST(end));
        U_EGARCH_STUDENTST = tcdf(stdresid,parameters_EGARCH_STUDENTST(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_EGARCH_STUDENTST,.05,'unifcdf',0,1);
        U_EGARCH_STUDENTST_mean = U_EGARCH_STUDENTST-mean(U_EGARCH_STUDENTST);
        U_EGARCH_STUDENTST_mean2 = (U_EGARCH_STUDENTST-mean(U_EGARCH_STUDENTST)).^2;
        U_EGARCH_STUDENTST_mean3 = (U_EGARCH_STUDENTST-mean(U_EGARCH_STUDENTST)).^3;
        U_EGARCH_STUDENTST_mean4 = (U_EGARCH_STUDENTST-mean(U_EGARCH_STUDENTST)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_EGARCH_STUDENTST_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_EGARCH_STUDENTST;
        Output{j}.ParamsGARCH = [parameters_EGARCH_STUDENTST(1:end-nEGARCH_STUDENTST(j)-const-1); parameters_EGARCH_STUDENTST(end)];
        Output{j}.ParamsAR = parameters_EGARCH_STUDENTST(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_EGARCH_STUDENTST,1)-1);
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
        Output{j}.arlag = nEGARCH_STUDENTST(j);
        Output{j}.likelihoods = likelihood_EGARCH_STUDENTST;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'STUDENTST';
        Output{j}.Innovations = resid_EGARCH_STUDENTST;
        Output{j}.BIC = BIC_EGARCH_STUDENTST;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

        % 6.TGARCH-STUDENTST
    elseif BIC_opt{j}==11
        [parameters_TGARCH_STUDENTST, LLF_TGARCH_STUDENTST, stderrors_TGARCH_STUDENTST, robustSE_TGARCH_STUDENTST, ht_TGARCH_STUDENTST, scores_TGARCH_STUDENTST, resid_TGARCH_STUDENTST, likelihood_TGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'TGARCH','STUDENTST',nTGARCH_STUDENTST(j),const);
        BIC_TGARCH_STUDENTST = 2*LLF_TGARCH_STUDENTST+log(t)*size(parameters_TGARCH_STUDENTST,1);
        Tstatistic_TGARCH_STUDENTST=parameters_TGARCH_STUDENTST./diag(robustSE_TGARCH_STUDENTST).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_TGARCH_STUDENTST./sqrt(ht_TGARCH_STUDENTST);
        result_stderrors_TGARCH_STUDENTST=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der STUDENTST; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'tdis_cdf',parameters_TGARCH_STUDENTST(end));
        U_TGARCH_STUDENTST = tcdf(stdresid,parameters_TGARCH_STUDENTST(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_TGARCH_STUDENTST,.05,'unifcdf',0,1);
        U_TGARCH_STUDENTST_mean = U_TGARCH_STUDENTST-mean(U_TGARCH_STUDENTST);
        U_TGARCH_STUDENTST_mean2 = (U_TGARCH_STUDENTST-mean(U_TGARCH_STUDENTST)).^2;
        U_TGARCH_STUDENTST_mean3 = (U_TGARCH_STUDENTST-mean(U_TGARCH_STUDENTST)).^3;
        U_TGARCH_STUDENTST_mean4 = (U_TGARCH_STUDENTST-mean(U_TGARCH_STUDENTST)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_TGARCH_STUDENTST_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_TGARCH_STUDENTST;
        Output{j}.ParamsGARCH = [parameters_TGARCH_STUDENTST(1:end-nTGARCH_STUDENTST(j)-const-1); parameters_TGARCH_STUDENTST(end)];
        Output{j}.ParamsAR = parameters_TGARCH_STUDENTST(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_TGARCH_STUDENTST,1)-1);
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
        Output{j}.arlag = nTGARCH_STUDENTST(j);
        Output{j}.likelihoods = likelihood_TGARCH_STUDENTST;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'STUDENTST';
        Output{j}.Innovations = resid_TGARCH_STUDENTST;
        Output{j}.BIC = BIC_TGARCH_STUDENTST;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

        % 7.GJRGARCH-STUDENTST
    elseif BIC_opt{j}==12
        [parameters_GJRGARCH_STUDENTST, LLF_GJRGARCH_STUDENTST, stderrors_GJRGARCH_STUDENTST, robustSE_GJRGARCH_STUDENTST, ht_GJRGARCH_STUDENTST, scores_GJRGARCH_STUDENTST, resid_GJRGARCH_STUDENTST,likelihood_GJRGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'GJRGARCH','STUDENTST',nGJRGARCH_STUDENTST(j),const);
        BIC_GJRGARCH_STUDENTST = 2*LLF_GJRGARCH_STUDENTST+log(t)*size(parameters_GJRGARCH_STUDENTST,1);
        Tstatistic_GJRGARCH_STUDENTST=parameters_GJRGARCH_STUDENTST./diag(robustSE_GJRGARCH_STUDENTST).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_GJRGARCH_STUDENTST./sqrt(ht_GJRGARCH_STUDENTST);
        result_stderrors_GJRGARCH_STUDENTST=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der STUDENTST; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'tdis_cdf',parameters_GJRGARCH_STUDENTST(end));
        U_GJRGARCH_STUDENTST=tcdf(stdresid,parameters_GJRGARCH_STUDENTST(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_GJRGARCH_STUDENTST,.05,'unifcdf',0,1);
        U_GJRGARCH_STUDENTST_mean = U_GJRGARCH_STUDENTST-mean(U_GJRGARCH_STUDENTST);
        U_GJRGARCH_STUDENTST_mean2 = (U_GJRGARCH_STUDENTST-mean(U_GJRGARCH_STUDENTST)).^2;
        U_GJRGARCH_STUDENTST_mean3 = (U_GJRGARCH_STUDENTST-mean(U_GJRGARCH_STUDENTST)).^3;
        U_GJRGARCH_STUDENTST_mean4 = (U_GJRGARCH_STUDENTST-mean(U_GJRGARCH_STUDENTST)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_GJRGARCH_STUDENTST_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_GJRGARCH_STUDENTST;
        Output{j}.ParamsGARCH = [parameters_GJRGARCH_STUDENTST(1:end-nGJRGARCH_STUDENTST(j)-const-1); parameters_GJRGARCH_STUDENTST(end)];
        Output{j}.ParamsAR = parameters_GJRGARCH_STUDENTST(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_GJRGARCH_STUDENTST,1)-1);
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
        Output{j}.arlag = nGJRGARCH_STUDENTST(j);
        Outpur{j}.likelihoods = likelihood_GJRGARCH_STUDENTST;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.datat = data(:,j);
        Output{j}.BIC =  2*LLF_GJRGARCH_STUDENTST+ log(t)*size(parameters_GJRGARCH_STUDENTST,1);
        Output{j}.Innnovations = resid_GJRGARCH_STUDENTST;
        Output{j}.BIC = BIC_GJRGARCH_STUDENTST;
        Output{j}.dist = 'STUDENTST';
        Output{j}.Innovations = resid_GJRGARCH_STUDENTST;
        Output{j}.BIC = BIC_GJRGARCH_STUDENTST;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
                % 8.AVGARCH-STUDENTST
    elseif BIC_opt{j}==13
        [parameters_AVGARCH_STUDENTST, LLF_AVGARCH_STUDENTST, stderrors_AVGARCH_STUDENTST, robustSE_AVGARCH_STUDENTST, ht_AVGARCH_STUDENTST, scores_AVGARCH_STUDENTST, resid_AVGARCH_STUDENTST, likelihood_AVGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'AVGARCH','STUDENTST',nAVGARCH_STUDENTST(j),const);
        BIC_AVGARCH_STUDENTST = 2*LLF_AVGARCH_STUDENTST+log(t)*size(parameters_AVGARCH_STUDENTST,1);
        Tstatistic_AVGARCH_STUDENTST=parameters_AVGARCH_STUDENTST./diag(robustSE_AVGARCH_STUDENTST).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_AVGARCH_STUDENTST./sqrt(ht_AVGARCH_STUDENTST);
        result_stderrors_AVGARCH_STUDENTST=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der STUDENTST; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'tdis_cdf',parameters_AVGARCH_STUDENTST(end));
        U_AVGARCH_STUDENTST=zeros(size(stdresid,1),1);
        U_AVGARCH_STUDENTST=tcdf(stdresid,parameters_AVGARCH_STUDENTST(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_AVGARCH_STUDENTST,.05,'unifcdf',0,1);
        U_AVGARCH_STUDENTST_mean = U_AVGARCH_STUDENTST-mean(U_AVGARCH_STUDENTST);
        U_AVGARCH_STUDENTST_mean2 = (U_AVGARCH_STUDENTST-mean(U_AVGARCH_STUDENTST)).^2;
        U_AVGARCH_STUDENTST_mean3 = (U_AVGARCH_STUDENTST-mean(U_AVGARCH_STUDENTST)).^3;
        U_AVGARCH_STUDENTST_mean4 = (U_AVGARCH_STUDENTST-mean(U_AVGARCH_STUDENTST)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_AVGARCH_STUDENTST_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_AVGARCH_STUDENTST;
        Output{j}.ParamsGARCH = [parameters_AVGARCH_STUDENTST(1:end-nAVGARCH_STUDENTST(j)-const-1); parameters_AVGARCH_STUDENTST(end)];
        Output{j}.ParamsAR = parameters_AVGARCH_STUDENTST(size(Output{j}.ParamsGARCH,1):size(parameters_AVGARCH_STUDENTST,1)-1);
        Output{j}.LLF=LLF_AVGARCH_STUDENTST;
        Output{j}.stderrors=stderrors_AVGARCH_STUDENTST;
        Output{j}.robustSE=robustSE_AVGARCH_STUDENTST;
        Output{j}.ht=ht_AVGARCH_STUDENTST;
        Output{j}.Scores=scores_AVGARCH_STUDENTST;
        Output{j}.Innovations=resid_AVGARCH_STUDENTST;
        Output{j}.GARCH = 'AVGARCH';
        Output{j}.U = U_AVGARCH_STUDENTST(j);
        Output{j}.Tstat = Tstatistic_AVGARCH_STUDENTST;
        Output{j}.garchtype = 3;
        Output{j}.leverage = 1;
        Output{j}.errortype = 2;
        Output{j}.arlag = nAVGARCH_STUDENTST;
        Output{j}.likelihoods = likelihood_AVGARCH_STUDENTST;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'STUDENTST';
        Output{j}.Innovations = resid_AVGARCH_STUDENTST;
        Output{j}.BIC = BIC_AVGARCH_STUDENTST;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end


    elseif BIC_opt{j}==14
        [parameters_NGARCH_STUDENTST, LLF_NGARCH_STUDENTST, stderrors_NGARCH_STUDENTST, robustSE_NGARCH_STUDENTST, ht_NGARCH_STUDENTST, scores_NGARCH_STUDENTST, resid_NGARCH_STUDENTST, likelihood_NGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NGARCH','STUDENTST',nNGARCH_STUDENTST(j),const);
        BIC_NGARCH_STUDENTST = 2*LLF_NGARCH_STUDENTST+log(t)*size(parameters_NGARCH_STUDENTST,1);
        Tstatistic_NGARCH_STUDENTST=parameters_NGARCH_STUDENTST./diag(robustSE_NGARCH_STUDENTST).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_NGARCH_STUDENTST./sqrt(ht_NGARCH_STUDENTST);
        result_stderrors_NGARCH_STUDENTST=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der STUDENTST; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'tdis_cdf',parameters_NGARCH_STUDENTST(end));
        U_NGARCH_STUDENTST=tcdf(stdresid,parameters_NGARCH_STUDENTST(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_NGARCH_STUDENTST,.05,'unifcdf',0,1);
        U_NGARCH_STUDENTST_mean = U_NGARCH_STUDENTST-mean(U_NGARCH_STUDENTST);
        U_NGARCH_STUDENTST_mean2 = (U_NGARCH_STUDENTST-mean(U_NGARCH_STUDENTST)).^2;
        U_NGARCH_STUDENTST_mean3 = (U_NGARCH_STUDENTST-mean(U_NGARCH_STUDENTST)).^3;
        U_NGARCH_STUDENTST_mean4 = (U_NGARCH_STUDENTST-mean(U_NGARCH_STUDENTST)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_NGARCH_STUDENTST_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_NGARCH_STUDENTST;
        Output{j}.ParamsGARCH = [parameters_NGARCH_STUDENTST(1:end-nNGARCH_STUDENTST(j)-const-1); parameters_NGARCH_STUDENTST(end)];
        Output{j}.ParamsAR = parameters_NGARCH_STUDENTST(size(Output{j}.ParamsGARCH,1):size(parameters_NGARCH_STUDENTST,1)-1);
        Output{j}.LLF=LLF_NGARCH_STUDENTST;
        Output{j}.stderrors=stderrors_NGARCH_STUDENTST;
        Output{j}.robustSE=robustSE_NGARCH_STUDENTST;
        Output{j}.ht=ht_NGARCH_STUDENTST;
        Output{j}.Scores=scores_NGARCH_STUDENTST;
        Output{j}.Innovations=resid_NGARCH_STUDENTST;
        Output{j}.GARCH = 'NGARCH';
        Output{j}.U = U_NGARCH_STUDENTST(j);
        Output{j}.Tstat = Tstatistic_NGARCH_STUDENTST;
        Output{j}.garchtype = 4;
        Output{j}.leverage = 1;
        Output{j}.errortype = 2;
        Output{j}.arlag = nNGARCH_STUDENTST;
        Output{j}.likelihoods = likelihood_NGARCH_STUDENTST;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'STUDENTST';
        Output{j}.Innovations = resid_NGARCH_STUDENTST;
        Output{j}.BIC = BIC_NGARCH_STUDENTST;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

    elseif BIC_opt{j}==15
        [parameters_NAGARCH_STUDENTST, LLF_NAGARCH_STUDENTST, stderrors_NAGARCH_STUDENTST, robustSE_NAGARCH_STUDENTST, ht_NAGARCH_STUDENTST, scores_NAGARCH_STUDENTST, resid_NAGARCH_STUDENTST, likelihood_NAGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NAGARCH','STUDENTST',nNAGARCH_STUDENTST(j),const);
        BIC_NAGARCH_STUDENTST = 2*LLF_NAGARCH_STUDENTST+log(t)*size(parameters_NAGARCH_STUDENTST,1);
        Tstatistic_NAGARCH_STUDENTST=parameters_NAGARCH_STUDENTST./diag(robustSE_NAGARCH_STUDENTST).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_NAGARCH_STUDENTST./sqrt(ht_NAGARCH_STUDENTST);
        result_stderrors_NAGARCH_STUDENTST=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der STUDENTST; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'tdis_cdf',parameters_NAGARCH_STUDENTST(end));
        U_NAGARCH_STUDENTST=tcdf(stdresid,parameters_NAGARCH_STUDENTST(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_NAGARCH_STUDENTST,.05,'unifcdf',0,1);
        U_NAGARCH_STUDENTST_mean = U_NAGARCH_STUDENTST-mean(U_NAGARCH_STUDENTST);
        U_NAGARCH_STUDENTST_mean2 = (U_NAGARCH_STUDENTST-mean(U_NAGARCH_STUDENTST)).^2;
        U_NAGARCH_STUDENTST_mean3 = (U_NAGARCH_STUDENTST-mean(U_NAGARCH_STUDENTST)).^3;
        U_NAGARCH_STUDENTST_mean4 = (U_NAGARCH_STUDENTST-mean(U_NAGARCH_STUDENTST)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 NAullhypothese wird akzeptiert
        [H_NAGARCH_STUDENTST_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_NAGARCH_STUDENTST;
        Output{j}.ParamsGARCH = [parameters_NAGARCH_STUDENTST(1:end-nNAGARCH_STUDENTST(j)-const-1); parameters_NAGARCH_STUDENTST(end)];
        Output{j}.ParamsAR = parameters_NAGARCH_STUDENTST(size(Output{j}.ParamsGARCH,1):size(parameters_NAGARCH_STUDENTST,1)-1);
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
        Output{j}.arlag = nNAGARCH_STUDENTST(j);
        Output{j}.likelihoods = likelihood_NAGARCH_STUDENTST;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'STUDENTST';
        Output{j}.Innovations = resid_NAGARCH_STUDENTST;
        Output{j}.BIC = BIC_NAGARCH_STUDENTST;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

    elseif BIC_opt{j}==16
        [parameters_APGARCH_STUDENTST, LLF_APGARCH_STUDENTST, stderrors_APGARCH_STUDENTST, robustSE_APGARCH_STUDENTST, ht_APGARCH_STUDENTST, scores_APGARCH_STUDENTST, resid_APGARCH_STUDENTST, likelihood_APGARCH_STUDENTST, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'APGARCH','STUDENTST',nAPGARCH_STUDENTST(j),const);
        BIC_APGARCH_STUDENTST = 2*LLF_APGARCH_STUDENTST+log(t)*size(parameters_APGARCH_STUDENTST,1);
        Tstatistic_APGARCH_STUDENTST=parameters_APGARCH_STUDENTST./diag(robustSE_APGARCH_STUDENTST).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_APGARCH_STUDENTST./sqrt(ht_APGARCH_STUDENTST);
        result_stderrors_APGARCH_STUDENTST=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der STUDENTST; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'tdis_cdf',parameters_APGARCH_STUDENTST(end));
        U_APGARCH_STUDENTST=tcdf(stdresid,parameters_APGARCH_STUDENTST(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_APGARCH_STUDENTST,.05,'unifcdf',0,1);
        U_APGARCH_STUDENTST_mean = U_APGARCH_STUDENTST-mean(U_APGARCH_STUDENTST);
        U_APGARCH_STUDENTST_mean2 = (U_APGARCH_STUDENTST-mean(U_APGARCH_STUDENTST)).^2;
        U_APGARCH_STUDENTST_mean3 = (U_APGARCH_STUDENTST-mean(U_APGARCH_STUDENTST)).^3;
        U_APGARCH_STUDENTST_mean4 = (U_APGARCH_STUDENTST-mean(U_APGARCH_STUDENTST)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 NAullhypothese wird akzeptiert
        [H_APGARCH_STUDENTST_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_APGARCH_STUDENTST;
        Output{j}.ParamsGARCH = [parameters_APGARCH_STUDENTST(1:end-nAPGARCH_STUDENTST(j)-const-1); parameters_APGARCH_STUDENTST(end)];
        Output{j}.ParamsAR = parameters_APGARCH_STUDENTST(size(Output{j}.ParamsGARCH,1):size(parameters_APGARCH_STUDENTST,1)-1);
        Output{j}.LLF=LLF_APGARCH_STUDENTST;
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
        Output{j}.arlag = nAPGARCH_STUDENTST(j);
        Output{j}.likelihoods = likelihood_APGARCH_STUDENTST;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'STUDENTST';
        Output{j}.Innovations = resid_APGARCH_STUDENTST;
        Output{j}.BIC = BIC_APGARCH_STUDENTST;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end


        % Verteilung GED
    elseif BIC_opt{j}==17
        % 1.GARCH-GED
        [parameters_GARCH_GED, LLF_GARCH_GED, stderrors_GARCH_GED, robustSE_GARCH_GED, ht_GARCH_GED, scores_GARCH_GED, resid_GARCH_GED, likelihood_GARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'GARCH','GED',nGARCH_GED(j),const);
        BIC_GARCH_GED = 2*LLF_GARCH_GED+log(t)*size(parameters_GARCH_GED,1);
        Tstatistic_GARCH_GED=parameters_GARCH_GED./diag(robustSE_GARCH_GED).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_GARCH_GED./sqrt(ht_GARCH_GED);
        result_stderrors_GARCH_GED=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der GED; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'gedcdf',parameters_GARCH_GED(end));
        U_GARCH_GED= gedcdf(stdresid,parameters_GARCH_GED(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_GARCH_GED,.05,'unifcdf',0,1);
        U_GARCH_GED_mean = U_GARCH_GED-mean(U_GARCH_GED);
        U_GARCH_GED_mean2 = (U_GARCH_GED-mean(U_GARCH_GED)).^2;
        U_GARCH_GED_mean3 = (U_GARCH_GED-mean(U_GARCH_GED)).^3;
        U_GARCH_GED_mean4 = (U_GARCH_GED-mean(U_GARCH_GED)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_GARCH_GED_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_GARCH_GED;
        Output{j}.ParamsGARCH = [parameters_GARCH_GED(1:end-nGARCH_GED(j)-const-1); parameters_GARCH_GED(end)];
        Output{j}.ParamsAR = parameters_GARCH_GED(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_GARCH_GED,1)-1);
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
        Output{j}.arlag = nGARCH_GED(j);
        Output{j}.likelihoods = likelihood_GARCH_GED;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GED';
        Output{j}.Innovations = resid_GARCH_GED;
        Output{j}.BIC = BIC_GARCH_GED;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

        % % % 5.EGARCH-GED
    elseif BIC_opt{j}==18
        [parameters_EGARCH_GED, LLF_EGARCH_GED, stderrors_EGARCH_GED, robustSE_EGARCH_GED, ht_EGARCH_GED, scores_EGARCH_GED, resid_EGARCH_GED, likelihood_EGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'EGARCH','GED',nEGARCH_GED(j),const);
        BIC_EGARCH_GED = 2*LLF_EGARCH_GED+log(t)*size(parameters_EGARCH_GED,1);
        Tstatistic_EGARCH_GED=parameters_EGARCH_GED./diag(robustSE_EGARCH_GED).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_EGARCH_GED./sqrt(ht_EGARCH_GED);
        result_stderrors_EGARCH_GED=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der GED; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'gedcdf',parameters_EGARCH_GED(end));
        U_EGARCH_GED = gedcdf(stdresid,parameters_EGARCH_GED(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_EGARCH_GED,.05,'unifcdf',0,1);
        U_EGARCH_GED_mean = U_EGARCH_GED-mean(U_EGARCH_GED);
        U_EGARCH_GED_mean2 = (U_EGARCH_GED-mean(U_EGARCH_GED)).^2;
        U_EGARCH_GED_mean3 = (U_EGARCH_GED-mean(U_EGARCH_GED)).^3;
        U_EGARCH_GED_mean4 = (U_EGARCH_GED-mean(U_EGARCH_GED)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_EGARCH_GED_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_EGARCH_GED;
        Output{j}.ParamsGARCH = [parameters_EGARCH_GED(1:end-nEGARCH_GED(j)-const-1); parameters_EGARCH_GED(end)];
        Output{j}.ParamsAR = parameters_EGARCH_GED(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_EGARCH_GED,1)-1);
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
        Output{j}.arlag = nEGARCH_GED(j);
        Output{j}.likelihoods = likelihood_EGARCH_GED;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GED';
        Output{j}.Innovations = resid_EGARCH_GED;
        Output{j}.BIC = BIC_EGARCH_GED;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end


        % 6.TGARCH-GED
    elseif BIC_opt{j}==19
        [parameters_TGARCH_GED, LLF_TGARCH_GED, stderrors_TGARCH_GED, robustSE_TGARCH_GED, ht_TGARCH_GED, scores_TGARCH_GED, resid_TGARCH_GED, likelihood_TGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'TGARCH','GED',nTGARCH_GED(j),const);
        BIC_TGARCH_GED = 2*LLF_TGARCH_GED+log(t)*size(parameters_TGARCH_GED,1);
        Tstatistic_TGARCH_GED=parameters_TGARCH_GED./diag(robustSE_TGARCH_GED).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_TGARCH_GED./sqrt(ht_TGARCH_GED);
        result_stderrors_TGARCH_GED=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der GED; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'gedcdf',parameters_TGARCH_GED(end));
        U_TGARCH_GED= gedcdf(stdresid,parameters_TGARCH_GED(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_TGARCH_GED,.05,'unifcdf',0,1);
        U_TGARCH_GED_mean = U_TGARCH_GED-mean(U_TGARCH_GED);
        U_TGARCH_GED_mean2 = (U_TGARCH_GED-mean(U_TGARCH_GED)).^2;
        U_TGARCH_GED_mean3 = (U_TGARCH_GED-mean(U_TGARCH_GED)).^3;
        U_TGARCH_GED_mean4 = (U_TGARCH_GED-mean(U_TGARCH_GED)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_TGARCH_GED_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_TGARCH_GED;
        Output{j}.ParamsGARCH = [parameters_TGARCH_GED(1:end-nTGARCH_GED(j)-const-1); parameters_TGARCH_GED(end)];
        Output{j}.ParamsAR = parameters_TGARCH_GED(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_TGARCH_GED,1)-1);
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
        Output{j}.arlag = nTGARCH_GED(j);
        Output{j}.likelihoods = likelihood_TGARCH_GED;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GED';
        Output{j}.Innovations = resid_TGARCH_GED;
        Output{j}.BIC = BIC_TGARCH_GED;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

        % 7.GJRGARCH-GED
    elseif BIC_opt{j}==20
        [parameters_GJRGARCH_GED, LLF_GJRGARCH_GED, stderrors_GJRGARCH_GED, robustSE_GJRGARCH_GED, ht_GJRGARCH_GED, scores_GJRGARCH_GED, resid_GJRGARCH_GED, likelihood_GJRGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'GJRGARCH','GED',nGJRGARCH_GED(j),const);
        BIC_GJRGARCH_GED = 2*LLF_GJRGARCH_GED+log(t)*size(parameters_GJRGARCH_GED,1);
        Tstatistic_GJRGARCH_GED=parameters_GJRGARCH_GED./diag(robustSE_GJRGARCH_GED).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_GJRGARCH_GED./sqrt(ht_GJRGARCH_GED);
        result_stderrors_GJRGARCH_GED=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der GED; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'gedcdf',parameters_GJRGARCH_GED(end));
        U_GJRGARCH_GED=gedcdf(stdresid,parameters_GJRGARCH_GED(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_GJRGARCH_GED,.05,'unifcdf',0,1);
        U_GJRGARCH_GED_mean = U_GJRGARCH_GED-mean(U_GJRGARCH_GED);
        U_GJRGARCH_GED_mean2 = (U_GJRGARCH_GED-mean(U_GJRGARCH_GED)).^2;
        U_GJRGARCH_GED_mean3 = (U_GJRGARCH_GED-mean(U_GJRGARCH_GED)).^3;
        U_GJRGARCH_GED_mean4 = (U_GJRGARCH_GED-mean(U_GJRGARCH_GED)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_GJRGARCH_GED_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_GJRGARCH_GED;
        Output{j}.ParamsGARCH = [parameters_GJRGARCH_GED(1:end-nGJRGARCH_GED(j)-const-1); parameters_GJRGARCH_GED(end)];
        Output{j}.ParamsAR = parameters_GJRGARCH_GED(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_GJRGARCH_GED,1)-1);
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
        Output{j}.arlag = nGJRGARCH_GED(j);
        Output{j}.likelihoods = likelihood_GJRGARCH_GED;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GED';
        Output{j}.Innovations = resid_GJRGARCH_GED;
        Output{j}.BIC = BIC_GJRGARCH_GED;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

        %                 AVGARCH-GED
    elseif BIC_opt{j}==21
        [parameters_AVGARCH_GED, LLF_AVGARCH_GED, stderrors_AVGARCH_GED, robustSE_AVGARCH_GED, ht_AVGARCH_GED, scores_AVGARCH_GED, resid_AVGARCH_GED, likelihood_AVGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'AVGARCH','GED',nAVGARCH_GED(j),const);
        BIC_AVGARCH_GED = 2*LLF_AVGARCH_GED+log(t)*size(parameters_AVGARCH_GED,1);
        Tstatistic_AVGARCH_GED=parameters_AVGARCH_GED./diag(robustSE_AVGARCH_GED).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_AVGARCH_GED./sqrt(ht_AVGARCH_GED);
        result_stderrors_AVGARCH_GED=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der GED; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'gedcdf',parameters_AVGARCH_GED(end));
        U_AVGARCH_GED=gedcdf(stdresid,parameters_AVGARCH_GED(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_AVGARCH_GED,.05,'unifcdf',0,1);
        U_AVGARCH_GED_mean = U_AVGARCH_GED-mean(U_AVGARCH_GED);
        U_AVGARCH_GED_mean2 = (U_AVGARCH_GED-mean(U_AVGARCH_GED)).^2;
        U_AVGARCH_GED_mean3 = (U_AVGARCH_GED-mean(U_AVGARCH_GED)).^3;
        U_AVGARCH_GED_mean4 = (U_AVGARCH_GED-mean(U_AVGARCH_GED)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_AVGARCH_GED_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_AVGARCH_GED;
        Output{j}.ParamsGARCH = [parameters_AVGARCH_GED(1:end-nAVGARCH_GED(j)-const-1); parameters_AVGARCH_GED(end)];
        Output{j}.ParamsAR = parameters_AVGARCH_GED(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_AVGARCH_GED,1)-1);
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
        Output{j}.leverage = 1;
        Output{j}.errortype = 3;
        Output{j}.arlag = nAVGARCH_GED(j);
        Output{j}.likelihoods = likelihood_AVGARCH_GED;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GED';
        Output{j}.Innovations = resid_AVGARCH_GED;
        Output{j}.BIC = BIC_AVGARCH_GED;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end


    elseif BIC_opt{j}==22
        [parameters_NGARCH_GED, LLF_NGARCH_GED, stderrors_NGARCH_GED, robustSE_NGARCH_GED, ht_NGARCH_GED, scores_NGARCH_GED, resid_NGARCH_GED, likelihood_NGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NGARCH','GED',nNGARCH_GED(j),const);
        BIC_NGARCH_GED = 2*LLF_NGARCH_GED+log(t)*size(parameters_NGARCH_GED,1);
        Tstatistic_NGARCH_GED=parameters_NGARCH_GED./diag(robustSE_NGARCH_GED).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_NGARCH_GED./sqrt(ht_NGARCH_GED);
        result_stderrors_NGARCH_GED=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der GED; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'gedcdf',parameters_NGARCH_GED(end));
        U_NGARCH_GED=gedcdf(stdresid,parameters_NGARCH_GED(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_NGARCH_GED,.05,'unifcdf',0,1);
        U_NGARCH_GED_mean = U_NGARCH_GED-mean(U_NGARCH_GED);
        U_NGARCH_GED_mean2 = (U_NGARCH_GED-mean(U_NGARCH_GED)).^2;
        U_NGARCH_GED_mean3 = (U_NGARCH_GED-mean(U_NGARCH_GED)).^3;
        U_NGARCH_GED_mean4 = (U_NGARCH_GED-mean(U_NGARCH_GED)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_NGARCH_GED_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_NGARCH_GED;
        Output{j}.ParamsGARCH = [parameters_NGARCH_GED(1:end-nNGARCH_GED(j)-const-1); parameters_NGARCH_GED(end)];
        Output{j}.ParamsAR = parameters_NGARCH_GED(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_NGARCH_GED,1)-1);
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
        Output{j}.arlag = nNGARCH_GED(j);
        Output{j}.likelihoods = likelihood_NGARCH_GED;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GED';
        Output{j}.Innovations = resid_NGARCH_GED;
        Output{j}.BIC = BIC_NGARCH_GED;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

    elseif BIC_opt{j}==23
        [parameters_NAGARCH_GED, LLF_NAGARCH_GED, stderrors_NAGARCH_GED, robustSE_NAGARCH_GED, ht_NAGARCH_GED, scores_NAGARCH_GED, resid_NAGARCH_GED, likelihood_NAGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NAGARCH','GED',nNAGARCH_GED(j),const);
        BIC_NAGARCH_GED = 2*LLF_NAGARCH_GED+log(t)*size(parameters_NAGARCH_GED,1);
        Tstatistic_NAGARCH_GED=parameters_NAGARCH_GED./diag(robustSE_NAGARCH_GED).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_NAGARCH_GED./sqrt(ht_NAGARCH_GED);
        result_stderrors_NAGARCH_GED=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der GED; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'gedcdf',parameters_NAGARCH_GED(end));
        U_NAGARCH_GED=gedcdf(stdresid,parameters_NAGARCH_GED(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_NAGARCH_GED,.05,'unifcdf',0,1);
        U_NAGARCH_GED_mean = U_NAGARCH_GED-mean(U_NAGARCH_GED);
        U_NAGARCH_GED_mean2 = (U_NAGARCH_GED-mean(U_NAGARCH_GED)).^2;
        U_NAGARCH_GED_mean3 = (U_NAGARCH_GED-mean(U_NAGARCH_GED)).^3;
        U_NAGARCH_GED_mean4 = (U_NAGARCH_GED-mean(U_NAGARCH_GED)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 NAullhypothese wird akzeptiert
        [H_NAGARCH_GED_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_NAGARCH_GED;
        Output{j}.ParamsGARCH = [parameters_NAGARCH_GED(1:end-nNAGARCH_GED(j)-const-1); parameters_NAGARCH_GED(end)];
        Output{j}.ParamsAR = parameters_NAGARCH_GED(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_NAGARCH_GED,1)-1);
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
        Output{j}.arlag = nNAGARCH_GED(j);
        Output{j}.likelihoods = likelihood_NAGARCH_GED;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GED';
        Output{j}.Innovations = resid_NAGARCH_GED;
        Output{j}.BIC = BIC_NAGARCH_GED;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

    elseif BIC_opt{j}==24
        [parameters_APGARCH_GED, LLF_APGARCH_GED, stderrors_APGARCH_GED, robustSE_APGARCH_GED, ht_APGARCH_GED, scores_APGARCH_GED, resid_APGARCH_GED, likelihood_APGARCH_GED, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'APGARCH','GED',nAPGARCH_GED(j),const);
        BIC_APGARCH_GED = 2*LLF_APGARCH_GED+log(t)*size(parameters_APGARCH_GED,1);
        Tstatistic_APGARCH_GED=parameters_APGARCH_GED./diag(robustSE_APGARCH_GED).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_APGARCH_GED./sqrt(ht_APGARCH_GED);
        result_stderrors_APGARCH_GED=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der GED; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'gedcdf',parameters_APGARCH_GED(end));
        U_APGARCH_GED=gedcdf(stdresid,parameters_APGARCH_GED(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_APGARCH_GED,.05,'unifcdf',0,1);
        U_APGARCH_GED_mean = U_APGARCH_GED-mean(U_APGARCH_GED);
        U_APGARCH_GED_mean2 = (U_APGARCH_GED-mean(U_APGARCH_GED)).^2;
        U_APGARCH_GED_mean3 = (U_APGARCH_GED-mean(U_APGARCH_GED)).^3;
        U_APGARCH_GED_mean4 = (U_APGARCH_GED-mean(U_APGARCH_GED)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 NAullhypothese wird akzeptiert
        [H_APGARCH_GED_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_APGARCH_GED;
        Output{j}.ParamsGARCH = [parameters_APGARCH_GED(1:end-nAPGARCH_GED(j)-const-1); parameters_APGARCH_GED(end)];
        Output{j}.ParamsAR = parameters_APGARCH_GED(size(Output{j}.ParamsGARCH,1)-1+1:size(parameters_APGARCH_GED,1)-1);
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
        Output{j}.arlag = nAPGARCH_GED(j);
        Output{j}.likelihoods = likelihood_APGARCH_GED;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'GED';
        Output{j}.Innovations = resid_APGARCH_GED;
        Output{j}.BIC = BIC_APGARCH_GED;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end



        % Verteilung SKEWT
        % 1.GARCH-SKEWT
    elseif BIC_opt{j}==25
        [parameters_GARCH_SKEWT, LLF_GARCH_SKEWT, stderrors_GARCH_SKEWT, robustSE_GARCH_SKEWT, ht_GARCH_SKEWT, scores_GARCH_SKEWT, resid_GARCH_SKEWT, likelihood_GJRGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'GARCH','SKEWT',nGARCH_SKEWT(j),const);
        BIC_GARCH_SKEWT = 2*LLF_GARCH_SKEWT+log(t)*size(parameters_GARCH_SKEWT,1);
        Tstatistic_GARCH_SKEWT=parameters_GARCH_SKEWT./diag(robustSE_GARCH_SKEWT).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_GARCH_SKEWT./sqrt(ht_GARCH_SKEWT);
        result_stderrors_GARCH_SKEWT=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der SKEWT; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'skewtdis_cdf',parameters_GARCH_SKEWT(end-1), parameters_GARCH_SKEWT(end));
        U_GARCH_SKEWT = skewtdis_cdf(stdresid,parameters_GARCH_SKEWT(end-1),parameters_GARCH_SKEWT(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_GARCH_SKEWT,.05,'unifcdf',0,1);
        U_GARCH_SKEWT_mean = U_GARCH_SKEWT-mean(U_GARCH_SKEWT);
        U_GARCH_SKEWT_mean2 = (U_GARCH_SKEWT-mean(U_GARCH_SKEWT)).^2;
        U_GARCH_SKEWT_mean3 = (U_GARCH_SKEWT-mean(U_GARCH_SKEWT)).^3;
        U_GARCH_SKEWT_mean4 = (U_GARCH_SKEWT-mean(U_GARCH_SKEWT)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_GARCH_SKEWT_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_GARCH_SKEWT;
        Output{j}.ParamsGARCH = [parameters_GARCH_SKEWT(1:end-nGARCH_SKEWT(j)-const-2); parameters_GARCH_SKEWT(end-1:end)];
        Output{j}.ParamsAR = parameters_GARCH_SKEWT(size(Output{j}.ParamsGARCH,1)-2+1:size(parameters_GARCH_SKEWT,1)-2);
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
        Output{j}.arlag = nGARCH_SKEWT(j);
        Output{j}.likelihoods = likelihood_GARCH_SKEWT;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'SKEWT';
        Output{j}.Innovations = resid_GARCH_SKEWT;
        Output{j}.BIC = BIC_GARCH_SKEWT;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

        % % % 5.EGARCH-SKEWT
    elseif BIC_opt{j}==26
        [parameters_EGARCH_SKEWT, LLF_EGARCH_SKEWT, stderrors_EGARCH_SKEWT, robustSE_EGARCH_SKEWT, ht_EGARCH_SKEWT, scores_EGARCH_SKEWT, resid_EGARCH_SKEWT, likelihood_EGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'EGARCH','SKEWT',nEGARCH_SKEWT(j),const);
        BIC_EGARCH_SKEWT = 2*LLF_EGARCH_SKEWT+log(t)*size(parameters_EGARCH_SKEWT,1);
        Tstatistic_EGARCH_SKEWT=parameters_EGARCH_SKEWT./diag(robustSE_EGARCH_SKEWT).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_EGARCH_SKEWT./sqrt(ht_EGARCH_SKEWT);
        result_stderrors_EGARCH_SKEWT=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der SKEWT; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'skewtdis_cdf',parameters_EGARCH_SKEWT(end-1), parameters_EGARCH_SKEWT(end));
        U_EGARCH_SKEWT = skewtdis_cdf(stdresid,parameters_EGARCH_SKEWT(end-1),parameters_EGARCH_SKEWT(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_EGARCH_SKEWT,.05,'unifcdf',0,1);
        U_EGARCH_SKEWT_mean = U_EGARCH_SKEWT-mean(U_EGARCH_SKEWT);
        U_EGARCH_SKEWT_mean2 = (U_EGARCH_SKEWT-mean(U_EGARCH_SKEWT)).^2;
        U_EGARCH_SKEWT_mean3 = (U_EGARCH_SKEWT-mean(U_EGARCH_SKEWT)).^3;
        U_EGARCH_SKEWT_mean4 = (U_EGARCH_SKEWT-mean(U_EGARCH_SKEWT)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_EGARCH_SKEWT_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_EGARCH_SKEWT;
        Output{j}.ParamsGARCH = [parameters_EGARCH_SKEWT(1:end-nEGARCH_SKEWT(j)-const-2); parameters_EGARCH_SKEWT(end-1:end)];
        Output{j}.ParamsAR = parameters_EGARCH_SKEWT(size(Output{j}.ParamsGARCH,1)-2+1:size(parameters_EGARCH_SKEWT,1)-2);
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
        Output{j}.arlag = nEGARCH_SKEWT(j);
        Output{j}.likelihoods = likelihood_EGARCH_SKEWT;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'SKEWT';
        Output{j}.Innovations = resid_EGARCH_SKEWT;
        Output{j}.BIC = BIC_EGARCH_SKEWT;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

        % 6.TGARCH-SKEWT
    elseif BIC_opt{j}==27
        [parameters_TGARCH_SKEWT, LLF_TGARCH_SKEWT, stderrors_TGARCH_SKEWT, robustSE_TGARCH_SKEWT, ht_TGARCH_SKEWT, scores_TGARCH_SKEWT, resid_TGARCH_SKEWT, likelihood_TGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'TGARCH','SKEWT',nTGARCH_SKEWT(j),const);
        BIC_TGARCH_SKEWT = 2*LLF_TGARCH_SKEWT+log(t)*size(parameters_TGARCH_SKEWT,1);
        Tstatistic_TGARCH_SKEWT=parameters_TGARCH_SKEWT./diag(robustSE_TGARCH_SKEWT).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_TGARCH_SKEWT./sqrt(ht_TGARCH_SKEWT);
        result_stderrors_TGARCH_SKEWT=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der SKEWT; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'skewtdis_cdf',parameters_TGARCH_SKEWT(end-1), parameters_TGARCH_SKEWT(end));
        U_TGARCH_SKEWT = skewtdis_cdf(stdresid,parameters_TGARCH_SKEWT(end-1),parameters_TGARCH_SKEWT(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_TGARCH_SKEWT,.05,'unifcdf',0,1);
        U_TGARCH_SKEWT_mean = U_TGARCH_SKEWT-mean(U_TGARCH_SKEWT);
        U_TGARCH_SKEWT_mean2 = (U_TGARCH_SKEWT-mean(U_TGARCH_SKEWT)).^2;
        U_TGARCH_SKEWT_mean3 = (U_TGARCH_SKEWT-mean(U_TGARCH_SKEWT)).^3;
        U_TGARCH_SKEWT_mean4 = (U_TGARCH_SKEWT-mean(U_TGARCH_SKEWT)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_TGARCH_SKEWT_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_TGARCH_SKEWT;
        Output{j}.ParamsGARCH = [parameters_TGARCH_SKEWT(1:end-nTGARCH_SKEWT(j)-const-2); parameters_TGARCH_SKEWT(end-1:end)];
        Output{j}.ParamsAR = parameters_TGARCH_SKEWT(size(Output{j}.ParamsGARCH,1)-2+1:size(parameters_TGARCH_SKEWT,1)-2);
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
        Output{j}.arlag = nTGARCH_SKEWT(j);
        Output{j}.likelihoods = likelihood_TGARCH_SKEWT;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'SKEWT';
        Output{j}.Innovations = resid_TGARCH_SKEWT;
        Output{j}.BIC = BIC_TGARCH_SKEWT;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

        % 7.GJRGARCH-SKEWT
    elseif BIC_opt{j}==28
        [parameters_GJRGARCH_SKEWT, LLF_GJRGARCH_SKEWT, stderrors_GJRGARCH_SKEWT, robustSE_GJRGARCH_SKEWT, ht_GJRGARCH_SKEWT, scores_GJRGARCH_SKEWT, resid_GJRGARCH_SKEWT, likelihood_GJRGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,1,1,'GJRGARCH','SKEWT',nGJRGARCH_SKEWT(j),const);
        BIC_GJRGARCH_SKEWT = 2*LLF_GJRGARCH_SKEWT+log(t)*size(parameters_GJRGARCH_SKEWT,1);
        Tstatistic_GJRGARCH_SKEWT=parameters_GJRGARCH_SKEWT./diag(robustSE_GJRGARCH_SKEWT).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_GJRGARCH_SKEWT./sqrt(ht_GJRGARCH_SKEWT);
        result_stderrors_GJRGARCH_SKEWT=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der SKEWT; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'skewtdis_cdf',parameters_GJRGARCH_SKEWT(end-1), parameters_GJRGARCH_SKEWT(end));
        U_GJRGARCH_SKEWT = skewtdis_cdf(stdresid,parameters_GJRGARCH_SKEWT(end-1),parameters_GJRGARCH_SKEWT(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_GJRGARCH_SKEWT,.05,'unifcdf',0,1);
        U_GJRGARCH_SKEWT_mean = U_GJRGARCH_SKEWT-mean(U_GJRGARCH_SKEWT);
        U_GJRGARCH_SKEWT_mean2 = (U_GJRGARCH_SKEWT-mean(U_GJRGARCH_SKEWT)).^2;
        U_GJRGARCH_SKEWT_mean3 = (U_GJRGARCH_SKEWT-mean(U_GJRGARCH_SKEWT)).^3;
        U_GJRGARCH_SKEWT_mean4 = (U_GJRGARCH_SKEWT-mean(U_GJRGARCH_SKEWT)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_GJRGARCH_SKEWT_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_GJRGARCH_SKEWT;
        Output{j}.ParamsGARCH = [parameters_GJRGARCH_SKEWT(1:end-nGJRGARCH_SKEWT(j)-const-2); parameters_GJRGARCH_SKEWT(end-1:end)];
        Output{j}.ParamsAR = parameters_GJRGARCH_SKEWT(size(Output{j}.ParamsGARCH,1)-2+1:size(parameters_GJRGARCH_SKEWT,1)-2);
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
        Output{j}.leverage = 1;
        Output{j}.errortype = 4;
        Output{j}.arlag = nGJRGARCH_SKEWT(j);
        Output{j}.likelihoods = likelihood_GJRGARCH_SKEWT;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'SKEWT';
        Output{j}.Innovations = resid_GJRGARCH_SKEWT;
        Output{j}.BIC = BIC_GJRGARCH_SKEWT;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

        %                 AVGARCH-SKEWT
    elseif BIC_opt{j}==29
        [parameters_AVGARCH_SKEWT, LLF_AVGARCH_SKEWT, stderrors_AVGARCH_SKEWT, robustSE_AVGARCH_SKEWT, ht_AVGARCH_SKEWT, scores_AVGARCH_SKEWT, resid_AVGARCH_SKEWT, likelihood_AVGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'AVGARCH','SKEWT',nAVGARCH_SKEWT(j),const);
        BIC_AVGARCH_SKEWT = 2*LLF_AVGARCH_SKEWT+log(t)*size(parameters_AVGARCH_SKEWT,1);
        Tstatistic_AVGARCH_SKEWT=parameters_AVGARCH_SKEWT./diag(robustSE_AVGARCH_SKEWT).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_AVGARCH_SKEWT./sqrt(ht_AVGARCH_SKEWT);
        result_stderrors_AVGARCH_SKEWT=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der SKEWT; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'skewtdis_cdf',parameters_AVGARCH_SKEWT(end-1), parameters_AVGARCH_SKEWT(end));
        U_AVGARCH_SKEWT=zeros(size(stdresid,1),1);
        U_AVGARCH_SKEWT = skewtdis_cdf(stdresid,parameters_AVGARCH_SKEWT(end-1),parameters_AVGARCH_SKEWT(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_AVGARCH_SKEWT,.05,'unifcdf',0,1);
        U_AVGARCH_SKEWT_mean = U_AVGARCH_SKEWT-mean(U_AVGARCH_SKEWT);
        U_AVGARCH_SKEWT_mean2 = (U_AVGARCH_SKEWT-mean(U_AVGARCH_SKEWT)).^2;
        U_AVGARCH_SKEWT_mean3 = (U_AVGARCH_SKEWT-mean(U_AVGARCH_SKEWT)).^3;
        U_AVGARCH_SKEWT_mean4 = (U_AVGARCH_SKEWT-mean(U_AVGARCH_SKEWT)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_AVGARCH_SKEWT_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_AVGARCH_SKEWT;
        Output{j}.ParamsGARCH = [parameters_AVGARCH_SKEWT(1:end-nAVGARCH_SKEWT(j)-const-2);parameters_AVGARCH_SKEWT(end-1:end)] ;
        Output{j}.ParamsAR = parameters_AVGARCH_SKEWT(size(Output{j}.ParamsGARCH,1)-2+1:size(parameters_AVGARCH_SKEWT,1)-2);
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
        Output{j}.leverage = 1;
        Output{j}.errortype = 4;
        Output{j}.arlag = nAVGARCH_SKEWT(j);
        Output{j}.likelihoods = likelihood_AVGARCH_SKEWT;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'SKEWT';
        Output{j}.Innovations = resid_AVGARCH_SKEWT;
        Output{j}.BIC = BIC_AVGARCH_SKEWT;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end


    elseif BIC_opt{j}==30
        [parameters_NGARCH_SKEWT, LLF_NGARCH_SKEWT, stderrors_NGARCH_SKEWT, robustSE_NGARCH_SKEWT, ht_NGARCH_SKEWT, scores_NGARCH_SKEWT, resid_NGARCH_SKEWT, likelihood_NGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NGARCH','SKEWT',nNGARCH_SKEWT(j),const);
        BIC_NGARCH_SKEWT = 2*LLF_NGARCH_SKEWT+log(t)*size(parameters_NGARCH_SKEWT,1);
        Tstatistic_NGARCH_SKEWT=parameters_NGARCH_SKEWT./diag(robustSE_NGARCH_SKEWT).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_NGARCH_SKEWT./sqrt(ht_NGARCH_SKEWT);
        result_stderrors_NGARCH_SKEWT=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der SKEWT; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'skewtdis_cdf',parameters_NGARCH_SKEWT(end-1), parameters_NGARCH_SKEWT(end));
        U_NGARCH_SKEWT=zeros(size(stdresid,1),1);
        U_NGARCH_SKEWT = skewtdis_cdf(stdresid,parameters_NGARCH_SKEWT(end-1),parameters_NGARCH_SKEWT(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_NGARCH_SKEWT,.05,'unifcdf',0,1);
        U_NGARCH_SKEWT_mean = U_NGARCH_SKEWT-mean(U_NGARCH_SKEWT);
        U_NGARCH_SKEWT_mean2 = (U_NGARCH_SKEWT-mean(U_NGARCH_SKEWT)).^2;
        U_NGARCH_SKEWT_mean3 = (U_NGARCH_SKEWT-mean(U_NGARCH_SKEWT)).^3;
        U_NGARCH_SKEWT_mean4 = (U_NGARCH_SKEWT-mean(U_NGARCH_SKEWT)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 Nullhypothese wird akzeptiert
        [H_NGARCH_SKEWT_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_NGARCH_SKEWT;
        Output{j}.ParamsGARCH = [parameters_NGARCH_SKEWT(1:end-nNGARCH_SKEWT(j)-const-2);parameters_NGARCH_SKEWT(end-1:end)];
        Output{j}.ParamsAR = parameters_NGARCH_SKEWT(size(Output{j}.ParamsGARCH,1)-2+1:size(parameters_NGARCH_SKEWT,1)-2);
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
        Output{j}.errortype = 4;
        Output{j}.arlag = nNGARCH_SKEWT(j);
        Output{j}.likelihoods = likelihood_NGARCH_SKEWT;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'SKEWT';
        Output{j}.Innovations = resid_NGARCH_SKEWT;
        Output{j}.BIC = BIC_NGARCH_SKEWT;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

    elseif BIC_opt{j}==31
        [parameters_NAGARCH_SKEWT, LLF_NAGARCH_SKEWT, stderrors_NAGARCH_SKEWT, robustSE_NAGARCH_SKEWT, ht_NAGARCH_SKEWT, scores_NAGARCH_SKEWT, resid_NAGARCH_SKEWT, likelihood_NAGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'NAGARCH','SKEWT',nNAGARCH_SKEWT(j),const);
        BIC_NAGARCH_SKEWT = 2*LLF_NAGARCH_SKEWT+log(t)*size(parameters_NAGARCH_SKEWT,1);
        Tstatistic_NAGARCH_SKEWT=parameters_NAGARCH_SKEWT./diag(robustSE_NAGARCH_SKEWT).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_NAGARCH_SKEWT./sqrt(ht_NAGARCH_SKEWT);
        result_stderrors_NAGARCH_SKEWT=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der SKEWT; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'skewtdis_cdf',parameters_NAGARCH_SKEWT(end-1), parameters_NAGARCH_SKEWT(end));
        U_NAGARCH_SKEWT=zeros(size(stdresid,1),1);
        U_NAGARCH_SKEWT = skewtdis_cdf(stdresid,parameters_NAGARCH_SKEWT(end-1),parameters_NAGARCH_SKEWT(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_NAGARCH_SKEWT,.05,'unifcdf',0,1);
        U_NAGARCH_SKEWT_mean = U_NAGARCH_SKEWT-mean(U_NAGARCH_SKEWT);
        U_NAGARCH_SKEWT_mean2 = (U_NAGARCH_SKEWT-mean(U_NAGARCH_SKEWT)).^2;
        U_NAGARCH_SKEWT_mean3 = (U_NAGARCH_SKEWT-mean(U_NAGARCH_SKEWT)).^3;
        U_NAGARCH_SKEWT_mean4 = (U_NAGARCH_SKEWT-mean(U_NAGARCH_SKEWT)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 NAullhypothese wird akzeptiert
        [H_NAGARCH_SKEWT_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_NAGARCH_SKEWT;
        Output{j}.ParamsGARCH = [parameters_NAGARCH_SKEWT(1:end-nNAGARCH_SKEWT(j)-const-2);parameters_NAGARCH_SKEWT(end-1:end)];
        Output{j}.ParamsAR = parameters_NAGARCH_SKEWT(size(Output{j}.ParamsGARCH,1)-2+1:size(parameters_NAGARCH_SKEWT,1)-2);
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
        Output{j}.errortype = 4;
        Output{j}.arlag = nNAGARCH_SKEWT(j);
        Output{j}.likelihoods = likelihood_NAGARCH_SKEWT;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'SKEWT';
        Output{j}.Innovations = resid_NAGARCH_SKEWT;
        Output{j}.BIC = BIC_NAGARCH_SKEWT;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end

    elseif BIC_opt{j}==32
        [parameters_APGARCH_SKEWT, LLF_APGARCH_SKEWT, stderrors_APGARCH_SKEWT, robustSE_APGARCH_SKEWT, ht_APGARCH_SKEWT, scores_APGARCH_SKEWT, resid_APGARCH_SKEWT, likelihood_APGARCH_SKEWT, EXITFLAG]=ar_multigarch_grm(data(:,j),1,0,1,'APGARCH','SKEWT',nAPGARCH_SKEWT(j),const);
        BIC_APGARCH_SKEWT = 2*LLF_APGARCH_SKEWT+log(t)*size(parameters_APGARCH_SKEWT,1);
        Tstatistic_APGARCH_SKEWT=parameters_APGARCH_SKEWT./diag(robustSE_APGARCH_SKEWT).^0.5;
        % H0: keine autocorrelation - in diesem Fall der quadrierten
        % standardisierten Residuen
        stdresid = resid_APGARCH_SKEWT./sqrt(ht_APGARCH_SKEWT);
        result_stderrors_APGARCH_SKEWT=lmtest2(stdresid,10);
        % H0: angenommene Verteilugn für ML-Schätzung entspricht der wahren
        % Verteilung (in diesem Fall der SKEWT; 0: H0 wird nicht abgelehnt; 1: H0
        % wird abgelehnt
        [stat_Kolmogorov_resid, siglevel_Kolmogorov_resid, H_Kolmogorov_resid]=kolmogorov(stdresid,.05,'skewtdis_cdf',parameters_APGARCH_SKEWT(end-1), parameters_APGARCH_SKEWT(end));
        U_APGARCH_SKEWT=zeros(size(stdresid,1),1);
        U_APGARCH_SKEWT = skewtdis_cdf(stdresid,parameters_APGARCH_SKEWT(end-1),parameters_APGARCH_SKEWT(end));
        [stat_Kolmogorov_U, siglevel_Kolmogorov_U, H_Kolmogorov_U]=kolmogorov(U_APGARCH_SKEWT,.05,'unifcdf',0,1);
        U_APGARCH_SKEWT_mean = U_APGARCH_SKEWT-mean(U_APGARCH_SKEWT);
        U_APGARCH_SKEWT_mean2 = (U_APGARCH_SKEWT-mean(U_APGARCH_SKEWT)).^2;
        U_APGARCH_SKEWT_mean3 = (U_APGARCH_SKEWT-mean(U_APGARCH_SKEWT)).^3;
        U_APGARCH_SKEWT_mean4 = (U_APGARCH_SKEWT-mean(U_APGARCH_SKEWT)).^4;
        % H0: Zeitreihe ist strict white noise; H=0 NAullhypothese wird akzeptiert
        [H_APGARCH_SKEWT_swn,pValue,Qstat,CriticalValue] = lbqtest(stdresid,10,.05,[]);
        Output{j}.Params=parameters_APGARCH_SKEWT;
        Output{j}.ParamsGARCH = [parameters_APGARCH_SKEWT(1:end-nAPGARCH_SKEWT(j)-const-2);parameters_APGARCH_SKEWT(end-1:end)];
        Output{j}.ParamsAR = parameters_APGARCH_SKEWT(size(Output{j}.ParamsGARCH,1)-2+1:size(parameters_APGARCH_SKEWT,1)-2);
        Output{j}.LLF=LLF_APGARCH_SKEWT;
        Output{j}.stderrors=stderrors_APGARCH_SKEWT;
        Output{j}.robustSE=robustSE_APGARCH_SKEWT;
        Output{j}.ht=ht_APGARCH_SKEWT;
        Output{j}.Scores=scores_APGARCH_SKEWT;
        Output{j}.Innovations=resid_APGARCH_SKEWT;
        Output{j}.GARCH = 'APGARCH';
        Output{j}.U = U_APGARCH_SKEWT;
        Output{j}.Tstat = Tstatistic_APGARCH_SKEWT;
        Output{j}.garchtype = 6;
        Output{j}.leverage = 2;
        Output{j}.errortype = 4;
        Output{j}.arlag = nAPGARCH_SKEWT(j);
        Output{j}.likelihoods = likelihood_APGARCH_SKEWT;
        Output{j}.EXITFLAG = EXITFLAG;
        Output{j}.data = data(:,j);
        Output{j}.dist = 'SKEWT';
        Output{j}.Innovations = resid_APGARCH_SKEWT;
        Output{j}.BIC = BIC_APGARCH_SKEWT;
        Output{j}.stdresid = stdresid;
        Output{j}.statKolmResid = stat_Kolmogorov_resid;
        Output{j}.siglevelKolmResid = siglevel_Kolmogorov_resid;
        Output{j}.HKolmResid = H_Kolmogorov_resid;
        Output{j}.statKolmU= stat_Kolmogorov_U;
        Output{j}.siglevelKolmU = siglevel_Kolmogorov_U;
        Output{j}.HKolmU = H_Kolmogorov_U;
        if const == 1
            Output{j}.const = 1;
        else
            Output{j}.const = 0;
        end
    end
end
