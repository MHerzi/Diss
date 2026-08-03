
% DynamicType : dynamische Struktur - 'DCC','ADCC','AGDCC'
% ModelType        : Modell - 'Copula','MultiGARCH'
% Purpose           : 'backtest','full'
% Beispiel für die Schätzung von multivariaten GARCH und multivariaten
% Copulas

% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% setzen der bestimmten Inputs
Spec = setInputs(daten);

% vor jedem neuen Lauf müssen die Variablen parameters und parametersALL
% gelöscht werden (lösche sie also zur Sicherheit immer)
clear parameters parametersALL

[t,k]=size(daten);


% für historische simulation genügt die maximale AR-lag Länge
if strcmp(Spec.ModelType,'HistSim_Copula') || strcmp(Spec.ModelType,'HistSim_Gauss')
    for i=1:k
        if strcmp(Spec.ModelType,'HistSim_Copula')
            maxarlag(i)=GARCHOutput{i}.arlag;
        elseif strcmp(Spec.ModelType,'HistSim_Gauss')
            maxarlag(i)=GARCHOutput_Gauss{i}.arlag;
        end
    end
    maxarlag = max(maxarlag);

    if strcmp(Spec.ModelType,'HistSim_Gauss') || strcmp(Spec.ModelType,'HistSim_Copula')
        if strcmp(Spec.Sim,'Window')
            [VaR] = HS_VaR(daten,Spec,maxarlag,'on');
            return
        elseif strcmp(Spec.Sim,'Pearson')
            [VaR] = HS_VaR_Pearson(daten,Spec,maxarlag,'on');
            return
        end
    end
end

% Für den Backtest: Anzahl der Perioden, die für den out-of-sample backtest
% benutzt werden sollen (für diese Anzahl der Perioden werden univariaten
% GARCH-Koeffizienten konstant gehalten und jeweils 1-Tages Prognosen
% gemacht - Variable Spec.ForecastNumb)
% ab diesem Wert beginnt die out-of-sample-periode (Spec.Forecaststart); die vorherigen
% Beobachtungen werden für die Schätzung des ersten Modells benutzt
% Setze der Startwert so, dass die Differenz zwischen Start und Ende geteilt durch die forecast-Period
% eine natürliche Zahl ergibt, also z.B. sample 100 Daten; start=50 und
% Spec.ForecastNumb=5; ergibt 10 verschiedene Ergebnisse für die jeweilige
% out-of-sample Schätzung
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------


% bringe für multivariate GARCH-Funktionen Lag-Terme in Vektoren
if isempty(Spec.archP)
    archP=ones(1,k);
elseif isscalar(Spec.archP)
    archP=ones(1,k)*Spec.archP;
else
    archP = Spec.archP(:)';
    if numel(archP) ~= k
        error('Diss:Main:InvalidArchOrder', ...
            'Spec.archP must be scalar or contain one value per series.');
    end
end

if isempty(Spec.garchQ)
    garchQ=ones(1,k);
elseif isscalar(Spec.garchQ)
    garchQ=ones(1,k)*Spec.garchQ;
else
    garchQ = Spec.garchQ(:)';
    if numel(garchQ) ~= k
        error('Diss:Main:InvalidGarchOrder', ...
            'Spec.garchQ must be scalar or contain one value per series.');
    end
end

% -------------------------------------------------------------------------
% -----------------------------------------------------------------------
% Schätzung der Marginalmodelle

% Marginalmodelle (für DCC Modell und alle Derivate selbigens muss die
% Normalverteilung für die Schätzung der Marginalverteilung angenommen
% werden - alle Modelle werden mit archP=1 und garchQ=1 geschätzt
% Für die BEstimmung des optimalen Modells schätze Modelle über komplette
% Datenreihe
if strcmp(Spec.univariate,'on')
    [GARCHOutput GARCHOutput_Gauss] = AR_MarginalModel(daten,Spec.const,Spec.arlag);
end

% Bringe die Innovations und die Varianzen auf die gleiche Länge, durch
% unterschiedliche AR-terme gehen unterschiedlich viel Beobachtungen
% verloren
% ----------------für die copulas---------------------------------
if strcmp(Spec.ModelType,'MultiCopula') || strcmp(Spec.ModelType,'VineCopula') || strcmp(Spec.ModelType,'MultiMixCopula')  || strcmp(Spec.ModelType,'DVine-Mix')
    maxarlag=zeros(k,1);
    arlag=zeros(k,1);
    ardiff=zeros(k,1);
    for i=1:k
        maxarlag(i)=GARCHOutput{i}.arlag;
        arlag(i) = GARCHOutput{i}.arlag;
    end
    maxarlag = max(maxarlag);
    for i=1:k
        if GARCHOutput{i}.arlag<maxarlag
            ardiff(i) = maxarlag-GARCHOutput{i}.arlag;
        else ardiff(i) = 0;
        end
        ht_new(:,i) = GARCHOutput{i}.ht(ardiff(i)+1:end);
        resid_new(:,i) = GARCHOutput{i}.Innovations(ardiff(i)+1:end);
        stdresid(:,i) = resid_new(:,i)./sqrt(ht_new(:,i));
        scores_new{i} = GARCHOutput{i}.Scores(ardiff(i)+1:end,:);
        daten_new(:,i) = daten(maxarlag+1:end,i);
    end

    if strcmp(Spec.tails,'pareto')
        [U_new,upareto] = paretocdf(stdresid);
    elseif strcmp(Spec.tails,'empirical')
        for i=1:k
            U_new(:,i) = empiricalCDF(stdresid(:,i));
        end
        upareto=[];
    else
        for i=1:k
            U_new(:,i) = GARCHOutput{i}.U(ardiff(i)+1:end);
            upareto=[]; %Platzhalter falls keine paretotails angewandt wird
        end
    end
    [t,k] = size(stdresid);

    % für die multivariaten GARCH
elseif strcmp(Spec.ModelType,'MultiGARCH')
    maxarlag_Gauss=zeros(k,1);
    arlag_Gauss=zeros(k,1);
    ardiff_Gauss=zeros(k,1);
    for i=1:k
        maxarlag_Gauss(i)=GARCHOutput_Gauss{i}.arlag;
        arlag_Gauss(i) = GARCHOutput_Gauss{i}.arlag;
    end
    maxarlag_Gauss = max(maxarlag_Gauss);

    for i=1:k
        if GARCHOutput_Gauss{i}.arlag<maxarlag_Gauss
            ardiff_Gauss(i) = maxarlag_Gauss-GARCHOutput_Gauss{i}.arlag;
        else ardiff_Gauss(i) = 0;
        end
        ht_new_Gauss(:,i) = GARCHOutput_Gauss{i}.ht(ardiff_Gauss(i)+1:end);
        resid_new_Gauss(:,i) = GARCHOutput_Gauss{i}.Innovations(ardiff_Gauss(i)+1:end);
        stdresid_Gauss(:,i) = resid_new_Gauss(:,i)./sqrt(ht_new_Gauss(:,i));
        scores_new_Gauss{i} = GARCHOutput_Gauss{i}.Scores(ardiff_Gauss(i)+1:end,:);
        daten_new_Gauss(:,i) = daten(maxarlag_Gauss+1:end,i);
        U_new_Gauss(:,i) = GARCHOutput_Gauss{i}.U(ardiff_Gauss(i)+1:end);
        [t,k] = size(stdresid_Gauss);
    end
end


% Portfoliooptimierung
% if strcmp(Spec.PortOpt,'on')
%     if isempty(rnd_ret) %random returns müssen vorhanden sein
%         error('Es werden Szenario-Returns zur Optimierung benötigt');
%     end
%     DatenBacktest = daten(end-size(rnd_ret,2)+1:end,:); %nehme die letzten Original Portfoliodaten, die auch das Backtest-Sample bilden
%     for i=1:size(rnd_ret,2)
%         R0(i) = ((1/k)*ones(1,k))*exp(DatenBacktest(i,:)'); %als Minimum Retunr nehme den Return des gleichgeichteten Portfolios
%         R0(i) = log(R0(i));
%         [OptPortVaR(i,1),weights(i,:)]=CVaROptimization_grm(rnd_ret{i}, Spec.beta, Spec.LB, Spec.UB, R0(i),DatenBacktest(i,:));
%     end
%     %     korrigiere Rundungsfehler
%     weights(weights<0)=0;
%     return
% end


% -------------------------------------------------------------------------
%teste auf dynamische Korrelation
if strcmp(Spec.ModelType,'MultiGARCH')
    [pval_Gauss, stat_Gauss]=dcc_mvgarch_test_grm(stdresid_Gauss,10);
elseif strcmp(Spec.ModelType,'DeltaNormal')
%     kein Test
else
    [pval, stat]=dcc_mvgarch_test_grm(stdresid,10);
end
% ----------------------------------------------------------------------

% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

if strcmp(Spec.ModelType,'MultiGARCH') == 1
    %     Schätze multivariate GARCH-Modelle über den gesamten Zeitraum
    if strcmp(Spec.purpose,'full') == 1
        if strcmp(Spec.DynamicType,'DCC')==1
            display('----------------------------------------')
            display('%%%%% Estimating the DCC-MGARCH %%%%%')
            display('----------------------------------------')
            [dccparameters, negativeLikelihood, exitFlag] = Est_DCC(stdresid_Gauss, Spec.dccP, Spec.dccQ);
        elseif strcmp(Spec.DynamicType,'ADCC')==1
            display('----------------------------------------')
            display('%%%%% Estimating the ADCC-MGARCH %%%%%')
            display('----------------------------------------')
            [dccparameters, negativeLikelihood, exitFlag] = Est_ADCC(stdresid_Gauss, Spec.dccP, Spec.dccQ);
        elseif strcmp(Spec.DynamicType,'GDCC')==1
            display('----------------------------------------')
            display('%%%%% Estimating the GDCC-MGARCH %%%%%')
            display('----------------------------------------')
            [dccparameters, negativeLikelihood, exitFlag] = Est_GDCC(stdresid_Gauss, Spec.dccP, Spec.dccQ);
        elseif strcmp(Spec.DynamicType,'AGDCC')==1
            display('----------------------------------------')
            display('%%%%% Estimating the AGDCC-MGARCH %%%%%')
            display('----------------------------------------')
            [dccparameters, negativeLikelihood, exitFlag] = Est_AGDCC(stdresid_Gauss, Spec.dccP, Spec.dccQ);
        end
        h=1;
        g=1;
        H=zeros(size(ht_new_Gauss,1),k);
        errortype = zeros(1,k);
        garchtype = zeros(1,k);
        asymmG = zeros(1,k);
        for i=1:k
            parameters(h:h+size(GARCHOutput_Gauss{i}.ParamsGARCH,1)-1) = GARCHOutput_Gauss{i}.ParamsGARCH;
            parametersALL(g:g+size(GARCHOutput_Gauss{i}.Params,1)-1) = GARCHOutput_Gauss{i}.Params;
            h = h + size(GARCHOutput_Gauss{i}.ParamsGARCH,1);
            g = g + size(GARCHOutput_Gauss{i}.Params,1);
            H(:,i) = ht_new_Gauss(:,i);
            errortype(i) = GARCHOutput_Gauss{i}.errortype;
            garchtype(i) = GARCHOutput_Gauss{i}.garchtype;
            asymmG(i) = GARCHOutput_Gauss{i}.leverage;
        end
        if size(parameters,2)>1
            parameters=parameters';
        end
        if size(dccparameters,2)>1
            parameters=[parameters;dccparameters'];
        else
            parameters=[parameters;dccparameters];
        end
        if size(parametersALL,2)>1
            parametersALL = parametersALL';
        end
        parametersALL = [parametersALL; dccparameters];

        if strcmp(Spec.DynamicType,'DCC')==1
            [logL, Rt, likelihoods, Qt] = DCC_full_likelihood(parameters, resid_new_Gauss, archP, garchQ, asymmG, garchtype, errortype, Spec.dccP, Spec.dccQ);
            likelihoods=-likelihoods;
            logL=-logL;
            [aic,bic]=aicbic(logL,size(dccparameters,1),t);
            dccG=0;
            if strcmp(Spec.stderrors,'an')
                display('----------------------------------------')
                display('%%%%% Calc of StdErrors %%%%%')
                display('----------------------------------------')
                [Tstat, stderrors] = stderrors_AR_DCC(parametersALL, daten, archP,garchQ,asymmG,Spec.dccP,Spec.dccQ,dccG,garchtype,errortype,GARCHOutput_Gauss,Spec.const);
            end
        elseif strcmp(Spec.DynamicType,'ADCC')==1
            [logL, Rt, likelihoods, Qt]=ADCC_full_likelihood(parameters, resid_new_Gauss, archP, garchQ, asymmG, garchtype, errortype, Spec.dccP, Spec.dccQ, Spec.dccG);
            likelihoods=-likelihoods;
            logL=-logL;
            [aic,bic]=aicbic(logL,size(dccparameters,1),t);
            if strcmp(Spec.stderrors,'an')
                display('----------------------------------------')
                display('%%%%% Calc of StdErrors %%%%%')
                display('----------------------------------------')
                [Tstat, stderrors] = stderrors_AR_ADCC(parametersALL, daten, archP,garchQ,asymmG,Spec.dccP,Spec.dccQ,Spec.dccG,garchtype,errortype,GARCHOutput_Gauss,Spec.const);
            end
        elseif strcmp(Spec.DynamicType,'GDCC')==1
            [logL, Rt, likelihoods, Qt] = GDCC_full_likelihood(parameters, resid_new_Gauss, archP, garchQ, asymmG, garchtype, errortype, Spec.dccP, Spec.dccQ);
            likelihoods=-likelihoods;
            logL=-logL;
            [aic,bic]=aicbic(logL,size(dccparameters,1),t);
            if strcmp(Spec.stderrors,'an')
                display('----------------------------------------')
                display('%%%%% Calc of StdErrors %%%%%')
                display('----------------------------------------')
                [Tstat, stderrors] = stderrors_AR_GDCC(parametersALL, daten, archP,garchQ,asymmG,Spec.dccP,Spec.dccQ,Spec.dccG,garchtype,errortype,GARCHOutput_Gauss,Spec.const);
            end
        elseif strcmp(Spec.DynamicType,'AGDCC')==1
            [logL, Rt, likelihoods, Qt] = AGDCC_full_likelihood(parameters, resid_new_Gauss, archP, garchQ, asymmG, garchtype, errortype, Spec.dccP, Spec.dccQ, Spec.dccG);
            likelihoods=-likelihoods;
            logL=-logL;
            [aic,bic]=aicbic(logL,size(dccparameters,1),t);
            display('----------------------------------------')
            display('%%%%% Calc of StdErrors %%%%%')
            display('----------------------------------------')
            if strcmp(Spec.stderrors,'an')
                [Tstat, stderrors] = stderrors_AR_AGDCC(parametersALL, daten, archP,garchQ,asymmG,Spec.dccP,Spec.dccQ,Spec.dccG,garchtype,errortype,GARCHOutput_Gauss,Spec.const);
            end
        end
        [GOF_Stat,zt] = GOF_MVGARCH(stdresid_Gauss,Rt);
        return
    end
    if strcmp(Spec.purpose,'backtest') == 1
        if strcmp(Spec.ModelType,'MultiGARCH')
            %     Schätze multivariate GARCH-Modelle über den Backtest Zeitraum
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Für den Backtest:
            % Bestimme GARCH-Modelle über Zeitfenster, wobei bei einem bestimmten
            % Zeitpunkt angefangen wird und dann sukzessive Perioden (Spec.ForecastNumb) hinzugefügt werden
            %             der Zeitpunkt des ersten forecast richtet sich nach der
            %             Anzahl der Ar-Lags (Bsp BRIC: Start=526; maxarlag=1;
            %             MArginalModel: Daten von 2:526 werden zur Schätzung genommen;
            %             dann steht in 526 der erste forecast mit foecastP=104 steht
            %             dann in 629 der letzte VaR-forecast für t=629

            if strcmp(Spec.uniBacktest,'on')
                g=1;
                c=size(daten,1);
                for i=Spec.ForecastStart:Spec.uniforecastP:c-Spec.uniforecastP
                    [GARCHOutput_Backtest_Gauss{g}] = AR_MarginalModel_add(daten(1:i,:),Spec.archP,Spec.garchQ,GARCHOutput_Gauss);
                    g=g+1;
                end
                [AR_pred, ht_pred, resid] = univariateForecastBacktest(daten,GARCHOutput_Backtest_Gauss,Spec.ForecastStart,Spec.uniforecastP,Spec);
            end
            %             start_new=Spec.ForecastStart-maxarlag_Gauss; %durch AR-lags gehen Beobachtungen verloren
            [Rt_pred,GARCHOutput_Backtest,dccparameters_backtest] = BacktestMultiGARCH(stdresid_Gauss,Spec.ForecastStart,daten,GARCHOutput_Gauss,Spec.ForecastNumb,Spec, archP, garchQ,GARCHOutput_Backtest_Gauss);
            %     Berechne VaR für den Backtest Zeitraum
            clear rnd_ret
            [VaR,rnd_ret] = VaR_MultiGARCH(AR_pred,ht_pred,GARCHOutput_Gauss,Rt_pred,Spec.SimNumb,daten,'on',Spec);
            return
        end
    end
end

% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% Berechnung der MultiCopula (elliptisch)

if strcmp(Spec.ModelType,'MultiCopula')
    if strcmp(Spec.purpose,'full')
        if strcmp(Spec.CopulaType,'Gauss')
            family=cell(1);
            family{1}='gaussian';
            [CopParam_tv, Weights_tv, copparameters, LL_Mix, LL, AIC, BIC, CopParam_1, stderrors, startinvals] = copulafitmix_tv_grm2(family, U_new, Spec.DynamicType, Spec.stderrors, 'on');
        elseif strcmp(Spec.CopulaType,'t')
            family{1}='t';
            [CopParam_tv, Weights_tv, copparameters, LL_Mix, LL, AIC, BIC, CopParam_1, stderrors, startinvals] = copulafitmix_tv_grm2(family, U_new, Spec.DynamicType, Spec.stderrors, 'on');
            %         die komplette Likelihood besteht aus der Likelihood der
            %         GARCH-Modelle und der Likelihood der Copula
        end
        for i=1:k
            GARCH_LL(i) = GARCHOutput{i}.LLF;
        end
        GARCH_LL = -sum(GARCH_LL);%wandle negative LL in positive
        LL_complete = LL + GARCH_LL;
        [aic_all,bic_all]=aicbic(LL_complete,size(copparameters,1),t);
        [GOF_Stat,zt] = GOF_Copula(stdresid,Weights_tv,CopParam_tv,copparameters,Spec,upareto);
        % -------------------------------------------------------------------------------------------------------------------------------------------------------------
        %         if strcmp(Spec.stderrors,'an')
        %             if strcmp(Spec.CopulaType,'Gauss')
        %                 family=cell(1);
        %                 family{1}='gaussian';
        %                 [CopParam_tv, Weights_tv, copparameters, LL_Mix, LL, AIC, BIC, CopParam_1, SE] = copulafitmix_tv_grm2(family, U_new, Spec.DynamicType, 'an');
        %             elseif strcmp(Spec.CopulaType,'t')
        %                 family{1}='t';
        %                 [CopParam_tv, Weights_tv, copparameters, LL_Mix, LL, AIC, BIC, CopParam_1] = copulafitmix_tv_grm2(family, U_new, Spec.DynamicType, 'an');
        %             end
        %         end
        return
    elseif strcmp(Spec.purpose,'backtest')
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Für den Backtest:
        % Bestimme GARCH-Modelle über Zeitfenster, wobei bei einem bestimmten
        % Zeitpunkt angefangen wird und dann sukzessive Perioden (Spec.ForecastNumb) hinzugefügt werden
        %         das letzte Modell muss für Gesamtlänge Datensatz - Spec.ForecastNumb
        %         Perioden geschätzt werden, da genau Spec.ForecastNumb vorhersagen gemacht
        %         werden sollen
        if strcmp(Spec.uniBacktest,'on')
            g=1;
            c=size(daten,1);
            for i=Spec.ForecastStart:Spec.uniforecastP:c-Spec.uniforecastP
                [GARCHOutput_Backtest{g}] = AR_MarginalModel_add(daten(1:i,:),Spec.archP,Spec.garchQ,GARCHOutput);
                g=g+1;
            end
            [AR_pred, ht_pred, resid] = univariateForecastBacktest(daten,GARCHOutput_Backtest,Spec.ForecastStart,Spec.uniforecastP,Spec);
        end
        %         [AR_pred, ht_pred] = univariate_pred_AR_Ht_1P(data,GARCHOutput_Backtest,1);

        %         In den Backtest-Copula kommen schon um AR-Länge korrigierte
        %         Variablen deshalb muss maxarlag abgezogen werden
        %         start_new=Spec.ForecastStart-maxarlag;
        clear rnd_ret
        [Rt_pred,copparameters_backtest,rnd_ret,VaR] = BacktestMultiCopula_multistep(U_new,stdresid,GARCHOutput,GARCHOutput_Backtest,AR_pred,ht_pred,'on',daten,Spec,upareto);
        %         [VaR,SimReturn] = VaR_MultiCopula(AR_pred,ht_pred,GARCHOutput,Rt_pred,Spec.SimNumb,daten,Spec.ForecastStart,GARCHOutput_Backtest,Spec,params_backtest,'on',maxarlag,U_rnd,Spec.uniforecastP);
        return
    end
end

% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% Berechnung der Vine

if strcmp(Spec.ModelType,'VineCopula')
    if strcmp(Spec.purpose,'full')
        %     -----------------determine form of data for Copulas-----------------
        % von Hand schauen in data_oredering steht dann die Reihenfolge der Daten
        % data_ordering = Vinecheckdata()
        if isempty(data_ordering)
            error('Ordne zuerst die Daten nach bivariate Abhängigkeit');
        end
        %         data_ordering = Vinecheckdata()
        clear CopulaSpec StrOutput LogL %lösche noch existierende Variable damit auf jeden Fall eine neue in der Funktion kreiert wird
        [LogL, StrOutput, CopulaSpec] = fitCopulaVine_multistep_grm(U_new_ordered);
        for i=1:k
            GARCH_LL(i) = GARCHOutput{i}.LLF;
        end
        GARCH_LL = -sum(GARCH_LL);%wandle negative LL in positive
        LL_complete = LogL + GARCH_LL;
        [aic,bic] = aicbic(LL_complete,k*(k-1)/2*size(StrOutput.VineParams{1}{1},1),t);
        [GOF_Stat,zt] = GOFVine(U_new_ordered,CopulaSpec,StrOutput);
        return
    elseif strcmp(Spec.purpose,'backtest')
        %         [LogL, StrOutput, CopulaSpec] = fitCopulaVine_multistep_grm(U_new);
        if strcmp(Spec.uniBacktest,'on')
            %             !!!! Achtung : GARCHOutput_Backtest auch nach der obigen
            %             Reihenfolge ordnen!!!!
            g=1;
            c = size(daten,1);
            for i=Spec.ForecastStart:Spec.uniforecastP:c-Spec.uniforecastP
                [GARCHOutput_Backtest{g}] = AR_MarginalModel_add(daten(1:i,:),archP(1),garchQ(1),GARCHOutput);
                g=g+1;
            end
            [AR_pred, ht_pred, resid] = univariateForecastBacktest(daten,GARCHOutput_Backtest,Spec.ForecastStart,Spec.ForecastNumb,Spec);
        end
        CopulaSpec = setCopulaVineLLinputs_grm(k);
        clear rnd_ret
        [Rt_pred,VineOutput_DMulti,CopulaSpec,rnd_ret,VaR] = BacktestVine_multistep(U_new_ordered,Spec.ForecastStart,CopulaSpec,GARCHOutput_ordered,GARCHOutput_Backtest_ordered, AR_pred_ordered,ht_pred_ordered,'on',daten,Spec,upareto);
        %         [VaR,SimReturn] = VineVaR(USim,GARCHOutput_ordered,GARCHOutput_Backtest_ordered,Spec.uniforecastP,Spec.SimNumb,AR_pred_ordered,ht_pred_ordered,'on',daten);
        return
    end
end

%     Berechnung DVine mixture
if strcmp(Spec.ModelType,'DVine-Mix')
    if ~exist('CopulaSpec', 'var')
        CopulaSpec=[];
    end
    if strcmp(Spec.purpose,'full')
        if isempty(data_ordering)
            error('Ordne zuerst die Daten nach bivariate Abhängigkeit');
        end
        %         data_ordering = Vinecheckdata()
        CopulaSpec=[];
        [LogL, VineOutput, CopulaSpec, Vinephi]= fitCopulaVine_multistep_mix_grm(U_new_ordered,CopulaSpec,Spec.CopStat,'off');
        GARCH_LL = zeros(k,1);
        for i=1:k
            GARCH_LL(i) = GARCHOutput{i}.LLF;
        end
        GARCH_LL = -sum(GARCH_LL);%wandle negative LL in positive
        LL_complete = LogL + GARCH_LL;
    elseif strcmp(Spec.purpose,'backtest')
        CopulaSpec = setCopulaVineLLinputs_grm(k);
        if strcmp(Spec.uniBacktest,'on')
            g=1;
            c = size(daten,1);
            for i=Spec.ForecastStart:Spec.uniforecastP:c-Spec.uniforecastP
                [GARCHOutput_Backtest{g}] = AR_MarginalModel_add(daten(1:i,:),Spec.archP,Spec.garchQ,GARCHOutput);
                g=g+1;
            end
        end
        clear rnd_ret
        [Output,VineOutput_back,CopulaSpec,rnd_ret]  = BacktestVine_multistep_mix(U_new, GARCHOutput, GARCHOutput_Backtest, CopulaSpec, daten, AR_pred, ht_pred, Spec, Spec.uniforecastP,upareto);
    end
end

%     Berechnung Multi-Mix Copula
if strcmp(Spec.ModelType,'MultiMixCopula')
    if strcmp(Spec.purpose,'full')
        [CopParam_tv, Weights_tv, tv_faktor, LL_Mix, LL, AIC, BIC, CopParam_1, stderrors, startinvals] = copulafitmix_tv_grm2(Spec.family, U_new, Spec.Dynamic, Spec.stderrors, 'on');
        for i=1:k
            GARCH_LL(i) = GARCHOutput{i}.LLF;
        end
        GARCH_LL = -sum(GARCH_LL);%wandle negative LL in positive
        LL_complete = LL + GARCH_LL;
        LL_Mix_complete = LL_Mix + GARCH_LL;
        if size(Spec.family,2)>1
            [AIC,BIC] = aicbic(LL_Mix_complete,size(tv_faktor,1),t);
        else
            [AIC,BIC] = aicbic(LL_complete,size(tv_faktor,1),t);
        end
        return
    elseif strcmp(Spec.purpose,'backtest')
        if strcmp(Spec.uniBacktest,'on')
            g=1;
            c = size(daten,1);
            for i=Spec.ForecastStart:Spec.uniforecastP:c
                [GARCHOutput_Backtest{g}] = AR_MarginalModel_add(daten(1:i,:),Spec.archP,Spec.garchQ,GARCHOutput);
                g=g+1;
            end
            [AR_pred, ht_pred, resid] = univariateForecastBacktest(daten,GARCHOutput_Backtest,Spec.ForecastStart,Spec.ForecastNumb,Spec);
        end
        clear rnd_ret
        [VaR,rnd_ret,CopParam_tv_back, Weights_tv_back, tv_faktor_back, CopParam_pred, weights_pred] = Backtest_VaR_Dynamic_Mixture_multistep(Spec.family,U_new,daten,GARCHOutput,GARCHOutput_Backtest,AR_pred,ht_pred,'on',Spec,upareto);
        return
    end
end


if strcmp(Spec.ModelType,'DeltaNormal')
%     es gehen die p-ersten Zeilen durch AR(p)-Schätzung verloren, obwohl
%     hier nicht mit AR-Modellen gebarbeitet wird korrigiere zum Vergleich
%     trotzdem
    VaR = MVNVaR(daten(Spec.archP+1:end,:),Spec,'on');
    datenBack=daten(end-750+1:end,:);
    [VaRnumb] = VaR250(VaR,datenBack,250);
    [VaRExc] = VaRExc(VaR,Spec,daten,[],[]);
    return
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % Splitte die Verletzungen in jährliche returns auf
datenBack=daten(end-750+1:end,:);
[VaRnumb] = VaR250(VaR,datenBack,250);
% % % %
% % % % % Berechne Exceedance VaR und Expected Shortfall
[Exc] = VaRExc(VaR,Spec,daten,rnd_ret,B);

% % % %
% % % Anderson-Darling und Kolmogorov g.o.f. test
% % % für Copulas außer Vine
if strcmp(Spec.family{1},'clayton') || strcmp(Spec.family{1},'rotclayton')
    copparameters=[];
elseif size(Spec.family,2)>1
    copparameters=tv_faktor;
    Spec.DynamicType=Spec.Dynamic;
end
[GOF_Stat,zt] = GOF_Copula(stdresid,Weights_tv,CopParam_tv,copparameters,Spec,upareto);
% %

% % %
% % %
% % % %  % Berechne Portfoliooptimierung
% % % % if strcmp(Spec.PortOpt,'on')
%     if isempty(rnd_ret)
%         error('Es werden Szenario-Returns zur Optimierung benötigt');
%     end
%     %     Originaldaten die mit dem Backtest-sample verglichen werden
%     datenBacktest = daten(end-size(rnd_ret,2)+1:end,:);
%     %     bestimme die Returns
%     Spec.beta=.99;
%     Spec.UB=.25;
%     Spec.LB=0;
%     for i=1:size(rnd_ret,2)
%        R0(i) = mean(rnd_ret{i})*ones(k,1)*1/k; %gleichgewichteter mean Return als min Return
% %         VaR0 = quantile(sort(rnd_ret{i})*ones(k,1)*1/k,1-Spec.beta); %als VaR constraint nehme den VaR des gleichgeichteten Portfolios
% %         VaR0 = -VaR0; %VaR ist als positive Zahl definiert
%                 if Spec.beta==.99
%                     VaR0(i) = -VaR.VaRPortfolio991(i);
%                 elseif Spec.beta==.95
%                     VaR0(i) = -VaR.VaRPortfolio951(i);
%                 elseif Spec.beta==.9
%                     VaR0(i) = -VaR.VaRPortfolio901(i);
%                 end
%         [OptCVaR(i,:),weights(i,:)]=CVaROptimization_grm2(rnd_ret{i}, R0(i), VaR0(i), Spec.beta,  Spec.UB, Spec.LB);
%     end
%     %     %     korrigiere Rundungsfehler z.B., wenn LB=0 sind Gewichte
%     %     manchmal negativ (sehr gering)
%     weights(weights<Spec.LB) = Spec.LB;
%     weights(weights>Spec.UB) = Spec.UB;
%     for i=1:size(rnd_ret,2)
%         OptPuL(i,1) = weights(i,1:k)*exp(datenBacktest(i,:)');
%         EqualPuL(i,1) = 1/k*ones(1,k)*exp(datenBacktest(i,:)');
%         EqualPuL(i) = log(EqualPuL(i));
%         OptPuL(i) = log(OptPuL(i));
%     end
%     %     rechne geometrisch annualisierten mean return für gleichgeichtetes
%     %     und optimiertes Portfolio aus
%     %     Gesamtrendite
%     EqualPuL_all = sum(EqualPuL);
%     OptPuL_all = sum(OptPuL);
%     EqualPuL_ann = ((1+EqualPuL_all)^(1/3)-1);
%     OptPuL_ann = ((1+OptPuL_all)^(1/3)-1);
% %     clear rnd_ret %damit bei abspeichern nicht zuviel Platz verbraucht wird lösche die random returns
% % % % %     Output = OptVaRStat(weights(:,end),daten,Spec);
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % % % % % % % % % % % % % % %
% % % % % Portfoliooptimierung Patton
% % % % datenBacktest = daten(end-size(rnd_ret,2)+1:end,:);
% % % % options = optimset('fmincon');
% % % % options = optimset(options,'Algorithm','interior-point','Hessian','bfgs');
% % % % options = optimset(options,'Display','iter');
% % % % [t,k]=size(rnd_ret{1});
% % % % weight0 = ones(k,1)*(1/k);
% % % % % Upper und Lower Bounds für Gewichte
% % % % UB = ones(1,k)*Spec.UB - 1e-6;
% % % % LB = ones(1,k)*Spec.LB + 1e-6;
% % % % Aeq  = ones(1,k); %summe der Gewichte A*w=b muss eins ergeben
% % % % beq  = 1;
% % % % RA = 7; %Koeffizient der relativen Risikoaversion
% % % % weights(1,:) = fmincon('Patton_CRRA',weight0,[],[],Aeq,beq,LB,UB,[],options,rnd_ret{1},RA);
% % % % for i=2:size(rnd_ret,2)
% % % %     weights(i,:) = fmincon('Patton_CRRA',weights(i-1,:)',[],[],Aeq,beq,LB,UB,[],options,rnd_ret{i},RA);
% % % % end
% % % % weights(weights<Spec.LB) = Spec.LB;
% % % %     weights(weights>Spec.UB) = Spec.UB;
% % % %     for i=1:size(rnd_ret,2)
% % % %         OptPuL(i,1) = weights(i,1:k)*exp(datenBacktest(i,:)');
% % % %         EqualPuL(i,1) = 1/k*ones(1,k)*exp(datenBacktest(i,:)');
% % % %         EqualPuL(i) = log(EqualPuL(i));
% % % %         OptPuL(i) = log(OptPuL(i));
% % % %     end
% % % %     %     rechne geometrisch annualisierten mean return für gleichgeichtetes
% % % %     %     und optimiertes Portfolio aus
% % % %     %     Gesamtrendite
% % % %     EqualPuL_all = sum(EqualPuL);
% % % %     OptPuL_all = sum(OptPuL);
% % % %     EqualPuL_ann = ((1+EqualPuL_all)^(1/3)-1);
% % % %     OptPuL_ann = ((1+OptPuL_all)^(1/3)-1);
% % % %     Output = OptVaRStat(weights(:,end),daten,Spec);
% % % % end
% % % %  %  Portfolio Optimierug für MVGARCH und elliptische Copulas
% % % % % für out-of-sample
% % % % ConSet = portcons('AssetLims',ones(1,k)*Spec.LB,ones(1,k)*Spec.UB,'Custom',ones(1,k),1);
% % % % for i=1:Spec.ForecastNumb
% % % %     if iscell(Rt_pred{i})
% % % %         Rt_pred_convert(:,:,i)=Rt_pred{i}{1};
% % % %     else
% % % %         Rt_pred_convert = Rt_pred;
% % % %     end
% % % %     ExpCovariance = corr2cov(sqrt(ht_pred(i,:)),Rt_pred_convert(:,:,i));
% % % %     ExpReturn = AR_pred(i,:);
% % % %     [PortRisk(i,:), PortReturn(i,:), PortWts(:,:,i)] = portopt(ExpReturn,ExpCovariance, 1, [], ConSet);
% % % %     datenBacktest = daten(end-size(PortWts,1)+1:end,:);
% % % %     PortWts(PortWts<Spec.LB) = Spec.LB;
% % % %     PortWts(PortWts>Spec.UB) = Spec.UB;
% % % %     for i=1:size(PortWts,1)
% % % %         OptPuL(i,1) = PortWts(i,1:k)*exp(datenBacktest(i,:)');
% % % %         EqualPuL(i,1) = 1/k*ones(1,k)*exp(datenBacktest(i,:)');
% % % %         EqualPuL(i) = log(EqualPuL(i));
% % % %         OptPuL(i) = log(OptPuL(i));
% % % %     end
% % % %     %     rechne geometrisch annualisierten mean return für gleichgeichtetes
% % % %     %     und optimiertes Portfolio aus
% % % %     %     Gesamtrendite
% % % %     EqualPuL_all = sum(EqualPuL);
% % % %     OptPuL_all = sum(OptPuL);
% % % %     EqualPuL_ann = ((1+EqualPuL_all)^(1/3)-1);
% % % %     OptPuL_ann = ((1+OptPuL_all)^(1/3)-1);
% % % %     Output = OptVaRStat(weights(:,end),daten,Spec);
% % % % end
