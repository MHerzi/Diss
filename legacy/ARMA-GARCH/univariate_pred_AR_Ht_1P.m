function [ar_pred, ht_pred] = univariate_pred_AR_Ht_1P(data,GARCH,const)

% mache ein-Tages vohersage für die GARCH-Modelle
% d.h. schätze für jede Vorhersage ein eigenes GARCH-Modell

[t,k] = size(data);
% ------------------------------------------------------
% AR-forecast
% nehme die AR-Koeffizienten aus dem GARCH-Modell und mache damit die
% Vorhersage; y_t+1 = omega+y_t;
% erzeuge eine gelaggte Datenmatrix
y=cell(k,1);
for i=1:k
    if GARCH{i}.arlag==1 %bei einem AR(1)-Modell dürfen die Daten nicht gelaggt werden
        if const==1
            y{i} = [ones(size(GARCH{i}.data,1),1) GARCH{i}.data]; %setze eine Spalte von Einsne dazu, wenn mit Konstante geschätzt wurde
        else
            y{i} = GARCH{i}.data;
        end
    elseif GARCH{i}.arlag>1
        if const==1
            y{i} = [ones(size(GARCH{i}.data),1) GARCH{i}.data];
        else
            y{i} = GARCH{i}.data;
        end
        y{i} = mlag(GARCH{i}.data,GARCH{i}.arlag-1);
        y{i} = trimr(y{i},GARCH{i}.data);
    end
end
% für die Vorhersage nehmen den letzten Datenzeitpunkt und mache regression
for i=1:k
    ar_pred(:,i) = GARCH{i}.ParamsAR'*y{i}(end,:)';
end


% ------------------------------------------------------------------------
% GARCH-process
% ------------------------------------------------------------------------
% create new ht_variable which last element can be used in recursive
% estimation of predicted ht's
ht=cell(k,1);
resid_t_neg=cell(k,1);
for i=1:k
    ht{i} = GARCH{i}.ht;
    %     make transformation for variance if necessary (for formulas see
    %     Cappiello et al(2006)
    if strcmp(GARCH{i}.GARCH,'TGARCH')
        ht{i} = sqrt(ht{i});
    elseif strcmp(GARCH{i}.GARCH,'AVGARCH')
        ht{i} = sqrt(ht{i});
    elseif strcmp(GARCH{i}.GARCH,'NGARCH')
        ht{i} = ht{i}.^(2/GARCH{i}.ParamsGARCH(4));
    elseif strcmp(GARCH{i}.GARCH,'APGARCH')
        ht{i} = ht{i}.^(2/GARCH{i}.ParamsGARCH(5));
    end
end

% nehme für die einperiodige Vorhersage den letzen Datenzeitpunkt und mache
% damit rekursiven forecast
ht_pred=zeros(1,k);
for i=1:k
    if strcmp(GARCH{i}.GARCH,'GARCH') == 1
        ht_pred(:,i) = GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*GARCH{i}.Innovations(end).^2 + GARCH{i}.ParamsGARCH(3)*ht{i}(end);
    elseif strcmp(GARCH{i}.GARCH,'EGARCH') == 1
        %             !!! EGARCH parameters = ht = omega + gamma + alpha + beta !!!
        ht_pred(:,i) = exp(GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(3)*abs(GARCH{i}.Innovations(end))./sqrt(ht_pred{i}(j-1)) + GARCH{i}.ParamsGARCH(2)*GARCH{i}.Innovations(end)./sqrt(ht{i}(end)) + GARCH{i}.ParamsGARCH(4)*log(ht{i}(end)));
    elseif strcmp(GARCH{i}.GARCH,'TGARCH') == 1
        if GARCH{i}.Innovations(end)>=0 %setzte Innovationen größer gleich als Null sind auf Null
            negresid=0;
        else
            negresid=GARCH{i}.Innovations(end);
        end
        ht_pred(:,i) =    GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*abs(GARCH{i}.Innovations(end)) + GARCH{i}.ParamsGARCH(3)*negresid+GARCH{i}.ParamsGARCH(4)*ht{i}(end);
    elseif strcmp(GARCH{i}.GARCH,'GJRGARCH') == 1
        if GARCH{i}.Innovations(end)>=0
            negresid=0;
        else
            negresid=GARCH{i}.Innovations(end);
        end
        ht_pred(:,i) = GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*GARCH{i}.Innovations(end).^2 + GARCH{i}.ParamsGARCH(3)*negresid.^2+GARCH{i}.ParamsGARCH(4)*ht{i}(end);
    elseif strcmp(GARCH{i}.GARCH,'AVGARCH') == 1
        ht_pred(:,i) = GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*abs(GARCH{i}.Innovations(end)) + GARCH{i}.ParamsGARCH(3)*ht{i}(end);
    elseif strcmp(GARCH{i}.GARCH,'NGARCH') == 1
        %             ht_pred{i}(j,:)= ((GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*abs(resid_t{i}(j-1,1))^(GARCH{i}.ParamsGARCH(4)) +  GARCH{i}.ParamsGARCH(3)*ht_pred{i}(j-1)^GARCH{i}.ParamsGARCH(4))^(1/GARCH{i}.ParamsGARCH(4)))^2;
        ht_pred(:,i) = ((GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*abs(GARCH{i}.Innovations(end))^(GARCH{i}.ParamsGARCH(4)) +  GARCH{i}.ParamsGARCH(3)*ht{i}(end)^GARCH{i}.ParamsGARCH(4)));
    elseif strcmp(GARCH{i}.GARCH,'NAGARCH') == 1
        ht_pred(:,i) = GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*GARCH{i}.Innovations(end)+GARCH{i}.ParamsGARCH(4)*sqrt(ht{i}(end))^2 + GARCH{i}.ParamsGARCH(3)*ht{i}(end);
    elseif strcmp(GARCH{i}.GARCH,'APGARCH') == 1
        if GARCH{i}.Innovations(end)>=0
            negresid=0;
        else
            negresid=GARCH{i}.Innovations(end);
        end
        ht_pred(:,i) = (GARCH{i}.ParamsGARCH(1) + GARCH{i}.ParamsGARCH(2)*abs(GARCH{i}.Innovations(end))^GARCH{i}.ParamsGARCH(4) + GARCH{i}.ParamsGARCH(3)*abs(negresid)^GARCH{i}.ParamsGARCH(4) + GARCH{i}.ParamsGARCH(3)*ht{i}(end));
    end
end

% make transformations to ensure output is VARIANCE and NOT standard
% deviation
for i=1:k
    if strcmp(GARCH{i}.GARCH,'TGARCH') || strcmp(GARCH{i}.GARCH,'AVGARCH')
        ht_pred(:,i) = ht_pred(:,i).^2;
    elseif strcmp(GARCH{i}.GARCH,'NGARCH')
        ht_pred(:,i)=ht_pred(:,i).^(2/(GARCH{i}.ParamsGARCH(4)));
    elseif strcmp(GARCH{i}.GARCH,'APGARCH')
        ht_pred(:,i)=ht_pred(:,i).^(2/(GARCH{i}.ParamsGARCH(5)));
    end
end



