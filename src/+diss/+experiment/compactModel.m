function compact = compactModel(model, retainTrainingData)
%COMPACTMODEL Remove repeated histories from stored backtest window models.

arguments
    model (1, 1) struct
    retainTrainingData (1, 1) logical = false
end

compact = model;
if retainTrainingData
    compact.IsCompact = false;
    return
end
if isfield(compact, 'TrainingReturns')
    compact = rmfield(compact, 'TrainingReturns');
end
if isfield(compact, 'Marginal')
    largeFields = {'Residuals', 'ConditionalVariances', ...
        'StandardizedResiduals'};
    for index = 1:numel(largeFields)
        field = largeFields{index};
        if isfield(compact.Marginal, field)
            compact.Marginal = rmfield(compact.Marginal, field);
        end
    end
end
compact.IsCompact = true;

end
