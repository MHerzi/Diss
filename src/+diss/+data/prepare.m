function dataset = prepare(data, options)
%PREPARE Convert supported input formats into a validated return dataset.

arguments
    data
    options (1, 1) struct
end

originalType = string(class(data));

if istimetable(data)
    ensureNumericVariables(data);
    returns = data{:, :};
    time = data.Properties.RowTimes;
    variableNames = string(data.Properties.VariableNames);
elseif istable(data)
    ensureNumericVariables(data);
    returns = data{:, :};
    time = [];
    variableNames = string(data.Properties.VariableNames);
elseif isnumeric(data) && ismatrix(data) && ~isempty(data)
    returns = data;
    time = [];
    variableNames = compose("Series%d", 1:size(data, 2));
else
    error('diss:data:UnsupportedType', ...
        'Input data must be a numeric matrix, table or timetable.');
end

if ~isreal(returns)
    error('diss:data:ComplexReturns', ...
        'Return observations must be real-valued.');
end

returns = double(returns);
originalObservationCount = size(returns, 1);
observationIndex = (1:originalObservationCount)';
invalidRows = any(~isfinite(returns), 2);
removedRows = find(invalidRows);

switch options.missingAction
    case "error"
        if any(invalidRows)
            error('diss:data:NonfiniteReturns', ...
                ['Return data contain NaN or Inf in row %d. Set ', ...
                'config.data.missingAction to "omitRows" to remove ', ...
                'incomplete observations explicitly.'], removedRows(1));
        end
    case "omitRows"
        returns(invalidRows, :) = [];
        observationIndex(invalidRows) = [];
        if ~isempty(time)
            time(invalidRows, :) = [];
        end
    otherwise
        error('diss:data:InvalidMissingAction', ...
            'Unsupported missing-data action: %s.', options.missingAction);
end

if size(returns, 1) < 2
    error('diss:data:TooFewObservations', ...
        'At least two complete return observations are required.');
end

dataset = struct();
dataset.Returns = returns;
dataset.Time = time;
dataset.VariableNames = reshape(variableNames, 1, []);
dataset.ObservationIndex = observationIndex;
dataset.RemovedRows = removedRows;
dataset.ObservationCount = size(returns, 1);
dataset.SeriesCount = size(returns, 2);
dataset.OriginalType = originalType;

end

function ensureNumericVariables(data)
numericVariables = varfun(@isnumeric, data, 'OutputFormat', 'uniform');
if ~all(numericVariables)
    invalidName = data.Properties.VariableNames{find(~numericVariables, 1)};
    error('diss:data:NonnumericVariable', ...
        'Table variable %s is not numeric.', invalidName);
end
end
