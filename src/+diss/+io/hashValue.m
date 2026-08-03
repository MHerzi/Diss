function digest = hashValue(value)
%HASHVALUE Return a deterministic SHA-256 digest for JSON-compatible data.

canonical = canonicalize(value);
encoded = unicode2native(jsonencode(canonical), 'UTF-8');
engine = java.security.MessageDigest.getInstance('SHA-256');
engine.update(typecast(uint8(encoded(:)), 'int8'));
rawDigest = typecast(engine.digest(), 'uint8');
digest = lower(strjoin(compose("%02x", rawDigest(:)'), ""));

end

function output = canonicalize(input)
if isstruct(input)
    names = sort(fieldnames(input));
    output = orderfields(input, names);
    for element = 1:numel(output)
        for index = 1:numel(names)
            name = names{index};
            output(element).(name) = canonicalize(output(element).(name));
        end
    end
elseif iscell(input)
    output = input;
    for index = 1:numel(input)
        output{index} = canonicalize(input{index});
    end
elseif isdatetime(input) || isduration(input) || iscalendarDuration(input)
    output = string(input);
else
    output = input;
end
end
