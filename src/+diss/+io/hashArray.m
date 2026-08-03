function digest = hashArray(values)
%HASHARRAY Return a SHA-256 digest for a real numeric or logical array.

arguments
    values {mustBeNumericOrLogical}
end

shape = uint64(size(values));
classBytes = unicode2native(class(values), 'UTF-8');
shapeBytes = typecast(shape(:), 'uint8');
if islogical(values)
    valueBytes = uint8(values(:));
else
    valueBytes = typecast(values(:), 'uint8');
end
digest = sha256([classBytes(:); shapeBytes(:); valueBytes(:)]);

end

function mustBeNumericOrLogical(value)
if ~(isnumeric(value) || islogical(value)) || ~isreal(value)
    error('diss:io:UnsupportedHashArray', ...
        'Array hashing requires real numeric or logical input.');
end
end

function digest = sha256(bytes)
engine = java.security.MessageDigest.getInstance('SHA-256');
engine.update(typecast(uint8(bytes(:)), 'int8'));
rawDigest = typecast(engine.digest(), 'uint8');
digest = lower(strjoin(compose("%02x", rawDigest(:)'), ""));
end
