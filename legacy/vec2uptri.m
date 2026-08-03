function out = vec2uptri(data)
% Helper-Function: Transforms vector into upper triangular matrix

% error-checking
if size(data,2) >1 || isscalar(data)
    error('k x 1 vector as input needed')
end

k = size(data,1);

transformeddata = zeros(.5*k*(k-1),.5*k*(k-1));






