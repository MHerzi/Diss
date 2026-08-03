function matrices = dccDiagonalMatrixSequence(values)
%DCCDIAGONALMATRIXSEQUENCE Convert k-by-p values to k-by-k-by-p pages.
    validateattributes(values, {'double'}, ...
        {'2d', 'real', 'finite'}, mfilename, 'values');
    [seriesCount, order] = size(values);
    matrices = zeros(seriesCount, seriesCount, order, 'like', values);
    for lag = 1:order
        matrices(:, :, lag) = diag(values(:, lag));
    end
end
