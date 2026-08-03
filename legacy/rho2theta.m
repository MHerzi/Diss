function theta = rho2theta(rho)
%RHO2THETA Return the strict upper triangle of a correlation matrix.
%   Coefficients retain the legacy row-wise order: rho12, rho13, ...,
%   rho1k, rho23, ..., rho(k-1)k.

    validateattributes(rho, {'double'}, ...
        {'2d', 'real', 'finite', 'square', 'nonempty'}, mfilename, 'rho');
    if norm(rho - rho', 'fro') > 1e-10
        error('Diss:Correlation:NonSymmetricMatrix', ...
            'rho must be symmetric.');
    end
    transposedRho = rho.';
    theta = transposedRho(tril(true(size(rho, 1)), -1));
end
