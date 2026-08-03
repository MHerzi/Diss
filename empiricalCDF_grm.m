function probabilities = empiricalCDF_grm(data, evaluationPoints)
%EMPIRICALCDF_GRM Backward-compatible wrapper for empiricalCDF.
    if nargin < 2
        probabilities = empiricalCDF(data);
    else
        probabilities = empiricalCDF(data, evaluationPoints);
    end
end
