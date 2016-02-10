function [ basis ] = pcomp( data, scale )
%PCOMP Performs PCA on the data
%   Calls pcomp_std or pcomp_trick depending on the shape of the data

    N = size(data, 1);
    F = size(data, 2);
    
    if N > F
        basis = pcomp_std(data, scale);
    else
        basis = pcomp_trick(data, scale);
    end
end

