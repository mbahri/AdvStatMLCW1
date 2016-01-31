function [ basis ] = pcomp( data, scale)
%PCOMP Returns the change of basis matrix for PCA
    % Compute the mean-centered matrix of the observations

    U = bsxfun(@minus, data, mean(data));
    [basis, L] = eig(U'*U/(size(U,1) - 1), 'vector');
    [L, Indi] = sort(L, 'descend');
    
    switch scale
        case 'no'
            basis = basis(:,Indi);
        case 'yes'
            basis = basis(:,Indi);
            basis = basis * diag(L)^-1;
        otherwise
            basis = basis(:,Indi);
    end
end

