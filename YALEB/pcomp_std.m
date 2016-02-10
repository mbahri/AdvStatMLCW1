function basis = pcomp_std( data, scale)
%PCOMP Returns the change of basis matrix for PCA

    % Compute the mean-centered matrix of the observations
    U = bsxfun(@minus, data, mean(data));
    
    % Find the eigenvectors and eigenvalues of the covariance matrix
    [B, L, ~] = svd(U'*U/(size(U,1) - 1));
    
    % Whitened PCA: rescale the basis by the inverse of the square root of
    % diag(eigenvalues)
    switch scale
        case 'no'
            basis = B;
        case 'yes'
            basis = B * L^-(1/2);
        otherwise
            basis = B;
    end    
end

