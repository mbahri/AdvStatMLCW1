function basis = pcomp_trick( data, scale)
%PCOMP Returns the change of basis matrix for PCA using the kernel trick

    % Compute the mean-centered matrix of the observations
    U = bsxfun(@minus, data, mean(data));
    U_t = U';
    
    N = size(U,1) - 1;
    
    % Find the eigenvectors and eigenvalues of the covariance matrix
    [B, L, ~] = svd(U*U_t/N);
    B = B(:,1:N);
    L = L(1:N, 1:N);
    % Get the eigenvectors, divide by sqrt(N) to normalize
    Phi = U_t*B/sqrt(N);
    
    % Whitened PCA: rescale the basis by the inverse of the square root of
    % diag(eigenvalues)
    if scale
        basis = Phi * L^-1;
    else
        basis = Phi * L^-(1/2);
    end
end

