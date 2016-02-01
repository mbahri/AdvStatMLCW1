function [ basis, values ] = pcomp( data, scale)
%PCOMP Returns the change of basis matrix for PCA
    % Compute the mean-centered matrix of the observations
    
%     function [V newX D] = myPCA(X)
%         X = bsxfun(@minus, X, mean(X,1));           %# zero-center
%         C = (X'*X)./(size(X,1)-1);                  %'# cov(X)
% 
%         [V D] = eig(C);
%         [D order] = sort(diag(D), 'descend');       %# sort cols high to low
%         V = V(:,order);
% 
%         newX = X*V(:,1:end);
%     end

    U = bsxfun(@minus, data, mean(data));
    [basis, L] = eig(U'*U/(size(U,1) - 1), 'vector');
    [L, Indi] = sort(L, 'descend');
    
    switch scale
        case 'no'
            basis = basis(:,Indi);
        case 'yes'
            basis = basis(:,Indi);
            basis = basis * diag(L)^-(1/2);
        otherwise
            basis = basis(:,Indi);
    end
    
    values = L;
end

