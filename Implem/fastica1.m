 function [ W ] = fastica1( obs )
%FASTICA Implements the FastICA algorithm
    
    % Data whitening using PCA
    whiteningMatrix = pcomp(obs, 'yes');
    X = bsxfun(@minus, obs, mean(obs));
%     X = obs - ones(size(obs,1),1) * mean(obs);
    X = (X*whiteningMatrix)';
    
    % Quantities of interest:
    N = size(X, 2);     % Number of observations
    F = size(X, 1);     % Number of features
    
    % Tolerance
    e = 1e-4;
    MAXITER = 500;
    
    % Vector of ones
    OO = ones(N,1);
    
    % Start with random orthonormal vectors
    W = orth(randn(F,F));
%     E = norm(eye(F)-W'*W,'fro');
%     W = randn(F,F);
        
    for i = 1:F
        k = 0;
        fprintf('Computing the %dth independent component.\n', i);
        % ith independent component
        WW = W(:,i);
        % Copy to allow comparison between two iterations
        WWp = zeros(F,1);
        % Loop until WW doesn't change from an iteration to another
        while norm(WW - WWp) > e && k < MAXITER
            k = k + 1;
%             fprintf('Iteration %d for %d\n', k, i);
%             fprintf('%f\n', norm(WW - WWp));
            % Update the copy
            WWp = WW;
%             fprintf('WWp: %f\n', norm(WWp));
            % Avoids computing this product twice
            Y = WW'*X;
            % Newton step
%             WW = (X*tanh(Y)' - (OO' - (tanh(Y).^2))*OO*WW)/N;
            WW = (X*(Y.^4)' - (4*Y.^3)*OO*WW)/N;

            % Gram-Schmidt step
            for j = 1:(i-1)
%                 fprintf('Orthogonalizing wrt %d\n', j);
                WW = WW - W(:,j)*WW'*W(:,j);
            end
            % Normalize WW
            WW = WW/sqrt(WW'*WW);
        end
        if k >= MAXITER
            fprintf('Max number of iterations reached.\n');
%         else
%             fprintf('Converged in %d iterations\n', k);
        end
        % Update the matrix
        W(:,i) = WW;
    end
%     S = (W'*X)'*whiteningMatrix';
    W = (W * whiteningMatrix')';
 end
