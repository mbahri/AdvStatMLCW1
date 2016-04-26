function [ Uc, Ur, T, E ] = brpca( X, lambda, r )

% FOR TESTING ONLY!!!
%warning('off','all')

%BRPCA Solves bilinear robust PCA via ALM
% min ||Uc|| + ||Ur|| + sum(labmda_n*||En||_1)
% s.t for all n Xn = UcTnUr^T + En

% Constant
Ir = eye(r);

% Dimensions of the input
[n, m] = size(X{1});
Nobs = length(X);
% Sparse error terms
E = mat2cell(zeros(n, m*Nobs), n, repmat(m, 1, Nobs));
% Low-rank subspace
% T = zeros(r, r, Nobs);
T = mat2cell(repmat(Ir, 1, Nobs), r, repmat(r, 1, Nobs));
% Lagrange multipliers
Y = mat2cell(zeros(n, m*Nobs), n, repmat(m, 1, Nobs));
% Left basis
Uc = eye(n,r);
% Right basis
Ur = eye(m,r);

norms = cellfun(@norm, X);

% Penalty parameter
% mu = Nobs*1.25/sum(norms); % this one can be tuned
mu = 1.25
% Upper bound on the penalty
mu_bar = mu * 1e7;
% Coefficient for updating the penalty parameter
rho = 1.5 ;

% Convergence
converged = false;
niter = 0;
MAXITER = 10;
tol = 1e-5;

while ~converged && niter<MAXITER
    niter = niter + 1
    
    % Update Uc first
    Uc_num = @(X, Y, E, T) ((mu * X + Y - mu*E)*Ur*T');
    Uc_denom = @(T) (mu*T*(Ur'*Ur)*T');
    
    Uc = cellsum(cellfun(Uc_num, X, Y, E, T, 'UniformOutput', false))/...
        (2*Ir + cellsum(cellfun(Uc_denom, T, 'UniformOutput', false)));
    
    % Then Ur
    Ur_num = @(X, Y, E, T) ((mu * X' - Y' - mu*E')*Uc*T);
    Ur_denom = @(T) (mu*T'*(Uc'*Uc)*T);
    
    Ur = cellsum(cellfun(Ur_num, X, Y, E, T, 'UniformOutput', false))/...
        (2*Ir + cellsum(cellfun(Ur_denom, T, 'UniformOutput', false)));
    
    % Now update T
    K = pinv(kron(Ur, Uc));
%     solve_for_T = @(Y, X, E) reshape(K\reshape(( (1/mu)*Y + X - E), m*n, 1), r, r);
    solve_for_T = @(Y, X, E) reshape(K*reshape(( (1/mu)*Y + X - E), m*n, 1), r, r);
    T = cellfun(solve_for_T, Y, X, E, 'UniformOutput', false);
    
    % Update E
    update_basis = @(T) (Uc * T * Ur');
    A = cellfun(update_basis, T, 'UniformOutput', false); % A = UcTnUr^T

    update_E = @(X, A, Y)(soft_shrinkage(X, A, Y, mu, lambda));
    E = cellfun(update_E, X, A, Y, 'UniformOutput', false);

    % Update the lagrange multiplier
    update_lagrange = @(Y, X, A, E) (Y - mu*(X - A - E));
    Y = cellfun(update_lagrange, Y, X, A, E, 'UniformOutput', false);
    
    mu = min(mu_bar, rho * mu)
end

end

function [E] = soft_shrinkage(X, A, Y, mu, lambda)
temp_T = X - A + (1/mu)*Y;
E = max(temp_T - lambda/mu, 0);
E = E + min(temp_T + lambda/mu, 0);
end

function [M] = cellsum(A)
catA=cat(3, A{:});
M = sum(catA,3);
end