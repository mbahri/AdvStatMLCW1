 function [ W ] = lpp_heat( obs, T, center)
%LPP Locality preserving projections with heat kernel
    %   T = 1e7 works well
        
    % Quantities of interest:
    N = size(obs, 1);      % Number of observations
    
    if center
        % Global mean
        mu = mean(obs);
        % Mean-centered data
        X = bsxfun(@minus, obs, mu);
    else
        X = obs;
    end
        
    % Distances squared
    DotProd = X*X';
    NormsSq = repmat(diag(DotProd),1,N);

    Dist = NormsSq + NormsSq' - 2 * DotProd;
    
    % Construct S with heat kernel and D by summing
    S = exp(- Dist / T) - eye(N,N); % Remove the ones on the diagonal
    D = diag(sum(S,1));             % Find D
    
    % Square root of D
    SqD = D^(1/2);
        
    [V,L,~] = svd(SqD*X*(X')*SqD);
    U = X' * SqD * V * L^-1;
    
    Xp = U'*X';
    [Q, ~, ~] = svd(Xp*(D-S)*Xp');
    
    % Reverse the vectors
    Ind = 190:-1:1;
    Q = Q(:,Ind);
    
    W = U*Q;
 end
