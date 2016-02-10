 function [ W ] = lpp_knn( obs, K, center)
%LPP Locality preserving projections with standard knn

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
    
    % Construct S and D
    S = zeros(N,N);
    % Find indices of the K nearest neighbours of each element
%     [Neig, Dist] = knnsearch(X,X,'K',K,'Distance','cosine');
    [Neig, ~] = knnsearch(X,X,'K',K);

    % Set S(i,j) to 1 if j is a neighbour of i
    for i=1:N
        S(i,Neig(i,2:end)) = 1;
    end
    S = max(S, S');     % Make S symmetric
    D = diag(sum(S,1)); % Find D
    
    SqD = D^(1/2);      % Square root of D
    
    X = X';
    
    [V,L,~] = svd(SqD*(X')*X*SqD);
    U = X * SqD * V * L^-1;
    
    Xp = U'*X;
    [Q, ~, ~] = svd(Xp*(D-S)*Xp');
    
    % Reverse the vectors
    Ind = 190:-1:1;
    Q = Q(:,Ind);
    
    W = U*Q;
 end
