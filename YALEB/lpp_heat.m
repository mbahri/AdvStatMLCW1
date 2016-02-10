 function [ W ] = lpp_heat( obs, K, T, center)
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
    
    % Construct S and D
    S = zeros(N,N);
    % Find indices of the K nearest neighbours of each element and their
    % distances
%     [Neig, Dist] = knnsearch(X,X,'K',K,'Distance','cosine');
    [Neig, Dist] = knnsearch(X,X,'K',K);

    % Heat kernel
    Dist = exp(- (Dist .^ 2) / T);

    % Set S(i,j) to 1 if j is a neighbour of i
    for i=1:N
       for j=2:size(Neig,2)
           k = Neig(i,j);
           S(i,k) = Dist(i,j);
       end
    end
    S = (S + S') / 2;       % Make S symmetric
    D = diag(sum(S > 0,1)); % Find D
    
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
