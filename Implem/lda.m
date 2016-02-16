 function [ W ] = lda( obs, labels )
%LDA Performs Fisher linear discriminant analysis
    % Compute the mean-centered matrix of the observations
    
    % We first group the observations by class. This is already the case
    % for YaleB but it might not be true in general.
%     data = sortrows([labels, obs]);
%     labels = data(:,1);
%     data = data(:,2:end);

    data = obs;

    % Here we use the "unique" function to get the start index of the
    % different classes.
    [~, b, ~] = unique(labels);
    % We append the last index of the class vector + 1 for the next step
    b = [b ; length(labels) + 1];
    % We use pdist2 to compute the pairwise differences of the start
    % indices, this gives us the number of elements in each class. We will
    % use this to split the data matrix into class matrices.
    % pdist2 computes all the pairwise differences and returns a matrix.
    % The values that we need are not on the main diagonal of this matrix
    % (which is full of zeros since it computes the diff for the pair x, x)
    % but on the diagonal right above the main.
    cardinals = diag(pdist2(b, b), -1);
    
    % Quantities of interest:
    F = size(data, 2);      % Number of features
    N = size(data, 1);      % Number of observations
    C = size(cardinals, 1); % Number of classes
    % Global mean
    mu = mean(data);
    % Mean-centered data
    X = bsxfun(@minus, data, mu);
    
    % Compute the matrix M
    E_matrices = arrayfun(@(N)(ones(N,1)*ones(1,N)/N), cardinals, ...
        'UniformOutput', 0);
    M = blkdiag(E_matrices{:});
    
    % Compute Xw
    Xw = X'*(eye(N,N) - M);
    
    % Eigenvalue decomposition of Xw'*Xw,
    % using svd might be more numerically stable.
    [ Vw, Lw, ~ ] = svd(Xw'*Xw);
    Vw = Vw(:,1:end-C);
    Lw = Lw(1:end-C, 1:end-C);
    
    % U
    U = Xw * Vw * Lw^-1;
    % Xb and Q = Vb
    Xb = U'*X'*M;
    [ Vb, ~, ~ ] = svd(Xb*Xb');
    Vb = Vb(:,1:C-1);
    
    % W
    W = U*Vb;
 end