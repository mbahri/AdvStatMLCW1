function [ basis, values ] = lda( obs, labels )
%LDA Performs Fisher linear discriminant analysis
    % Compute the mean-centered matrix of the observations
    
    % We first group the observations by class. This is already the case
    % for YaleB but it might not be true in general.
    data = sortrows([labels, obs]);
    data = data(:,2:end);
    
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

    % Split the data matrix into submatrices (one for each class), here we
    % provide the number of rows to retain in each array
    classmats = mat2cell(data, cardinals);
    
    % Quantities of interest:
    K = length(cardinals);  % Number of classes
    F = size(data, 2);      % Number of features
    
    % Global mean
    mu = mean(data);
    % Class means
    classmeans = cellfun(@mean, classmats, 'UniformOutput', 0);
    
    % Compute between-class scatter matrix by summing the individual
    % matrices
    Sb = zeros(F, F);
    for i = 1:K
        Scatt = (classmeans{i} - mu);
        Sb = Sb + Scatt'*Scatt;
    end
    
    % Compute within-class scatter matrices, avoid recomputing the means,
    % divides by the cardinals
    cardcell = mat2cell(cardinals, ones(1,K));
    scattermats = cellfun(@scatter, classmats, classmeans, cardcell,'UniformOutput', 0);
    
    % Within-class scatter matrix as the sum of all the individual matrices
    Sw = zeros(F, F);
    for i = 1:K
        Sw = Sw + scattermats{i};
    end
    
    % Eigenvalue decomposition of Sw^-1 * Sb, we use pinv to avoid issues,
    % using svd might be more numerically stable.
    [ basis, values, ~ ] = svd(pinv(Sw) * Sb);
end

function [U] = center(mat, mu)
    U = bsxfun(@minus, mat, mu);
end

function [U] = scatter(mat, mu, card)
    cent = center(mat, mu);
    U = (cent' * cent) / card;
end
