function [ basis, values ] = lda( data, labels )
%LDA Performs Fisher linear discriminant analysis
    % Compute the mean-centered matrix of the observations
    
    [~, b, ~] = unique(labels);
    b = [b ; length(labels) + 1];
    cardinals = diag(pdist2(b, b), -1);

    classmats = mat2cell(data, cardinals);
    
    % Quantities of interest:
    K = length(cardinals);  % Number of classes
    F = size(data, 2);      % Number of features
    
    mu = mean(data);
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
    
    % Eigenvalue decomposition of Sw^-1 * Sb
    [ basis, values ] = eig(pinv(Sw) * Sb, 'vector');
end

function [U] = center(mat, mu)
    U = bsxfun(@minus, mat, mu);
end

function [U] = scatter(mat, mu, card)
    cent = center(mat, mu);
    U = (cent' * cent) / card;
end
