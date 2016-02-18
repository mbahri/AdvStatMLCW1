load('correct_pca')
load('correct_pca_white')
load('correct_lda')
load('correct_lpp_heat')
load('correct_lpp_knn_7')
load('correct_ica')

figure
N = length(error_pca);
Nnpp = size(error_lpp_heat, 2);
Nl = size(error_lda,2);

X = 5*(1:N);
Xlpp = 5*(1:Nnpp);
Xl = 5*(1:Nl);

plot(X, mean(error_pca), ...
    X, mean(error_pca_white), ...
    X, mean(error_ica));
hold on;
    plot(Xlpp, mean(error_lpp_heat), ...
    Xlpp, mean(error_lpp_knn_7));
plot(Xl, mean(error_lda));
legend('PCA', 'Whitened PCA', 'FastICA', 'LPP heat kernel', 'LPP KNN 7', 'LDA');
title('Error rate vs number of dimensions - PIE');