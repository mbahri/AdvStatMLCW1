load('correct_pca')
error_pca = mean(error);
load('correct_pca_white')
error_pca_white = mean(error);
load('correct_lda')
error_lda = mean(error);
load('correct_lpp_heat')
error_lpp_heat = mean(error);
load('correct_lpp_knn_7')
error_lpp_knn_7 = mean(error);
load('correct_ica')
error_ica = mean(error);

figure
N = length(error_pca);
Nnpp = size(error_lpp_heat, 2);
Nl = size(error_lda,2);

X = 1:N;
Xlpp = 1:Nnpp;
Xl = 1:Nl;

plot(X, error_pca,...
    X, error_pca_white,...
    X, error_ica);
hold on;
    plot(Xlpp, error_lpp_heat, ...
    Xlpp, error_lpp_knn_7);
plot(Xl, error_lda);
legend('PCA', 'Whitened PCA', 'FastICA', 'LPP heat kernel', 'LPP KNN 7', 'LDA');
title('Error rate vs number of dimensions - YALE-B');