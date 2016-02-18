clear; clc; close all force

addpath('../Implem/')

% Suppress annoying warnings about deprecated function
warning('off', 'bioinfo:knnclassify:incompatibility');

load YaleB_32x32;

[nSmp,nFea] = size(fea);
fea1 = zeros(nSmp, 4*nFea);
for ii = 1:nSmp
    temp = fea(ii,:);
    temp = reshape(temp, 32, 32);
    temp = imresize(temp, 2);
    temp = temp(:);
    fea1(ii,:) = temp;
end

[nSmp,nFea] = size(fea1);

% Waitbars to show progress
h1 = waitbar(0,'Global');
h2 = waitbar(0,'Current permutation');

dim = 1; %%check recognition rate every dim dimensions (change it appropriatly for PCA, LDA etc)

error = [];
for jj = 1:20  %%%run for 20 random pertrurbations
    waitbar(0, h2, 'Current permutation');
    waitbar(jj/20, h1, sprintf('Global: permutation %d/20', jj));
    
    eval(['load 5Train/' num2str(jj)]); %%% load the pertrurbation number jj

    fea_Train = fea1(trainIdx,:);  %%take the training data
    fea_Test = fea1(testIdx,:);    %%take the test data

    gnd_Train = gnd(trainIdx);
    gnd_Test = gnd(testIdx);

    fprintf('[%d] - Computing the transformation matrix.\n', jj);
    U_reduc = pcomp(fea_Train);
%     U_reduc = pcomp(fea_Train, 'whiten', true);
%     U_reduc = lda(fea_Train, gnd_Train);
%     U_reduc = lpp_heat(fea_Train);
%     U_reduc = lpp_knn(fea_Train, 'k', 7);
%     U_reduc = fastica_lowdim(fea_Train);

    fprintf('[%d] - Matrix computation done.\n', jj);
    
    oldfea = fea_Train*U_reduc;  %%training data 
    
    newfea = fea_Test*U_reduc;   %%test data

    mg = mean(oldfea, 1);  %%compute the training mean
    mg_oldfea = repmat(mg,  size(oldfea,1), 1);
    oldfea = oldfea - mg_oldfea; %%subtract the mean 

    mg_newfea = repmat(mg,  size(newfea,1), 1);
    newfea = newfea - mg_newfea;  %%subtract the mean 

    len     = 1:dim:size(newfea, 2);
    correct = zeros(1, length(1:dim:size(newfea, 2)));
    for ii = 1:length(len)  %%for each dimension perform classification
        waitbar(ii/length(len), h2, sprintf('Current: iteration %d/%d', ii, length(len)));
        fprintf('[%d] - Computing class. rate - iteration %d\n', jj, ii);
        ii;
        Sample = newfea(:, 1:len(ii));
        Training = oldfea(:, 1:len(ii));
        Group = gnd_Train;
        k = 1;
        distance = 'cosine';
        Class = knnclassify(Sample, Training , Group, k, distance); %%nearest neighbor classification

        correct(ii) = length(find(Class-gnd_Test == 0));
    end

    correct = correct./length(gnd_Test); %%compute the correct classification rate
    error = [error; 1- correct];

end

fprintf('Max score: %f\n', max(correct));

close(h1);
close(h2);

plot(mean(error,1)); %%plotting the error 
save('correct_pca.mat', 'correct', 'error');
