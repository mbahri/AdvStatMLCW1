clear; clc; close all force

addpath('../Implem/')

% Suppress annoying warnings about deprecated function
warning('off', 'bioinfo:knnclassify:incompatibility');

load Train5_64;
load fea64;
load gnd64;

fea = fea64; clear fea64;
gnd = gnd64; clear gnd64;
Train = Train5_64; clear Train5_64;

fea1 = fea;

% Waitbars to show progress
h1 = waitbar(0,'Global');
h2 = waitbar(0,'Current permutation');

dim =5; %%check recognition rate every dim dimensions (change it appropriatly for PCA, LDA etc

error = [];
for jj = 1:20
    waitbar(0, h2, 'Current permutation');
    waitbar(jj/20, h1, sprintf('Global: permutation %d/20', jj));

    TrainIdx = Train(jj, :);
    TestIdx = 1:size(fea, 1);
    TestIdx(TrainIdx) = [];

    fea_Train = fea1(TrainIdx,:);
    gnd_Train = gnd(TrainIdx);
    [gnd_Train ind] = sort(gnd_Train, 'ascend');
    fea_Train = fea_Train(ind, :);

    fea_Test = fea1(TestIdx,:);
    gnd_Test = gnd(TestIdx);

    fprintf('[%d] - Computing the transformation matrix.\n', jj);
    %U_reduc = pcomp(fea_Train, 'whiten', true);
%     U_reduc = pcomp(fea_Train);
%     U_reduc = lda(fea_Train, gnd_Train);
%     U_reduc = lpp_heat(fea_Train);
%     U_reduc = lpp_knn(fea_Train, 'k', 7);
     U_reduc = fastica_lowdim(fea_Train);
    
    fprintf('[%d] - Matrix computation done.\n', jj);

    oldfea = fea_Train*U_reduc;
    newfea = fea_Test*U_reduc;

    mg = mean(oldfea, 1);
    mg_oldfea = repmat(mg,  size(oldfea,1), 1);
    oldfea = oldfea - mg_oldfea;

    mg_newfea = repmat(mg,  size(newfea,1), 1);
    newfea = newfea - mg_newfea;

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
        Class = knnclassify(Sample, Training , Group, k, distance);

        correct(ii) = length(find(Class-gnd_Test == 0));
    end

    correct = correct./length(gnd_Test);
    error = [error; 1- correct];
  
end

fprintf('Max score: %f\n', max(correct));

close(h1);
close(h2);

plot(mean(error,1)); %%plotting the error 
correct_ica = correct;
error_ica = error;
save('correct_ica.mat', 'correct_ica', 'error_ica');
