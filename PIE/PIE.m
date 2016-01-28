clear; clc; close all

load Train5_64;
load fea64;
load gnd64;

fea = fea64; clear fea64;
gnd = gnd64; clear gnd64;
Train = Train5_64; clear Train5_64;

fea1 = fea;


error = [];
dim =2000; %%check recognition rate every dim dimensions (change it appropriatly for PCA, LDA etc
for jj = 1:20
    jj

    TrainIdx = Train(jj, :);
    TestIdx = 1:size(fea, 1);
    TestIdx(TrainIdx) = [];

    fea_Train = fea1(TrainIdx,:);
    gnd_Train = gnd(TrainIdx);
    [gnd_Train ind] = sort(gnd_Train, 'ascend');
    fea_Train = fea_Train(ind, :);

    fea_Test = fea1(TestIdx,:);
    gnd_Test = gnd(TestIdx);

      U_reduc = eye(size(64*64,64*64));  %%change it to PCA, LDA, etc

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

plot(mean(error,1)); %%plotting the error 