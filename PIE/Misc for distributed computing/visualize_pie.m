% Generate images of the 3 first components for different methods

clear; clc;

load('fea64');
load('gnd64');

load('Train5_64');
trainIdx = Train5_64(1,:);

feat = fea64(trainIdx,:);
gndt = gnd64(trainIdx);

W = pcomp(feat);

for i=1:3
    temp = reshape(W(:,i), 64, 64); %%reshape to 32x32
    temp=temp-min(temp(:)); % shift data such that the smallest element of A is 0
    temp=temp/max(temp(:)); % normalize the shifted data to 1 
    
    imwrite(temp, sprintf('pie_pca%d.png', i) );
end

W = pcomp(feat,'whiten',true);

for i=1:3
    temp = reshape(W(:,i), 64, 64); %%reshape to 32x32
    temp=temp-min(temp(:)); % shift data such that the smallest element of A is 0
    temp=temp/max(temp(:)); % normalize the shifted data to 1 
    
    imwrite(temp, sprintf('pie_pca_white%d.png', i) );
end

W = lda(feat, gndt);

for i=1:3
    temp = reshape(W(:,i), 64, 64); %%reshape to 32x32
    temp=temp-min(temp(:)); % shift data such that the smallest element of A is 0
    temp=temp/max(temp(:)); % normalize the shifted data to 1 
    
    imwrite(temp, sprintf('pie_lda%d.png', i) );
end

W = lpp_heat(feat);

for i=1:3
    temp = reshape(W(:,i), 64, 64); %%reshape to 32x32
    temp=temp-min(temp(:)); % shift data such that the smallest element of A is 0
    temp=temp/max(temp(:)); % normalize the shifted data to 1 
    
    imwrite(temp, sprintf('pie_lpp_heat%d.png', i) );
end

W = lpp_knn(feat, 'k', 7);

for i=1:3
    temp = reshape(W(:,i), 64, 64); %%reshape to 32x32
    temp=temp-min(temp(:)); % shift data such that the smallest element of A is 0
    temp=temp/max(temp(:)); % normalize the shifted data to 1 
    
    imwrite(temp, sprintf('pie_lpp_knn%d.png', i) );
end

W = fastica_lowdim(feat);

for i=1:3
    temp = reshape(W(:,i), 64, 64); %%reshape to 32x32
    temp=temp-min(temp(:)); % shift data such that the smallest element of A is 0
    temp=temp/max(temp(:)); % normalize the shifted data to 1 
    
    imwrite(temp, sprintf('pie_ica%d.png', i) );
end

