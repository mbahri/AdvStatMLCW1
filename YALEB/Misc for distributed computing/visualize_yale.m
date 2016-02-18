% Generate images of the 3 first components for different methods

clear; clc;

load('YaleB_32x32');
load('5Train/1');

feat = fea(trainIdx,:);
gndt = gnd(trainIdx);

W = pcomp(feat);

for i=1:3
    temp = reshape(W(:,i), 32, 32); %%reshape to 32x32
    temp = imresize(temp, 2);     %%resize to 64x64
    temp=temp-min(temp(:)); % shift data such that the smallest element of A is 0
    temp=temp/max(temp(:)); % normalize the shifted data to 1 
    
    imwrite(temp, sprintf('yale_pca%d.png', i) );
end

W = pcomp(feat,'whiten',true);

for i=1:3
    temp = reshape(W(:,i), 32, 32); %%reshape to 32x32
    temp = imresize(temp, 2);     %%resize to 64x64
    temp=temp-min(temp(:)); % shift data such that the smallest element of A is 0
    temp=temp/max(temp(:)); % normalize the shifted data to 1 
    
    imwrite(temp, sprintf('yale_pca_white%d.png', i) );
end

W = lda(feat, gndt);

for i=1:3
    temp = reshape(W(:,i), 32, 32); %%reshape to 32x32
    temp = imresize(temp, 2);     %%resize to 64x64
    temp=temp-min(temp(:)); % shift data such that the smallest element of A is 0
    temp=temp/max(temp(:)); % normalize the shifted data to 1 
    
    imwrite(temp, sprintf('yale_lda%d.png', i) );
end

W = lpp_heat(feat);

for i=1:3
    temp = reshape(W(:,i), 32, 32); %%reshape to 32x32
    temp = imresize(temp, 2);     %%resize to 64x64
    temp=temp-min(temp(:)); % shift data such that the smallest element of A is 0
    temp=temp/max(temp(:)); % normalize the shifted data to 1 
    
    imwrite(temp, sprintf('yale_lpp_heat%d.png', i) );
end

W = lpp_knn(feat, 'k', 7);

for i=1:3
    temp = reshape(W(:,i), 32, 32); %%reshape to 32x32
    temp = imresize(temp, 2);     %%resize to 64x64
    temp=temp-min(temp(:)); % shift data such that the smallest element of A is 0
    temp=temp/max(temp(:)); % normalize the shifted data to 1 
    
    imwrite(temp, sprintf('yale_lpp_knn%d.png', i) );
end

W = fastica_lowdim(feat);

for i=1:3
    temp = reshape(W(:,i), 32, 32); %%reshape to 32x32
    temp = imresize(temp, 2);     %%resize to 64x64
    temp=temp-min(temp(:)); % shift data such that the smallest element of A is 0
    temp=temp/max(temp(:)); % normalize the shifted data to 1 
    
    imwrite(temp, sprintf('yale_ica%d.png', i) );
end

