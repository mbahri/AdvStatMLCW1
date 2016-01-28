clear; clc; %close all

load YaleB_32x32; %%load database YaleB 32x32 resolution

[nSmp,nFea] = size(fea);
fea1 = zeros(nSmp, 4*nFea);
for ii = 1:nSmp
    temp = fea(ii,:);
    temp = reshape(temp, 32, 32); %%reshape to 32x32
    temp = imresize(temp, 2);     %%resize to 64x64
    imshow(temp,[]);              %% show the image
    pause;                        %% press a key      
    temp = temp(:);
    fea1(ii,:) = temp;
end

