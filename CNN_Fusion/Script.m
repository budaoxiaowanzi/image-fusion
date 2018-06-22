close all;
clear all;
clc;

A  = imread('sourceimages/children_1.tif');
B  = imread('sourceimages/children_2.tif');
if size(A)~=size(B)
    error('two images are not the same size.');
end
figure,imshow(A);figure,imshow(B);

model_name = 'model/cnnmodel.mat';

F=CNN_Fusion(A,B,model_name);

figure,imshow(F);
imwrite(F,'results/fused_cnn.tif');
