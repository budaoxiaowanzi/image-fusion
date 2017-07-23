%%%the MATLAB code for the paper "Image matting for fusion of multi-focus images
%%%in dynamic scenes" Author: Xudong Kang
%%%Donot hesitate to contact me if you meet any problems when implementing
%%%this code.
%%%Author: Xudong Kang;                                                            
%%%Email:xudong_kang@163.com
%%%Homepage:http://xudongkang.weebly.com

%%% Input I: two or more than two multifocus images
%%% Output F: a fused image

%%%% gray image fusion
I = load_images( '.\sourceimages\colourset',1); 
F = IFM(I);
imshow(F);
% %%%% color image fusion
% I = load_images( '.\sourceimages\grayset',1); 
% F = IFM(I);
% figure,imshow(F);