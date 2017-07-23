clc,clear
%%%% gray image fusion
% I = load_images( '.\sourceimages\grayset',1); 
% F = GFF(I);
% imshow(F);
%%%% color image fusion
I = load_images( '.\Sourceimages\colourset',1); 
F = GFF(I);
figure,imshow(F);