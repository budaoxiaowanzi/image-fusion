% script_fusionOthers.m
% -------------------------------------------------------------------
% 
% Date:    10/04/2013
% Last modified: 1/11/2013
% -------------------------------------------------------------------

function Fusion_test_for_registered_img()

%     clear
    close all
    clc

    %% ------ Input the images ----------------
    % ------------- The Gray ----------------
     path1 = '.\image\registered-images\book_A.bmp';
     path2 = '.\image\registered-images\book_B.bmp';
%     path1 = '.\image\registered-images\flower_A.png';
%     path2 = '.\image\registered-images\flower_B.png';
%     path1 = '.\image\registered-images\desk_A.tif';   
%     path2 = '.\image\registered-images\desk_B.tif';
%     path1 = '.\image\registered-images\pepsi_A.tif';
%     path2 = '.\image\registered-images\pepsi_B.tif';
%     path1 = '.\image\registered-images\cameramanleft.jpg';
%     path2 = '.\image\registered-images\cameramanright.jpg';
    % -----------------------------------------
    [img1, img2] = PickName(path1, path2, 0);
    paraShow.fig = 'Input 1';
    paraShow.title = 'Org1';
    ShowImageGrad(img1, paraShow)
    paraShow.fig = 'Input 2';
    paraShow.title = 'Org2';
    ShowImageGrad(img2, paraShow)
    %% ---- The parameters -----
    % ----------- the multi scale -----
    para.Scale.lsigma = 4;
    para.Scale.ssigma = 0.5;
    para.Scale.alpha = 0.5;
    % -------------- the Merge parameter fusion of registered images-------------
    para.Merge.per = 0.5;
    para.Merge.margin = 1.5*para.Scale.lsigma;
    para.Merge.method = 2;
    % ------------- the Reconstruct parameter -----------
    para.Rec.iter = 500;
    para.Rec.res = 1e-6;
    para.Rec.modify = 5;
    para.Rec.iniMode = 'weight';   
    
    %% ---- MWGF implementation ------
    imgRec = MWGFusion(img1, img2, para);

    % --- Show the result ------
    paraShow.fig = 'fusion result';
    paraShow.title = 'MWGF';
    ShowImageGrad(imgRec, paraShow);
    imwrite(uint8(imgRec), 'result.jpg', 'jpeg');
    
   
end
