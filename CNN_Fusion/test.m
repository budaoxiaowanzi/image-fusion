pet = imread('/Users/wangyaping/Downloads/PET-MRIfeature/data/case2/pet_reg.png');
MRI = dicomread('/Users/wangyaping/Downloads/PET-MRIfeature/data/case2/T2reg.dcm');
%MRI =(MRI - min(MRI(:)))/(max(MRI(:))-min(MRI(:)));
pet = im2double(pet);
MRI = im2double(MRI);
figure,imshow(pet);
figure,imshow(MRI,[]);