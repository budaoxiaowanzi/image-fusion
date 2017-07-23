function [ F ] = IFM(I)
if size(I,3)==3
G=rgb2gray_n(I);
else
G=I;
end
% morphological filtering
D= MorF(G);
% trimap generation
T= TriG(D);
% alpha estimation (here the closed form matting method is used for the
% MATLAB implementation of the proposed method), the matting method actually
% has a little influence to the fusion performance
Alpha=AlpE(I,T);
F=AlpF(I,Alpha);

function [G] = rgb2gray_n( I )
N=size(I,4);
for i=1:N
    G(:,:,i)=rgb2gray(I(:,:,:,i));
end
function [D] = MorF( I )
B=strel('disk',2);
N = size(I,3);
D = zeros(size(I,1),size(I,2),N); % Assign memory
for i = 1:N
    db=I(:,:,i)-imopen(I(:,:,i),B);
    dd=imclose(I(:,:,i),B)-I(:,:,i);
    D(:,:,i) = max(db,dd);
end
D=uint8(D);
function [T] = TriG(D)
[w h N]=size(D);
for i=1:N
   r=zeros(w,h);
   ED=D;
   ED(:,:,i)=[];
   [max_D]=max(ED,[],3);
   r(find(D(:,:,i)>max_D))=1; 
   r=medfilt2(r,[8 8]);
   r=bwmorph(r,'skel',5);
   r=medfilt2(r,[8 8]);
   Enumber=[1:N];
   Enumber(Enumber==i)=[];
   Minu_D=D(:,:,i)-max(D(:,:,Enumber),[],3);
   r(Minu_D>127)=1;
   R(:,:,i)=r;
end

for j=1:N-1
    t=0.5.*ones(w,h);
    ER=R;
    ER(:,:,j)=[];
   [max_R]=max(ER,[],3);
   t(find(R(:,:,j)==1&max_R==0))=1; 
   t(find(R(:,:,j)==0&max_R==1))=0;   
   T(:,:,j)=t;
end
function [Alpha] = AlpE( I,T )
[r c N] = size(T);
I=double(I)/255;
%%%parameters of the matting method
if (~exist('thr_alpha','var'))
  thr_alpha=[];
end
if (~exist('epsilon','var'))
  epsilon=[];
end
if (~exist('win_size','var'))
  win_size=[];
end

if (~exist('levels_num','var'))
    
    if r*c>768*768
    levels_num=4;
    elseif r*c>512*512
    levels_num=3;
    elseif r*c>256*256
        levels_num=2;
    else
        levels_num=1;
    end
end  
if (~exist('active_levels_num','var'))
  active_levels_num=1;
end
%%%%
for i=1:N
    cM=zeros(r,c);
    cM(T(:,:,i)~=0.5)=1;
    cV=zeros(r,c);
    cV(T(:,:,i)==1)=1;
    if size(I,4)>1
    Alpha(:,:,i)=solveAlphaC2F(I(:,:,:,i),cM,cV,levels_num, ...
                    active_levels_num,thr_alpha,epsilon,win_size);
    else
    Alpha(:,:,i)=solveAlphaC2F(I(:,:,i),cM,cV,levels_num, ...
                    active_levels_num,thr_alpha,epsilon,win_size);
    end
end
function [F] = AlpF(I,Alpha)
N=size(Alpha,3);
Alp=Alpha;
Alp(:,:,N+1)=1-double(uint8(sum(Alpha,3)*255))/255;
I=double(I)/255;
for i=1:N+1
    if size(I,4)>1
        F(:,:,:,i)=I(:,:,:,i).*repmat(Alp(:,:,i),[1 1 3]);
    else
        F(:,:,i)=I(:,:,i).*Alp(:,:,i);
    end
end
if size (I,4)>1
F=uint8(sum(F,4)*255);
else
F=uint8(sum(F,3)*255);
end
