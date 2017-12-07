%% init
close all
addpath('../..');
addpath('../../2016-Correctness');
addpath('../modefilt2');

load('../../AllImages.mat');

imgNumber=710;

xx=0;

limg=imread( AllImages(imgNumber).LImage);
rimg=imread( AllImages(imgNumber).RImage);
lGTdisp=uint8( imread( AllImages(imgNumber).LDispOcc))/3;
% maskGT=uint8( imread( AllImages(imgNumber).    ))/3;

disrange= min(AllImages(imgNumber).maxDisp,255);
tau=1;


if(xx>0)
    limg=limg(1:xx,1:xx,:);
    rimg=rimg(1:xx,1:xx,:);
end


w=size(limg,1);
h=size(limg,2);

%% process ncc

tic

[ imgL_d, imgR_d, Cost,CostR, CostVolume,CostVolumeR] =   NCCAll( limg,rimg,[0 disrange]);


disp('Cost compute time for NCCAll:')
toc

imgL_d=uint8(imgL_d);
%% process error for ncc
% nccError=sum(sum(abs(imgL_d - lGTdisp) > tau))/(w*h);
% disp(['ncc error ',num2str(nccError*100),'%']);
%% cost correction
% Cost(Cost>disrange)=disrange;
% Cost(Cost<-disrange)=-disrange;
%% GPC
[w ,h]=size(Cost);
for i=1:w
    for j=1:h
        if(Cost(i,j)>0.5 && Cost(i,j)<0.95)
            newCost=reshape( CostVolume(i,j,:),[],1);
            [minValue, minIndex]=min(newCost);
            newCost=ones(1,disrange).*2;
            newCost(minIndex)=minValue;
            CostVolume(i,j,:)=newCost;
        end
    end
end

%% process mrf
 
 for i=1:10
     ii=10^i;
      disp(['start processsing ii= ',num2str( ii)]);
    mrf=uint8(FastPD2(CostVolume,disrange, imgL_d,ii));
    % mrf=modefilt2(imgL_d);
    %% prcoess error of mrf
    [ nccError , imgMask , badPixels] = EvaluateDisp(AllImages(imgNumber),imgL_d,tau) ;
    [ mrfError , imgMask , badPixels] = EvaluateDisp(AllImages(imgNumber),mrf,tau) ;
    % mrfError=sum(sum(abs(mrf - lGTdisp) > tau))/(w*h);
    % nccError=sum(sum(abs(imgL_d - lGTdisp) > tau))/(w*h);
 
    
    
    disp(['ncc error ',num2str(nccError*100),'%']);
    disp(['mrf error ',num2str(mrfError*100),'% for ii=' num2str( ii)]);
    
     
end
%% show result
subplot(221);
imshow(limg);
title('left image ');

subplot(222);
imshow(rimg);
title('right image  ');

subplot(223);
imshow(imgL_d,[]);
title('nnc  ');

subplot(224);
imshow(mrf,[]);
title('mrf');
