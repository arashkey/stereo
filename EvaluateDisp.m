function [ dispError , imgMask , badPixels] = EvaluateDisp(ImageStruct,LEstDisp,tau,stateCheck) 
% stateCheck an input option to select NonOccluded =1 or Occluded =2 
% this function calculates the error rate of Left Estimated image (pixles which their errors are
% greater than tua) except Occluded and unknown areas

%if stateCheck not set, check  NonOccluded pixels by default
if nargin<4
	stateCheck=1;
end

LEstDisp=double(LEstDisp);
if ImageStruct.type
    switch ImageStruct.datasetName
        case 'Middlebury2014'
            %reading left groundtruth
            imgGT = readpfm(ImageStruct.LDispOcc);
            imgMask = imread(ImageStruct.LMask);
            imgMask = imgMask == 255;
            
            %Error thresholds need to be converted accordingly, e.g., a threshold of
            %2.0 at full resolution would correspond to a threshold of 0.5 at quarter resolution.
            if ImageStruct.ImageName(end)=='Q'
                tau=tau/4;
            elseif ImageStruct.ImageName(end)=='H'
                tau=tau/2;
            else
                tau=tau;
            end
                
            badPixels = abs(LEstDisp - imgGT) > tau;
            badPixels(~imgMask) = 0;
            dispError= sum(badPixels(:))/sum(imgMask(:));
            
        case 'Middlebury2006'
            if ImageStruct.ImageName(end)=='t'
                scale=3;
            elseif ImageStruct.ImageName(end)=='h'
                scale=2;
            else
                scale=1;
            end
            
            imgGT=double(imread(ImageStruct.LDispOcc))/scale;
            imgGTr=double(imread(ImageStruct.RDispOcc))/scale;                 
            imgMask = imgGT ~= 0;%ignoring unknown pixels
            imgMask(GetOccludedArea(imgGT,imgGTr))=0;       %also ignoring occluded pixels
            %Occlusion maps can be generated by crosschecking the pair of disparity maps.
            badPixels=abs(LEstDisp-imgGT) > tau;%thid size
            badPixels(~imgMask) = 0;
            dispError=sum(badPixels(:))/sum(imgMask(:));
        
        case 'Middlebury2005'
            if ImageStruct.ImageName(end)=='t'
                scale=3;
            elseif ImageStruct.ImageName(end)=='h'
                scale=2;
            else
                scale=1;
            end
            
            imgGT=double(imread(ImageStruct.LDispOcc))/scale;
            imgGTr=double(imread(ImageStruct.RDispOcc))/scale;                 
            imgMask = imgGT ~= 0;%ignoring unknown pixels
            imgMask(GetOccludedArea(imgGT,imgGTr))=0;       %also ignoring occluded pixels
            %Occlusion maps can be generated by crosschecking the pair of disparity maps.
            badPixels=abs(LEstDisp-imgGT) > tau;%thid size
            badPixels(~imgMask) = 0;
            dispError=sum(badPixels(:))/sum(imgMask(:));
        
        case 'KITTI2012'        
            %only few pixels are different in occluded images!
            I = imread(ImageStruct.LDispNoc);
            imgGT = double(I)/256;
            imgMask = imgGT ~= 0;
            
            badPixels = abs(LEstDisp - imgGT) > tau;
            badPixels(~imgMask) = 0;
            dispError= sum(badPixels(:))/sum(imgMask(:));
            
        case 'Sintel'
            DISP = imread(ImageStruct.LDispOcc);
            I_r = double(DISP(:,:,1));
            I_g = double(DISP(:,:,2));
            I_b = double(DISP(:,:,3));
            imgGT = I_r * 4 + I_g / (2^6) + I_b / (2^14);
            imgMask = imread(ImageStruct.LMask);
            imgMask = imgMask == 255;
            
            badPixels = abs(LEstDisp - imgGT) > tau;
            badPixels(~imgMask) = 0;
            dispError= sum(badPixels(:))/sum(imgMask(:));
    end
    
else
    error('there is no ground truth availible for the image');
end
end