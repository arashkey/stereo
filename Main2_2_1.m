%% DESCRIPTION
% Dataset: any
% images: from dataset
% features: DD LRC MED ai TS
% algorithms: implemented
% classifier: TreeBagger


%% clear
close all;
clear ;
clc;

%% Initialization
%loading image names and locations

DatasetDir;

%this code for start prallel processing
tic
% p=gcp;
% if(p.Connected==0)
%     parpool
%     disp 'parallel processing starting in:'
%     toc
% end
addpath('2016-Correctness'); %features should be floats in range [0 1]

%loading all functions in arrays
%FunctionsDir;
algoFunc{1}=str2func('NCCAll');
numberOfFeature=0;
%algosNum =[1];% [ 4 5 9 10 11] ;                                 %<<<-----------------------HARD CODED
%select desired algorithms from the list below and put its number in the list
%1-ADSM  2-ARWSM 3-BMSM  4-BSM   5-ELAS  6-FCVFSM   7-SGSM  8-SSCA  9-WCSM
%10-MeshSM 11-NCC

%featuresNum= [1 2 3];
%select desired features from the list below and put its number in the list
%   1-CE    2-CANNY 3-FNVE  4-GRAYCONNECTED 5-HA    6-HARRISCORNERPOINTS    7-HOG   8-LABELEDREGIONS    9-RD    10-SOBEL    11-SP   12-SURFF

% featureFunc{1}=str2func('DD');%overriding features
% featureFunc{2}=str2func('LRC');
% featureFunc{3}=str2func('MED');
%featureFunc{4}=str2func('DB');
%MMN, AML, LRD are not usefull here since we do not have cost volumes of other algorithms


%% reading and calculating errors and making features and NCC for images (left and right)

disp ('calculating disparities...');
data=struct;
%dispData=struct ;
tau=1; %error threshold

%real image mumbers in AllImages
trainImageList=[709];                                   %<<<-----------------------HARD CODED
testImageList=[710];                                        %<<<-----------------------HARD CODED
imagesList = [ trainImageList ,testImageList];

err=zeros(1,size(imagesList,2));
for imgNum=1:size(imagesList,2) %local image numbers
    imgL=imread(AllImages(imagesList(imgNum)).LImage);
    imgR=imread(AllImages(imagesList(imgNum)).RImage);
    dispRange=AllImages(imagesList(imgNum)).maxDisp;
    
    fileName=strcat('./Results/',num2str(imagesList(imgNum)) ,'_',AllImages(imagesList(imgNum)).ImageName , '_NCCALL.mat');
    %display(fileName);
    if exist(fileName,'file')
        load(fileName);
    else
        [w , h]=size(imgL);
        tic;
        [dispL, dispR,Cost,CostR, CostVolume,CostVolumeR]=NCCAll(imgL,imgR,[1 dispRange]);
        data.TimeCosts=toc;
        
        disp(['time for calculate picture ' AllImages(imagesList(imgNum)).ImageName  ...
            ' w=' num2str(w) ' h=' num2str(h) ': ' num2str( data.TimeCosts) ' seconds'])
        
        
        data.DisparityLeft=dispL;
        data.DisparityRight=dispR;
        data.Cost=Cost;
        % data.CostR=CostR;
        data.CostVolume=CostVolume;
        data.CostVolumeR=CostVolumeR;
        data.ErrorRates=EvaluateDisp(AllImages(imagesList(imgNum)),double(data.DisparityLeft),tau);
        
        save(fileName,'data');
    end
    sortedCostVol =sort(data.CostVolume,3);
    sortedCostVolR =sort(data.CostVolumeR,3);
    
    err(imgNum)=EvaluateDisp(AllImages(imagesList(imgNum)),data.DisparityLeft,tau);%data.ErrorRates;
    %save features matrix
    [w,h]=size(data.DisparityLeft);
    data.Features=zeros(w,h,numberOfFeature);
    
    fIndex=1;
    data.Features(:,:,fIndex)=MM(sortedCostVol);
    fIndex=fIndex+1;
    
    data.Features(:,:,fIndex)=DD(data.DisparityLeft);
    fIndex=fIndex+1;
    
    data.Features(:,:,fIndex)=MED(data.DisparityLeft);
    fIndex=fIndex+1;
    
    data.Features(:,:,fIndex)=LRC(data.DisparityLeft,data.DisparityRight);
    fIndex=fIndex+1;
    
    data.Features(:,:,fIndex)=data.Cost;
    fIndex=fIndex+1;
%     data.Features(:,:,fIndex)=AML([w, h],data.CostVolume,sortedCostVol);
%     fIndex=fIndex+1;
%     data.Features(:,:,fIndex)=LRD(data.DisparityLeft,dispRange, sortedCostVol,sortedCostVolR);
%         fIndex=fIndex+1;
    
    %     data.DD=DD(data.DisparityLeft);
    %     data.LRC=LRC(data.DisparityLeft,data.DisparityRight);
    %     data.MED=MED(data.DisparityLeft);
    
    numberOfFeature=fIndex-1;
    if(exist('dispData','var')==0)
        dispData=struct(data);
    end
    
    dispData(imgNum)=data;
    %     end
    disp([num2str(imagesList(imgNum)) ' done']);
end
clear   aNum data fileName imgL imgR sortedCostVol sortedCostVolR currentImageInfo Cost CostR dispL dispR CostVolume CostVolumeR
%checking every result
% for i=1:m
% imshow(dispData(i,2).left,[]);
% waitforbuttonpress();
% cla;
% end




%% TreeBagger

treesCount=50;
%init train and test number of pixcel
imgPixelCountTrain=0;
imgPixelCountTest=0;
NumberOfTrainImage=size (trainImageList,2);


for i=1:NumberOfTrainImage
    [w, h]=size(dispData(i).DisparityLeft);
    imgPixelCountTrain=imgPixelCountTrain+(w*h);
end

for i=1:size (testImageList,2)
    %test after train, so add i with train number of image, becuse test set
    %after train set
    [w, h]=size(dispData(i+NumberOfTrainImage).DisparityLeft);
    imgPixelCountTest=imgPixelCountTest+(w*h);
end

trainInput=zeros(imgPixelCountTrain,numberOfFeature);
trainClass=zeros(imgPixelCountTrain,1);

disp 'start init of array for classifier'
tic;
currentIndex=1;
for imgNum=1:NumberOfTrainImage
    currentImageInfo=dispData(imgNum);
    %[w, h]=size(currentImageInfo.DisparityLeft);
    
    
    
    imgGT = GetGT(AllImages(imagesList(imgNum)));
    
    truePixles = abs(dispData(imgNum).DisparityLeft - imgGT) <= 1;
    truePixles =truePixles(:);
    %merge all feature for this image and make feature matrix
    %[m n NumberOfFeatures] ==>  [m*n NumberOfFeatures]
 
    temp=reshape(permute( currentImageInfo.Features,[2 1 3]),[],numberOfFeature);
    %get the image Mask for remove known or oculuded pixcel
    [ dispError , imgMask , badPixels] = EvaluateDisp(AllImages(imagesList(imgNum)),dispData(imgNum).DisparityLeft,1);
    %make img mask liner for * to feature array
    
     imgMask=imgMask(:);
    % select feature that in the image mask 
     temp=temp(imgMask>0,:);
     truePixles=truePixles(imgMask>0);
    [wh,f]=size(temp);
    trainInput(currentIndex:wh+currentIndex-1,:)=temp;
    
    trainClass(currentIndex:wh+currentIndex-1)=    truePixles(:);
    
    currentIndex=currentIndex+wh ;
    %     for i=1:w
    %         for j=1:h
    %             fNumber=1;%this var for stop static indexing of features
    %             trainInput(currentIndex,fNumber)=currentImageInfo.MM(i,j);
    %             fNumber=fNumber+1;
    %             trainInput(currentIndex,fNumber)=currentImageInfo.LRD(i,j);
    %             fNumber=fNumber+1;
    %             trainInput(currentIndex,fNumber)=currentImageInfo.AML(i,j);
    %             fNumber=fNumber+1;
    %             trainInput(currentIndex,fNumber)=currentImageInfo.DD(i,j);
    %             fNumber=fNumber+1;
    %             trainInput(currentIndex,fNumber)=currentImageInfo.LRC(i,j);
    %             fNumber=fNumber+1;
    %             trainInput(currentIndex,fNumber)=currentImageInfo.MED(i,j);
    %
    %             trainClass(currentIndex)=    truePixles(i,j);
    %             currentIndex=currentIndex+1;
    %         end
    %     end
    
end
disp 'init of array for classifire fished in:'
toc;

disp 'start learning classifier'
tic
%remove other array for learning
trainClass=trainClass(1:currentIndex-1,:);
trainInput=trainInput(1:currentIndex-1,:);
%RFs(i).model=TreeBagger(treesCount,X,Y,'OOBPrediction','on');
%to store TreeBagger models
RFmodel=compact (TreeBagger(treesCount,trainInput,trainClass,'Method','classification','MinLeafSize',5000,'MergeLeaves','on' ));
%RFs(i).model=TreeBagger(treesCount,X,Y);
%RFs(i).treeErrors = oobError(RFs(i).model);%out of bag error
%tr10 = RFs(i).model.Trees{10};
%view(tr10,'Mode','graph');

disp 'classifier learned is finished in:'
toc;
clear currentImageInfo testInput trainInput imgGT trainClass currentImageInfo truePixles trainInput trainClass    

%% testing...
disp('testing...');
tic
testInput=zeros(imgPixelCountTest,numberOfFeature);
testClass=zeros(imgPixelCountTest,1);

disp('testing init start...');
tic
NumberOfTestImage=size (testImageList,2);
currentIndex=1;
for imgNum=NumberOfTrainImage+1:NumberOfTrainImage+NumberOfTestImage
    currentImageInfo=dispData(imgNum);
    [w, h]=size(currentImageInfo.DisparityLeft);
    
    imgGT = GetGT(AllImages(imagesList(imgNum)));
    truePixles = abs(dispData(imgNum).DisparityLeft - imgGT) <= 1;
      
    %merge all feature for this image and make feature matrix
    temp=reshape(permute( currentImageInfo.Features,[2 1 3]),[],numberOfFeature);
    testInput(currentIndex:w*h+currentIndex-1,:)=temp;
    
    testClass(currentIndex:w*h+currentIndex-1)=    truePixles(:);
    
    currentIndex=currentIndex+w*h;
    %     for i=1:w
    %         for j=1:h
    %             fNumber=1;%this var for stop static indexing of features
    %             testInput(currentIndex,fNumber)=currentImageInfo.MMN(i,j);
    %             fNumber=fNumber+1;
    %             testInput(currentIndex,fNumber)=currentImageInfo.LRD(i,j);
    %             fNumber=fNumber+1;
    %             testInput(currentIndex,fNumber)=currentImageInfo.AML(i,j);
    %             fNumber=fNumber+1;
    %             testInput(currentIndex,fNumber)=currentImageInfo.DD(i,j);
    %             fNumber=fNumber+1;
    %             testInput(currentIndex,fNumber)=currentImageInfo.LRC(i,j);
    %             fNumber=fNumber+1;
    %             testInput(currentIndex,fNumber)=currentImageInfo.MED(i,j);
    %
    %             testClass(currentIndex)=    truePixles(i,j);
    %             currentIndex=currentIndex+1;
    %         end
    %     end
    
end
disp('testing init finish in:');
toc
clear currentImageInfo  imgGT  currentImageInfo truePixles   temp

disp 'predict is start ...'
tic
[labels,confidence] = predict(RFmodel,testInput);
finalScores=confidence(:,2);
%show confidence
figure;
%get the first test image size
[w, h]=size(dispData(NumberOfTrainImage+1).DisparityLeft);     
temp=finalScores(1:w*h);

    [ dispError , imgMask , badPixels] = EvaluateDisp(AllImages(imagesList(NumberOfTrainImage+1)),dispData(NumberOfTrainImage+1).DisparityLeft,1);

imshow(reshape(temp,h,w)' .* imgMask,[]);
figure;
imshow(reshape(temp,h,w)' .* imgMask);


[values, indices]=max(finalScores);
disp ([ 'max final score: values:' num2str(values) ' indices:' num2str(indices)])
disp 'predict is finished in:'
toc
clear  testInput

accuricyOfClassifier=sum(sum(abs(str2num( cell2mat(labels)) - testClass)))*100/size(testClass,1);
disp (['testing finished. with error rate=%' num2str( accuricyOfClassifier) '' ])
return
%% GCPs
disp 'start GCPs...'
tic 
%getting 1 class confidence
temp=confidence(:,2);
temp=temp(:);
indexOfGCPSelect=find(temp>0.5&temp<0.95);
index=1;
for imgNum=NumberOfTrainImage+1:NumberOfTrainImage+NumberOfTestImage
    currentImageInfo=dispData(imgNum);
    [w, h]=size(currentImageInfo.DisparityLeft);
    for i=1:w
        for j=1:h
            %if pixcel in the GCP selection
           if( sum(ismembertol(indexOfGCPSelect,index))==1)
              [~, minIndex]=min(currentImageInfo.CostVolume(i,j,:));
              b=currentImageInfo.CostVolume(i,j,:);
              b=b(:);
              b([1:minIndex-1 minIndex+1:end])=2;
              currentImageInfo.CostVolume(i,j,:)=b;
            end
            index = index+1;
        end
    end
    
end


clear temp

disp 'finish GCP on :'
toc
%% MRF
disp 'start MRF...'
tic 
addpath('PostProcessing\FastPD\');
for imgNum=NumberOfTrainImage+1:NumberOfTrainImage+NumberOfTestImage
    currentImageInfo=dispData(imgNum);
    dispRange=AllImages(imagesList(imgNum)).maxDisp;
    imgL=imread(AllImages(imagesList(imgNum)).LImage);
    imgR=imread(AllImages(imagesList(imgNum)).RImage);
%     mrf=uint8(FastPD(imgL,imgR,currentImageInfo.DisparityLeft,currentImageInfo.DisparityRight ...
%         ,currentImageInfo.CostVolume,dispRange));
%     currentImageInfo.DisparityLeft=mrf;
    figure;
    imshow(currentImageInfo.DisparityLeft,[]);
end

disp 'finish MRF on :'
toc


disp('Job Done.');