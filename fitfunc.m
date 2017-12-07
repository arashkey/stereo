function [ErrorOuput]=fitfunc(inputx)

inputx=round(inputx);

disp(['input x=' num2str(inputx ) ])
%%
%loading image names and locations
DatasetDir;

%loading all functions in arrays
FunctionsDir;


%select desired Confidence Measures from the list below and put its number in the list
%   1-AML  2-DB 3-DD 4-HGM 5-LRC 6-LRD 7-MED 8-MM

%% reading or calculating errors for images (left and right)

disp ('calculating disparities...');
data=struct;
dispData=struct;
errThreshold=1; %error threshold                                  %<<<-----------------------HARD CODED
addpath ('2016-Correctness');
addpath('PostProcessing/FastPD')
addpath('PostProcessing/modefilt2');
%real image mumbers in AllImages
trainImageList=[704:709,711:716];%[702:710, 711:719];                                   %<<<-----------------------HARD CODED
testImageList=[710];%[693:701];                                        %<<<-----------------------HARD CODED
imagesList = [ trainImageList ,testImageList];

for imgNum=1:size(imagesList,2) %local image numbers
    imgL=imread(AllImages(imagesList(imgNum)).LImage);
    imgR=imread(AllImages(imagesList(imgNum)).RImage);
    algoCount=0;
    for aNum=1
        algoCount=algoCount+1;
        fileName=strcat('./Results/',num2str(imagesList(imgNum)) ,'_',AllImages(imagesList(imgNum)).ImageName , '_'  ,'NCCALL',  '.mat');
        if exist(fileName,'file')
            load(fileName);
        else
            [dispL, dispR,Cost,CostR, CostVolume,CostVolumeR]=NCCAll(imgL,imgR,[1 AllImages(imagesList(imgNum)).maxDisp]);
            data.DisparityLeft=dispL;
            data.DisparityRight=dispR;
            data.Cost=Cost;
            data.CostVolume=CostVolume;
            data.CostVolumeR=CostVolumeR;
            %data.ErrorRates=EvaluateDisp(AllImages(imagesList(imgNum)),double(dispL),errThreshold);
            save(fileName,'data');
        end
        %err(algoCount,imgNum)=EvaluateDisp(AllImages(imagesList(imgNum)),data.DisparityLeft,errThreshold);%data.ErrorRates;
        dispData(algoCount,imgNum).left=data.DisparityLeft;
        dispData(algoCount,imgNum).right=data.DisparityRight;
        dispData(algoCount,imgNum).Cost=data.Cost;
        dispData(algoCount,imgNum).CostVolume=data.CostVolume;
        dispData(algoCount,imgNum).CostVolumeR=data.CostVolumeR;
    end
    disp([num2str(imagesList(imgNum)) 'done']);
end
clear algoCount aNum data fileName

%% making the dataset and features
k=1;%size(algosNum,2); %number of active matchers
disp('making dataset...');
totalPCount=0;
trainCount=0;
%samples=struct;


for imgNum=1:size(imagesList,2)
    width=size(dispData(1,imgNum).left,1);
    height=size(dispData(1,imgNum).left,2);
    imgPixelCount(imgNum)=width*height;
end
samplesNum=sum(imgPixelCount);




input=zeros(samplesNum,8,k);                                %<<<-----------------------HARD CODED
class=zeros(samplesNum,k);
load('confParam.mat');%params for fn_confidence_measure
for imgNum=1:size(imagesList,2)
    imgL=imread(AllImages(imagesList(imgNum)).LImage);
     maxDisparity=AllImages(imagesList(imgNum)).maxDisp;
      M=size(imgL,1); N=size(imgL,2);
    conf = fn_confidence_measure(imgL, dispData(1,imgNum).CostVolume,dispData(1,imgNum).CostVolumeR, maxDisparity , 1, confParam);
   
    disp(['working on img ' num2str(imagesList(imgNum)) ]);
    A=zeros(M ,N,8);
    for ii=1:8
        if(inputx(ii)<=27)
            A(:,:,ii)=reshape(conf(inputx(ii),:),[M N]);
        elseif(inputx(ii)==28)
            A(:,:,ii)=cmFunc{3}(dispData(1,imgNum).left);
        elseif(inputx(ii)==29)
            A(:,:,ii)=cmFunc{7}(dispData(1,imgNum).left);
        elseif(inputx(ii)==30)
            sortedCostVol =sort(dispData(1,imgNum).CostVolume,3); 
            A(:,:,ii)= cmFunc{8}(sortedCostVol);
        elseif(inputx(ii)==31)
            A(:,:,ii)=cmFunc{2}(dispData(1,imgNum).left);
        elseif(inputx(ii)==32)
            A(:,:,ii)=dispData(1,imgNum).Cost;
       
        end
    end
    %pre-calculting all DDs LRCs and MEDs
    disp('Getting all confidence measures!' );
%     i=1;
    %     DD=cmFunc{3}(dispData(i,imgNum).left);
%     LRC=cmFunc{5}(dispData(i,imgNum).left,dispData(i,imgNum).right );
%     MED=cmFunc{7}(dispData(i,imgNum).left);
%     sortedCostVol =sort(dispData(i,imgNum).CostVolume,3);
%     MM=cmFunc{8}(sortedCostVol);
%     DB=cmFunc{2}(dispData(i,imgNum).left);
%     %LRD=cmFunc{6}(dispData(i,imgNum).left,dispData(i,imgNum).CostVolume,dispData(i,imgNum).CostVolumeR);
%     
%     imgL=imread(AllImages(imagesList(imgNum)).LImage);
%     maxDisparity=AllImages(imagesList(imgNum)).maxDisp;
%     M=size(imgL,1); N=size(imgL,2);
%     
%     %     DD=reshape(conf(?,:),[M N]);
%     %     LRC=reshape(conf(15,:),[M N]);
%     %     MED=reshape(conf(?,:),[M N]);
%     %     MM=reshape(conf(4,:),[M N]);
%     %     DB=reshape(conf(?,:),[M N]);
%     
%     %               LRD=reshape(conf(8,:),[M N]);
%     %              AML=reshape(conf(11,:),[M N]);
%     
%     LRD=reshape(conf(x(1),:),[M N]);
%     AML=reshape(conf(x(2),:),[M N]);
    
    %imgGT = GetGT(AllImages(imagesList(imgNum)));
    [~,imgMask,badPixels]=EvaluateDisp(AllImages(imagesList(imgNum)),dispData(1,imgNum).left,errThreshold);
      i=1;
    pCount=totalPCount;%number of pixels (samples)
    %showMeasures;
    
    %making data
    disp(['making data for algorithm number ', num2str(i)]);
    for x=1:size(dispData(i,imgNum).left,1)
        for y=1:size(dispData(i,imgNum).left,2)
            %in 2016-correctness.. Occluded pixels are ignored during training.
            if ~(imgMask(x,y)==0 && imgNum<=size(trainImageList,2))
                pCount=pCount+1;
                
                %only using its own features                %<<<-----------------------HARD CODED
                input(pCount,1,i)=squeeze(A(x,y,1));
                input(pCount,2,i)=squeeze(A(x,y,2));
                input(pCount,3,i)=squeeze(A(x,y,3));
                input(pCount,4,i)=squeeze(A(x,y,4));
                input(pCount,5,i)=squeeze(A(x,y,5));%dispData(1,imgNum).Cost(x,y));
                input(pCount,6,i)=squeeze(A(x,y,6));
                input(pCount,7,i)=squeeze(A(x,y,7));
                input(pCount,8,i)=squeeze(A(x,y,8));
                class(pCount,i)= ~badPixels(x,y);%whether the disparity assigned to that pixel was correct (1) or not (0)
            end
        end
    end
    
    totalPCount=pCount;
    if imgNum<=size(trainImageList,2)
        trainCount=trainCount+sum(imgMask(:));
    end
    disp([num2str(imagesList(imgNum)) ' done']);
end
clear width height agreementMat DD LRC MED AML LRD MM DB imgGT pCount tmpCount diff


%% TreeBagger
imgPixelCountTrain=imgPixelCount(1:size(trainImageList,2));
imgPixelCountTest=imgPixelCount(1+size(trainImageList,2):end);
%permutedIndices=randperm( sum(imgPixelCountTrain));
permutedIndices=randperm( trainCount);
portion=1;%in 0.25 the avg error increses 0.0002 and avg AUC increses 0.0006 (for 702:711)
%sampleCount=uint32( portion*sum(imgPixelCountTrain));
sampleCount=uint32( portion*trainCount);
trainIndices=permutedIndices (1:sampleCount);

RFs=struct;%to store TreeBagger models
treesCount=50;
%train and test sets
%train and test sets
trainInput=input(trainIndices,:,:);
trainClass=class(trainIndices,:);
testInput=input(1+trainCount:totalPCount,:,:);
%testClass=class(1+trainCount:totalPCount,:);
clear input class

for i=1:k
    X=trainInput(:,:,i);
    Y=trainClass(:,i);
    disp(['training RF number ' num2str(i)]);
    %RFs(i).model=TreeBagger(treesCount,X,Y,'OOBPrediction','on');
    RFs(i).model=compact (TreeBagger(treesCount,X,Y,'MinLeafSize',5000 ));%,'MergeLeaves','on'
    %RFs(i).model=TreeBagger(treesCount,X,Y);
    %RFs(i).treeErrors = oobError(RFs(i).model);%out of bag error
    %tr10 = RFs(i).model.Trees{10};
    %view(tr10,'Mode','graph');
end

%testing...
disp('testing...');

for i=1:k
    [~,confidence] = predict(RFs(i).model,testInput(:,:,i));
    %[RFs(i).labels,RFs(i).scores] = predict(RFs(i).model,testInput(:,:,i),'Trees',10:20);
    finalScores(i,:)=confidence(:,2);
    %finalLabels(i,:)=labels;
end
%[values, indices]=max(finalScores);
values=finalScores';

%getting results per image
Results=struct;




testImgNum=1;

ind1=sum(imgPixelCountTest(1:testImgNum-1));
ind2=ind1+imgPixelCountTest(testImgNum);
imgNum=testImgNum+size(imgPixelCountTrain,2);
[imgW ,imgH]=size(dispData(1,imgNum).left);

Results(testImgNum).Values=reshape(values(1+ind1:ind2),[imgH imgW ])';

finalDisp=dispData(1,imgNum).left;
%% GPC
CostVolume=dispData(imgNum).CostVolume;
imageData=AllImages(testImageList(i));
disrange=imageData.maxDisp;
GPCMask=zeros(imgW,imgH);
for i=1:imgW
    for j=1:imgH
        if(Results(testImgNum).Values(i,j)>0.6 && Results(testImgNum).Values(i,j)<0.95)
            GPCMask(i,j)=1;
            newCost=reshape( CostVolume(i,j,:),[],1);
            [minValue, minIndex]=min(newCost);
            newCost=ones(1,disrange).*2;
            newCost(minIndex)=minValue;
            CostVolume(i,j,:)=newCost;
        end
    end
end



% mrf=double(FastPD(CostVolume,disrange,finalDisp  ));
mrf=modefilt2(finalDisp);
mrf = mrf.*~GPCMask;
finalDisp=finalDisp.*GPCMask;
finalDisp=mrf+finalDisp;

ErrorOuput=EvaluateDisp(AllImages(imagesList(imgNum)),finalDisp,errThreshold);