%% DESCRIPTION
% Dataset: any
% images: from dataset
% features: DD LRC MED MM DB LRD AML NCC
% algorithms: NCC
% classifier: TreeBagger


%% Initialization
close all;
clear;
clc;


%% loading image names and locations
DatasetDir;

%loading all functions in arrays
FunctionsDir;

cfNum = [] ;
%select desired Confidence Measures from the list below and put its number in the list
%   1-AML  2-DB 3-DD 4-HGM 5-LRC 6-LRD 7-MED 8-MM

%% reading or calculating errors for images (left and right)
for ii=15:22
    disp ('calculating disparities...');
    data=struct;
    dispData=struct;
    errThreshold=1; %error threshold                                  %<<<-----------------------HARD CODED
    addpath ('2016-Correctness');
    addpath('PostProcessing/FastPD')
    addpath('PostProcessing/modefilt2');
    %real image mumbers in AllImages
    trainImageList=[693:693+ii];%[702:710, 711:719];                                   %<<<-----------------------HARD CODED
    testImageList=[715:719];%[693:701];                                        %<<<-----------------------HARD CODED
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
            %         dispData(algoCount,imgNum).Cost=data.Cost;
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
    
    
    
    
    input=zeros(samplesNum,9,k);                                %<<<-----------------------HARD CODED
    class=zeros(samplesNum,k);
    load('confParam.mat');%params for fn_confidence_measure
    for imgNum=1:size(imagesList,2)
        disp(['working on img ' num2str(imagesList(imgNum)) ]);
        
        %pre-calculting all DDs LRCs and MEDs
        disp('Getting all confidence measures!' );
        i=1;
        %      DD=cmFunc{3}(dispData(i,imgNum).left);
        
        %     LRC=cmFunc{5}(dispData(i,imgNum).left,dispData(i,imgNum).right );
        %     MED=cmFunc{7}(dispData(i,imgNum).left);
        %     sortedCostVol =sort(dispData(i,imgNum).CostVolume,3);
        %     MM=cmFunc{8}(sortedCostVol);
        DB=cmFunc{2}(dispData(i,imgNum).left);
        %LRD=cmFunc{6}(dispData(i,imgNum).left,dispData(i,imgNum).CostVolume,dispData(i,imgNum).CostVolumeR);
        
        imgL=imread(AllImages(imagesList(imgNum)).LImage);
        maxDisparity=AllImages(imagesList(imgNum)).maxDisp;
        M=size(imgL,1); N=size(imgL,2);
        conf = fn_confidence_measure(imgL, dispData(i,imgNum).CostVolume,dispData(i,imgNum).CostVolumeR, maxDisparity , 1, confParam);
        
        A1=reshape(conf(26,:),[M N]);
        A2=reshape(conf(14,:),[M N]);
        A3=reshape(conf(20,:),[M N]);
        A4=reshape(conf(27,:),[M N]);
        A5=reshape(conf(8,:),[M N]);
        A6=reshape(conf(9,:),[M N]);
        A7=reshape(conf(22,:),[M N]);
        A8=edge(rgb2gray(imgL),'canny');
        %     DD=reshape(conf(?,:),[M N]);
        %     LRC=reshape(conf(15,:),[M N]);
        %     MED=reshape(conf(?,:),[M N]);
        %     MM=reshape(conf(4,:),[M N]);
        %     DB=reshape(conf(?,:),[M N]);
        
        %               LRD=reshape(conf(8,:),[M N]);
        %              AML=reshape(conf(11,:),[M N]);
        
        %     LRD=reshape(conf(5,:),[M N]);
        %     AML=reshape(conf(6+1,:),[M N]);
        
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
                    input(pCount,1,i)=squeeze(A1(x,y));
                    input(pCount,2,i)=squeeze(DB(x,y));
                    input(pCount,3,i)=squeeze(A2(x,y));
                    input(pCount,4,i)=squeeze(A3(x,y));
                    input(pCount,5,i)=squeeze(A4(x,y));
                    input(pCount,6,i)=squeeze(A5(x,y));
                    input(pCount,7,i)=squeeze(A6(x,y));
                    input(pCount,8,i)=squeeze(A7(x,y));
                    input(pCount,9,i)=squeeze(A8(x,y));
                    
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
    %[Confidence, indices]=max(finalScores);
    Confidence=finalScores';
    
    %% getting results per image
    
    Results=struct;
    for testImgNum=1:size(imgPixelCountTest,2)
        ind1=sum(imgPixelCountTest(1:testImgNum-1));
        ind2=ind1+imgPixelCountTest(testImgNum);
        imgNum=testImgNum+size(imgPixelCountTrain,2);
        [imgW ,imgH]=size(dispData(1,imgNum).left);
        
        Results(testImgNum).Confidence=reshape(Confidence(1+ind1:ind2),[imgH imgW ])';
        
        finalDisp=dispData(1,imgNum).left;
        % GPC
        CostVolume=dispData(imgNum).CostVolume;
        imageData=AllImages(testImageList(testImgNum));
        disrange=imageData.maxDisp;
        GPCMask=zeros(imgW,imgH);
        for i=1:imgW
            for j=1:imgH
                if(Results(testImgNum).Confidence(i,j)>0.6 && Results(testImgNum).Confidence(i,j)<0.95)
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
        postErrorWithGPC=modefilt2( finalDisp );
        postErrorWithGPC = postErrorWithGPC.*~GPCMask;
        temp=finalDisp.*GPCMask;
        postErrorWithGPC=postErrorWithGPC+temp;
        
        %     postErrorGPCEdgeCorrection=edgeCorrection(postErrorWithGPC,imageData ,[ii ii] );
        
        %Results(testImgNum).FinalDisp=postErrorWithGPC;
        %     Results(testImgNum).postErrorGPCEdgeCorrection=postErrorGPCEdgeCorrection;
        Results(testImgNum).Error=EvaluateDisp(AllImages(imagesList(imgNum)),finalDisp,errThreshold);
        Results(testImgNum).PostError=EvaluateDisp(AllImages(imagesList(imgNum)),postErrorWithGPC,errThreshold) ;
        [roc,pers]=GetROC(AllImages(imagesList(imgNum)),finalDisp,Results(testImgNum).Confidence,errThreshold);
        %     Results(testImgNum).postErrorEdge=EvaluateDisp(AllImages(imagesList(imgNum)),postErrorGPCEdgeCorrection,errThreshold) ;
        
        Results(testImgNum).ROC=roc;
        %The trapz function overestimates the value of the integral when f(x) is concave up.
        Results(testImgNum).AUC=GetAUC(roc,pers); %perfect AUC is err-(1-err)*ln(1-err)
        fprintf('for ii=%d \t  testImgNum=%d \t disp error = %f \t disp PostError = %f \t  AUC=%f \n',ii,testImageList(testImgNum) ,Results(testImgNum).Error, Results(testImgNum).PostError,Results(testImgNum).AUC  );
        
        fid=fopen('result.txt','a+');
        fprintf(fid,'for ii=%d \t  testImgNum=%d \t disp error = %f \t disp PostError = %f \t  AUC=%f \n',ii,testImageList(testImgNum) ,Results(testImgNum).Error, Results(testImgNum).PostError,Results(testImgNum).AUC  );
        fclose(fid);
        
    end
end
figure;imshow(Results(end).Confidence);
figure;imshow(Results(end).FinalDisp  ,[] );
clear alldisps alldispsDif X Y roc pers imgGT imgNum i j x y labels confidence finalScores ind1 ind2 imgW imgH ind val
load chirp % chirp handel  gong
sound(y,Fs);    disp('Job Done.');