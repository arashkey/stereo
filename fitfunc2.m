function [ErrorOuput,finalDisp]=fitfunc2(inputx)

addpath('PostProcessing/ICM')
inputx=round(inputx);
errThreshold=1;

covar=inputx(1);
max_diff=inputx(2); 
weight_diff=inputx(3);
iterations=round( inputx(4));

disp(['input covar=' num2str( covar ) ', max_diff=' num2str(max_diff) ', weight_diff=' num2str(weight_diff,2) ', iterations=' num2str(iterations) ]) 
a=load('dispData.mat');
ErrorOuput=0;

for i=1:size(a,2)
    leftD=a.tempData(i).leftD;
    AllImage=a.tempData(i).AllImage;
   
    finalDisp=ICM(leftD,covar, max_diff, weight_diff, iterations);
    ErrorOuput=ErrorOuput+EvaluateDisp(AllImage,finalDisp,errThreshold);
end

