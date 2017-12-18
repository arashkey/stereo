function result=edgeCorrection(postErrorWithGPC,imageData,blockSize)
%this function for correct edge of disparity
if nargin<3
    blockSize=[9 9];
end
halfBlockSize=floor(blockSize/2);

[w,h]=size(postErrorWithGPC);


edgeForPost=edge(postErrorWithGPC,'canny');
edgeLeft=edge(rgb2gray(imread( imageData.LImage)),'canny');
dispRange=imageData.maxDisp;
edgeRight=edge(rgb2gray(imread( imageData.RImage)),'canny');

edgeDisparity=zeros(w,h);
for i=1:w
    for j=1:h
        if(edgeLeft(i,j)==0)
            continue;
        end
        for ii=i:min(i+dispRange,w)
            if(edgeRight(ii,j)==1)
                edgeDisparity(i,j)=ii-i;
            end
        end
    end
end
moveDisparity=zeros(w,h,2);%move disparity with edge [x y]
imshow(edgeForPost+ edgeLeft *0.5 +edgeRight*0.2 )
for i=1:w-blockSize(1)
    for j=1:h-blockSize(2)
        if(i<halfBlockSize(1) ||  j<halfBlockSize(2) ||  edgeForPost(i,j)==0)
            continue;
        end
        theBlock= postErrorWithGPC(i-halfBlockSize(1)+1:i+halfBlockSize(1)+1,j-halfBlockSize(2)+1:j+halfBlockSize(2)+1) ;
        theedgeDisparityBlock= edgeDisparity(i-halfBlockSize(1)+1:i+halfBlockSize(1)+1,j-halfBlockSize(2)+1:j+halfBlockSize(2)+1) ;
        [theBlockw,theBlockh]=size(theBlock);
        sumBlock=sum(theBlock(:));
        if( sumBlock==0)
            continue;
        end
        for ii1=0:halfBlockSize(1)
            mask=padarray(zeros(2*ii1+1,2*ii1+1),[1 1],1);
            [ maskw, maskh]=size(mask);
            mask=padarray(mask,[(theBlockw-maskw)/2, (theBlockh-maskh)/2],0);
            maskBlock=theBlock.*mask;
            maskDisparity=(theedgeDisparityBlock).*mask;
            if(sum(maskBlock(:))>0)
                maskDisparity(maskDisparity==0)=99999;
                [x, y ]=min(maskDisparity);
                [z, p ]=min(x);
                moveDisparity(i,j,:)=   halfBlockSize- [y(p) p];
                break;
            end
        end
        
    end
end
%% show mask
result =postErrorWithGPC;
for i=1:w
    for j=1:h
        if(sum(moveDisparity(i,j,:)>0))
            x = moveDisparity(i,j,1) ;
            y = moveDisparity(i,j,2);
            if(i+x<1)
                x=1;
            end
             if(j+y<1)
                y=1;
            end
            try
            replaceData=result(i+x+round(x/(x+0.00001)),j+y+round(y/(y+0.00001)));
            catch ex
                disp (ex)
            end
            while(x~=0 || y~=0)
                result(max(i+x,1),max(j+y,1))=replaceData;
                if(x~=0)
                    x=x-(x/x);
                end
                if(y~=0)
                    y=y-y/y;
                end
            end
            result(i-moveDisparity(i,j,1),j-moveDisparity(i,j,2))=result(i,j);
        else
            result(i,j)=postErrorWithGPC(i,j);
        end
    end
end

subplot(121);
imshow(result,[]);
subplot(122);
imshow(postErrorWithGPC,[]);