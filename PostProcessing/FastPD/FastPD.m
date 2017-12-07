function MRF_labeling = FastPD(CostVolume, numlabels ,disparity)
CostVolume=CostVolume+abs(min(CostVolume(:)));
tic
DEBUG=true;
if (DEBUG)
    input_file = 'tempinput.bin';
    output_file = 'tempoutput.bin';
else
    input_file = tempname('.');
    output_file = tempname('.');
end
delete(input_file, output_file);

fid = fopen(input_file, 'wb');
if fid == -1
    error(['Cannot open file ' results_fname]);
end

w = size(CostVolume, 1);
h = size(CostVolume, 2);
numpoints = h * w;

numpairs = 2 * h * w - h - w;
maxIters = 100;
type = 'int32';

fwrite(fid, numpoints, type);
fwrite(fid, numpairs, type);
fwrite(fid, numlabels, type);
fwrite(fid, maxIters, type);

% HACK
% avg = mean(cost3D(:));
% stdev = std(cost3D(:));
% cost3D(cost3D> avg + 2*stdev) = avg + 2 * stdev;
% cost3D = normalize(cost3D);


if (DEBUG)
    disp 'about to start writing'
    toc
    disp ' '
end


%     % label costs at each point
%     s = zeros(h, w, numlabels);
%     for l = 0 : (numlabels - 1)
%       %  diff = l/numlabels - d;
%      %   diff = diff .* diff;
%
% %        temp = diff;
%         temp = 10 * abs(l * ones(h, w) - round((numlabels - 1) * d));
%         temp(d == 0) = 0;
%         s(:, :, l + 1) = temp;
%     end
%	fread( _lcosts, sizeof(Real), _numpoints*_numlabels, fp );
% CostVolume=reshape(CostVolume,[],size(CostVolume,3));
% for i=1:size(CostVolume,1)
%     for j=1:size(CostVolume,2)
%         for k=1:size(CostVolume,3)
%             fwrite(fid, CostVolume);
%         end
%     end
% end 

fwrite(fid,uint32( CostVolume*10000), type);

if (DEBUG)
    disp 'wrote label costs'
    toc
    disp ' '
end

% pairs (each pixel is neighbor with 4 adjacent)
for i = 1:h
    row = w * (i - 1);
    %        for j = 1:(w-1)
    %            fwrite(fid, row + j - 1, type);
    %            fwrite(fid, row + j, type);
    %        end
    temp = [row, kron((row + 1):(row + w - 2), [1 1]), w - 1];
    
    
    %	fread( _pairs , sizeof(int ), _numpairs*2          , fp );
    fwrite(fid, temp, type);
end
temp = zeros(1, 2 * w * (h - 1));
temp(1:2:end) = 0 : (w * (h - 1) - 1);
temp(2:2:end) = w : (w * h - 1);


%	fread( _pairs , sizeof(int ), _numpairs*2          , fp );
fwrite(fid, temp, type);
%    for i = 1:(h-1)
%        row1 = w * (i - 1);
%        row2 = w * i;
%        for j = 1:w
%            fwrite(fid, row1 + j - 1, type);
%            fwrite(fid, row2 + j - 1, type);
%        end
%    end

if (DEBUG)
    disp 'wrote pairs'
    toc
    disp ' '
end

% inter-label costs (0 if same, 1 if adjacent, 2 otherwise)
labelcosts = 2 * ones(numlabels, numlabels) - ...
    2 * eye(numlabels) - ...
    1 * diag(ones(numlabels - 1, 1), 1) - ...
    1 * diag(ones(numlabels - 1, 1), -1);


%	fread( _dist  , sizeof(Real), _numlabels*_numlabels, fp );
fwrite(fid, labelcosts, type);

if (DEBUG)
    disp 'wrote label costs'
    toc
    disp ' '
end

% graph edge costs (const everywhere)
%	fread( _wcosts, sizeof(Real), _numpairs            , fp );

%defocos => fwrite(fid, ones(1, numpairs), type);

wcosts=zeros(w,h);
landaC=3.6;



% parfor i=1:w
%     for j=1:h
%         x=double(imgL3D(i,j,:));
%
%         result=    power(imgR3D(:,:,1)-x(:,:,1),2) ...
%             + power(imgR3D(:,:,2)-x(:,:,2),2) ...
%             + power(imgR3D(:,:,3)-x(:,:,3),2) ;
%         oghlidosDistanceRGB= sqrt(double(result)          );
%         expCompute=exp( - oghlidosDistanceRGB/landaC );
%         wcosts(i,j)=  max(  [  expCompute(:) ; 0.0003]  );
%     end
% end
fwrite(fid,disparity, type);

% graph edge costs (cost if falls on image edges)
%    for i = 1:h
%        for j = 1:(w-1)
%            if xor(d(i,j),d(i,j+1))
%                fwrite(fid, 100, type);
%            else
%                fwrite(fid, 20, type);
%            end


%            fwrite(fid, 100 * exp(-abs(img(i, j) - img(i, j+1))^2));
%            fwrite(fid, round(exp(-(img(i, j) - img(i, j+1))^2)));
%        end
%    end
%    for i = 1:(h-1)
%        for j = 1:w
%            if xor(d(i,j), d(i+1,j))
%                fwrite(fid, 100, type);
%            else
%                fwrite(fid, 20, type);
%            end


%            fwrite(fid, 100 * exp(-abs(img(i, j) - img(i+1, j))^2));
%            fwrite(fid, round(exp(-(img(i, j) - img(i+1, j))^2)));
%        end
%    end

if (DEBUG)
    disp 'wrote edge costs'
    toc
    disp ' '
end

fclose(fid);

% run FastPD
commandStr = ['"'  mfilename('fullpath') '/../FastPD.exe" ' input_file ' ' output_file]
result=system(commandStr);
if(result~=0)
    error(['FastPD.exe return error:' num2str(result)]);
end
MRF_labeling = get_MRF_labeling(output_file);
MRF_labeling = reshape(MRF_labeling, w,h);
%    MRF_labeling = max(max(MRF_labeling)) - MRF_labeling;

% clean up
%if (~DEBUG)
delete(input_file, output_file);
%end

end
