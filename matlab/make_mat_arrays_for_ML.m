close all; clear; clc;

pullbackStr = 'TRF-13';

pixShiftInfo = dir(['C:\Users\Chaitanya\Documents\MATLAB\Converted Labels\Images And Labels for Machine Learning\Validation\' pullbackStr '*Pixel_Shifted.tif']);
labelInfo = dir(['C:\Users\Chaitanya\Documents\MATLAB\Converted Labels\Images And Labels for Machine Learning\Validation\'  pullbackStr '*Labels.tif']);

b = {pixShiftInfo.name};
c = b(1,:);
d = char(c);
e = d(:,15:end);

[~, reindex] = sort( str2double( regexp( cellstr(e), '\d+', 'match', 'once' )));
pixShiftInfo = pixShiftInfo(reindex) ;
labelInfo = labelInfo(reindex);


data_array_all = zeros(numel(pixShiftInfo) * 496, 203);
total_alines = 0;

% Make sure that the info files match 
for k = 1:numel(pixShiftInfo)
    
    total_alines = total_alines + 1;
    
    fprintf('Frame: %d / %d \n', k, numel(pixShiftInfo));
    
    label = imread(['C:\Users\Chaitanya\Documents\MATLAB\Converted Labels\Images And Labels for Machine Learning\Validation\' labelInfo(k).name]);
    image = imread(['C:\Users\Chaitanya\Documents\MATLAB\Converted Labels\Images And Labels for Machine Learning\Validation\' pixShiftInfo(k).name]);
    
    for row = 1:size(image,1)
        if label(row) == 1
            if mean(image(row,:)) ~= 0
                data_array_all((k-1)*size(image, 1) + row,:) = [image(row,:) 1 0 0];
            else 
                data_array_all((k-1)*size(image, 1) + row,:) = [image(row,:) 0 0 1];
            end
        
        elseif label(row) == 2
            if mean(image(row,:)) ~= 0
                data_array_all((k-1)*size(image, 1) + row,:) = [image(row,:) 0 1 0];
            else    
                data_array_all((k-1)*size(image, 1) + row,:) = [image(row,:) 0 0 1];            
            end
            
        elseif label(row) == 3
            if mean(image(row,:)) ~= 0
                data_array_all((k-1)*size(image, 1) + row,:) = [image(row,:) 0 0 1];              
            else 
                data_array_all((k-1)*size(image, 1) + row,:) = [image(row,:) 0 0 1];
            end
            
        else
            data_array_all((k-1)*size(image, 1) + row,:) = [image(row,:) 0 0 1];
        end
        
    end
end

data_array_all(sum(data_array_all, 2) == 0) = [];

ALine_Label_Matrix = data_array_all;
save([pullbackStr '.mat'], 'ALine_Label_Matrix');