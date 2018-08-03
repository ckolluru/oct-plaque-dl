close all; clear; clc;

pullbackStr = 'TRF-71';

pixShiftInfo = dir(['C:\Users\Chaitanya\Documents\MATLAB\Converted Labels\All Images And Labels\' pullbackStr '*Pixel_Shifted.tif']);
labelInfo = dir(['C:\Users\Chaitanya\Documents\MATLAB\Converted Labels\All Images And Labels\' pullbackStr '*Labels.tif']);

counter = 0;
ALine_Label_Matrix = cell(541, 2);
    
% Make sure that the info files match up
for k = 1:numel(pixShiftInfo)
    label = imread(['C:\Users\Chaitanya\Documents\MATLAB\Converted Labels\All Images And Labels\' labelInfo(k).name]);
    image = imread(['C:\Users\Chaitanya\Documents\MATLAB\Converted Labels\All Images And Labels\' pixShiftInfo(k).name]);
    
    counter = counter + 1;
    
    for row = 1:size(image,1)
       if label(row) == 1
           if mean(image(row,:)) ~= 0
               ALine_Label_Matrix{counter,1} = [ALine_Label_Matrix{counter,1}; image(row,:)];
               ALine_Label_Matrix{counter,2} = [ALine_Label_Matrix{counter,2}; [1 0 0]];
           end
       end
       if label(row) == 2
           if mean(image(row,:)) ~= 0
               ALine_Label_Matrix{counter,1} = [ALine_Label_Matrix{counter,1}; image(row,:)];
               ALine_Label_Matrix{counter,2} = [ALine_Label_Matrix{counter,2}; [0 1 0]];
           end
       end
       if label(row) == 3
           if mean(image(row,:)) ~= 0
               ALine_Label_Matrix{counter,1} = [ALine_Label_Matrix{counter,1}; image(row,:)];
               ALine_Label_Matrix{counter,2} = [ALine_Label_Matrix{counter,2}; [0 0 1]];
           end
       end
    end
    
end

% Remove empty image frames
ALine_Label_Matrix = ALine_Label_Matrix(~cellfun(@isempty, ALine_Label_Matrix(:,1)), :);    

fprintf('Total Images Processed: %d \n', size(ALine_Label_Matrix, 1));
fprintf('Total ALines Processed: %d \n', size(cell2mat(ALine_Label_Matrix(1:end,1)), 1));

save(['Pullbacks_Cell_Array\' pullbackStr '.mat'], 'ALine_Label_Matrix');
