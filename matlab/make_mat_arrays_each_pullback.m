function make_mat_arrays_each_pullback(index)

    pullbacksList = dir('C:\Users\Chaitanya\Documents\MATLAB\Pullbacks_Mat_Files_All_Alines\TRF-*.mat');

    pullbackStr = pullbacksList(index).name;
    pullbackStr = pullbackStr(1:end-4);

    pixShiftInfo = dir(['C:\Users\Chaitanya\Documents\MATLAB\Converted Labels\All Images And Labels\' pullbackStr '*Pixel_Shifted.tif']);
    labelInfo = dir(['C:\Users\Chaitanya\Documents\MATLAB\Converted Labels\All Images And Labels\' pullbackStr '*Labels.tif']);

    b = {pixShiftInfo.name};
    c = b(1,:);
    d = char(c);
    e = d(:,15:end);

    [~, reindex] = sort( str2double( regexp( cellstr(e), '\d+', 'match', 'once' )));
    pixShiftInfo = pixShiftInfo(reindex) ;
    labelInfo = labelInfo(reindex);

    sample_image = imread(['C:\Users\Chaitanya\Documents\MATLAB\Converted Labels\All Images And Labels\' pixShiftInfo(1).name]);

    data_array_all = zeros(numel(pixShiftInfo) * size(sample_image, 1), 203);

    % Make sure that the info files match 
    for k = 1:numel(pixShiftInfo)

        fprintf('Frame: %d / %d \n', k, numel(pixShiftInfo));

        label = imread(['C:\Users\Chaitanya\Documents\MATLAB\Converted Labels\All Images And Labels\' labelInfo(k).name]);
        image = imread(['C:\Users\Chaitanya\Documents\MATLAB\Converted Labels\All Images And Labels\' pixShiftInfo(k).name]);

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

    ALine_Label_Matrix = data_array_all;

    save(['Pullbacks_Mat_Files_All_Alines\' pullbackStr '.mat'], 'ALine_Label_Matrix');

    actual_labels = ALine_Label_Matrix(:,201:203);
    actual_labels_reshape = reshape(actual_labels, [size(image, 1), numel(pixShiftInfo), 3]);
    figure, imshow(uint8(actual_labels_reshape)*255, []);

end