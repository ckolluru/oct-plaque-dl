% Get directory list containing validation set
clear;

folder = 'C:\Users\Chaitanya\Documents\OCT deep learning dataset\Split sample cryo\';
dirList = dir('C:\Users\Chaitanya\Documents\OCT deep learning dataset\Split sample cryo\Vessel*');

for k = 1:numel(dirList)
    contents_in_oct_raw = dir(['C:\Users\Chaitanya\Documents\OCT deep learning dataset\Split sample cryo\' dirList(k).name '\OCT Raw\*.oct']);
    raw_file_names{k} = contents_in_oct_raw(1).name;
end

frame_nums = {240:257; 205:244; 113:160; 210:224; 346:410; 242:316; 84:158; 204:250; 208:247};

%% Save pixel shifted images
% for k = 1:numel(dirList)
%     
%     currentDir = dirList(k).name;
%     currentRawFileName = raw_file_names{k};    
%     currentFrameSet = frame_nums{k};
%     
%     Guidewire_Positions = zeros(numel(currentFrameSet), 2);
%     
%     for frame = 1:numel(currentFrameSet)
%         tic;
%         
%         currentFrame = currentFrameSet(frame);
%         
%         % Read the raw image
%         im = imread([folder currentDir '\OCT Raw\' currentRawFileName], currentFrame);
%         
%         % Remove guidewire
%         [guidewire_positions, im_gw_removed] = remove_guidewire_block(double(im));
% 
%         fprintf('Guidewire positions: %d %d \n', guidewire_positions(1), guidewire_positions(2));
% 
%         % Account for offset
%         if guidewire_positions(1) < 4
%             guidewire_positions(1) = 4;
%         end
%         if guidewire_positions(2) < 4
%             guidewire_positions(2) = 4;
%         end
%         if guidewire_positions(1) > size(im,1)-4
%             guidewire_positions(1) = size(im,1)-4;
%         end
%         if guidewire_positions(2) > size(im,1)-4
%             guidewire_positions(2) = size(im,1)-4;
%         end
% 
%         % Detect lumen
%         lumenPixels = lumen_detection_block(double(im), guidewire_positions, true);
%             
%         % Pixel shift
%         [im_pix_shift, ~] = pixel_shifting_block(im, lumenPixels, false);
% 
%         % Remove guidewire A-lines
%         if guidewire_positions(2) - guidewire_positions(1) < 100
%             im_pix_shift(max(1,guidewire_positions(1)-5):...
%                          min(guidewire_positions(2)+5, size(im,1)),:) = 0;
%             im_pix_shift = log(double(im_pix_shift) + 1.0);
% 
%             roi1 = im_pix_shift(1:max(1,guidewire_positions(1) -5), :);
%             roi2 = im_pix_shift(min(guidewire_positions(2)+5, size(im,1)):size(im,1), :);
% 
%             roi1_filtered = imgaussfilt(roi1, 1, 'FilterSize', [7 7], 'Padding', 'symmetric');
%             roi2_filtered = imgaussfilt(roi2, 1, 'FilterSize', [7 7], 'Padding', 'symmetric'); 
% 
%             im_pix_shift(1:max(1,guidewire_positions(1) -5), :) = roi1_filtered;
%             im_pix_shift(min(guidewire_positions(2)+5, size(im,1)):size(im,1), :)  = roi2_filtered;       
% 
%         else
%             im_pix_shift(1:guidewire_positions(1),:) = 0;
%             im_pix_shift(guidewire_positions(2):size(im,1),:) = 0;
% 
%             im_pix_shift = log(double(im_pix_shift) + 1.0);
% 
%             roi1 = im_pix_shift(guidewire_positions(1):guidewire_positions(2), :);
%             roi1_filtered = imgaussfilt(roi1, 1, 'FilterSize', [7 7], 'Padding', 'symmetric');
% 
%             im_pix_shift(guidewire_positions(1):guidewire_positions(2), :) = roi1_filtered;
%         end
%         
%         % Remove log filtering
%         im_pix_shift = exp(im_pix_shift) - 1.0;
% 
%         Guidewire_Positions(frame,:) = guidewire_positions;
% 
%         imwrite(uint16(im_pix_shift), [folder '\Pixel Shifted Images\' dirList(k).name '_' int2str(currentFrame) '_Pixel_Shifted.tif']);
%         
%         fprintf('One frame took %f seconds', toc);
%         
%         close all;
%     end
%  
%     save([folder '\Guidewire Positions\' dirList(k).name '_Guidewire.mat'], 'Guidewire_Positions');
% end


%% Make labels
% for k = 1:numel(dirList)
%     
%     currentDir = dirList(k).name;
%     currentRawFileName = raw_file_names{k};    
%     currentFrameSet = frame_nums{k};
%     
%     dirList2 = dir([folder currentDir '\OCT Labels\*.tif']);
%     
%     for frame = 1:numel(currentFrameSet)
%         
%         currentFrame = currentFrameSet(frame);
%         
%         label_xy = imread([folder currentDir '\OCT Labels\' dirList2(frame).name]);
%         rt_image = imread([folder currentDir '\OCT Raw\' currentRawFileName], 1);
%             
%         label_xy = imrotate(label_xy, -90);
%     
%         label_rt = polartorect_fast(double(label_xy), size(rt_image, 2), size(rt_image, 1));
%         label_rt = label_rt';
%         label_alines = zeros(size(rt_image, 1), 1);  
%         
%         for rows = 1:size(label_rt, 1)
%             if ~isempty(find(label_rt(rows,:) == 4))
%                 label_alines(rows, 1) = 1;
%             elseif ~isempty(find(label_rt(rows,:) == 3))
%                 label_alines(rows,1) = 2;
%             else
%                 label_alines(rows,1) = 0;
%             end
%         end
%         
%         imwrite(uint8(label_alines), [folder '\Labels\' dirList(k).name '_' num2str(currentFrame) '_Labels.tif']);
%     end
%     
% end
%     
%% Make the validation matrix 
labelList = dir([folder 'Labels\*.tif']);
pixShiftList = dir([folder '\Pixel Shifted Images\*.tif']);

b = {pixShiftList.name};
c = b(1,:);
d = char(c);
e = d(:,15:end);

[~, reindex] = sort( str2double( regexp( cellstr(e), '\d+', 'match', 'once' )));
pixShiftList = pixShiftList(reindex) ;
labelList = labelList(reindex);
    
sample_image = imread([folder 'Pixel Shifted Images\' pixShiftList(1).name]);

% Assuming all OCT images in the cryo dataset have 504 A-lines
total_alines = 0;
for k = 1:numel(dirList)
     total_alines = total_alines + (504 * numel(frame_nums{k}));
end

data_array_all = zeros(total_alines, size(sample_image, 2) + 3);
aline_counter = 0;
compare_boolean = true;

for k = 1:numel(labelList)
    
    if ~strcmp(pixShiftList(k).name(1:end-18), labelList(k).name(1:end-11))
        fprintf('Strings do not match. Exiting.. \n');
        compare_boolean = false;
        break;
    end
    
    label = imread([folder '\Labels\' labelList(k).name]);
    image = imread([folder '\Pixel Shifted Images\' pixShiftList(k).name]);
        
    for row = 1:size(image,1)
        aline_counter = aline_counter + 1;
        if label(row) == 1
            if mean(image(row,:)) ~= 0
                data_array_all(aline_counter,:) = [image(row,:) 1 0 0];
            else 
                data_array_all(aline_counter,:) = [image(row,:) 0 0 1];
            end

        elseif label(row) == 2
            if mean(image(row,:)) ~= 0
                data_array_all(aline_counter,:) = [image(row,:) 0 1 0];
            else 
                data_array_all(aline_counter,:) = [image(row,:) 0 0 1];
            end

        elseif label(row) == 3
            if mean(image(row,:)) ~= 0
                data_array_all(aline_counter,:) = [image(row,:) 0 0 1];
            else 
                data_array_all(aline_counter,:) = [image(row,:) 0 0 1];  
            end

        else
            data_array_all(aline_counter,:) = [image(row,:) 0 0 1];
        end

    end
end

if compare_boolean
    ALine_Label_Matrix = data_array_all;
    save([folder 'Split_Sample_Cryo.mat'], 'ALine_Label_Matrix');
end

