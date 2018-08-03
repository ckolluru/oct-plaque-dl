close all; clear; clc;

original_directory = 'C:\Users\Chaitanya\Documents\MATLAB\Converted Labels\All Images And Labels\';
segments_directory = 'C:\Users\Chaitanya\Documents\MATLAB\Converted Labels\Images And Labels for 2D processing\';

load('Frame_Numbers.mat');
load('ALines_In_Frame.mat');
load('PullbackStrings.mat');

for k = 1:numel(pullbackStrings)
    
    segment_frames = frame_nums(k, :);
    
    % Delete all empty cells
    segment_frames(cellfun('isempty', segment_frames)) = [];
    frames_in_segment = zeros(size(segment_frames, 2), 1);
    
    for segment = 1:size(segment_frames, 2)
        frames_in_segment(segment) = size(segment_frames{1,segment}, 2);
    end
      
    filenames_for_pullback = dir([original_directory pullbackStrings(k) '*_Pixel_Shifted.tif']);

    for segment = 1:size(segment_frames, 2)
        
        start_frame = sum(frames_in_segment(1:segment-1)) + 1;
        stop_frame = start_frame + frames_in_segment(segment) - 1;
        
        CRF_Segment_Color = CRF_Color(:,start_frame:stop_frame,:);
        
        % Morphological processing on CRF_Color
        binarized_version = CRF_Segment_Color(:,:,1) ~= 1;
        binarized_version_open = bwareaopen(binarized_version, 10);
        for row = 1:size(CRF_Segment_Color, 1)
            for col = 1:size(CRF_Segment_Color, 2)
                if binarized_version_open(row,col) == 0
                    CRF_Segment_Color(row,col,:) = [1 0 0];
                end
            end
        end


        binarized_version = CRF_Segment_Color(:,:,2) ~= 1;
        binarized_version_open = bwareaopen(binarized_version, 10);
        for row = 1:size(CRF_Segment_Color, 1)
            for col = 1:size(CRF_Segment_Color, 2)
                if binarized_version_open(row,col) == 0
                    CRF_Segment_Color(row,col,:) = [0 1 0];
                end
            end
        end

        binarized_version = CRF_Segment_Color(:,:,3) ~= 1;
        binarized_version_open = bwareaopen(binarized_version, 10);
        for row = 1:size(CRF_Segment_Color, 1)
            for col = 1:size(CRF_Segment_Color, 2)
                if binarized_version_open(row,col) == 0
                    CRF_Segment_Color(row,col,:) = [0 0 1];
                end
            end
        end    
        
        CRF_Color(:,start_frame:stop_frame,:) = CRF_Segment_Color;
        CRF_Color_GW = CRF_Color;
        actual_reshape_GW = actual_reshape;
        
        for frame = start_frame:stop_frame
            gw_pos = Guidewire_Positions(frame,:);
%             For TRF-01
            if frame >=43 && frame <= 49
                gw_pos(2) = 331;
            end
            
            if gw_pos(2) - gw_pos(1) < 150
                CRF_Color_GW(gw_pos(1):gw_pos(2),frame,:) = zeros(numel(gw_pos(1):gw_pos(2)), 1, 3);
                actual_reshape_GW(gw_pos(1):gw_pos(2), frame,:) = zeros(numel(gw_pos(1):gw_pos(2)), 1, 3);
            else
                CRF_Color_GW(1:gw_pos(1), frame,:) = zeros(numel(1:gw_pos(1)), 1, 3);
                CRF_Color_GW(gw_pos(2):end, frame,:) = zeros(numel(gw_pos(2):size(CRF_Color,1)), 1, 3);
                actual_reshape_GW(1:gw_pos(1), frame,:) = zeros(numel(1:gw_pos(1)), 1, 3);
                actual_reshape_GW(gw_pos(2):end, frame,:) = zeros(numel(gw_pos(2):size(actual_reshape_GW,1)), 1, 3);
                
            end
        end
        
        subplot(1, 2*size(segment_frames, 2), 2*segment - 1);
        imshow(actual_reshape_GW(:,start_frame:stop_frame,:), []);
        axis normal;
        
        subplot(1, 2*size(segment_frames, 2), 2*segment);
        imshow(CRF_Color_GW(:,start_frame:stop_frame, :), []);
        axis normal;
        
        CRF_Color = CRF_Color_GW;
        actual_reshape = actual_reshape_GW;
    end
    
    
    for row = 1:size(CRF_Color, 1)
        for col = 1:size(CRF_Color, 2)
            
            if CRF_Color(row,col,1) == 1
                CRF_New(row,col) = 1;
            elseif CRF_Color(row,col,2) == 1
                CRF_New(row,col) = 2;
            elseif CRF_Color(row,col,3) == 1
                CRF_New(row,col) = 3;
            elseif sum(CRF_Color(row,col,:)) == 0
                CRF_New(row,col) = 0;
            end
            
            if actual_reshape(row,col,1) == 1
                actual_New(row,col) = 1;
            elseif actual_reshape(row,col,2) == 1
                actual_New(row,col) = 2;
            elseif actual_reshape(row,col,3) == 1
                actual_New(row,col) = 3;
            elseif sum(actual_reshape(row,col,:)) == 0
                actual_New(row,col) = 0;
            end
        end
    end
    
    suptitle(pullbacks(k).name(1:end-4));
    
    saveas(gcf, [directory 'Images\' pullbacks(k).name(1:end-4) '.tif']);
    predict_reshape_again = reshape(CRF_New, size(CRF_New, 1) * size(CRF_New, 2), 1);
    actual_reshape_again = reshape(actual_New, size(actual_New, 1) * size(actual_New,2), 1);
    
    actual_reshape_index_full = [actual_reshape_index_full; actual_reshape_again];
    predict_reshape_index_full = [predict_reshape_index_full; predict_reshape_again];
    
end

Confusion_Matrix = confusionmat(actual_reshape_index_full, predict_reshape_index_full);
fprintf('\n Confusion Matrix: \n');
disp(Confusion_Matrix);

% Remove guidewire class since we don't really care about it
Confusion_Matrix(:,1) = [];
Confusion_Matrix(1,:) = [];

for k = 1:3
    Confusion_Matrix(k,:) = Confusion_Matrix(k,:)/sum(Confusion_Matrix(k,:));
end

fprintf('Confusion Matrix in Percentage: \n');
disp(Confusion_Matrix* 100);