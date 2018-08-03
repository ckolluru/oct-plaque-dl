close all; clear; clc;

pullbacks = dir('C:\Users\Chaitanya\Documents\MATLAB\Pullbacks_Mat_Files_All_Alines\TRF-*.mat');

directory = 'C:\Users\Chaitanya\Documents\MATLAB\Pullbacks_Mat_Files_Results\';

load('Frame_Numbers.mat');
load('ALines_In_Frame.mat');
load('PullbackStrings.mat');

actual_reshape_index_full = [];
predict_reshape_index_full = [];

dice_mean_calcium_pullbacks = zeros(numel(pullbacks), 1);
dice_mean_lipid_pullbacks = zeros(numel(pullbacks), 1);
dice_mean_other_pullbacks = zeros(numel(pullbacks), 1);

dice_std_calcium_pullbacks = zeros(numel(pullbacks), 1);
dice_std_lipid_pullbacks = zeros(numel(pullbacks), 1);
dice_std_other_pullbacks = zeros(numel(pullbacks), 1);

for k = 1:numel(pullbacks)
    load([directory 'Ground truth En Face View\' pullbacks(k).name(1:end-4) '_Ground_Truth_En_Face_View.mat']);
    load([directory 'Predictions En Face View\' pullbacks(k).name(1:end-4) '_Predictions_En_Face_View.mat']);
    load(['Converted Labels\Guidewire_positions\' pullbackStrings{k} '_Guidewire.mat']);
    
    total_num_of_frames = 0;
    segment_frames = frame_nums(k, :);
    
    % Delete all empty cells
    segment_frames(cellfun('isempty', segment_frames)) = [];
    frames_in_segment = zeros(size(segment_frames, 2), 1);
    
    predict_reshape_filtered = zeros(alines_in_frame(k), sum(frames_in_segment), 3);
    
    for segment = 1:size(segment_frames, 2)
        frames_in_segment(segment) = size(segment_frames{1,segment}, 2);
    end
      
    if size(actual_reshape, 2) ~= sum(frames_in_segment)
        fprintf('Incorrect number in excel sheet \n');
        fprintf('Total_Num_Of_Frames: %d \n', total_num_of_frames);
        fprintf('Number of frames in Mat file: %d \n', size(actual_reshape, 2));
        fprintf('Pullback string: %s \n', pullbacks(k).name(1:end-4));
        break;
    end
    
    predict_reshape_filtered = predict_reshape;
    predict_final = zeros(size(predict_reshape_filtered, 1), size(predict_reshape_filtered, 2));
    predict_final_color = zeros(size(predict_reshape_filtered, 1), size(predict_reshape_filtered,2), 3);
   
    for row = 1:size(predict_reshape_filtered, 1)
        for col = 1:size(predict_reshape_filtered, 2)
            
            if predict_reshape_filtered(row, col, 1) >= predict_reshape_filtered(row, col, 2)
                if predict_reshape_filtered(row, col, 1) >= predict_reshape_filtered(row, col, 3)
                    predict_final(row,col) = 1;
                end
            end
            
            if predict_reshape_filtered(row, col, 2) >= predict_reshape_filtered(row, col, 1)
                if predict_reshape_filtered(row, col, 2) >= predict_reshape_filtered(row, col, 3)
                    predict_final(row,col) = 2;
                end
            end
            
            if predict_reshape_filtered(row, col, 3) >= predict_reshape_filtered(row, col, 2)
                if predict_reshape_filtered(row, col, 3) >= predict_reshape_filtered(row, col, 1)
                    predict_final(row,col) = 3;
                end
            end            
        end
    end
    
    for segment = 1:size(segment_frames, 2)        
        start_frame = sum(frames_in_segment(1:segment-1)) + 1;
        stop_frame = start_frame + frames_in_segment(segment) - 1;
        
        predict_final_segment = predict_final(:, start_frame:stop_frame);
        
        for frame = start_frame:stop_frame
            gw_pos = Guidewire_Positions(frame,:);
            
            if gw_pos(2) - gw_pos(1) < 100
                predict_final(gw_pos(1):gw_pos(2),frame) = 0;
                actual_reshape(gw_pos(1):gw_pos(2),frame,:) = zeros(numel(gw_pos(1):gw_pos(2)), 1, 3);
            else
                predict_final(1:gw_pos(1), frame) = 0;
                predict_final(gw_pos(2):end, frame) = 0;
                actual_reshape(1:gw_pos(1),frame,:) = zeros(numel(1:gw_pos(1)), 1, 3);
                actual_reshape(gw_pos(2):end, frame,:) = zeros(numel(gw_pos(2):size(actual_reshape,1)), 1, 3);
            end
        end        
   end
    
    for row = 1:size(predict_reshape_filtered, 1)
        for col = 1:size(predict_reshape_filtered, 2)
            if predict_final(row,col) == 1
                predict_final_color(row,col,:) = [1 0 0];
            elseif predict_final(row,col) == 2
                predict_final_color(row,col,:) = [0 1 0];
            elseif predict_final(row,col) == 3
                predict_final_color(row,col,:) = [0 0 1];
            elseif predict_final(row,col) == 0
                predict_final_color(row,col,:) = [0 0 0];
            end
        end
    end
                
    actual_reshape_again = reshape(actual_reshape, size(actual_reshape, 1) * size(actual_reshape, 2), 3);
    actual_reshape_final = zeros(size(actual_reshape_again,1), 1);
    for row = 1:size(actual_reshape_again,1)
        if actual_reshape_again(row,1) == 1
            actual_reshape_final(row) = 1;
        elseif actual_reshape_again(row,2) == 1
            actual_reshape_final(row) = 2;
        elseif actual_reshape_again(row,3) == 1
            actual_reshape_final(row) = 3;
        elseif sum(actual_reshape_again(row)) == 0
            actual_reshape_final(row) = 0;
        end
    end
    
    predict_reshape_again = reshape(predict_final, size(predict_final, 1) * size(predict_final, 2), 1);
    
    save(['Pullbacks_Mat_Files_Results\Predictions Noise Cleaned\' pullbacks(k).name(1:end-4) '_Predictions_Cleaned.mat'], 'predict_reshape_again');
    
    figure;
    for segment = 1:size(segment_frames, 2)
        
        start_frame = sum(frames_in_segment(1:segment-1)) + 1;
        stop_frame = start_frame + frames_in_segment(segment) - 1;
        
        subplot(1, 3*size(segment_frames, 2), 3*segment - 2);
        imshow(actual_reshape(:,start_frame:stop_frame,:), []);
        axis normal;
        
        subplot(1, 3*size(segment_frames, 2), 3*segment - 1);
        imshow(predict_reshape(:,start_frame:stop_frame, :), []);
        axis normal;
        
        subplot(1, 3*size(segment_frames, 2), 3*segment);
        imshow(predict_final_color(:,start_frame:stop_frame, :), []);
        axis normal;
        
    end
    
    dice_calcium = NaN(size(actual_reshape, 2), 1);
    dice_lipid = NaN(size(actual_reshape, 2), 1);
    dice_other = NaN(size(actual_reshape, 2), 1);
    
    for frame = 1:size(actual_reshape, 2)
        predictions = predict_final(:, frame);
        actuals = actual_reshape(:, frame, :);
        act = zeros(size(actuals,1), size(actuals,2));
        
        for rows = 1:size(predictions, 1)
            for cols = 1:size(predictions,2)                
                if actuals(rows,cols,1) == 1
                    act(rows,cols) = 1;
                elseif actuals(rows,cols,2) == 1
                    act(rows,cols) = 2;
                elseif actuals(rows,cols,3) == 1
                    act(rows,cols) = 3;
                end                
            end
        end     
        
        % Intersections
        inter_calcium = find(predictions == 1 & act == 1);
        inter_lipid = find(predictions == 2 & act == 2);
        inter_other = find(predictions == 3 & act == 3);
        
        % Union
        union_calcium = find(predictions == 1 | act == 1);
        union_lipid = find(predictions == 2 | act == 2);
        union_other = find(predictions == 3 | act == 3);
        
        if numel(union_calcium) ~= 0
            dice_calcium(frame) = 2* numel(inter_calcium)/(numel(inter_calcium) + numel(union_calcium));
        end
        
        if numel(union_lipid) ~= 0
            dice_lipid(frame) = 2* numel(inter_lipid)/(numel(inter_lipid) + numel(union_lipid));
        end
        
        if numel(union_other) ~= 0
            dice_other(frame) = 2* numel(inter_other)/(numel(inter_other) + numel(union_other));
        end       
        
    end
    
    fprintf('Mean Dice Calcium for this pullback: %f +/- %f \n', nanmean(dice_calcium), nanstd(dice_calcium));
    fprintf('Mean Dice Lipid for this pullback: %f +/- %f \n', nanmean(dice_lipid), nanstd(dice_lipid));
    fprintf('Mean Dice Other for this pullback: %f +/- %f \n', nanmean(dice_other), nanstd(dice_other));
    
    dice_mean_calcium_pullbacks(k) = nanmean(dice_calcium);
    dice_mean_lipid_pullbacks(k) = nanmean(dice_lipid);
    dice_mean_other_pullbacks(k) = nanmean(dice_other);
    
    dice_std_calcium_pullbacks(k) = nanstd(dice_calcium);
    dice_std_lipid_pullbacks(k) = nanstd(dice_lipid);
    dice_std_other_pullbacks(k) = nanstd(dice_other);
    
    suptitle(pullbacks(k).name(1:end-4));
    
    saveas(gcf, [directory 'Images\' pullbacks(k).name(1:end-4) '.tif']);
    
    if k == 31
        a = 1;
    end
    
    actual_reshape_index_full = [actual_reshape_index_full; actual_reshape_final];
    predict_reshape_index_full = [predict_reshape_index_full; predict_reshape_again];
    
end

Confusion_Matrix = confusionmat(actual_reshape_index_full, predict_reshape_index_full);
fprintf('\n Confusion Matrix: \n');
disp(Confusion_Matrix);

Confusion_Matrix(1,:) = [];
Confusion_Matrix(:,1) = [];

for k = 1:size(Confusion_Matrix,1)
    Confusion_Matrix(k,:) = Confusion_Matrix(k,:)/sum(Confusion_Matrix(k,:));
end

fprintf('Confusion Matrix in Percentage: \n');
disp(Confusion_Matrix* 100);