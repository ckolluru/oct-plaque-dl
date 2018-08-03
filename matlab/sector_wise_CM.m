close all; clear; clc;

pullbacks = dir('C:\Users\Chaitanya\Documents\MATLAB\Pullbacks_Mat_Files_All_Alines\TRF-*.mat');

directory = 'C:\Users\Chaitanya\Documents\MATLAB\Pullbacks_Mat_Files_Results\';
filename = 'cnn_no_norm_kernel_size_11_crf_5_19';
%filename = 'ann_feature_norm_hidden1_100_hidden2_50_hidden3_25_crf_5_19';

load('Frame_Numbers.mat');
load('ALines_In_Frame.mat');
load('PullbackStrings.mat');
load('pullbackShortStrings.mat');
load('Test_Pullback_Indices_In_Each_Fold.mat');

actual_reshape_index_full = [];
predict_reshape_index_full = [];

dice_mean_calcium_pullbacks = zeros(numel(pullbacks), 1);
dice_mean_lipid_pullbacks = zeros(numel(pullbacks), 1);
dice_mean_other_pullbacks = zeros(numel(pullbacks), 1);

dice_std_calcium_pullbacks = zeros(numel(pullbacks), 1);
dice_std_lipid_pullbacks = zeros(numel(pullbacks), 1);
dice_std_other_pullbacks = zeros(numel(pullbacks), 1);

sector_CM = zeros(4,4);

for k = 1:numel(pullbacks)   
    
    pullbackString_short = pullbackShortStrings{k};
    
    load([directory 'Ground truth En Face View\' pullbacks(k).name(1:end-4) '_Ground_Truth_En_Face_View.mat']);
    load([directory 'Predictions_CRF_Noise_Cleaned\' pullbacks(k).name(1:end-4) '_Predictions_En_Face_View.mat']);
    load([directory 'Predictions En Face View\' pullbackString_short '_Predictions_En_Face_view.mat']);
    load(['C:\Users\Chaitanya\Documents\MATLAB\Pullbacks_Mat_Files_All_Alines\ALines Zero\' pullbackString_short '.mat']);
    
    load(['Converted Labels\Guidewire_positions\' pullbackStrings{k} '_Guidewire.mat']);
    
    total_num_of_frames = 0;
    segment_frames = frame_nums(k, :);
    
    % Delete all empty cells
    segment_frames(cellfun('isempty', segment_frames)) = [];
    frames_in_segment = zeros(size(segment_frames, 2), 1);
    
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
    
    CRF_Color = zeros(size(CRF_Results, 1), size(CRF_Results,2), 3);
    CRF_New = 5*ones(size(CRF_Results));
    actual_New = 5*ones(size(CRF_Results));
   
    for row = 1:size(CRF_Results, 1)
        for col = 1:size(CRF_Results, 2)            
            if CRF_Results(row,col) == 0
                CRF_Color(row,col,:) = [1 0 0];
            elseif CRF_Results(row,col) == 1
                CRF_Color(row,col,:) = [0 1 0];
            elseif CRF_Results(row,col) == 2
                CRF_Color(row,col,:) = [0 0 1];
            end
        end
    end
    
    %actual_reshape_again = reshape(actual_reshape, size(actual_reshape, 1) * size(actual_reshape, 2), 3);
    %[actual_reshape_column, actual_reshape_index] = max(actual_reshape_again, [], 2);
    
    figure;  
    
    CRF_Color_GW = zeros(size(CRF_Color));
    actual_reshape_GW = zeros(size(actual_reshape));
    predict_reshape_GW = zeros(size(predict_reshape));
        
    CRF_Color_GW_Confusion_Mat = zeros(size(CRF_Color));
    actual_reshape_GW_Confusion_Mat = zeros(size(actual_reshape));
    predict_reshape_GW_Confusion_Mat = zeros(size(predict_reshape));
    
    for segment = 1:size(segment_frames, 2)
                
        start_frame = sum(frames_in_segment(1:segment-1)) + 1;
        stop_frame = start_frame + frames_in_segment(segment) - 1;
        
        CRF_Segment_Color = CRF_Color(:,start_frame:stop_frame,:);
        
        % Morphological processing on CRF_Color
        
        % Close calcium and lipid chunks
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
                        
        % Remove islands of calcium and lipid
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
        
        for frame = start_frame:stop_frame
            gw_pos = Guidewire_Positions(frame,:);
            
            crf_size = size(CRF_Color_GW,1);
            
            for row = 1:size(CRF_Color, 1)
                if alines_zero(row, frame) == 0
                    CRF_Color_GW_Confusion_Mat(row, frame, :) = CRF_Color(row, frame, :);
                    actual_reshape_GW_Confusion_Mat(row, frame, :) = actual_reshape(row, frame, :);
                    predict_reshape_GW_Confusion_Mat(row, frame, :) = predict_reshape(row, frame, :);
                end
            end
            
            if gw_pos(2) - gw_pos(1) < 150
                CRF_Color_GW(1:max(1,gw_pos(1)-5), frame, :) = CRF_Color(1:max(1,gw_pos(1)-5), frame, :);
                CRF_Color_GW(min(gw_pos(2)+5, crf_size):crf_size, frame, :) = CRF_Color(min(gw_pos(2)+5, crf_size):crf_size, frame, :);
                
                actual_reshape_GW(1:max(1,gw_pos(1)-5), frame, :) = actual_reshape(1:max(1,gw_pos(1)-5), frame, :);
                actual_reshape_GW(min(gw_pos(2)+5, crf_size):crf_size, frame, :) = actual_reshape(min(gw_pos(2)+5, crf_size):crf_size, frame, :);
                                
                predict_reshape_GW(1:max(1,gw_pos(1)-5), frame, :) = predict_reshape(1:max(1,gw_pos(1)-5), frame, :);
                predict_reshape_GW(min(gw_pos(2)+5, crf_size):crf_size, frame, :) = predict_reshape(min(gw_pos(2)+5, crf_size):crf_size, frame, :);
                
%                 CRF_Color_GW(max(1,gw_pos(1)-4):min(gw_pos(2)+4,size(CRF_Color_GW,1)),frame,:) = zeros(numel(max(1,gw_pos(1)-4):min(gw_pos(2)+4,size(CRF_Color_GW,1))), 1, 3);
%                 actual_reshape_GW(max(1,gw_pos(1)-4):min(gw_pos(2)+4,size(CRF_Color_GW,1)), frame,:) = zeros(numel(max(1,gw_pos(1)-4):min(gw_pos(2)+4,size(CRF_Color_GW,1))), 1, 3);
%                 predict_reshape_GW(max(1,gw_pos(1)-4):min(gw_pos(2)+4,size(CRF_Color_GW,1)), frame,:) = zeros(numel(max(1,gw_pos(1)-4):min(gw_pos(2)+4,size(CRF_Color_GW,1))), 1, 3);
                
            else
%                 if max(1,gw_pos(1)-5) ~= 1
%                     start_aline = max(1,gw_pos(1)-5) - 1;
%                     CRF_Color_GW(1:start_aline, frame,:) = zeros(numel(1:start_aline), 1, 3);                                    
%                     actual_reshape_GW(1:start_aline, frame,:) = zeros(numel(1:start_aline), 1, 3);
%                     predict_reshape_GW(1:start_aline, frame, :) = zeros(numel(1:start_aline), 1, 3);
%                 end
%                 
%                 if min(gw_pos(2)+5,size(CRF_Color_GW,1)) ~= size(CRF_Color_GW,1)
%                     stop_aline = min(gw_pos(2)+5,size(CRF_Color_GW,1)) + 1;
%                     CRF_Color_GW(stop_aline:end, frame,:) = zeros(numel(stop_aline:size(CRF_Color,1)), 1, 3);
%                     actual_reshape_GW(stop_aline:end, frame,:) = zeros(numel(stop_aline:size(actual_reshape_GW,1)), 1, 3);
%                     predict_reshape_GW(stop_aline:end, frame, :) = zeros(numel(stop_aline:size(predict_reshape,1)), 1, 3);
%                 end

                CRF_Color_GW(max(1,gw_pos(1)-5):min(gw_pos(2)+5,crf_size), frame, :) = CRF_Color(max(1,gw_pos(1)-5):min(gw_pos(2)+5,crf_size), frame, :);                
                actual_reshape_GW(max(1,gw_pos(1)-5):min(gw_pos(2)+5,crf_size), frame, :) = actual_reshape(max(1,gw_pos(1)-5):min(gw_pos(2)+5,crf_size), frame, :);
                predict_reshape_GW(max(1,gw_pos(1)-5):min(gw_pos(2)+5,crf_size), frame, :) = predict_reshape(max(1,gw_pos(1)-5):min(gw_pos(2)+5,crf_size), frame, :);
                
            end
        end
        
        subplot(1, 3*size(segment_frames, 2), 3*segment - 2);
        imshow(actual_reshape_GW(:,start_frame:stop_frame,:), []);
        axis normal;
        
        subplot(1, 3*size(segment_frames, 2), 3*segment - 1);
        imshow(predict_reshape_GW(:,start_frame:stop_frame, :), []);
        axis normal;
        
        subplot(1, 3*size(segment_frames, 2), 3*segment);
        imshow(CRF_Color_GW(:,start_frame:stop_frame, :), []);
        axis normal;
        
        if k == 31
            a = 1;
        end
        
        CRF_Color2 = CRF_Color;
        CRF_Color(:, start_frame:stop_frame, :) = CRF_Color_GW(:, start_frame:stop_frame, :);
        actual_reshape(:, start_frame:stop_frame, :) = actual_reshape_GW(:, start_frame:stop_frame, :);
        predict_reshape(:, start_frame:stop_frame, :) = predict_reshape_GW(:, start_frame:stop_frame, :);
    end
    
    for row = 1:size(CRF_Color2, 1)
        for col = 1:size(CRF_Color2, 2)
            if CRF_Color2(row,col,1) == 1
                CRF_Results(row,col) = 0;
            elseif CRF_Color2(row,col,2) == 1
                CRF_Results(row,col) = 1;
            else
                CRF_Results(row,col) = 2;
            end
        end
    end
    
    % Save final results after morphological closing
    save([directory 'Predictions_CRF_Noise_Cleaned_Morph\' pullbacks(k).name(1:end-4) '_Predictions_En_Face_View.mat'], 'CRF_Results');
    
%     for row = 1:size(CRF_Color, 1)
%         for col = 1:size(CRF_Color, 2)
%             
%             if CRF_Color(row,col,1) == 1
%                 CRF_New(row,col) = 1;
%             elseif CRF_Color(row,col,2) == 1
%                 CRF_New(row,col) = 2;
%             elseif CRF_Color(row,col,3) == 1
%                 CRF_New(row,col) = 3;
%             elseif sum(CRF_Color(row,col,:)) == 0
%                 CRF_New(row,col) = 0;
%             end
%             
%             if actual_reshape(row,col,1) == 1
%                 actual_New(row,col) = 1;
%             elseif actual_reshape(row,col,2) == 1
%                 actual_New(row,col) = 2;
%             elseif actual_reshape(row,col,3) == 1
%                 actual_New(row,col) = 3;
%             elseif sum(actual_reshape(row,col,:)) == 0
%                 actual_New(row,col) = 0;
%             end
%         end
%     end

    for row = 1:size(CRF_Color_GW_Confusion_Mat, 1)
        for col = 1:size(CRF_Color_GW_Confusion_Mat, 2)
            
            if CRF_Color_GW_Confusion_Mat(row,col,1) == 1
                CRF_New(row,col) = 1;
            elseif CRF_Color_GW_Confusion_Mat(row,col,2) == 1
                CRF_New(row,col) = 2;
            elseif CRF_Color_GW_Confusion_Mat(row,col,3) == 1
                CRF_New(row,col) = 3;
            elseif sum(CRF_Color_GW_Confusion_Mat(row,col,:)) == 0
                CRF_New(row,col) = 0;
            end
            
            if actual_reshape_GW_Confusion_Mat(row,col,1) == 1
                actual_New(row,col) = 1;
            elseif actual_reshape_GW_Confusion_Mat(row,col,2) == 1
                actual_New(row,col) = 2;
            elseif actual_reshape_GW_Confusion_Mat(row,col,3) == 1
                actual_New(row,col) = 3;
            elseif sum(actual_reshape_GW_Confusion_Mat(row,col,:)) == 0
                actual_New(row,col) = 0;
            end
        end
    end

    suptitle(pullbacks(k).name(1:end-4));
    
    saveas(gcf, [directory 'Images\' pullbacks(k).name(1:end-4) '.tif']);
    predict_reshape_again = reshape(CRF_New, size(CRF_New, 1) * size(CRF_New, 2), 1);
    actual_reshape_again = reshape(actual_New, size(actual_New, 1) * size(actual_New,2), 1);
    
    sector_gt = zeros(8, size(actual_New, 2));
    sector_pred = zeros(8, size(CRF_New, 2));
    
    for col = 1:size(CRF_New, 2)
        for row = 1:8
            start_aline = (size(CRF_New, 1)/8 * (row-1)) + 1;
            stop_aline = start_aline + (size(CRF_New,1)/8) - 1;
            
            sector_gt(row,col) = mode(actual_New(start_aline:stop_aline, col));
            sector_pred(row,col) = mode(CRF_New(start_aline:stop_aline, col));
            
        end
    end
           
    sector_CM = sector_CM + confusionmat(sector_gt(:), sector_pred(:), 'Order', [0 1 2 3]);
    
    actual_reshape_index_full = [actual_reshape_index_full; actual_reshape_again];
    predict_reshape_index_full = [predict_reshape_index_full; predict_reshape_again];
    
    disp(pullbackStrings(k));
    
    save(['Pullbacks_Mat_Files_Results\Final Predictions In-vivo Pullbacks\Pullback_' int2str(k) '.mat'], 'predict_reshape_again');
    CM_Ind_Pullback(:,:,k) = confusionmat(actual_reshape_again, predict_reshape_again, 'Order', [0 1 2 3]);

    fprintf('\n Confusion Matrix: \n');
    disp(CM_Ind_Pullback(:,:,k));

    for index = 1:size(CM_Ind_Pullback, 1)
        CM_Ind_Pullback_Percent(index,:, k) = CM_Ind_Pullback(index,:,k)/sum(CM_Ind_Pullback(index,:,k));
    end    

    fprintf('Confusion Matrix in Percentage: \n');
    disp(CM_Ind_Pullback_Percent(:,:,k)* 100);
    
end

% Compute mean and std. dev percentage for confusion matrix
fprintf('Confusion Matrix Mean Percentage across pullbacks \n');
disp(nanmean(CM_Ind_Pullback, 3));

fprintf('Confusion Matrix Std Deviation Percentage across pullbacks \n');
disp(nanstd(CM_Ind_Pullback, 0, 3));

Confusion_Matrix = confusionmat(actual_reshape_index_full, predict_reshape_index_full);
fprintf('\n Confusion Matrix: \n');
disp(Confusion_Matrix);

% Remove guidewire class since we don't really care about it
Confusion_Matrix(:,1) = [];
Confusion_Matrix(1,:) = [];

% Print F1 score metrics
metrics(Confusion_Matrix);
for k = 1:3
    Confusion_Matrix(k,:) = Confusion_Matrix(k,:)/sum(Confusion_Matrix(k,:));
end

% fprintf('Confusion Matrix in Percentage: \n');
% disp(Confusion_Matrix* 100);

fprintf('Mean and standard error over folds \n');
CM_Ind_Pullback(:,1,:) = [];
CM_Ind_Pullback(1,:,:) = [];

CM_folds = zeros(3,3,10);
for fold = 1:10
    
    if fold == 10
        test_folds = test_folds_full(fold,1:3);
    else
        test_folds = test_folds_full(fold,:);
    end
    
    CM_folds(:,:,fold) = sum(CM_Ind_Pullback(:,:,test_folds), 3);
    
    for row = 1:3
        CM_folds_percent(row, :, fold) = CM_folds(row,:,fold)/sum(CM_folds(row,:,fold));
    end
end

CM_folds_mean = mean(CM_folds_percent, 3) * 100;
CM_folds_std = std(CM_folds_percent, [], 3) * 100/sqrt(10);

disp(CM_folds_mean);
disp(CM_folds_std);

save(['Pullbacks_Mat_Files_Results\Confusion Matrices\' filename '_Mean_CM.mat'], 'CM_folds_mean');
save(['Pullbacks_Mat_Files_Results\Confusion Matrices\' filename '_Std_CM.mat'], 'CM_folds_std');
