clear; close all;

% Create En Face Views

frame_nums = {240:257; 205:244; 113:160; 210:224; 346:410; 242:316; 84:158; 204:250; 208:247};
dirList = dir('C:\Users\Chaitanya\Documents\OCT deep learning dataset\Split sample cryo\Vessel*');

for k = 1:numel(dirList)
    contents_in_oct_raw = dir(['C:\Users\Chaitanya\Documents\OCT deep learning dataset\Split sample cryo\' dirList(k).name '\OCT Raw\*.oct']);
    raw_file_names{k} = contents_in_oct_raw(1).name;
end

load('Pullbacks_Mat_Files_Results\Predictions Cryo\alines_validation_prediction.mat');
folder = 'C:\Users\Chaitanya\Documents\OCT deep learning dataset\Split sample cryo\';

load([folder 'Split_Sample_Cryo.mat']);
actual = ALine_Label_Matrix(:,201:203);
predict = alines_validation_prediction;

% Vessel 59 is ordered as 100 to 158 and then 84 to 99, so fix that.
actual_59_frame_84_99 = actual((320*504)+1: (336*504),:);
actual_59_frame_100_158 = actual((261*504)+1:(320*504),:);

actual((261*504)+1:(277*504),:) = actual_59_frame_84_99;
actual((277*504)+1:(336*504),:) = actual_59_frame_100_158;

predict_59_frame_84_99 = predict((320*504)+1: (336*504),:);
predict_59_frame_100_158 = predict((261*504)+1:(320*504),:);

predict((261*504)+1:(277*504),:) = predict_59_frame_84_99;
predict((277*504)+1:(336*504),:) = predict_59_frame_100_158;

%Remove temporary variables
clear actual_59_frame_84_99 actual_59_frame_100_158 predict_59_frame_84_99 predict_59_frame_100_158

%View en-face views before cleaning
aline_counter = 0;
for k = 1:numel(dirList)
    
    pullback_actual_label = actual(aline_counter + 1:aline_counter + 504*numel(frame_nums{k}),:);
    pullback_predicted_label = predict(aline_counter + 1:aline_counter + 504*numel(frame_nums{k}),:);

    actual_reshape = reshape(pullback_actual_label, 504, size(pullback_actual_label,1)/504, 3);
    predict_reshape = reshape(pullback_predicted_label, 504, size(pullback_predicted_label,1)/504, 3);
    
    figure;
    suptitle(dirList(k).name);
    subplot(1,2,1), imshow(actual_reshape, []);
    subplot(1,2,2), imshow(predict_reshape, []);
    
    aline_counter = aline_counter + (504*numel(frame_nums{k}));
    
    save(['Pullbacks_Mat_Files_Results\Predictions Cryo En Face View\' dirList(k).name '_Ground_Truth_En_Face_View.mat'], 'actual_reshape');
    save(['Pullbacks_Mat_Files_Results\Predictions Cryo En Face View\' dirList(k).name '_Predictions_En_Face_View.mat'], 'predict_reshape');

end

%% CRF noise cleaning
aline_counter = 0;
confusion_matrix = zeros(4,4);
actual_full = [];
pullback_crf_full = [];

for k = 1:numel(dirList)

    load(['Pullbacks_Mat_Files_Results\Predictions Cryo CRF Noise Cleaned\' dirList(k).name '_Predictions_En_Face_View.mat']);
    
    % Guidewire Positions
    load([folder 'Guidewire Positions\' dirList(k).name '_Guidewire.mat']);
    
    pullback_actual_label = actual(aline_counter + 1:aline_counter + 504*numel(frame_nums{k}),:);
    pullback_predicted_label = predict(aline_counter + 1:aline_counter + 504*numel(frame_nums{k}),:);
    pullback_crf_label = CRF_Results + 1; % CRF gives 0,1,2 instead of 1,2,3
    
    pullback_crf_label_color = zeros(size(pullback_crf_label, 1), size(pullback_crf_label, 2), 3);
    
    for row = 1:size(pullback_crf_label, 1)
        for col = 1:size(pullback_crf_label, 2)
            if pullback_crf_label(row,col) == 1
                pullback_crf_label_color(row,col,:) = [1 0 0];
            end
            if pullback_crf_label(row,col) == 2
                pullback_crf_label_color(row,col,:) = [0 1 0];
            end
            if pullback_crf_label(row,col) == 3
                pullback_crf_label_color(row,col,:) = [0 0 1];
            end
        end
    end
    
    actual_reshape = reshape(pullback_actual_label, 504, size(pullback_actual_label,1)/504, 3);
    predict_reshape = reshape(pullback_predicted_label, 504, size(pullback_predicted_label,1)/504, 3);
    actual_label = zeros(size(actual_reshape,1), size(actual_reshape,2));
    
    for row = 1:size(actual_reshape, 1)
        for col = 1:size(actual_reshape, 2)
            if actual_reshape(row,col,1) == 1
                actual_label(row,col) = 1;
            end
            if actual_reshape(row,col,2) == 1
                actual_label(row,col) = 2;
            end
            if actual_reshape(row,col,3) == 1
                actual_label(row,col) = 3;
            end
        end
    end
    
    % Close calcium and lipid chunks
    binarized_version = pullback_crf_label_color(:,:,1) ~= 1;
    binarized_version_open = bwareaopen(binarized_version, 10);
    for row = 1:size(pullback_crf_label_color, 1)
        for col = 1:size(pullback_crf_label_color, 2)
            if binarized_version_open(row,col) == 0
                pullback_crf_label_color(row,col,:) = [1 0 0];
            end
        end
    end


    binarized_version = pullback_crf_label_color(:,:,2) ~= 1;
    binarized_version_open = bwareaopen(binarized_version, 10);
    for row = 1:size(pullback_crf_label_color, 1)
        for col = 1:size(pullback_crf_label_color, 2)
            if binarized_version_open(row,col) == 0
                pullback_crf_label_color(row,col,:) = [0 1 0];
            end
        end
    end 

    % Remove islands of calcium and lipid
    binarized_version = pullback_crf_label_color(:,:,3) ~= 1;
    binarized_version_open = bwareaopen(binarized_version, 10);
    for row = 1:size(pullback_crf_label_color, 1)
        for col = 1:size(pullback_crf_label_color, 2)
            if binarized_version_open(row,col) == 0
                pullback_crf_label_color(row,col,:) = [0 0 1];
            end
        end
    end  
    
    for row = 1:size(pullback_crf_label_color, 1)
        for col = 1:size(pullback_crf_label_color, 2)
            if pullback_crf_label_color(row,col,1) == 1
                pullback_crf_label(row,col) = 1;
            end
            if pullback_crf_label_color(row,col,2) == 1
                pullback_crf_label(row,col) = 2;
            end
            if pullback_crf_label_color(row,col,3) == 1
                pullback_crf_label(row,col) = 3;
            end
        end
    end
    
    % Fix for guidewire positions
    actual_reshape_GW = actual_reshape; 
    predict_reshape_GW = predict_reshape;
    pullback_crf_label_color_GW = pullback_crf_label_color;
    
    for col = 1:size(actual_reshape, 2)
        gw_pos = Guidewire_Positions(col,:);
        
        for row = 1:size(actual_reshape,1)
            
            if gw_pos(2) - gw_pos(1) < 150
                actual_label(gw_pos(1):gw_pos(2),col) = 0;
                pullback_crf_label(gw_pos(1):gw_pos(2),col) = 0;
                pullback_crf_label_color_GW(gw_pos(1):gw_pos(2),col,:) = zeros(numel(gw_pos(1):gw_pos(2)), 1, 3);
                actual_reshape_GW(gw_pos(1):gw_pos(2), col,:) = zeros(numel(gw_pos(1):gw_pos(2)), 1, 3);
                predict_reshape_GW(gw_pos(1):gw_pos(2), col,:) = zeros(numel(gw_pos(1):gw_pos(2)), 1, 3);
                
            else
                actual_label(1:gw_pos(1), col) = 0;
                actual_label(gw_pos(2):end, col) = 0;
                pullback_crf_label(1:gw_pos(1), col) = 0;
                pullback_crf_label(gw_pos(2):end, col) = 0;
                pullback_crf_label_color_GW(1:gw_pos(1), col,:) = zeros(numel(1:gw_pos(1)), 1, 3);
                pullback_crf_label_color_GW(gw_pos(2):end, col,:) = zeros(numel(gw_pos(2):size(CRF_Color,1)), 1, 3);
                actual_reshape_GW(1:gw_pos(1), col,:) = zeros(numel(1:gw_pos(1)), 1, 3);
                actual_reshape_GW(gw_pos(2):end, col,:) = zeros(numel(gw_pos(2):size(actual_reshape_GW,1)), 1, 3);
                predict_reshape_GW(1:gw_pos(1), col, :) = zeros(numel(1:gw_pos(1)), 1, 3);
                predict_reshape_GW(gw_pos(2):end, col, :) = zeros(numel(gw_pos(2):size(predict_reshape,1)), 1, 3);
            end
        end
    end
    
            
    figure;
    suptitle(dirList(k).name);
    subplot(1,3,1), imshow(actual_reshape_GW, []);
    subplot(1,3,2), imshow(predict_reshape_GW, []);
    subplot(1,3,3), imshow(pullback_crf_label_color_GW, []);
    
    aline_counter = aline_counter + (504*numel(frame_nums{k}));
    
    %confusion_matrix = confusion_matrix + confusionmat(actual_label(:), double(pullback_crf_label(:)));
    
    % If training with 2 additional cryo vessels, remove them from the
    % confusion matrix results
    if k ~= 1 && k ~= 5
        confusion_matrix = confusion_matrix + confusionmat(actual_label(:), double(pullback_crf_label(:)), 'Order', [0 1 2 3]);
                    
        actual_full = [actual_full; actual_label(:)];
        pullback_crf_full = [pullback_crf_full; pullback_crf_label(:)];
    end

    
    save(['Pullbacks_Mat_Files_Results\Predictions Cryo CRF Noise Cleaned Morph\' dirList(k).name '.mat'], 'pullback_crf_label' );
    save(['Pullbacks_Mat_Files_Results\Predictions Cryo CRF Noise Cleaned Morph\' dirList(k).name '_Actual_Labels.mat'], 'actual_label');
end

fprintf('Confusion Matrix \n');
disp(confusion_matrix);

disp('\n Metrics \n');
actual_full(actual_full == 0) = [];
pullback_crf_full(pullback_crf_full == 0) = [];

[cmatrix, result] = confusion.getMatrix(actual_full, pullback_crf_full);


confusion_matrix(1,:) = [];
confusion_matrix(:,1) = [];

confusion_matrix_percent = zeros(size(confusion_matrix));

for row = 1:size(confusion_matrix, 1)
    confusion_matrix_percent(row,:) = confusion_matrix(row,:)/sum(confusion_matrix(row,:));
end

fprintf('Confusion Matrix in Percentage \n');
disp(confusion_matrix_percent* 100);


%% Visualize XY images

for k = 1:numel(dirList)
        
    frame_numbers_this_pullback = frame_nums{k};
    raw_file_name_this_pullback = raw_file_names{k};

    addpath(genpath('C:\Users\Chaitanya\Documents\MATLAB\Helper functions\'));
    
    load(['Pullbacks_Mat_Files_Results\Predictions Cryo CRF Noise Cleaned Morph\' dirList(k).name]);
    predict_final = pullback_crf_label;

    % Load actual labels for this pullback
    load(['Pullbacks_Mat_Files_Results\Predictions Cryo CRF Noise Cleaned Morph\' dirList(k).name '_Actual_Labels.mat']);
    actual_final = actual_label;
    
    % Do noise cleaning, get final label
    predict_final_colors = zeros(size(predict_final,1),size(predict_final, 2), 3);
    actual_final_colors = zeros(size(actual_final,1), size(actual_final,2), 3);
    
    for row = 1:size(actual_final_colors, 1)
        for col = 1:size(actual_final_colors, 2)
            if actual_final(row,col) == 0
                actual_final_colors(row,col,:) = [0 0 0];
            end
            if actual_final(row,col) == 1
                actual_final_colors(row,col,:) = [1 0 0];
            end
            if actual_final(row,col) == 2
                actual_final_colors(row,col,:) = [0 1 0];
            end
            if actual_final(row,col) == 3
                actual_final_colors(row,col,:) = [0 0 1];
            end
            
            if predict_final(row,col) == 0
                predict_final_colors(row,col,:) = [0 0 0];
            end
            if predict_final(row,col) == 1
                predict_final_colors(row,col,:) = [1 0 0];
            end
            if predict_final(row,col) == 2
                predict_final_colors(row,col,:) = [0 1 0];
            end
            if predict_final(row,col) == 3
                predict_final_colors(row,col,:) = [0 0 1];
            end
        end
    end

    aline_counter = 0;

    for frame = 1:numel(frame_numbers_this_pullback)
        
        current_frame_num = frame_numbers_this_pullback(frame);
            
        % Print frame details
        fprintf('Current Pullback %s, Frame: %s \n', dirList(k).name , int2str(current_frame_num));

        im_raw = imread([folder dirList(k).name '\OCT Raw\' raw_file_name_this_pullback], current_frame_num);
        im_raw = log(double(im_raw) + 1.0);
        im_raw = (im_raw - min(im_raw(:)))/ (max(im_raw(:)) - min(im_raw(:)));  

        im_raw_cart = rectopolar_fast(im2double(im_raw)', 1024);
            
        im_raw_cart = repmat(im_raw_cart, 1, 1, 3);
        im_raw_cart = imrotate(im_raw_cart, 90);
        
        for row = 1:size(predict_final_colors, 1)
            for col = 1:size(predict_final_colors, 2)
                if sum(predict_final_colors(row,col,:)) == 0
                    predict_final_colors(row,col,:) = [1 1 1];
                end
                if sum(actual_final_colors(row,col,:)) == 0
                    actual_final_colors(row,col,:) = [1 1 1];
                end
            end
        end
        
        prediction = predict_final_colors(:, frame, :);            
        ground_truth = actual_final_colors(:, frame, :);
        
        elements_needed = floor(0.7*size(im_raw, 2)):floor(0.72*size(im_raw,2));
        prediction = repmat(prediction, 1, numel(elements_needed), 1);

        prediction_padded = zeros(size(im_raw, 1), size(im_raw, 2), 3);
        prediction_padded(:, elements_needed, :) = prediction;

        prediction_cart(:,:,1) = rectopolar_fast(im2double(prediction_padded(:,:,1)'), 1024);
        prediction_cart(:,:,2) = rectopolar_fast(im2double(prediction_padded(:,:,2)'), 1024);
        prediction_cart(:,:,3) = rectopolar_fast(im2double(prediction_padded(:,:,3)'), 1024);

        prediction_cart = imrotate(prediction_cart, 90);

        [row_cal, col_cal] = find(prediction_cart(:,:,1) ~= 0);
        [row_lip, col_lip] = find(prediction_cart(:,:,2) ~= 0);
        [row_nor, col_nor] = find(prediction_cart(:,:,3) ~= 0);

        [row_gw, col_gw]   = find(prediction_cart(:,:,1) ~= 0 & ...
                                  prediction_cart(:,:,2) ~= 0 & ...
                                  prediction_cart(:,:,3) ~= 0);

        elements_needed = floor(0.8*size(im_raw, 2)):floor(0.82*size(im_raw,2));
        ground_truth = repmat(ground_truth, 1, numel(elements_needed), 1);

        ground_truth_padded = zeros(size(im_raw, 1), size(im_raw, 2), 3);
        ground_truth_padded(:, elements_needed, :) = ground_truth;

        gt_cart = ones(1024);
        gt_cart(:,:,1) = rectopolar_fast(im2double(ground_truth_padded(:,:,1)'), 1024);
        gt_cart(:,:,2) = rectopolar_fast(im2double(ground_truth_padded(:,:,2)'), 1024);
        gt_cart(:,:,3) = rectopolar_fast(im2double(ground_truth_padded(:,:,3)'), 1024);

        gt_cart = imrotate(gt_cart, 90);

        [row_gt_cal, col_gt_cal] = find(gt_cart(:,:,1) ~= 0);
        [row_gt_lip, col_gt_lip] = find(gt_cart(:,:,2) ~= 0);
        [row_gt_nor, col_gt_nor] = find(gt_cart(:,:,3) ~= 0);


        [row_gt_gw, col_gt_gw]   = find(gt_cart(:,:,1) ~= 0 & ...
                                        gt_cart(:,:,2) ~= 0 & ...
                                        gt_cart(:,:,3) ~= 0);
%         % Predictions
%         for cal = 1:numel(row_cal)
%             im_raw_cart(row_cal(cal), col_cal(cal), :) = [1 0 0];        
%         end
% 
%         for lip =  1:numel(row_lip)
%             im_raw_cart(row_lip(lip), col_lip(lip), :) = [0 1 0];
%         end
% 
%         for nor = 1:numel(row_nor)
%             im_raw_cart(row_nor(nor), col_nor(nor), :) = [0 0 1];
%         end
% 
%         for gw = 1:numel(row_gw)
%             im_raw_cart(row_gw(gw), col_gw(gw),:) = [0 0 0];
%         end
        
        % Ground truth
        for cal = 1:numel(row_gt_cal)
            im_raw_cart(row_gt_cal(cal), col_gt_cal(cal), :) = [1 0 0];        
        end

        for lip =  1:numel(row_gt_lip)
            im_raw_cart(row_gt_lip(lip), col_gt_lip(lip), :) = [0 1 0];
        end

        for nor = 1:numel(row_gt_nor)
            im_raw_cart(row_gt_nor(nor), col_gt_nor(nor), :) = [0 0 1];
        end

        for gw = 1:numel(row_gt_gw)
            im_raw_cart(row_gt_gw(gw), col_gt_gw(gw), :) = [0 0 0];
        end

        %figure, imshow(im_raw_cart, []);
        imwrite(im_raw_cart, ['Pullbacks_Mat_Files_Results\Predictions Cryo XY Images + Labels\' dirList(k).name '_Frame_' int2str(current_frame_num) '.tif']);

    end
        
end

