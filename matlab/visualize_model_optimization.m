close all; clear; clc;

%%
current_fold = 6;
load('Ten_Fold_All_Alines\Fold6_Test.mat');

%% Calcium
load('Pullbacks_Mat_Files_Results\Saliency Maps\Original_Image_Calcium_TRF_04_300.mat');
load('Pullbacks_Mat_Files_Results\Saliency Maps\Saliency_Map_Calcium_TRF_04_300.mat');
load('Pullbacks_Mat_Files_Results\Saliency Maps\Lumen_Pixels_Calcium_TRF_04_300.mat');
oct_rt = imread('C:\Users\Chaitanya\Documents\OCT deep learning dataset\Transform Data -- Raw Files\TRF-04-0M-1-RCA-PRE.oct', 300);
image_name = 'Calcium_TRF_04_300';
gw_pos = [376, 406];
start_aline = (28+144+14+242-97+1+300-281)* 496;
stop_aline = start_aline + 496;
pullback_number = 4;
start_aline_in_pullback = 165*496 + 1;
stop_aline_in_pullback = 166*496;

%% Lipid
% load('Pullbacks_Mat_Files_Results\Saliency Maps\Original_Image_Lipid_TRF_23_Prox_301.mat');
% load('Pullbacks_Mat_Files_Results\Saliency Maps\Saliency_Map_Lipid_TRF_23_Prox_301.mat');
% load('Pullbacks_Mat_Files_Results\Saliency Maps\Lumen_Pixels_Lipid_TRF_23_Prox_301.mat');
% oct_rt = imread('C:\Users\Chaitanya\Documents\OCT deep learning dataset\Transform Data -- Raw Files\TRF-23-0M-2-LAD-PRE-Prox.oct', 301);
% image_name = 'Lipid_TRF_23_Prox_301';
% % 
% load('Pullbacks_Mat_Files_Results\Saliency Maps\Original_Image_Lipid_TRF_26_Dist_159.mat');
% load('Pullbacks_Mat_Files_Results\Saliency Maps\Saliency_Map_Lipid_TRF_26_Dist_159.mat');
% load('Pullbacks_Mat_Files_Results\Saliency Maps\Lumen_Pixels_Lipid_TRF_26_Dist_159.mat');
% oct_rt = imread('C:\Users\Chaitanya\Documents\OCT deep learning dataset\Transform Data -- Raw Files\TRF-26-0M-1-LAD-PRE-Dist.oct', 159);
% image_name = 'Lipid_TRF_26_Dist_159';
% gw_pos = [307, 353];
% start_aline = (28+144+14+304+53)* 496;
% stop_aline = start_aline + 496;
% pullback_number = 20;
% start_aline_in_pullback = (496*54) + 1;
% stop_aline_in_pullback = 496*55;

%% Other
% load('Pullbacks_Mat_Files_Results\Saliency Maps\Original_Image_Other_TRF_26_Dist_107.mat');
% load('Pullbacks_Mat_Files_Results\Saliency Maps\Saliency_Map_Other_TRF_26_Dist_107.mat');
% load('Pullbacks_Mat_Files_Results\Saliency Maps\Lumen_Pixels_Other_TRF_26_Dist_107.mat');
% oct_rt = imread('C:\Users\Chaitanya\Documents\OCT deep learning dataset\Transform Data -- Raw Files\TRF-26-0M-1-LAD-PRE-Dist.oct', 107);
% image_name = 'Other_TRF_26_Dist_107';

%%
OCT_image = zeros(496, 976);
Saliency_map = zeros(496, 976);
ground_truth_from_matrix = ALine_Label_Test_Matrix(start_aline:stop_aline-1,201:203);
ground_truth = zeros(size(ground_truth_from_matrix, 1), 1, size(ground_truth_from_matrix,2));
load(['Pullbacks_Mat_Files_Results\Final Predictions In-vivo Pullbacks\Pullback_' int2str(pullback_number) '.mat']);
predicted_label = predict_reshape_again(start_aline_in_pullback:stop_aline_in_pullback);
prediction = zeros(size(predicted_label,1), 1, 3);

for row = 1:size(ground_truth_from_matrix, 1)
    ground_truth(row,1,:) = ground_truth_from_matrix(row,:);
end

for row = 1:size(predicted_label,1)
    if predicted_label(row) == 1
        prediction(row,:,:) = [1 0 0];
    elseif predicted_label(row) == 2
        prediction(row,:,:) = [0 1 0];
    elseif predicted_label(row) == 3
        prediction(row,:,:) = [0 0 1];
    else
        prediction(row,:,:) = [0 0 0];
    end
end

if gw_pos(2) - gw_pos(1) < 100
    prediction(max(1,gw_pos(1)):...
                 min(gw_pos(2), size(prediction,1)),1,:) = ones(numel(max(1,gw_pos(1)):...
                                                                        min(gw_pos(2), size(prediction,1))), 1, 3);
else
    prediction(1:gw_pos(1),1,:) = ones(numel(1:gw_pos(1)), 1, 3);
    prediction(gw_pos(2):end,1,:) = ones(numel(gw_pos(2):size(prediction, 1)), 1, 3);
end

if gw_pos(2) - gw_pos(1) < 100
    ground_truth(max(1,gw_pos(1)):...
                 min(gw_pos(2), size(ground_truth,1)),1,:) = ones(numel(max(1,gw_pos(1)):...
                                                                          min(gw_pos(2), size(ground_truth,1))), 1, 3);
else
    ground_truth(1:gw_pos(1),1,:) = ones(numel(1:gw_pos(1)), 1, 3);
    ground_truth(gw_pos(2):end,1,:) = ones(numel(gw_pos(2):size(ground_truth, 1)), 1, 3);
end
            
for row = 1:496
    OCT_image(row, lumenPixels(row):lumenPixels(row) + 199) = ALines_Prediction(row, 6:205);
    Saliency_map(row, lumenPixels(row):lumenPixels(row) + 199) = Saliency_Map(row, 6:205);
end

Saliency_map(gw_pos(1):gw_pos(2), :) = 0;

figure, imshow(OCT_image, []), title('OCT image RT');
figure, imshow(Saliency_map, []), title('Saliency map RT');

% Convert to XY
oct_xy = rectopolar_fast(log(double(oct_rt + 1.0))', 1024);
oct_xy = imrotate(oct_xy, 90);

sal_map_xy(:,:) = rectopolar_fast(im2double(Saliency_map(:,:))', 1024);
sal_map_xy = imrotate(sal_map_xy, 90);

figure;
imshow(oct_xy, []);
F = getframe(gcf);
[X, Map] = frame2im(F);

imwrite(oct_xy, ['Pullbacks_Mat_Files_Results\Saliency Maps\' image_name '_Original_Image.png']);

figure;
imshow(oct_xy, []);
green = cat(3, ones(size(oct_xy)), zeros(size(oct_xy)), zeros(size(oct_xy))); 
hold on 
h = imshow(green); 
hold off 
set(h, 'AlphaData', sal_map_xy) 

F = getframe(gcf);
[X, Map] = frame2im(F);

imwrite(X, ['Pullbacks_Mat_Files_Results\Saliency Maps\' image_name '_Saliency_Map.png']);


elements_needed = floor(0.7*size(oct_rt, 2)):floor(0.72*size(oct_rt,2));
prediction = repmat(prediction, 1, numel(elements_needed), 1);

prediction_padded = zeros(size(oct_rt, 1), size(oct_rt, 2), 3);
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

elements_needed = floor(0.8*size(oct_rt, 2)):floor(0.82*size(oct_rt,2));
ground_truth = repmat(ground_truth, 1, numel(elements_needed), 1);

ground_truth_padded = zeros(size(oct_rt, 1), size(oct_rt, 2), 3);
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
                            
oct_rt = log(double(oct_rt) + 1.0);                            
oct_rt = (oct_rt - min(oct_rt(:)))/ (max(oct_rt(:)) - min(oct_rt(:)));  

oct_xy = rectopolar_fast(im2double(oct_rt)', 1024);

oct_xy = repmat(oct_xy, 1, 1, 3);
oct_xy = imrotate(oct_xy, 90);

% Predictions
for cal = 1:numel(row_cal)
    oct_xy(row_cal(cal), col_cal(cal), :) = [1 0 0];        
end

for lip =  1:numel(row_lip)
    oct_xy(row_lip(lip), col_lip(lip), :) = [0 1 0];
end

for nor = 1:numel(row_nor)
    oct_xy(row_nor(nor), col_nor(nor), :) = [0 0 1];
end

for gw = 1:numel(row_gw)
    oct_xy(row_gw(gw), col_gw(gw),:) = [0 0 0];
end



% Ground truth
for cal = 1:numel(row_gt_cal)
    oct_xy(row_gt_cal(cal), col_gt_cal(cal), :) = [1 0 0];        
end

for lip =  1:numel(row_gt_lip)
    oct_xy(row_gt_lip(lip), col_gt_lip(lip), :) = [0 1 0];
end

for nor = 1:numel(row_gt_nor)
    oct_xy(row_gt_nor(nor), col_gt_nor(nor), :) = [0 0 1];
end

for gw = 1:numel(row_gt_gw)
    oct_xy(row_gt_gw(gw), col_gt_gw(gw), :) = [0 0 0];
end

figure, imshow(oct_xy, []);
green = cat(3, ones(size(oct_xy,1), size(oct_xy,2)), zeros(size(oct_xy,1), size(oct_xy,2)), zeros(size(oct_xy,1), size(oct_xy,2))); 
hold on 
h = imshow(green); 
hold off 
set(h, 'AlphaData', sal_map_xy) 
F = getframe(gcf);
[X, Map] = frame2im(F);

imwrite(X, ['Pullbacks_Mat_Files_Results\Saliency Maps\' image_name '_Saliency_Map_With_Ring.png'])