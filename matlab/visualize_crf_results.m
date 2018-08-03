gt_dir = dir('Pullbacks_Mat_Files_Results\Ground truth En Face View\*.mat');
pred_dir = dir('Pullbacks_Mat_Files_Results\Predictions En Face View\*.mat');
crf_dir = dir('Pullbacks_Mat_Files_Results\Predictions_CRF_Noise_Cleaned\*.mat');

for k = 1:numel(gt_dir)
    
    load(['Pullbacks_Mat_Files_Results\Ground truth En Face View\' gt_dir(k).name]);
    load(['Pullbacks_Mat_Files_Results\Predictions En Face View\' pred_dir(k).name]);
    load(['Pullbacks_Mat_Files_Results\Predictions_CRF_Noise_Cleaned\' crf_dir(k).name]);
    
    CRF_Color = zeros(size(CRF_Results, 1), size(CRF_Results, 2), 3);
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
    
    figure;
    subplot(1,3,1), imshow(actual_reshape, []);
    subplot(1,3,2), imshow(predict_reshape, []);
    subplot(1,3,3), imshow(CRF_Color, []);
    
end