% Comparing whether the CNN or ANN is better for the in vivo dataset
% Fibrocalcific: conf_interval: [0.0219, 0.0455]
% Fibrolipidic: conf_interval: [0.000372, 0.0277]
% Other: conf_interval: [0.0305, 0.0438]
% ANN always has a higher error rate, CNN is better

% Considering that they are different learning algorithms
load('Pullbacks_Mat_Files_Results\Confusion_Matrix_KFold_ANN.mat');
CM_ANN = Confusion_Matrix_All_Folds;

load('Pullbacks_Mat_Files_Results\Confusion_Matrix_KFold_CNN.mat');
CM_CNN = Confusion_Matrix_All_Folds;

for fold = 1:10
    
    cm_ann = CM_ANN(2:end, 2:end, fold);
    cm_cnn = CM_CNN(2:end, 2:end, fold);
    
    error_rate_ann(fold) = cm_ann(3,1) + cm_ann(3,2) + cm_ann(1,3) + cm_ann(2,3);
    error_rate_ann(fold) = error_rate_ann(fold)/sum(cm_ann(:));
    
    error_rate_cnn(fold) = cm_cnn(3,1) + cm_cnn(3,2) + cm_cnn(1,3) + cm_cnn(2,3);
    error_rate_cnn(fold) = error_rate_cnn(fold)/sum(cm_cnn(:));
    
end
    
mean_err_delta = sum(error_rate_ann - error_rate_cnn)/10;

std_dev = 0;

for fold = 1:10
    delta_i = error_rate_ann(fold) - error_rate_cnn(fold);
    
    std_dev = std_dev + (delta_i - mean_err_delta)* (delta_i - mean_err_delta);

end

std_dev = std_dev /(10*9);
std_dev = sqrt(std_dev);

conf_interval = [mean_err_delta - (2.262* std_dev), mean_err_delta + (2.262* std_dev)];

cm_ann_full = zeros(3,3);
cm_cnn_full = zeros(3,3);

% Considering that they are different classifiers
% Fibrocalcific: [0.0296, 0.0307]
% Fibrolipidic: [0.0107, 0.0118]
% Other: [0.0361, 0.0373]

for fold = 1:10
    cm_ann_full = cm_ann_full + CM_ANN(2:4,2:4,fold);
    cm_cnn_full = cm_cnn_full + CM_CNN(2:4, 2:4, fold);
end

error_rate_calc_ann = cm_ann_full(1,3) + cm_ann_full(2,3) + cm_ann_full(3,1) + cm_ann_full(3,2);
error_rate_calc_ann = error_rate_calc_ann/sum(cm_ann_full(:));

error_rate_calc_cnn = cm_cnn_full(1,3) + cm_cnn_full(2,3) + cm_cnn_full(3,1) + cm_cnn_full(3,2);
error_rate_calc_cnn = error_rate_calc_cnn/sum(cm_cnn_full(:));

F = error_rate_calc_ann - error_rate_calc_cnn;
V = 0;
V = V + (error_rate_calc_ann*(1-error_rate_calc_ann)/sum(cm_ann_full(:)));

conf_interval_2 = [F - (1.96* sqrt(V)), F + (1.96* sqrt(V))];

% Considering CNN and CNN_strip
% Fibrocalcific: [-0.0011, -3.611e-4]
% Fibrolipidic: [-5.76e-5, 7.79e-4]
% Other: [-0.0104, -0.0095]

cm_cnn_full = [262548, 30573, 36604; 19740, 363593, 41432; 80148, 114219, 1066948];
% This is actually strip
cm_ann_full = [240671, 32385, 56669; 27591, 343495, 53679; 48928, 93037, 1119350];

error_rate_calc_ann = cm_ann_full(1,2) + cm_ann_full(2,1) + cm_ann_full(2,3) + cm_ann_full(3,2);
error_rate_calc_ann = error_rate_calc_ann/sum(cm_ann_full(:));

error_rate_calc_cnn = cm_cnn_full(1,2) + cm_cnn_full(2,1) + cm_cnn_full(2,3) + cm_cnn_full(3,2);
error_rate_calc_cnn = error_rate_calc_cnn/sum(cm_cnn_full(:));

F = error_rate_calc_ann - error_rate_calc_cnn;
V = 0;
V = V + (error_rate_calc_ann*(1-error_rate_calc_ann)/sum(cm_ann_full(:)));

conf_interval_3 = [F - (1.96* sqrt(V)), F + (1.96* sqrt(V))];

