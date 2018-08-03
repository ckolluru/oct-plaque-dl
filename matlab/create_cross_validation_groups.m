close all; clear; clc;

pullbacks = dir('C:\Users\Chaitanya\Documents\MATLAB\Pullbacks_Mat_Files_All_Alines\TRF-*.mat');

load('PullbackStrings.mat');

num_alines = [46128, 115072, 105648, 150784, 93248, 71424, 11904, 92256,  15376, 25296, 135904, 11904,...
              24304, 30752, 8928, 14384, 13888, 31248, 109120, 50096, 4960, 157232, 65472, 41168, 74896,...
              65968, 29760, 71424, 29120, 5824, 27280, 81840, 25792,  21328, 9920, 71920, 24800, 40176,...
              5456, 28272, 30752, 16864, 33728, 13392, 7936, 6944, 46624, 16368];
          
num_pullbacks = numel(pullbacks);

num_folds = 10;

pullbacks_per_fold = ceil(num_pullbacks/num_folds);

pullbacks_in_last_fold = num_pullbacks - ((num_folds - 1) * pullbacks_per_fold) ;

pullback_indices = 1:num_pullbacks;

training_folds_full = zeros(10, 40);

for k = 1:num_folds
    
    rng(10+k, 'twister');
    
    if k~= num_folds
        fold_indices{k} = datasample(pullback_indices, 5, 'Replace', false);

        pullback_indices = setdiff(pullback_indices, fold_indices{k});
    else
        fold_indices{k} = pullback_indices;
    end
    
end

for k = 1:num_folds
    
    test_fold_indices = fold_indices{k};
    
    train_fold_indices = [];
    
    for j = 1:num_folds
        if j ~= k
            train_fold_indices = [train_fold_indices fold_indices{j}];
        end
    end
    
    rng(10+k, 'twister');
    rand_perm = datasample(1:numel(train_fold_indices), 5, 'Replace', false);
    
    validation_fold_indices = train_fold_indices(rand_perm);
    train_fold_indices = setdiff(train_fold_indices, validation_fold_indices);
    
    ALine_Label_Training_Matrix = zeros(sum(num_alines(train_fold_indices)), 203);
    ALine_Label_Validation_Matrix = zeros(sum(num_alines(validation_fold_indices)), 203);
    ALine_Label_Test_Matrix = zeros(sum(num_alines(test_fold_indices)), 203);
    
    fprintf('Fold %d \n', k);
    
    disp('Training Indices');    
    disp(train_fold_indices);    
    
    disp('Test Indices');    
    disp(test_fold_indices);
    
    if numel(test_fold_indices) == 5
        test_folds_full(k,:) = test_fold_indices;
    else
        test_folds_full(k,:) = [test_fold_indices 0 0];
    end
    
    if numel(test_fold_indices) == 5
        training_folds_full(k,1:38) = train_fold_indices;
    else
        training_folds_full(k,1:40) = train_fold_indices;
    end
    
    validation_folds_full(k,:) = validation_fold_indices;
    
    fprintf('Validation Fold Indices');
    disp(validation_fold_indices);
    
    % Make sure that none of these sets intersect
    num_intersect_1 = numel(intersect(train_fold_indices, test_fold_indices));
    num_intersect_2 = numel(intersect(train_fold_indices, validation_fold_indices));
    num_intersect_3 = numel(intersect(validation_fold_indices, test_fold_indices));
    
    fprintf('Number of intersections: %d \n', num_intersect_1 + num_intersect_2 + num_intersect_3);
    
    fprintf('Fold %d \n', k);
    
    train_aline_counter = 0;
    
    for i = train_fold_indices        
        load(['Pullbacks_Mat_Files_All_Alines\' pullbacks(i).name]);
        ALine_Label_Training_Matrix(train_aline_counter+1:train_aline_counter+size(ALine_Label_Matrix,1),:) = ALine_Label_Matrix;        
        train_aline_counter = train_aline_counter + size(ALine_Label_Matrix, 1);
    end
    
    validation_aline_counter = 0;
    
    for i = validation_fold_indices        
        load(['Pullbacks_Mat_Files_All_Alines\' pullbacks(i).name]);
        ALine_Label_Validation_Matrix(validation_aline_counter+1:validation_aline_counter+size(ALine_Label_Matrix,1),:) = ALine_Label_Matrix;        
        validation_aline_counter = validation_aline_counter + size(ALine_Label_Matrix, 1);
    end
    
    test_aline_counter = 0;
    
    for i = test_fold_indices        
        load(['Pullbacks_Mat_Files_All_Alines\' pullbacks(i).name]);
        ALine_Label_Test_Matrix(test_aline_counter+1:test_aline_counter+size(ALine_Label_Matrix, 1),:) = ALine_Label_Matrix;
        test_aline_counter = test_aline_counter + size(ALine_Label_Matrix, 1);
    end
    
    save(['Ten_Fold_All_Alines\Fold' int2str(k) '_Train.mat'], 'ALine_Label_Training_Matrix', '-v7.3');
    save(['Ten_Fold_All_Alines\Fold' int2str(k) '_Validation.mat'], 'ALine_Label_Validation_Matrix');
    save(['Ten_Fold_All_Alines\Fold' int2str(k) '_Test.mat'], 'ALine_Label_Test_Matrix');
    
end

save('Test_Pullback_Indices_In_Each_Fold.mat', 'test_folds_full');
save('Training_Pullback_Indices_In_Each_Fold.mat', 'training_folds_full');
save('Validation_Pullback_Indices_In_Each_Fold.mat', 'validation_folds_full');

