close all; clear; clc;

pullbacks = dir('C:\Users\Chaitanya\Documents\MATLAB\Pullbacks_Mat_Files_All_Alines\ML datasets\All Training\TRF-*.mat');

load('PullbackStrings.mat');

aline_depth = 200;

num_alines = [115072, 105648, 150784, 93248, 71424, 11904, 15376, 25296, 135904, 11904,...
              30752, 8928, 14384, 13888, 31248, 109120, 50096, 4960, 157232, 65472, 41168, 74896,...
              65968, 29760, 29120, 5824, 27280, 81840,  21328, 9920, 71920, 40176,...
              5456, 30752, 33728, 13392, 7936, 6944, 46624, 16368];

num_pullbacks = numel(pullbacks);

ALine_Label_Training_Matrix = zeros(sum(num_alines), aline_depth + 3);

train_aline_counter = 0;

for k = 1:40
    
    fprintf('Loading dataset %d \n', k);
    load(['Pullbacks_Mat_Files_All_Alines\' pullbacks(k).name]);
    fprintf('Loaded dataset \n');
    
    fprintf('Making matrix \n');
    ALine_Label_Training_Matrix(train_aline_counter+1:train_aline_counter+size(ALine_Label_Matrix,1),:) = ALine_Label_Matrix;        
    train_aline_counter = train_aline_counter + size(ALine_Label_Matrix, 1);
   
end

ALine_Label_Matrix = ALine_Label_Training_Matrix;
save('Training_for_ML_Full_Dataset.mat', 'ALine_Label_Matrix', '-v7.3');
