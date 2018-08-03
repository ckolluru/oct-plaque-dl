directory = 'Z:\Annotated Transform IVOCT Database\';

load('Frame_Numbers.mat');
frame_numbers = frame_nums(pullbackIndex,:);
    
load('Pullback_Folder.mat');
load('PullbackStrings.mat');

for k = 1:numel(pullback_folder)
    dirInfo = dir([directory pullback_folder '\OCT Labels\*.tif']);
    
    for col = 1:6
        frame_numbers = [frame_nums frame_nums{k}];
    end    
    
    if numel(frame_numbers) ~= numel(dirInfo)
        fprintf('Something is wrong \n');
    end
    
end

% Calcium = 4
% Lipid = 3

for k = 1:numel(frame_numbers)
    filename = dirInfo(k).name;
    current_frame = frame_numbers(k);
    
    label_xy = imread([directory pullback_folder '\OCT Labels\' dirInfo(k).name]);
    rt_image = imread(['C:\Users\Chaitanya\Documents\OCT deep learning dataset\Transform Data -- Raw Files\' pullback_string '.oct'], 1);
    label_xy = imrotate(label_xy, -90);
    
    label_rt = polartorect_fast(double(label_xy), size(rt_image, 2), size(rt_image, 1));
    label_rt = label_rt';
    label_alines = zeros(size(rt_image, 1), 1);    
    
    for rows = 1:size(label_rt, 1)
        if ~isempty(find(label_rt(rows,:) == 3))
            label_alines(rows, 1) = 2;
        elseif ~isempty(find(label_rt(rows,:) == 4))
            label_alines(rows,1) = 1;
        else
            label_alines(rows,1) = 0;
        end
    end
    
    imwrite(uint8(label_alines), ['C:\Users\Chaitanya\Documents\MATLAB\Converted Labels\Images And Labels for Machine Learning\Training\' pullback_string '_' num2str(current_frame) '_Labels.tif']);
    
end
