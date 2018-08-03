% Variables
oct_raw_dir = 'C:\Users\Chaitanya\Documents\OCT deep learning dataset\Transform Data -- Raw Files\';
oct_raw_filename = 'TRF-09-0M-1-LAD-PRE';
oct_raw_filetype = '.oct';
num_of_frames = 540;
vessel_name = oct_raw_filename;

addpath(genpath('C:\Users\Chaitanya\Documents\MATLAB\Helper functions'));
sample_im = imread([oct_raw_dir oct_raw_filename oct_raw_filetype], 1);

ALine_Test_Matrix_Full_Pullback = zeros(size(sample_im,1) * num_of_frames, 200);

for k = 1:num_of_frames
    im = imread([oct_raw_dir oct_raw_filename oct_raw_filetype], k);

    fprintf('Processing frame %d \n', k);
    % Remove guidewire
    [guidewire_positions, im_gw_removed] = remove_guidewire_block(double(im));
    
    fprintf('Guidewire positions: %d %d \n', guidewire_positions(1), guidewire_positions(2));

    % Account for offset
    if guidewire_positions(1) < 3
        guidewire_positions(1) = 3;
    end
    if guidewire_positions(2) < 3
        guidewire_positions(2) = 3;
    end
    if guidewire_positions(1) > size(im,1)-3
        guidewire_positions(1) = size(im,1)-3;
    end
    if guidewire_positions(2) > size(im,1)-3
        guidewire_positions(2) = size(im,1)-3;
    end

    % Detect lumen
    lumenPixels = lumen_detection_block(double(im), guidewire_positions, false);

    % Pixel shift
    [im_pix_shift, ~] = pixel_shifting_block(im, lumenPixels, false);

    % Remove guidewire A-lines
    if guidewire_positions(2) - guidewire_positions(1) < 100
        im_pix_shift(max(1,guidewire_positions(1)-5):...
                     min(guidewire_positions(2)+5, size(im,1)),:) = 0;
        im_pix_shift = log(double(im_pix_shift) + 1.0);
        
        roi1 = im_pix_shift(1:max(1,guidewire_positions(1) -5), :);
        roi2 = im_pix_shift(min(guidewire_positions(2)+5, size(im,1)):size(im,1), :);
        
        roi1_filtered = imgaussfilt(roi1, 1, 'FilterSize', [7 7], 'Padding', 'symmetric');
        roi2_filtered = imgaussfilt(roi2, 1, 'FilterSize', [7 7], 'Padding', 'symmetric'); 
        
        im_pix_shift(1:max(1,guidewire_positions(1) -5), :) = roi1_filtered;
        im_pix_shift(min(guidewire_positions(2)+5, size(im,1)):size(im,1), :)  = roi2_filtered;       
        
    else
        im_pix_shift(1:guidewire_positions(1),:) = 0;
        im_pix_shift(guidewire_positions(2):size(im,1),:) = 0;
        
        im_pix_shift = log(double(im_pix_shift) + 1.0);
        
        roi1 = im_pix_shift(guidewire_positions(1):guidewire_positions(2), :);
        roi1_filtered = imgaussfilt(roi1, 1, 'FilterSize', [7 7], 'Padding', 'symmetric');
        
        im_pix_shift(guidewire_positions(1):guidewire_positions(2), :) = roi1_filtered;
    end  
        
    start_index = (k-1) *size(sample_im, 1) + 1;
    stop_index = k * size(sample_im, 1);
    
    im_pix_shift = exp(im_pix_shift) - 1.0;
    
    ALine_Test_Matrix_Full_Pullback(start_index:stop_index, 1:200) = uint16(im_pix_shift);
    Pullback_volume(:,:,k) = log(double(im_pix_shift) + 1.0);
    
    Guidewire_positions(k, :) = [guidewire_positions(1) guidewire_positions(2)];
    
end

save('ALine_Test_Matrix_Full_Pullback.mat', 'ALine_Test_Matrix_Full_Pullback');
save('Guidewire_positions.mat', 'Guidewire_positions');