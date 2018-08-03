function fast_pixel_shift(oct_raw_filename, pullback_index) %, segments)
    
    tic;
    profile on;
    % Variables
    oct_raw_dir = 'C:\Users\Chaitanya\Documents\OCT deep learning dataset\Transform Data -- Raw Files\';
    oct_raw_filetype = '.oct';

    % Frame numbers
    load('Frame_Numbers.mat');
    frame_nums_this_pullback = frame_nums(pullback_index,:);
    
    % Delete empty cells
    frame_nums_this_pullback(cellfun('isempty', frame_nums_this_pullback)) = [];

    addpath(genpath('C:\Users\Chaitanya\Documents\MATLAB\Helper functions'));

    frame_counter = 0;
    
    im_pix_full = [];
    
    for segment = 1:size(frame_nums_this_pullback, 2)
        for frame = 1:size(frame_nums_this_pullback{segment}, 2)
            
            frame_counter = frame_counter + 1;
            current_frame_list = frame_nums_this_pullback{segment};
            current_frame_num = current_frame_list(frame);

            im(:,:,frame) = imread([oct_raw_dir oct_raw_filename oct_raw_filetype], current_frame_num);

            % Remove guidewire
            [guidewire_positions, im_gw_removed] = remove_guidewire_block(double(im(:,:,frame)));

            %fprintf('Guidewire positions: %d %d \n', guidewire_positions(1), guidewire_positions(2));

            % Account for offset
            if guidewire_positions(1) < 4
                guidewire_positions(1) = 4;
            end
            if guidewire_positions(2) < 4
                guidewire_positions(2) = 4;
            end
            if guidewire_positions(1) > size(im,1)-4
                guidewire_positions(1) = size(im,1)-4;
            end
            if guidewire_positions(2) > size(im,1)-4
                guidewire_positions(2) = size(im,1)-4;
            end
            
            GW_full(frame, :) = guidewire_positions';
            
        end        
        
        lumenPixels = lumen_detection_block(double(im), GW_full, false);
            
        frame_counter = 0;
        for frame = 1:size(frame_nums_this_pullback{segment}, 2)
            
            frame_counter = frame_counter + 1;
            current_frame_list = frame_nums_this_pullback{segment};
            current_frame_num = current_frame_list(frame);

            guidewire_positions = GW_full(frame,:);
            
            % Pixel shift
            [im_pix_shift, ~] = pixel_shifting_block(im(:,:,frame), lumenPixels(:, frame), false);
            
            % Remove guidewire A-lines
            if guidewire_positions(2) - guidewire_positions(1) < 150
                im_pix_shift(max(1,guidewire_positions(1)-5):...
                             min(guidewire_positions(2)+5, size(im,1)),:) = 0;
                im_pix_shift = log(double(im_pix_shift) + 1.0);

                roi1 = im_pix_shift(1:max(1,guidewire_positions(1)-5), :);
                roi2 = im_pix_shift(min(guidewire_positions(2)+5, size(im,1)):size(im,1), :);

                roi1_filtered = imgaussfilt(roi1, 1, 'FilterSize', [7 7], 'Padding', 'symmetric');
                roi2_filtered = imgaussfilt(roi2, 1, 'FilterSize', [7 7], 'Padding', 'symmetric'); 

                im_pix_shift(1:max(1,guidewire_positions(1)-5), :) = roi1_filtered;
                im_pix_shift(min(guidewire_positions(2)+5, size(im,1)):size(im,1), :)  = roi2_filtered;
                
            else
                im_pix_shift(1:max(1, guidewire_positions(1)-5),:) = 0;
                im_pix_shift(min(guidewire_positions(2)+5, size(im,1)):size(im,1),:) = 0;

                im_pix_shift = log(double(im_pix_shift) + 1.0);

                roi1 = im_pix_shift(max(1,guidewire_positions(1)-5):min(guidewire_positions(2)+5, size(im_pix_shift,1)), :);
                roi1_filtered = imgaussfilt(roi1, 1, 'FilterSize', [7 7], 'Padding', 'symmetric');

                im_pix_shift(max(1,guidewire_positions(1)-5):min(guidewire_positions(2)+5,size(im_pix_shift,1)), :) = roi1_filtered;
                
            end
            
            im_pix_shift = exp(im_pix_shift) - 1.0;

            Guidewire_Positions(frame_counter,:) = guidewire_positions;
            
            imwrite(uint16(im_pix_shift), ['Converted Labels\Faster Shifting\' oct_raw_filename '_' int2str(current_frame_num) '_Pixel_Shifted.tif']);
            
            %im_pix_full = [im_pix_full; im_pix_shift];
            
            %fprintf('One frame took %f seconds \n', toc);
        end
        
        fprintf('Total: %f seconds/image', toc/size(frame_nums_this_pullback{segment}, 2));
        
        profile viewer;
    end

    %save(['Converted Labels\Guidewire_positions\' oct_raw_filename '_Guidewire.mat'], 'Guidewire_Positions');
end