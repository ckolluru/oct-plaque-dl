function [oct_pixel_shifted_volume, validALines] = pixel_shifting_block(oct_volume, lumenPixels, displayPixShift)

    % Pixel shifting
    pixels_to_shift = 200; % 1.0 mm 
    number_of_rows = size(oct_volume,1);
    number_of_cols = size(oct_volume,2);
    
    oct_pixel_shifted_volume = zeros(number_of_rows, pixels_to_shift, size(oct_volume,3));
    validALines = ones(size(oct_volume,1),1);
    
    for frameNum = 1:size(oct_volume,3)
        
        % Get back border for this frame
        backBorder = lumenPixels(:,frameNum) + pixels_to_shift;
        I = double(zeros(number_of_rows, pixels_to_shift));

        for rows= 1:number_of_rows
            
            if (backBorder(rows) <= number_of_cols) % just number_of_cols for rt direct
                backBorderPixel = min(backBorder(rows), number_of_cols);     
            
                aLineI = oct_volume(rows, lumenPixels(rows,frameNum):(backBorderPixel-1), frameNum);

                
                % If there actually seems to be something, then add
                if (max(aLineI(1:15)) >= 3.0) % log  3.5 for rt direct % 85 for rt
                    I(rows, 1:numel(aLineI)) = aLineI;
                    
                else
                    validALines(rows) = 0;
                end
            else
                validALines(rows) = 0;
            end            
            
        end
        
        oct_pixel_shifted_volume(:,:,frameNum) = I;    

        if displayPixShift
            figure, imshow(log(double(oct_pixel_shifted_volume(:,:,frameNum)+1.0)), []); %title('Pixel shifted image'), impixelinfo;
        end
    end

end