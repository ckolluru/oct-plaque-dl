function lumenPixels = lumen_detection_block(oct_volume, guidewire_positions, lumenDisplay)

    % Lumen Detection 
    catheterPos = 80;
    lumen_connectivity_rect = 35;
    bandwidth = 60;                    
        
    %lumen_connectivity_rect = 200;
    %bandwidth = 60; 
    
    for frameNum = 1:size(oct_volume,3)
        
        lumenBorderFrame = lumen_segmentation_DP(oct_volume(:,:,frameNum),...
                           catheterPos, lumen_connectivity_rect, bandwidth);                

        lb = zeros(size(lumenBorderFrame,1),1);

        for row=1:size(lumenBorderFrame, 1)
             rInd = find(lumenBorderFrame(row,:), 1, 'first');
             lb(row,1) = rInd;
        end
        
        lumenPixels(:,frameNum) = lb;
        lumenPixels(:,frameNum) = correctForGuidewire(oct_volume(:,:,frameNum), guidewire_positions(frameNum,:), lumenPixels(:,frameNum), 3);
        
        if lumenDisplay 
            figure('units','normalized','outerposition',[0 0 1 1])
            imshow(log(oct_volume(:,:,frameNum)+1.0), []); hold on;      
            y_lumen = 1:numel(lumenPixels(:,frameNum)); 
            plot(lumenPixels(:,frameNum), y_lumen, 'r-', 'LineWidth', 1);
            %title('Lumen border overlay');
            hold off;
            %impixelinfo;
        end
    
    end    

end