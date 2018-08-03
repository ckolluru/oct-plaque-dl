function find_alines_set_to_zero

    dirList = dir('C:\Users\Chaitanya\Documents\MATLAB\Pullbacks_Mat_Files_All_Alines\TRF*.mat');
    load('ALines_In_Frame.mat');
    
    for k = 1:numel(dirList)
        load(['C:\Users\Chaitanya\Documents\MATLAB\Pullbacks_Mat_Files_All_Alines\' dirList(k).name]);
        alines_zero = zeros(alines_in_frame(k), size(ALine_Label_Matrix, 1)/alines_in_frame(k));
        count = 1;
        
        for frame = 1:size(alines_zero, 2)
            for i = 1:alines_in_frame(k)
                if sum(ALine_Label_Matrix(count,1:200)) == 0                
                    alines_zero(i, frame) = 1;
                end
                count = count + 1;
            end
        end
        
        figure, imshow(alines_zero, []);
        save(['Pullbacks_Mat_Files_All_Alines\ALines Zero\' dirList(k).name], 'alines_zero');
    end
end
