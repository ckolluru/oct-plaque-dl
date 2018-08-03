load('C:\Users\Chaitanya\Desktop\alines_validation_prediction.mat');
load('C:\Users\Chaitanya\Documents\MATLAB\Pullbacks_Mat_Files_All_Alines\ML datasets\Validation_for_ML.mat');

frames = [51, 28, 51, 49, 19, 32, 32, 21, 26];
act_full = [];
pred_full = [];

for k = 1:numel(frames);
    
    if k == 1
        start_aline = 1;
        stop_aline = frames(k)*496;
    else
        start_aline = (sum(frames(1:k-1)))*496 + 1;
        stop_aline = start_aline + (frames(k))*496 - 1;
    end
    
    predictions = alines_validation_prediction(start_aline:stop_aline, :);
    predictions_reshape = reshape(predictions, 496, size(predictions, 1)/496, 3);
    
    actual = ALine_Label_Matrix(start_aline:stop_aline, 201:203);
    actual_reshape = reshape(actual, 496, size(actual, 1)/496, 3);

    filter_size = [21 5];
    padding_size = floor(filter_size/2);
    predict_reshape_pad = zeros(size(predictions_reshape, 1) + filter_size(1)-1, size(predictions_reshape, 2) + filter_size(2) - 1, 3);

    predict_reshape_pad(:,:,1) = padarray(predictions_reshape(:,:,1), [padding_size(1) padding_size(2)], 'symmetric');
    predict_reshape_pad(:,:,1) = colfilt(predict_reshape_pad(:,:,1), filter_size, 'sliding', @median);
    predictions_reshape(:, :,1) = predict_reshape_pad(padding_size(1) + 1:end-padding_size(1), padding_size(2) + 1:end-padding_size(2),1);

    predict_reshape_pad(:,:,2) = padarray(predictions_reshape(:,:,2), [padding_size(1) padding_size(2)], 'symmetric');
    predict_reshape_pad(:,:,2) = colfilt(predict_reshape_pad(:,:,2), filter_size, 'sliding', @median);
    predictions_reshape(:, :,2) = predict_reshape_pad(padding_size(1) + 1:end-padding_size(1), padding_size(2) + 1:end-padding_size(2),2);

    predict_reshape_pad(:,:,3) = padarray(predictions_reshape(:,:,3), [padding_size(1) padding_size(2)], 'symmetric');
    predict_reshape_pad(:,:,3) = colfilt(predict_reshape_pad(:,:,3), filter_size, 'sliding', @median);
    predictions_reshape(:, :,3) = predict_reshape_pad(padding_size(1) + 1:end-padding_size(1), padding_size(2) + 1:end-padding_size(2),3);
    
    figure;
    subplot(1,3,1), imshow(actual_reshape, []);
    subplot(1,3,2), imshow(predictions_reshape, []);
    
    predictions_reshape_new = zeros(size(predictions_reshape, 1), size(predictions_reshape, 2), 3);
   
    for row = 1:size(predictions_reshape, 1)
        for col = 1:size(predictions_reshape, 2)
            predictions_reshape_new(row,col,:) = [0 0 1];
        end
    end
    
    
    for row = 1:size(predictions_reshape, 1)
        for col = 1:size(predictions_reshape, 2)
            if predictions_reshape(row,col,1) > predictions_reshape(row,col,2)
                if predictions_reshape(row,col,1) > predictions_reshape(row,col,3)
                        predictions_reshape_new(row,col,:) = [1 0 0];
                end
            end
            
            if predictions_reshape(row,col,2) > predictions_reshape(row,col,1)
                if predictions_reshape(row,col,2) > predictions_reshape(row,col,3)
                    predictions_reshape_new(row,col,:) = [0 1 0];
                end
            end
            
            if predictions_reshape(row,col,3) > predictions_reshape(row,col,2)
                if predictions_reshape(row,col,3) > predictions_reshape(row,col,1)
                    predictions_reshape_new(row,col,:) = [0 0 1];
                end
            end            
        end
    end
    
    subplot(1,3,3), imshow(predictions_reshape_new, []);
    
    for row = 1:size(predictions_reshape_new, 1)
        for col = 1:size(predictions_reshape_new, 2)
            if predictions_reshape_new(row,col,1) == 1
                pred(row,col) = 1;
            elseif predictions_reshape_new(row,col,2) == 1
                pred(row,col) = 2;
            else
                pred(row,col) = 3;
            end
        end
    end
    
    for row = 1:size(actual_reshape, 1)
        for col = 1:size(actual_reshape, 2)
            if actual_reshape(row,col,1) == 1
                act(row,col) = 1;
            elseif actual_reshape(row,col,2) == 1
                act(row,col) = 2;
            else
                act(row,col) = 3;
            end
        end
    end
    
    act_full = [act_full; act(:)];
    pred_full = [pred_full; pred(:)];
end

C = confusionmat(act_full, pred_full);
disp(C);

for row = 1:size(C,1)
    C(row,:) = C(row,:)/sum(C(row,:));
end

disp(C*100);
