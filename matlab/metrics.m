function metrics (confusion_matrix)

    for col = 1:3
        recall(col) = confusion_matrix(col,col)/sum(confusion_matrix(:,col));
    end
    
    for row = 1:3
        precision(row) = confusion_matrix(row,row)/sum(confusion_matrix(row,:));
    end
    
    for k = 1:3
        f1(k) = 2* precision(k) * recall (k) / (precision(k) + recall(k));
        fprintf('F1 score of class %d: %f \n', k, f1(k));
    end 
    
    fprintf('F1 score of all classes combined: %f \n', sum(f1));
    
end

