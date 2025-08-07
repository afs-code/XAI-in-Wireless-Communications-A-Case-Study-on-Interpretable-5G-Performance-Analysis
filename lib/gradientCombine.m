function [gradientTotal, averageGradTotal, averageSqGradTotal] = gradientCombine(gradientFirst, averageGradFirst, averageSqGradFirst, gradientSecond,averageGradSecond, averageSqGradSecond)
    
    gradientTotal = struct;
    averageGradTotal = struct;
    averageSqGradTotal = struct;

    fieldsAverageGrad = fieldnames(gradientFirst);
    for i = 1:numel(fieldsAverageGrad)
        gradientTotal.(['subnet' num2str(i)]) = gradientFirst.(fieldsAverageGrad{i});
        averageGradTotal.(['subnet' num2str(i)]) = averageGradFirst.(fieldsAverageGrad{i});
        averageSqGradTotal.(['subnet' num2str(i)]) = averageSqGradFirst.(fieldsAverageGrad{i});
    end

    fieldsAverageGradInt = fieldnames(gradientSecond);
    for i = 1:numel(fieldsAverageGradInt)
        gradientTotal.(['subnet' num2str(i + numel(fieldsAverageGrad))]) = gradientSecond.(fieldsAverageGradInt{i});
        averageGradTotal.(['subnet' num2str(i + numel(fieldsAverageGrad))]) = averageGradSecond.(fieldsAverageGradInt{i});
        averageSqGradTotal.(['subnet' num2str(i + numel(fieldsAverageGrad))]) = averageSqGradSecond.(fieldsAverageGradInt{i});
    end

end