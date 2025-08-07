function [varianceFeatures, sortedIndex, prunedIndex] = varianceInt(numSamples, numFeatures, xOutputs)

    % Learnables = struct;
    varianceValues = zeros(1,numFeatures);
    varianceList = zeros(1,numFeatures);
    % meanY = mean(valY);
    xOutputs = xOutputs;

    for i = 1:numFeatures
        meanOutputs = mean(xOutputs(i,:));
        varianceValues(i) = sum((xOutputs(i,:) - meanOutputs).^2);
    end
    
    varianceList = varianceValues ./ (numSamples - 1);
    [varianceFeatures, sortedIndex] = sort(varianceList, 'descend');
    
    % Variance value threshold for Interactions
    threshold = 10^-6;
    % threshold = varianceFeatures(2);
    filter = varianceList >= threshold;
    [~, prunedIndex] = sort(varianceList(filter), 'descend');

end