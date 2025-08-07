function [varianceFeatures, sortedIndex, prunedIndex] = variance(numSamples, numFeatures, xOutputs)

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
    
    % Variance value threshold for Main Effects
    threshold = 10^-4;
    % threshold = varianceFeatures(6);       % For GAMI-Net function
    % threshold = varianceFeatures(7);        % For custom function
    filter = varianceList >= threshold;
    [~, prunedIndex] = sort(varianceList(filter), 'descend');

end