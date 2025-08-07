function [net,activeMainEffectIndex] = prune(net, X, T, numFeatures, numLayers, numerNets, trainingCheck, lossThreshold)
    
    loss = [];
    %%% The generalization of this function will be handled in the future.
    [sortedIndex, ~] = getRank(net,numFeatures, numerNets, "mainEffect");
    net.mainEffectSwitcher = dlarray(zeros(numFeatures,1));
    [valLoss, ~] = evaluate(net, X, T, numLayers, numFeatures, numerNets, trainingCheck);
    loss =  [loss, valLoss];

    for idx = 1:numFeatures
        selectedIdx = sortedIndex(1:idx);
        mainEffectSwitcher = dlarray(zeros(numFeatures,1));
        mainEffectSwitcher(selectedIdx) = 1;
        net.mainEffectSwitcher = mainEffectSwitcher;
        [valLoss, ~, ~] = evaluate(net, X, T, numLayers, numFeatures, numerNets, trainingCheck);
        loss = [loss, valLoss];
    end
       
    [lossBest, bestIdx] = min(loss);
    lossRange = max(loss) - min(loss);
    if lossRange > 0
        if sum(((loss - lossBest) / lossRange) < lossThreshold) > 0
            bestIndices = find(extractdata((loss - lossBest) / lossRange) < lossThreshold);
            bestIdx = bestIndices(1);
        end
    end
        
    activeMainEffectIndex = sortedIndex(1:bestIdx-1);
    mainEffectSwitcher = dlarray(zeros(numFeatures,1));
    mainEffectSwitcher(activeMainEffectIndex) = 1;
    net.mainEffectSwitcher = mainEffectSwitcher;

end