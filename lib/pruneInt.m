function [net,activeInteractionIndex] = pruneInt(net, netMain, X, T, numLayers, features, interactions, clarityParameter, numerNets, trainingCheck, numerNetsInt, trainingCheckInt, lossThreshold)
    
    loss = [];
    numInteractions = size(interactions,1);
    %%% The generalization of this function will be handled in the future.
    [sortedIndex, ~] = getRank(net,numInteractions,numerNetsInt,"interaction");
    net.interactionSwitcher = dlarray(zeros(numInteractions,1));
    [valLoss, ~, ~] = evaluateInt(net, netMain, X, T, numLayers,features, interactions, clarityParameter, numerNets, trainingCheck, numerNetsInt, trainingCheckInt);
    loss =  [loss, valLoss];

    for idx = 1:numInteractions
        selectedIdx = sortedIndex(1:idx);
        interactionSwitcher = dlarray(zeros(numInteractions,1));
        interactionSwitcher(selectedIdx) = 1;
        net.interactionSwitcher = interactionSwitcher;
        [valLoss, ~, ~] = evaluateInt(net, netMain, X, T, numLayers, features, interactions, clarityParameter, numerNets, trainingCheck, numerNetsInt, trainingCheckInt);
        loss = [loss, valLoss];
    end
       
    [lossBest, bestIdx]= min(loss);
    lossRange = max(loss) - min(loss);
    if lossRange > 0
        if sum(((loss - lossBest) / lossRange) < lossThreshold) > 0
            bestIndices = find(extractdata((loss - lossBest) / lossRange) < lossThreshold);
            bestIdx = bestIndices(1);
        end
    end
        
    activeInteractionIndex = sortedIndex(1:bestIdx-1);
    interactionSwitcher = dlarray(zeros(numInteractions,1));
    interactionSwitcher(activeInteractionIndex) = 1;
    net.interactionSwitcher = interactionSwitcher;

end